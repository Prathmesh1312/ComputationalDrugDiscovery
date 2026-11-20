"""
pareto_scorer.py — Multi-Objective Derringer-Suich Desirability & NSGA-II Pareto Frontier Engine
"""
from typing import Any, Dict, List, Optional

import numpy as np


class MultiObjectiveScorer:
    """
    Computes Composite Desirability Scores using the Derringer-Suich geometric mean formula:
        D(x) = (d1^w1 * d2^w2 * ... * dn^wn)^(1 / sum(w))

    Supports configurable `target_mode`:
      - "peripheral" (default for non-CNS drugs): maximizes bioavailability, safety (1-hERG),
        QED, synthetic ease, and low BBB penetration (1-BBB) to avoid CNS side effects.
      - "cns": favors high BBB penetration (BBB).
      - "neutral": optimizes the 4 core non-BBB objectives (bioavailability, hERG, QED, SAScore).
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, target_mode: str = "neutral"):
        self.target_mode = target_mode
        if weights is not None:
            self.weights = weights
        elif target_mode == "peripheral":
            self.weights = {
                "bioavailability": 1.0,
                "safety_herg": 1.5,
                "drug_likeness_qed": 1.0,
                "synthetic_accessibility": 0.8,
                "bbb_profile": 0.4,
            }
        elif target_mode == "cns":
            self.weights = {
                "bioavailability": 1.0,
                "safety_herg": 1.5,
                "drug_likeness_qed": 1.0,
                "synthetic_accessibility": 0.8,
                "bbb_profile": 0.8,
            }
        else:
            self.weights = {
                "bioavailability": 1.0,
                "safety_herg": 1.5,
                "drug_likeness_qed": 1.0,
                "synthetic_accessibility": 0.8,
            }

    def compute_composite_desirability(self, candidate_profile: Dict[str, float]) -> float:
        """Map raw ADMET objectives to normalized desirabilities in [0, 1] and compute geometric mean."""
        f_val = candidate_profile.get("bioavailability", 50.0)
        d_bio = float(np.clip(f_val / 100.0, 0.0, 1.0))

        herg_prob = candidate_profile.get("herg", 0.5)
        d_safety = float(np.clip(1.0 - herg_prob, 0.0, 1.0))

        qed = candidate_profile.get("qed", 0.5)
        d_qed = float(np.clip(qed, 0.0, 1.0))

        sascore = candidate_profile.get("sascore", 3.0)
        d_sa = float(np.clip((10.0 - sascore) / 9.0, 0.0, 1.0))

        bbb_prob = candidate_profile.get("bbb", 0.5)
        d_bbb = (
            float(np.clip(bbb_prob, 0.0, 1.0))
            if self.target_mode == "cns"
            else float(np.clip(1.0 - bbb_prob, 0.0, 1.0))
        )

        desirabilities = {
            "bioavailability": d_bio,
            "safety_herg": d_safety,
            "drug_likeness_qed": d_qed,
            "synthetic_accessibility": d_sa,
            "bbb_profile": d_bbb,
        }

        log_sum = 0.0
        weight_sum = 0.0
        for k, w in self.weights.items():
            val = desirabilities.get(k, 0.5)
            log_sum += w * np.log(max(val, 1e-4))
            weight_sum += w

        composite_score = np.exp(log_sum / max(weight_sum, 1e-5))
        return float(composite_score)


class ParetoFrontier:
    """
    Order-independent Pareto Non-Dominated Sorting & Frontier Filtering across ADMET objectives.
    """

    @staticmethod
    def is_pareto_optimal(costs: np.ndarray) -> np.ndarray:
        """
        Find Pareto-optimal points where ALL objectives are MINIMIZED.
        Order-independent and safe against duplicate rows.
        """
        num_points = costs.shape[0]
        is_optimal = np.ones(num_points, dtype=bool)
        for i in range(num_points):
            dominated_by_any = np.any(
                np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
            )
            if dominated_by_any:
                is_optimal[i] = False
        return is_optimal

    @staticmethod
    def _build_cost_matrix(
        candidates: List[Dict[str, Any]], target_mode: str = "neutral"
    ) -> np.ndarray:
        cost_matrix = []
        for c in candidates:
            row = [
                -float(c.get("bioavailability", 0.0)),  # Maximize -> minimize -Bio
                float(c.get("herg", 1.0)),              # Minimize hERG hazard
                -float(c.get("qed", 0.0)),              # Maximize -> minimize -QED
                float(c.get("sascore", 10.0)),          # Minimize SAScore
            ]
            if target_mode == "peripheral":
                row.append(float(c.get("bbb", 0.5)))    # Minimize unwanted BBB penetration
            elif target_mode == "cns":
                row.append(-float(c.get("bbb", 0.5)))   # Maximize BBB penetration
            cost_matrix.append(row)
        return np.asarray(cost_matrix, dtype=np.float64)

    @classmethod
    def filter_pareto_candidates(
        cls, candidates: List[Dict[str, Any]], target_mode: str = "neutral"
    ) -> List[Dict[str, Any]]:
        """Return Front-1 non-dominated Pareto-optimal candidates."""
        if not candidates:
            return []
        costs = cls._build_cost_matrix(candidates, target_mode=target_mode)
        mask = cls.is_pareto_optimal(costs)
        return [candidates[i] for i in range(len(candidates)) if mask[i]]

    @classmethod
    def assign_pareto_ranks(
        cls, candidates: List[Dict[str, Any]], target_mode: str = "neutral"
    ) -> List[Dict[str, Any]]:
        """
        NSGA-II fast non-dominated peeling: assigns `pareto_rank` (1 = Pareto Front, 2, 3...)
        to every candidate dictionary in place and returns the list.
        """
        if not candidates:
            return candidates

        costs = cls._build_cost_matrix(candidates, target_mode=target_mode)
        remaining = np.arange(len(candidates))
        rank = 1

        while len(remaining) > 0:
            sub_costs = costs[remaining]
            front_mask = cls.is_pareto_optimal(sub_costs)
            for idx in remaining[front_mask]:
                candidates[int(idx)]["pareto_rank"] = rank
            remaining = remaining[~front_mask]
            rank += 1

        return candidates
