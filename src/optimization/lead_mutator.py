"""
lead_mutator.py — MedChem SMIRKS + BRICS Evolutionary Analogue Generator for Lead Optimization
"""
from typing import Any, Callable, Dict, List, Set

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

from ..utils.chemistry import (
    compute_lipinski_and_pains,
    compute_tanimoto_similarity,
    standardize_smiles,
)


class MolecularLeadMutator:
    """
    Applies medicinal chemistry transformations and BRICS fragment recombination to a seed lead
    molecule to explore chemically valid, synthesizable local chemical space.
    """

    # All deleted hydrogens / leaving atoms are unmapped so RDKit executes cleanly with zero warnings
    REACTION_SMIRKS = [
        # Aromatic C-H substitutions
        "[c:1][H]>>[c:1]F",
        "[c:1][H]>>[c:1]Cl",
        "[c:1][H]>>[c:1]C",
        "[c:1][H]>>[c:1]O",
        "[c:1][H]>>[c:1]OC",
        "[c:1][H]>>[c:1]C(F)(F)F",
        "[c:1][H]>>[c:1]N",
        "[c:1][H]>>[c:1]C#N",
        "[c:1][H]>>[c:1]C(=O)N",
        "[c:1][H]>>[c:1]S(=O)(=O)C",
        "[c:1][H]>>[c:1]CC",
        "[c:1][H]>>[c:1]OC(F)(F)F",
        # Aza-scan (aromatic CH -> pyridyl N)
        "[c:1][c:2]([H])[c:3]>>[c:1][n:2][c:3]",
        # Halogen swaps
        "[c:1]Cl>>[c:1]F",
        "[c:1]Br>>[c:1]F",
        # Carboxylic acid bioisosteres & esterification
        "[C:1](=O)[O:2][H]>>[C:1](=O)[O:2]C",
        "[C:1](=O)O[H]>>[C:1](=O)N",
        "[C:1](=O)O[H]>>[C:1]1=NNN=N1",
        # Aliphatic C-H / methyl / amine modifications
        "[C:1]([H])([H])[H]>>[C:1](F)(F)F",
        "[C:1]([H])([H])[H]>>[C:1]C",
        "[C;X4:1][H]>>[C:1]O",
        "[C;X4:1][H]>>[C:1]F",
        "[N:1]([H])[H]>>[N:1]C(=O)C",
    ]

    def __init__(self):
        self.reactions = []
        for smirks in self.REACTION_SMIRKS:
            rxn = AllChem.ReactionFromSmarts(smirks)
            if rxn is not None:
                self.reactions.append(rxn)

    def generate_brics_analogues(self, seed_smiles: str, max_brics: int = 10) -> List[str]:
        """Generate fragment-recombined analogues via RDKit BRICS decomposition/reassembly."""
        mol = Chem.MolFromSmiles(seed_smiles)
        if mol is None:
            return []
        try:
            frags = sorted(BRICS.BRICSDecompose(mol))
            if len(frags) < 2:
                return []
            frag_mols = [Chem.MolFromSmiles(f) for f in frags[:6]]
            frag_mols = [m for m in frag_mols if m is not None]
            if len(frag_mols) < 2:
                return []

            out: Set[str] = set()
            seed_heavy = mol.GetNumHeavyAtoms()
            for rebuilt in BRICS.BRICSBuild(frag_mols):
                try:
                    Chem.SanitizeMol(rebuilt)
                    if not (max(4, seed_heavy - 6) <= rebuilt.GetNumHeavyAtoms() <= seed_heavy + 12):
                        continue
                    smi = Chem.MolToSmiles(rebuilt, canonical=True, isomericSmiles=False)
                    std = standardize_smiles(smi, min_heavy=4, max_heavy=80)
                    if std is not None:
                        out.add(std)
                    if len(out) >= max_brics * 2:
                        break
                except Exception:
                    continue
            return sorted(out)[:max_brics]
        except Exception:
            return []

    def generate_analogues(
        self,
        seed_smiles: str,
        max_analogues: int = 30,
        include_brics: bool = True,
        filter_medchem: bool = True,
        min_similarity: float = 0.35,
        reference_smiles: str = None,
    ) -> List[str]:
        """
        Generates deterministic, chemically valid analogues around `seed_smiles`.
        Applies optional PAINS + Lipinski filtering and Tanimoto similarity threshold.
        """
        seed_std = standardize_smiles(seed_smiles, min_heavy=2, max_heavy=80)
        if seed_std is None:
            return [seed_smiles]
        ref_smi = reference_smiles or seed_std

        mol = Chem.MolFromSmiles(seed_std)
        if mol is None:
            return [seed_std]

        mol_h = Chem.AddHs(mol)
        candidates: Set[str] = set()

        for rxn in self.reactions:
            try:
                products = rxn.RunReactants((mol_h,))
                for prod_tuple in products:
                    for prod in prod_tuple:
                        try:
                            prod_clean = Chem.RemoveHs(prod)
                            Chem.SanitizeMol(prod_clean)
                            smi = Chem.MolToSmiles(prod_clean, canonical=True, isomericSmiles=False)
                            std = standardize_smiles(smi, min_heavy=4, max_heavy=80)
                            if std is not None and std != seed_std:
                                candidates.add(std)
                        except Exception:
                            continue
            except Exception:
                continue

        if include_brics:
            for b_smi in self.generate_brics_analogues(seed_std, max_brics=max(6, max_analogues // 3)):
                if b_smi != seed_std:
                    candidates.add(b_smi)

        filtered: List[str] = []
        for smi in sorted(candidates):
            sim = compute_tanimoto_similarity(ref_smi, smi)
            if sim < min_similarity:
                continue
            if filter_medchem:
                mc = compute_lipinski_and_pains(smi)
                if not mc["pains_pass"] or not mc["lipinski_pass"]:
                    continue
            filtered.append(smi)

        # Always keep seed molecule first, followed by highest-similarity deterministic analogues
        filtered.sort(key=lambda s: (-compute_tanimoto_similarity(ref_smi, s), s))
        return [seed_std] + filtered[: max(0, max_analogues - 1)]

    def evolve_leads(
        self,
        seed_smiles: str,
        score_fn: Callable[[List[str]], List[Dict[str, Any]]],
        generations: int = 2,
        pop_size: int = 25,
        top_parents: int = 6,
        min_similarity: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Multi-generation evolutionary lead optimization:
          - Generation 1: mutate seed lead + BRICS fragments -> batch-score via `score_fn`
          - Generations 2..G: mutate top Pareto / desirability parents -> score unseen valid analogues
        """
        seed_std = standardize_smiles(seed_smiles, min_heavy=2, max_heavy=80) or seed_smiles
        seen: Set[str] = set()
        pool: List[Dict[str, Any]] = []

        gen1_smiles = self.generate_analogues(
            seed_std,
            max_analogues=pop_size,
            include_brics=True,
            filter_medchem=True,
            min_similarity=min_similarity,
            reference_smiles=seed_std,
        )
        seen.update(gen1_smiles)
        pool.extend(score_fn(gen1_smiles))

        for _gen in range(2, generations + 1):
            ranked = sorted(pool, key=lambda x: x.get("desirability", 0.0), reverse=True)
            parents = [p["smiles"] for p in ranked[:top_parents]]
            new_smiles: List[str] = []

            for p_smi in parents:
                kids = self.generate_analogues(
                    p_smi,
                    max_analogues=max(10, pop_size // 2),
                    include_brics=False,
                    filter_medchem=True,
                    min_similarity=min_similarity,
                    reference_smiles=seed_std,
                )
                for k in kids:
                    if k not in seen:
                        seen.add(k)
                        new_smiles.append(k)
                    if len(new_smiles) >= pop_size:
                        break
                if len(new_smiles) >= pop_size:
                    break

            if not new_smiles:
                break
            pool.extend(score_fn(new_smiles))

        return pool
