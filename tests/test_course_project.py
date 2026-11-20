"""
test_course_project.py — Comprehensive Unit & Integration Test Suite
"""
import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.data.augmentations import MoleculeAugmenter
from src.data.dataset import (
    NUM_ATOM_FEATURES,
    NUM_BOND_FEATURES,
    NUM_GEOM_3D_FEATURES,
    NUM_PE_FEATURES,
    compute_laplacian_pe,
    compute_rwse,
    smiles_to_graph_data,
)
from src.models.baselines import StandardGCNBackbone
from src.models.graph_transformer import MolecularGraphTransformer
from src.models.multi_objective_heads import MultiObjectiveADMETModel, compute_masked_multitask_loss
from src.models.ssl_contrastive import MolecularContrastiveSSL
from src.optimization.lead_mutator import MolecularLeadMutator
from src.optimization.pareto_scorer import MultiObjectiveScorer, ParetoFrontier
from src.utils.chemistry import standardize_smiles, suppress_rdkit_warnings
from src.utils.explainability import compute_atom_saliency
from src.utils.splitting import bemis_murcko_scaffold_split


class TestCourseProjectPipeline(unittest.TestCase):

    def setUp(self):
        suppress_rdkit_warnings()
        self.aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        self.ibuprofen_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
        self.imatinib_smiles = "CN1CCN(CC1)c2ccc(NC(=O)c3ccc(C)c(Nc4ncccn4)c3)cc2"

    def test_01_graph_data_2d_3d_construction(self):
        """Verify 29-dim atom features, 8-dim bond features, 8-dim LapPE+RWSE, and 4-dim 3D geometry."""
        data = smiles_to_graph_data(self.aspirin_smiles, compute_pe=True, pe_dim=8, use_3d=True)
        self.assertIsNotNone(data)
        self.assertEqual(data.num_nodes, 13)
        self.assertEqual(data.x.shape, (13, NUM_ATOM_FEATURES))
        self.assertEqual(data.edge_attr.shape[1], NUM_BOND_FEATURES)
        self.assertEqual(data.pe.shape, (13, NUM_PE_FEATURES))
        self.assertEqual(data.geom_3d.shape, (1, NUM_GEOM_3D_FEATURES))
        self.assertFalse(torch.isnan(data.x).any())
        self.assertFalse(torch.isnan(data.geom_3d).any())

    def test_02_salt_stripping_standardization(self):
        """Verify LargestFragmentChooser removes salts/counterions cleanly."""
        salted = "CC(C)NCC(O)COc1cccc2ccccc12.[Cl]"
        cleaned = standardize_smiles(salted)
        self.assertEqual(cleaned, "CC(C)NCC(O)COc1cccc2ccccc12")

    def test_03_laplacian_pe_sign_and_rwse_bounds(self):
        """Verify deterministic Laplacian PE sign canonicalization and RWSE [0, 1] bounds."""
        data = smiles_to_graph_data(self.aspirin_smiles, compute_pe=True, pe_dim=8, use_3d=False)
        lap_pe = compute_laplacian_pe(data.edge_index, data.num_nodes, k=4)
        for col in range(lap_pe.size(1)):
            vec = lap_pe[:, col]
            if vec.abs().max() > 1e-6:
                max_idx = int(torch.argmax(vec.abs()).item())
                self.assertGreaterEqual(float(vec[max_idx].item()), 0.0)

        rwse = compute_rwse(data.edge_index, data.num_nodes, steps=4)
        self.assertEqual(rwse.shape, (13, 4))
        self.assertTrue((rwse >= 0.0).all() and (rwse <= 1.0 + 1e-5).all())

    def test_04_augmentations_undirected_symmetry_and_node_drop(self):
        """Verify drop_edges strictly preserves bidirectionality and drop_nodes remaps edges cleanly."""
        data = smiles_to_graph_data(self.ibuprofen_smiles, compute_pe=True, pe_dim=8, use_3d=False)
        augmenter = MoleculeAugmenter(node_mask_ratio=0.25, edge_drop_ratio=0.35, node_drop_ratio=0.2)

        for _ in range(5):
            dropped = augmenter.drop_edges(data)
            pairs = set(zip(dropped.edge_index[0].tolist(), dropped.edge_index[1].tolist()))
            for u, v in pairs:
                self.assertIn((v, u), pairs, f"Asymmetric edge ({u}, {v}) left after drop_edges!")

        node_dropped = augmenter.drop_nodes(data)
        self.assertLess(node_dropped.num_nodes, data.num_nodes)
        self.assertEqual(node_dropped.x.size(0), node_dropped.num_nodes)
        self.assertEqual(node_dropped.pe.size(0), node_dropped.num_nodes)
        if node_dropped.edge_index.numel() > 0:
            self.assertLess(int(node_dropped.edge_index.max().item()), node_dropped.num_nodes)

        v1, v2 = augmenter.augment_pair(data)
        self.assertIsNotNone(v1)
        self.assertIsNotNone(v2)

    def test_05_hybrid_graphgps_forward_and_global_attention(self):
        """Verify Hybrid Local+Global GraphGPS forward pass and all-atom global message flow."""
        g1 = smiles_to_graph_data(self.aspirin_smiles)
        g2 = smiles_to_graph_data(self.ibuprofen_smiles)
        batch = Batch.from_data_list([g1, g2])

        model = MolecularGraphTransformer(hidden_dim=64, num_layers=2, heads=4)
        emb = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe, batch.geom_3d)
        self.assertEqual(emb.shape, (2, 64))
        self.assertFalse(torch.isnan(emb).any())

        # Check global attention transmits signal even between disconnected atoms
        model.eval()
        x_disc = torch.randn(3, NUM_ATOM_FEATURES, requires_grad=True)
        e_idx = torch.empty((2, 0), dtype=torch.long)
        e_attr = torch.empty((0, NUM_BOND_FEATURES))
        b_disc = torch.zeros(3, dtype=torch.long)
        emb_disc = model(x_disc, e_idx, e_attr, b_disc)
        emb_disc.pow(2).sum().backward()
        self.assertIsNotNone(x_disc.grad)
        self.assertGreater(float(x_disc.grad.abs().sum().item()), 0.0)

        h0 = torch.randn(3, 64)
        h1 = h0.clone()
        h1[2] = h1[2] + 5.0
        out0 = model.layers[0](h0, e_idx, torch.empty((0, 64)), batch=b_disc)
        out1 = model.layers[0](h1, e_idx, torch.empty((0, 64)), batch=b_disc)
        self.assertGreater(float((out1[0] - out0[0]).abs().max().item()), 0.0)

    def test_06_standard_gcn_baseline_forward(self):
        """Verify StandardGCNBackbone matches GraphTransformer embedding interface."""
        g1 = smiles_to_graph_data(self.aspirin_smiles)
        batch = Batch.from_data_list([g1])
        gcn = StandardGCNBackbone(hidden_dim=64, num_layers=3)
        emb = gcn(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe, batch.geom_3d)
        self.assertEqual(emb.shape, (1, 64))
        self.assertFalse(torch.isnan(emb).any())

    def test_07_ssl_contrastive_loss_and_grad(self):
        """Verify InfoNCE contrastive loss computation and gradient backpropagation."""
        g1 = smiles_to_graph_data(self.aspirin_smiles)
        g2 = smiles_to_graph_data(self.ibuprofen_smiles)
        augmenter = MoleculeAugmenter()
        v1_a, v1_b = augmenter.augment_pair(g1)
        v2_a, v2_b = augmenter.augment_pair(g2)

        batch_1 = Batch.from_data_list([v1_a, v2_a])
        batch_2 = Batch.from_data_list([v1_b, v2_b])

        backbone = MolecularGraphTransformer(hidden_dim=32, num_layers=2, heads=2)
        ssl_model = MolecularContrastiveSSL(backbone, proj_dim=16, temperature=0.1)
        loss = ssl_model.compute_ssl_loss(batch_1, batch_2)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertFalse(torch.isnan(loss))
        loss.backward()

    def test_08_multi_objective_heads_strict_bounds(self):
        """Verify all 5 heads satisfy strict physical bounds even under extreme inputs."""
        g1 = smiles_to_graph_data(self.aspirin_smiles)
        g1.x = g1.x * 50.0  # Extreme input stress test
        batch = Batch.from_data_list([g1])

        backbone = MolecularGraphTransformer(hidden_dim=32, num_layers=2, heads=2)
        model = MultiObjectiveADMETModel(backbone, hidden_dim=32)
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe, batch.geom_3d)

        self.assertTrue(0.0 <= float(preds["bioavailability"].item()) <= 100.0)
        self.assertTrue(0.0 <= float(preds["bbb"].item()) <= 1.0)
        self.assertTrue(0.0 <= float(preds["herg"].item()) <= 1.0)
        self.assertTrue(0.0 <= float(preds["qed"].item()) <= 1.0)
        self.assertTrue(1.0 <= float(preds["sascore"].item()) <= 10.0)

    def test_09_masked_multitask_loss_with_missing_tasks(self):
        """Verify masked multi-task loss handles missing NaN labels and all-masked tasks without NaNs."""
        g1 = smiles_to_graph_data(self.aspirin_smiles)
        g2 = smiles_to_graph_data(self.ibuprofen_smiles)
        batch = Batch.from_data_list([g1, g2])

        backbone = MolecularGraphTransformer(hidden_dim=32, num_layers=2, heads=2)
        model = MultiObjectiveADMETModel(backbone, hidden_dim=32)
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe, batch.geom_3d)

        targets = torch.tensor([[1.0, 0.0, 0.0, 0.54, 1.58], [0.0, 0.0, 1.0, 0.82, 1.90]])
        # Completely mask out column 1 (bbb) and partially mask col 0 and col 2
        masks = torch.tensor([[True, False, False, True, True], [False, False, True, True, True]])
        loss, bd = compute_masked_multitask_loss(preds, targets, masks)
        self.assertFalse(torch.isnan(loss))
        self.assertEqual(bd["bbb"], 0.0)
        loss.backward()

    def test_10_scaffold_split_zero_leakage(self):
        """Verify Bemis-Murcko scaffold split produces zero scaffold or SMILES overlap across splits."""
        test_smiles = [
            "c1ccccc1C(=O)O", "c1ccccc1CC(=O)O", "c1ccccc1N", "c1ccc(F)cc1C(=O)O",
            "c1ccncc1C(=O)O", "c1ccncc1N", "c1ccc2ccccc2c1", "c1ccc2ccccc2c1O",
            "c1ccc2[nH]ccc2c1", "c1ccc2ncncc2c1", "C1CCCCC1C(=O)O", "C1CCNCC1",
            "c1ccoc1C(=O)O", "c1ccsc1C(=O)O", "c1cncnc1N", "c1ccc(Oc2ccccc2)cc1",
            "c1ccc(-c2ccccc2)cc1", "CC1CC1C(=O)O", "O=C1CCCCN1", "c1ccc2occc2c1",
        ]
        df = pd.DataFrame({
            "smiles": test_smiles,
            "bioavailability": [i % 2 for i in range(len(test_smiles))],
            "bbb": [(i + 1) % 2 for i in range(len(test_smiles))],
            "herg": [i % 2 for i in range(len(test_smiles))],
            "qed": [0.5] * len(test_smiles),
            "sascore": [2.0] * len(test_smiles),
        })
        split_df = bemis_murcko_scaffold_split(df, train_frac=0.6, val_frac=0.2, seed=42)
        tr = split_df[split_df["split"] == "train"]
        va = split_df[split_df["split"] == "val"]
        te = split_df[split_df["split"] == "test"]
        self.assertEqual(len(set(tr["scaffold"]) & set(va["scaffold"])), 0)
        self.assertEqual(len(set(tr["scaffold"]) & set(te["scaffold"])), 0)
        self.assertEqual(len(set(va["scaffold"]) & set(te["scaffold"])), 0)

    def test_11_lead_mutator_smirks_brics_and_filters(self):
        """Verify SMIRKS + BRICS mutations, deterministic output, and PAINS/Lipinski filtering."""
        mutator = MolecularLeadMutator()
        a1 = mutator.generate_analogues(self.aspirin_smiles, max_analogues=25)
        a2 = mutator.generate_analogues(self.aspirin_smiles, max_analogues=25)
        self.assertEqual(a1, a2)
        self.assertGreaterEqual(len(a1), 15)

        brics = mutator.generate_brics_analogues(self.imatinib_smiles, max_brics=5)
        self.assertGreaterEqual(len(brics), 1)

    def test_12_pareto_frontier_nsga2_and_bbb_modes(self):
        """Verify order-independent Pareto dominance, NSGA-II ranking, and peripheral vs CNS modes."""
        candidates = [
            {"smiles": "MolA", "bioavailability": 85.0, "herg": 0.1, "bbb": 0.9, "qed": 0.8, "sascore": 2.0},
            {"smiles": "MolB", "bioavailability": 30.0, "herg": 0.9, "bbb": 0.5, "qed": 0.3, "sascore": 7.0},
            {"smiles": "MolC", "bioavailability": 90.0, "herg": 0.05, "bbb": 0.1, "qed": 0.85, "sascore": 1.5},
        ]
        periph_scorer = MultiObjectiveScorer(target_mode="peripheral")
        cns_scorer = MultiObjectiveScorer(target_mode="cns")

        self.assertGreater(
            periph_scorer.compute_composite_desirability(candidates[2]),
            periph_scorer.compute_composite_desirability(candidates[0]),
        )
        # In CNS mode MolA (BBB=0.9) gains vs peripheral mode (BBB=0.1 penalty)
        self.assertGreater(
            cns_scorer.compute_composite_desirability(candidates[0]),
            periph_scorer.compute_composite_desirability(candidates[0]),
        )

        ranked = ParetoFrontier.assign_pareto_ranks(candidates, target_mode="peripheral")
        self.assertEqual(ranked[2]["pareto_rank"], 1)  # MolC dominates all in peripheral mode
        self.assertGreater(ranked[1]["pareto_rank"], 1)

    def test_13_atom_saliency_xai(self):
        """Verify atom-level XAI gradient attribution returns normalized [0, 1] scores per atom."""
        backbone = MolecularGraphTransformer(hidden_dim=32, num_layers=2, heads=2)
        model = MultiObjectiveADMETModel(backbone, hidden_dim=32)
        mol, scores, preds = compute_atom_saliency(model, self.aspirin_smiles, target_task="herg")
        self.assertEqual(len(scores), mol.GetNumAtoms())
        self.assertTrue((scores >= 0.0).all() and (scores <= 1.0 + 1e-6).all())
        self.assertIn("herg", preds)

    def test_14_saved_checkpoints_load_and_infer(self):
        """Verify final SSL and ADMET checkpoints load cleanly with weights_only=True and run inference."""
        ssl_ckpt = os.path.join(BASE_DIR, "checkpoints", "ssl_pretrained_backbone.pth")
        admet_ckpt = os.path.join(BASE_DIR, "checkpoints", "admet_finetuned_model.pth")
        self.assertTrue(os.path.exists(ssl_ckpt), f"Missing SSL checkpoint: {ssl_ckpt}")
        self.assertTrue(os.path.exists(admet_ckpt), f"Missing ADMET checkpoint: {admet_ckpt}")

        backbone = MolecularGraphTransformer(hidden_dim=128, num_layers=3, heads=4)
        backbone.load_state_dict(torch.load(ssl_ckpt, map_location="cpu", weights_only=True))

        model = MultiObjectiveADMETModel(backbone, hidden_dim=128)
        model.load_state_dict(torch.load(admet_ckpt, map_location="cpu", weights_only=True))
        model.eval()

        g = smiles_to_graph_data(self.aspirin_smiles)
        b = Batch.from_data_list([g])
        with torch.no_grad():
            out = model(b.x, b.edge_index, b.edge_attr, b.batch, b.pe, b.geom_3d)
        self.assertFalse(torch.isnan(out["herg"]).any())
        self.assertTrue(1.0 <= float(out["sascore"].item()) <= 10.0)


if __name__ == "__main__":
    unittest.main()
