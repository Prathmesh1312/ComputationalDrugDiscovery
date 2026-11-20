"""
test_web_api.py — FastAPI Integration Tests for the 4-Tab ADMET GraphGPS Workbench UI
"""
import os
import sys
import unittest

from fastapi.testclient import TestClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from app import app


class TestWebWorkbenchAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._ctx = TestClient(app)
        cls.client = cls._ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_01_index_html_and_tabs_markup(self):
        """Verify GET / serves clean HTML containing all 4 accessible tabs and hidden=until-found."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        html = res.text
        self.assertEqual(html.count('role="tab"'), 4)
        self.assertEqual(html.count('role="tabpanel"'), 4)
        self.assertIn('hidden="until-found"', html)
        self.assertNotIn("aria-controls", html)

    def test_02_health_and_presets(self):
        """Verify /api/health and /api/presets return expected runtime metadata and 4 drugs."""
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)
        hj = h.json()
        self.assertEqual(hj["status"], "ok")
        self.assertGreater(hj["macro_roc_auc"], 0.80)

        p = self.client.get("/api/presets")
        self.assertEqual(p.status_code, 200)
        presets = p.json()["presets"]
        self.assertEqual(len(presets), 4)
        self.assertEqual(presets[0]["name"], "Aspirin")

    def test_03_predict_valid_and_invalid_smiles(self):
        """Verify /api/predict returns bounded ADMET predictions, 3D descriptors, and 422 on bad SMILES."""
        res = self.client.post("/api/predict", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        preds = data["predictions"]
        self.assertTrue(0.0 <= preds["bioavailability"] <= 100.0)
        self.assertTrue(0.0 <= preds["bbb"] <= 1.0)
        self.assertTrue(0.0 <= preds["herg"] <= 1.0)
        self.assertTrue(0.0 <= preds["qed"] <= 1.0)
        self.assertTrue(1.0 <= preds["sascore"] <= 10.0)
        self.assertTrue(data["image_data_uri"].startswith("data:image/png;base64,"))
        self.assertIn("npr1", data["geometry_3d"])

        bad = self.client.post("/api/predict", json={"smiles": "NOT_A_MOLECULE_###"})
        self.assertEqual(bad.status_code, 422)

    def test_04_saliency_attribution(self):
        """Verify /api/saliency returns normalized [0, 1] heavy-atom attributions and PNG image."""
        res = self.client.post(
            "/api/saliency",
            json={"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "task": "bbb"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["image_data_uri"].startswith("data:image/png;base64,"))
        self.assertGreaterEqual(len(data["top_atoms"]), 1)
        for atom in data["top_atoms"]:
            self.assertTrue(0.0 <= atom["attribution"] <= 1.0 + 1e-6)

    def test_05_optimize_pareto_frontier(self):
        """Verify /api/optimize runs evolutionary mutation and returns Pareto Front-1 candidates."""
        res = self.client.post(
            "/api/optimize",
            json={
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "generations": 1,
                "max_analogues": 20,
                "mode": "peripheral",
                "min_similarity": 0.35,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(data["total_candidates"], 5)
        self.assertGreaterEqual(data["pareto_front1_count"], 1)
        self.assertEqual(data["candidates"][0]["pareto_rank"], 1)
        self.assertTrue(data["top_cards"][0]["image_data_uri"].startswith("data:image/png;base64,"))

    def test_06_benchmarks_endpoint(self):
        """Verify /api/benchmarks returns all 4 evaluated models and figure paths."""
        res = self.client.get("/api/benchmarks")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("SSL-GraphGPS (Ours)", data["metrics_summary"])
        self.assertGreaterEqual(len(data["figures"]), 6)


if __name__ == "__main__":
    unittest.main()
