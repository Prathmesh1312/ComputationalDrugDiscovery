"""
run_all.py — End-to-End Reproducible Pipeline Runner (Stages 0 -> 1 -> 2 -> 4 -> 3)
"""
import os
import subprocess
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable

STAGES = [
    ("Stage 0: Curate Datasets & Scaffold Split", [PYTHON, os.path.join(BASE_DIR, "scripts", "0_prepare_datasets.py")]),
    ("Stage 1: Self-Supervised GraphGPS Pre-Training", [PYTHON, os.path.join(BASE_DIR, "scripts", "1_pretrain_ssl.py"), "--epochs", "25", "--batch-size", "64"]),
    ("Stage 2a: Fine-Tune SSL-GraphGPS", [PYTHON, os.path.join(BASE_DIR, "scripts", "2_finetune_admet.py"), "--init", "ssl", "--epochs", "40", "--batch-size", "64"]),
    ("Stage 2b: Train GraphGPS-Scratch Ablation", [PYTHON, os.path.join(BASE_DIR, "scripts", "2_finetune_admet.py"), "--init", "scratch", "--epochs", "40", "--batch-size", "64"]),
    ("Stage 2c: Train Standard-GCN Baseline", [PYTHON, os.path.join(BASE_DIR, "scripts", "2_finetune_admet.py"), "--init", "gcn", "--epochs", "40", "--batch-size", "64"]),
    ("Stage 4: Evaluate & Generate Ablation Figures", [PYTHON, os.path.join(BASE_DIR, "scripts", "4_evaluate_and_ablation.py")]),
    ("Stage 3: Evolutionary Lead Optimization & XAI", [PYTHON, os.path.join(BASE_DIR, "scripts", "3_lead_optimization.py"), "--smiles", "CC(=O)Oc1ccccc1C(=O)O", "--generations", "2", "--mode", "peripheral", "--save-plots"]),
]


def main() -> None:
    overall_start = time.time()
    timings = []
    for name, cmd in STAGES:
        print(f"\n>>> Running {name}")
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=BASE_DIR)
        timings.append((name, time.time() - t0))

    print("\n" + "=" * 70)
    print("Pipeline Completed Successfully! Timing Summary:")
    print("=" * 70)
    for name, dt in timings:
        print(f"  {name:<48} {dt:>6.1f}s")
    print(f"  Total Wall Time:                                 {time.time() - overall_start:>6.1f}s")


if __name__ == "__main__":
    main()
