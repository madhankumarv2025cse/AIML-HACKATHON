import os
import sys
import subprocess
import time
import pandas as pd

from src.preprocess import normalize_name, normalize_address, normalize_country, extract_postal_code
from src.blocking import generate_candidates_for_sources
from src.train import parse_ground_truth, train_and_optimize
from src.predict import predict_test_set

def load_tsv(path: str) -> pd.DataFrame:
    """Load dataset using TSV format as specified."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required data file not found: {path}")
    print(f" -> Reading TSV: {path}...", flush=True)
    return pd.read_csv(path, sep="\t")

def run_pipeline(
    train_dir: str = "dataset/train",
    test_dir: str = "dataset/test",
    output_dir: str = "output"
):
    print("==================================================", flush=True)
    print("      BUSINESS ENTITY RESOLUTION PIPELINE         ", flush=True)
    print("==================================================", flush=True)

    # 1. Load Data
    print("\n[1/6] Loading training datasets...", flush=True)
    train_s1 = load_tsv(os.path.join(train_dir, "train_source1.tsv"))
    train_s2 = load_tsv(os.path.join(train_dir, "train_source2.tsv"))
    train_s3 = load_tsv(os.path.join(train_dir, "train_source3.tsv"))
    df_gt = load_tsv(os.path.join(train_dir, "train_ground_truth.tsv"))

    print(f" -> Train S1 shape: {train_s1.shape}", flush=True)
    print(f" -> Train S2 shape: {train_s2.shape}", flush=True)
    print(f" -> Train S3 shape: {train_s3.shape}", flush=True)

    print(" -> Parsing ground truth mapping...", flush=True)
    gt_map = parse_ground_truth(df_gt)
    print(f" -> Ground truth loaded for {len(gt_map):,} S1 entities.", flush=True)

    print("\n[2/6] Generating candidates (Blocking) on training data...", flush=True)
    train_blocking_started = time.perf_counter()
    train_candidates = generate_candidates_for_sources(train_s1, train_s2, train_s3)
    print(f" -> Training candidate generation time: {time.perf_counter() - train_blocking_started:.1f}s", flush=True)
    total_cand_pairs = sum(len(c) for c in train_candidates.values())
    print(f" -> Total candidate pairs generated: {total_cand_pairs:,}", flush=True)

    print("\n[3/6] Training ML Model & Optimizing Macro F0.5 Threshold...", flush=True)
    clf, best_threshold, best_f05 = train_and_optimize(
        train_s1, train_s2, train_s3, train_candidates, gt_map
    )
    print(f" -> Final Selected Threshold: {best_threshold:.2f}", flush=True)
    print(f" -> Best Validation Macro F0.5 Score: {best_f05:.4f}", flush=True)

    print("\n[4/6] Loading test datasets...", flush=True)
    test_s1 = load_tsv(os.path.join(test_dir, "test_source1.tsv"))
    test_s2 = load_tsv(os.path.join(test_dir, "test_source2.tsv"))
    test_s3 = load_tsv(os.path.join(test_dir, "test_source3.tsv"))

    print(f" -> Test S1 shape: {test_s1.shape}", flush=True)
    print(f" -> Test S2 shape: {test_s2.shape}", flush=True)
    print(f" -> Test S3 shape: {test_s3.shape}", flush=True)

    print("\n[5/6] Generating candidates & Predicting for Test set...", flush=True)
    test_blocking_started = time.perf_counter()
    test_candidates = generate_candidates_for_sources(test_s1, test_s2, test_s3)
    print(f" -> Test candidate generation time: {time.perf_counter() - test_blocking_started:.1f}s", flush=True)
    predict_test_set(
        clf=clf,
        threshold=best_threshold,
        df_s1=test_s1,
        df_s2=test_s2,
        df_s3=test_s3,
        candidates=test_candidates,
        output_dir=output_dir
    )

    print("\n[6/6] Validating submission outputs...", flush=True)
    validator_path = "utils/validate_submission.py"
    if os.path.exists(validator_path):
        cmd = [
            sys.executable,
            validator_path,
            "--matching", os.path.join(output_dir, "matching_results.tsv"),
            "--candidate", os.path.join(output_dir, "candidate_pairs.tsv"),
            "--test-dir", test_dir
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("Submission Validator Output:\n", res.stdout, flush=True)
        if res.stderr:
            print("Submission Validator Errors:\n", res.stderr, flush=True)
        if res.returncode == 0:
            print("SUCCESS: Submission validated successfully!", flush=True)
        else:
            print("WARNING: Submission validator exited with errors.", flush=True)
            raise SystemExit(res.returncode)
    else:
        print(f"Info: Validator script '{validator_path}' not found. Skipping validation.", flush=True)

    print("\n==================================================", flush=True)
    print("      PIPELINE COMPLETED SUCCESSFULLY             ", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    run_pipeline()
