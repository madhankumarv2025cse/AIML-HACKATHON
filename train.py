import gc
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features import compute_pair_features_fast, FEATURE_NAMES
from src.evaluate import evaluate_predictions

def parse_ground_truth(df_gt: pd.DataFrame) -> Dict[str, Set[str]]:
    """Fast vectorized ground truth parsing into map: source1_entity_id -> set of matched_entity_ids."""
    gt_map = {}
    s1_ids = df_gt['source1_entity_id'].astype(str)
    matched_strs = df_gt['matched_entity_ids'].fillna('').astype(str)
    for s1_id, matched_str in zip(s1_ids, matched_strs):
        if matched_str and matched_str != "nan":
            gt_map[s1_id] = set(x for x in matched_str.split(',') if x)
        else:
            gt_map[s1_id] = set()
    return gt_map

def build_training_dataset_fast(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    candidates: Dict[str, List[str]],
    gt_map: Dict[str, Set[str]],
    max_samples: int = 1000000
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[Tuple[str, int]]]]:
    """
    Construct high-performance numpy feature array X and binary labels y.
    """
    selected_candidates = {}
    selected_s1_ids = set()
    selected_target_ids = set()
    remaining_samples = max_samples
    for s1_id, candidate_ids in candidates.items():
        if remaining_samples <= 0:
            break
        selected_ids = candidate_ids[:remaining_samples]
        if not selected_ids:
            continue
        selected_candidates[s1_id] = selected_ids
        selected_s1_ids.add(s1_id)
        selected_target_ids.update(selected_ids)
        remaining_samples -= len(selected_ids)

    print(
        f"   [Train] Indexing {len(selected_s1_ids):,} S1 and "
        f"{len(selected_target_ids):,} target rows for the {max_samples:,}-pair budget...",
        flush=True,
    )
    dict_s1 = (
        df_s1[df_s1['entity_id'].isin(selected_s1_ids)]
        .set_index('entity_id')
        .to_dict('index')
    )
    df_targets = pd.concat([
        df_s2[['entity_id', 'norm_name', 'norm_address', 'norm_country', 'postal_code']],
        df_s3[['entity_id', 'norm_name', 'norm_address', 'norm_country', 'postal_code']],
    ], ignore_index=True)
    df_targets = df_targets[df_targets['entity_id'].isin(selected_target_ids)]
    dict_targets = df_targets.set_index('entity_id').to_dict('index')

    feature_rows = []
    labels = []
    s1_to_pair_indices = {}

    print("   [Train] Extracting pair features for training candidates...", flush=True)
    for s1_id, cand_set in selected_candidates.items():
        if s1_id not in dict_s1:
            continue
        row_s1 = dict_s1[s1_id]
        gt_matches = gt_map.get(s1_id, set())

        s1_pairs = []
        for cand_id in cand_set:
            if cand_id not in dict_targets:
                continue
            row_cand = dict_targets[cand_id]
            feats = compute_pair_features_fast(row_s1, row_cand, cand_id)
            label = 1 if cand_id in gt_matches else 0

            curr_idx = len(feature_rows)
            feature_rows.append(feats)
            labels.append(label)
            s1_pairs.append((cand_id, curr_idx))

        s1_to_pair_indices[s1_id] = s1_pairs

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    del dict_s1, df_targets, dict_targets, feature_rows, labels
    gc.collect()

    return X, y, s1_to_pair_indices

def train_and_optimize(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    candidates: Dict[str, List[str]],
    gt_map: Dict[str, Set[str]]
) -> Tuple[object, float, float]:
    """
    Train HistGradientBoostingClassifier and optimize probability threshold for Macro F0.5.
    """
    X, y, s1_to_pair_indices = build_training_dataset_fast(
        df_s1, df_s2, df_s3, candidates, gt_map
    )

    s1_ids = list(s1_to_pair_indices.keys())
    np.random.seed(42)
    np.random.shuffle(s1_ids)

    # 80/20 Entity Split for validation
    split_idx = int(0.8 * len(s1_ids))
    train_s1_ids = set(s1_ids[:split_idx])
    val_s1_ids = set(s1_ids[split_idx:])

    train_indices = []
    val_indices = []
    for s1_id in train_s1_ids:
        train_indices.extend([idx for _, idx in s1_to_pair_indices[s1_id]])
    for s1_id in val_s1_ids:
        val_indices.extend([idx for _, idx in s1_to_pair_indices[s1_id]])

    print(f"   [Train] Training validation model (Train pairs: {len(train_indices):,}, Val pairs: {len(val_indices):,})...", flush=True)
    X_train, y_train = X[train_indices], y[train_indices]
    X_val = X[val_indices]

    clf_val = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=6,
        early_stopping=True,
        random_state=42
    )
    clf_val.fit(X_train, y_train)

    val_probs = clf_val.predict_proba(X_val)[:, 1] if len(val_indices) > 0 else np.array([])
    val_idx_to_prob = dict(zip(val_indices, val_probs))
    val_gt_map = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}

    best_threshold = 0.70
    best_f05 = -1.0
    threshold_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    print("\n--- Validation Threshold Optimization ---", flush=True)
    for th in threshold_grid:
        val_preds_map = {}
        for s1_id in val_s1_ids:
            pred_matches = set()
            for cand_id, row_idx in s1_to_pair_indices[s1_id]:
                if val_idx_to_prob.get(row_idx, 0.0) >= th:
                    pred_matches.add(cand_id)
            val_preds_map[s1_id] = pred_matches

        f05 = evaluate_predictions(val_gt_map, val_preds_map)
        print(f"   Threshold: {th:.2f} -> Validation Macro F0.5: {f05:.4f}", flush=True)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = th

    print(f"\n[+] Selected Optimal Threshold: {best_threshold:.2f} (Macro F0.5: {best_f05:.4f})", flush=True)

    print("\n   [Train] Retraining final model on 100% of candidate dataset...", flush=True)
    clf_full = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=6,
        early_stopping=True,
        random_state=42
    )
    clf_full.fit(X, y)

    return clf_full, best_threshold, best_f05
