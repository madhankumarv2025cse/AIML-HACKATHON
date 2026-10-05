import os
import gc
import numpy as np
import pandas as pd
from typing import Dict, List
from src.features import compute_pair_features_fast

def predict_test_set(
    clf: object,
    threshold: float,
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    candidates: Dict[str, List[str]],
    output_dir: str = "output",
    chunk_size: int = 200000
):
    """
    Predict matches for test set and write matching_results.tsv and candidate_pairs.tsv in memory-friendly chunks.
    """
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print("   [Predict] Indexing test target records...")
    dict_s1 = df_s1.set_index('entity_id').to_dict('index')
    df_targets = pd.concat([df_s2[['entity_id', 'norm_name', 'norm_address', 'norm_country', 'postal_code']],
                          df_s3[['entity_id', 'norm_name', 'norm_address', 'norm_country', 'postal_code']]], 
                         ignore_index=True)
    dict_targets = df_targets.set_index('entity_id').to_dict('index')

    s1_ids = list(df_s1['entity_id'])
    total_s1 = len(s1_ids)

    # Open files for streaming TSV output
    with open(matching_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_path, 'w', encoding='utf-8') as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for start_i in range(0, total_s1, chunk_size):
            end_i = min(start_i + chunk_size, total_s1)
            chunk_s1_ids = s1_ids[start_i:end_i]
            print(f"   [Predict] Processing test chunk {start_i:,} to {end_i:,} of {total_s1:,}...")

            for s1_id in chunk_s1_ids:
                s1_row = dict_s1[s1_id]
                cand_set = candidates.get(s1_id, set())

                sorted_cand_list = sorted(list(cand_set))
                cand_str = ",".join(sorted_cand_list)
                f_cand.write(f"{s1_id}\t{cand_str}\n")

                if not sorted_cand_list:
                    f_match.write(f"{s1_id}\t\n")
                    continue

                feat_list = []
                valid_cands = []
                for cand_id in sorted_cand_list:
                    if cand_id in dict_targets:
                        feat = compute_pair_features_fast(s1_row, dict_targets[cand_id], cand_id)
                        feat_list.append(feat)
                        valid_cands.append(cand_id)

                if not feat_list:
                    f_match.write(f"{s1_id}\t\n")
                    continue

                X_chunk = np.array(feat_list, dtype=np.float32)
                probs = clf.predict_proba(X_chunk)[:, 1]

                matched_cands = [cand_id for cand_id, prob in zip(valid_cands, probs) if prob >= threshold]
                matched_cands = sorted(matched_cands)
                matched_str = ",".join(matched_cands)

                f_match.write(f"{s1_id}\t{matched_str}\n")

            gc.collect()

    print(f"[+] Saved matching results to: {matching_path}")
    print(f"[+] Saved candidate pairs to: {candidate_path}")
