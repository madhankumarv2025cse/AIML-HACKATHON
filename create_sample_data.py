import os
import pandas as pd

def generate_sample_dataset():
    os.makedirs("dataset/train", exist_ok=True)
    os.makedirs("dataset/test", exist_ok=True)
    os.makedirs("utils", exist_ok=True)

    # Train Source 1
    train_s1_data = [
        {"entity_id": "S1-001", "business_name": "Acme Global Solutions Ltd", "business_address": "100 Main St, Suite 200", "country": "USA"},
        {"entity_id": "S1-002", "business_name": "Infosys Technology Pvt Ltd", "business_address": "44 Electronics City, Hosur Road", "country": "India"},
        {"entity_id": "S1-003", "business_name": "L'Oreal Beauty Group SA", "business_address": "41 Rue Martre, Clichy", "country": "France"},
        {"entity_id": "S1-004", "business_name": "Lone Star Diner Inc", "business_address": "500 Texas Ave", "country": "USA"}
    ]

    # Train Source 2
    train_s2_data = [
        {"entity_id": "S2-001", "business_name": "Acme Global Solutions", "business_address": "100 Main Street Ste 200", "country": "United States"},
        {"entity_id": "S2-002", "business_name": "Infosys Tech Private Limited", "business_address": "44 Electronics City Hosur Rd", "country": "India"},
        {"entity_id": "S2-003", "business_name": "L'Oreal SA", "business_address": "41 Rue Martre", "country": "France"},
        {"entity_id": "S2-004", "business_name": "Random Unmatched Corp", "business_address": "999 Nowhere Rd", "country": "USA"}
    ]

    # Train Source 3
    train_s3_data = [
        {"entity_id": "S3-001", "business_name": "Acme Global Inc.", "business_address": "100 Main St #200", "country": "USA"},
        {"entity_id": "S3-002", "business_name": "Infosys Ltd", "business_address": "Electronics City Hosur Road", "country": "India"},
        {"entity_id": "S3-003", "business_name": "L'Oreal France", "business_address": "Clichy Rue Martre 41", "country": "France"}
    ]

    # Ground Truth
    train_gt_data = [
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-001,S3-001"},
        {"source1_entity_id": "S1-002", "matched_entity_ids": "S2-002,S3-002"},
        {"source1_entity_id": "S1-003", "matched_entity_ids": "S2-003,S3-003"},
        {"source1_entity_id": "S1-004", "matched_entity_ids": ""}
    ]

    # Test Datasets
    test_s1_data = [
        {"entity_id": "S1-101", "business_name": "Apex Tech Solutions LLC", "business_address": "123 Innovation Blvd", "country": "USA"},
        {"entity_id": "S1-102", "business_name": "Tata Consultancy Services Pvt Ltd", "business_address": "Bandra Kurla Complex", "country": "India"},
        {"entity_id": "S1-103", "business_name": "Bistro de Paris SARL", "business_address": "15 Rue de Rivoli, Paris", "country": "France"},
        {"entity_id": "S1-104", "business_name": "Unknown Solitary Enterprise", "business_address": "777 Secret Pass", "country": "USA"}
    ]

    test_s2_data = [
        {"entity_id": "S2-101", "business_name": "Apex Tech Solutions", "business_address": "123 Innovation Blvd", "country": "USA"},
        {"entity_id": "S2-102", "business_name": "TCS Private Ltd", "business_address": "BKC Mumbai", "country": "India"},
        {"entity_id": "S2-103", "business_name": "Bistro de Paris", "business_address": "15 Rue de Rivoli", "country": "France"}
    ]

    test_s3_data = [
        {"entity_id": "S3-101", "business_name": "Apex Tech Inc", "business_address": "123 Innovation Boulevard", "country": "USA"},
        {"entity_id": "S3-102", "business_name": "Tata Consultancy Services", "business_address": "Bandra Kurla Complex Mumbai", "country": "India"},
        {"entity_id": "S3-103", "business_name": "Paris Bistro SARL", "business_address": "15 Rue Rivoli Paris", "country": "France"}
    ]

    pd.DataFrame(train_s1_data).to_csv("dataset/train/train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(train_s2_data).to_csv("dataset/train/train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(train_s3_data).to_csv("dataset/train/train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(train_gt_data).to_csv("dataset/train/train_ground_truth.tsv", sep="\t", index=False)

    pd.DataFrame(test_s1_data).to_csv("dataset/test/test_source1.tsv", sep="\t", index=False)
    pd.DataFrame(test_s2_data).to_csv("dataset/test/test_source2.tsv", sep="\t", index=False)
    pd.DataFrame(test_s3_data).to_csv("dataset/test/test_source3.tsv", sep="\t", index=False)

    print("[+] Sample dataset generated in dataset/train and dataset/test")

def generate_sample_validator():
    validator_code = '''import sys
import argparse
import pandas as pd

def validate(matching_path, candidate_path, test_dir):
    print("[*] Validating submission format...")
    df_s1 = pd.read_csv(f"{test_dir}/test_source1.tsv", sep="\\t")
    df_match = pd.read_csv(matching_path, sep="\\t")
    df_cand = pd.read_csv(candidate_path, sep="\\t")

    assert list(df_match.columns) == ["source1_entity_id", "matched_entity_ids"], f"Invalid columns in matching: {df_match.columns}"
    assert list(df_cand.columns) == ["source1_entity_id", "candidate_entity_ids"], f"Invalid columns in candidate: {df_cand.columns}"

    assert len(df_match) == len(df_s1), f"Matching row count mismatch: {len(df_match)} vs {len(df_s1)}"
    assert len(df_cand) == len(df_s1), f"Candidate row count mismatch: {len(df_cand)} vs {len(df_s1)}"

    s1_ids = set(df_s1["entity_id"])
    assert set(df_match["source1_entity_id"]) == s1_ids, "Matching S1 IDs do not match test_source1"
    assert set(df_cand["source1_entity_id"]) == s1_ids, "Candidate S1 IDs do not match test_source1"

    for _, row in df_match.iterrows():
        matches = [x.strip() for x in str(row["matched_entity_ids"]).split(",") if x.strip() and x.strip() != "nan"]
        for m in matches:
            assert m.startswith("S2-") or m.startswith("S3-"), f"Invalid matched entity ID prefix: {m}"

    for idx, row in df_cand.iterrows():
        s1_id = row["source1_entity_id"]
        cands = set(x.strip() for x in str(row["candidate_entity_ids"]).split(",") if x.strip() and x.strip() != "nan")
        match_row = df_match[df_match["source1_entity_id"] == s1_id].iloc[0]
        matches = set(x.strip() for x in str(match_row["matched_entity_ids"]).split(",") if x.strip() and x.strip() != "nan")
        assert matches.issubset(cands), f"Match {matches - cands} not present in candidates for {s1_id}"

    print("[✓] Submission structure & contents validation passed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()
    validate(args.matching, args.candidate, args.test_dir)
'''
    with open("utils/validate_submission.py", "w") as f:
        f.write(validator_code)
    print("[+] Submission validator created in utils/validate_submission.py")

if __name__ == "__main__":
    generate_sample_dataset()
    generate_sample_validator()
