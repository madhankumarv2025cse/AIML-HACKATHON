# Business Entity Resolution Pipeline

A lightweight, reproducible, and efficient Machine Learning pipeline for Business Entity Resolution across 3 independent data sources (Source 1 reference, Source 2 noisy, Source 3 noisy).

## Architecture Overview

1. **Preprocessing (`src/preprocess.py`)**:
   - Lowercasing, Unicode NFKD diacritics stripping.
   - Business name normalization (& to and, punctuation removal, legal entity suffix stripping e.g., Ltd, Corp, LLC, Inc, SA, SARL).
   - Address normalization (abbreviation mapping for St/Rd/Ave/Blvd, numeric token extraction, postal code extraction).
   - Open-set country string normalization (works on any country including US, India, France, etc.).

2. **Candidate Generation / Blocking (`src/blocking.py`)**:
   - Multi-rule blocking (token indexing, char 4-gram prefix indexing, postal code exact matching).
   - Candidate UNION strategy per Source 1 entity to maximize candidate recall while avoiding quadratic comparison complexity.

3. **Feature Engineering (`src/features.py`)**:
   - Name similarity: Exact match, RapidFuzz ratio, partial ratio, token sort/set ratio, char n-gram Jaccard, token Jaccard, token containment, length diff/ratio, first word match.
   - Address similarity: Exact match, RapidFuzz ratio, token set ratio, char n-gram Jaccard, token Jaccard, address number Jaccard, postal code match.
   - Other: Country match indicator, target source indicators (S2 vs S3), combined heuristic score.

4. **Machine Learning Model & Threshold Optimization (`src/train.py`)**:
   - Model: `HistGradientBoostingClassifier` (lightweight, fast, handles non-linear interactions).
   - Validation: Entity-level split (80/20) with official Macro F0.5 metric evaluation across threshold grid `[0.45, 0.50, ..., 0.90]`.
   - Re-trains final classifier on 100% of training data using the optimal Macro F0.5 probability threshold.

5. **Test Prediction & Output Generation (`src/predict.py`)**:
   - Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv` strictly conforming to required schemas.

## Requirements

```bash
pip install -r requirements.txt
```

## Running the Pipeline

To run training, threshold optimization, test prediction, and submission validation:

```bash
python main.py
```

## Generated Outputs

- `output/matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`
- `output/candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`
