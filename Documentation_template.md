# Business Entity Resolution Methodology Documentation

## 1. Methodology Overview
The Business Entity Resolution system matches reference entities from Source 1 against noisy records from Source 2 and Source 3. The pipeline employs an end-to-end machine learning framework consisting of string normalization, inverted-index candidate blocking, feature extraction, gradient boosted tree classification, and entity-level Macro F0.5 threshold optimization.

## 2. Candidate Generation / Blocking Strategy
To eliminate quadratic pair evaluations \(O(N_1 \times (N_2 + N_3))\), candidate generation uses multi-index blocking:
- **Token Indexing**: Inverted index on sanitized name tokens of length \(\ge 3\).
- **Prefix Indexing**: Character 4-gram prefix indexing on normalized business names.
- **Postal/PIN Indexing**: Exact postal code matching when numbers are present.
- **Union Strategy**: Candidate sets from all blocking rules are merged per Source 1 entity, maintaining high recall while reducing candidate pairs per entity to a small tractable set.

## 3. Preprocessing & Normalization
- **Business Names**: Lowercasing, Unicode NFKD accent stripping, `&` to `and` standardization, non-alphanumeric character stripping, and legal suffix removal (`ltd`, `pvt`, `corp`, `inc`, `llc`, `sa`, `sarl`, `gmbh`, etc.).
- **Addresses**: Address abbreviation mapping (`road` \(\rightarrow\) `rd`, `street` \(\rightarrow\) `st`, `avenue` \(\rightarrow\) `ave`, etc.), number token isolation, and postal code extraction.
- **Country**: Open-set string normalization ensuring generalization across US, India, France, and unseen countries.

## 4. Feature Engineering
For each candidate pair \((S1, S2/S3)\), 26 discriminative features are extracted:
- **Name Signals**: Exact match, RapidFuzz ratio, partial ratio, token sort/set ratio, char 3-gram Jaccard, token Jaccard, token containment, name length difference/ratio, first word match.
- **Address Signals**: Exact match, RapidFuzz address ratio, token set ratio, char 3-gram Jaccard, token Jaccard, numeric token Jaccard, postal code match status.
- **Contextual & System Signals**: Country match binary indicator, source indicators (`is_s2`, `is_s3`), and weighted heuristic score.

## 5. Model Architecture & Training
- **Model**: `HistGradientBoostingClassifier` with max 150 iterations, learning rate 0.08, max depth 6, and random seed 42.
- **Validation**: 80/20 Entity-level split on Source 1 IDs to eliminate data leakage.
- **Class Imbalance**: Handles positive/negative pair ratio natively through gradient boosting split criterion.

## 6. Entity-level F0.5 Metric & Threshold Optimization
The official metric is macro F0.5 computed per Source 1 entity:
\[ F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}} \]
- Precision-weighted evaluation rewards high-confidence predictions and penalizes false positives.
- A threshold sweep over \([0.45, 0.90]\) selects the probability decision threshold that maximizes validation Macro F0.5.
