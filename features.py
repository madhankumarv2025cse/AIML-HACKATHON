import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from rapidfuzz import fuzz
from src.preprocess import extract_numbers

def compute_pair_features_fast(row1: dict, row2: dict, target_id: str) -> List[float]:
    """
    Compute fast numerical features for a candidate pair and return as float array list.
    """
    n1, n2 = row1['norm_name'], row2['norm_name']
    a1, a2 = row1['norm_address'], row2['norm_address']
    c1, c2 = row1['norm_country'], row2['norm_country']

    country_match = 1.0 if c1 and c2 and c1 == c2 else 0.0

    # Name features
    exact_name = 1.0 if n1 and n2 and n1 == n2 else 0.0
    name_ratio = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    name_partial_ratio = fuzz.partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0

    t1, t2 = n1.split(), n2.split()
    s1_tok, s2_tok = set(t1), set(t2)
    if s1_tok and s2_tok:
        inter = len(s1_tok.intersection(s2_tok))
        union = len(s1_tok.union(s2_tok))
        name_jaccard = float(inter / union)
        name_overlap = float(inter)
        name_containment = float(inter / min(len(s1_tok), len(s2_tok)))
    else:
        name_jaccard = name_overlap = name_containment = 0.0

    first_word_match = 1.0 if (t1 and t2 and t1[0] == t2[0]) else 0.0
    name_len_diff = float(abs(len(n1) - len(n2)))
    name_len_ratio = float(min(len(n1), len(n2)) / max(len(n1), len(n2))) if (n1 and n2) else 0.0

    # Address features
    exact_address = 1.0 if a1 and a2 and a1 == a2 else 0.0
    addr_ratio = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    addr_partial_ratio = fuzz.partial_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0

    at1, at2 = a1.split(), a2.split()
    as1_tok, as2_tok = set(at1), set(at2)
    if as1_tok and as2_tok:
        ainter = len(as1_tok.intersection(as2_tok))
        aunion = len(as1_tok.union(as2_tok))
        addr_jaccard = float(ainter / aunion)
        addr_overlap = float(ainter)
        addr_containment = float(ainter / min(len(as1_tok), len(as2_tok)))
    else:
        addr_jaccard = addr_overlap = addr_containment = 0.0

    nums1, nums2 = extract_numbers(a1), extract_numbers(a2)
    if nums1 and nums2:
        num_inter = len(nums1.intersection(nums2))
        num_jaccard = float(num_inter / len(nums1.union(nums2)))
    else:
        num_jaccard = 0.0

    pc1, pc2 = row1.get('postal_code', ''), row2.get('postal_code', '')
    if pc1 and pc2:
        postal_match = 1.0 if pc1 == pc2 else 0.0
    else:
        postal_match = -1.0

    addr_len_diff = float(abs(len(a1) - len(a2)))

    is_s2 = 1.0 if target_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if target_id.startswith('S3-') else 0.0
    combined_score = 0.6 * name_token_set + 0.4 * addr_token_set

    return [
        country_match, exact_name, name_ratio, name_partial_ratio, name_token_sort,
        name_token_set, name_jaccard, name_overlap, name_containment, first_word_match,
        name_len_diff, name_len_ratio, exact_address, addr_ratio, addr_partial_ratio,
        addr_token_set, addr_jaccard, addr_overlap, addr_containment, num_jaccard,
        postal_match, addr_len_diff, is_s2, is_s3, combined_score
    ]

FEATURE_NAMES = [
    'country_match', 'exact_name', 'name_ratio', 'name_partial_ratio', 'name_token_sort',
    'name_token_set', 'name_jaccard', 'name_overlap', 'name_containment', 'first_word_match',
    'name_len_diff', 'name_len_ratio', 'exact_address', 'addr_ratio', 'addr_partial_ratio',
    'addr_token_set', 'addr_jaccard', 'addr_overlap', 'addr_containment', 'num_jaccard',
    'postal_match', 'addr_len_diff', 'is_s2', 'is_s3', 'combined_score'
]
