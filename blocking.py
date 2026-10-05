import gc
import re
import pandas as pd
from array import array
from collections import defaultdict
from typing import Dict, Set, List

import numpy as np
GENERIC_TOKENS = {
    "and", "the", "for", "of", "in", "at", "to", "a", "an", "st", "rd", "ave", "dr",
    "co", "company", "inc", "corp", "corporation", "ltd", "limited", "llc", "pvt", "private",
    "group", "holdings", "enterprises", "services", "solutions", "international", "intl",
    "store", "shop", "restaurant", "hotel", "center", "centre", "auto", "cafe", "tech",
    "technology", "technologies", "trading", "industries", "consulting", "logistics"
}
NAME_STOPWORDS = GENERIC_TOKENS | {"india", "road", "street"}
ADDRESS_STOPWORDS = {
    "and", "the", "for", "of", "in", "at", "to", "road", "rd", "street", "st",
    "avenue", "ave", "boulevard", "blvd", "drive", "dr", "lane", "ln", "suite",
    "ste", "apartment", "apt", "building", "bldg", "floor", "fl", "square", "sq",
    "highway", "hwy", "junction", "jct", "plaza", "plz", "parkway", "pkwy", "route",
    "rt", "near", "india", "company", "co", "unit", "block", "sector"
}

def fast_normalize_series(series: pd.Series) -> pd.Series:
    """Fast vectorized pandas string normalization."""
    s = series.fillna('').astype(str).str.lower()
    s = s.str.replace(r'\s*&\s*', ' and ', regex=True)
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(r'\b(ltd|limited|pvt|private|corp|corporation|co|company|inc|llc|sa|sarl|gmbh)\b', '', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def fast_country_series(series: pd.Series) -> pd.Series:
    """Fast vectorized country string normalization."""
    s = series.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', '', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def fast_address_series(series: pd.Series) -> pd.Series:
    """Fast vectorized address normalization."""
    s = series.fillna('').astype(str).str.lower()
    s = s.str.replace(r'\s*&\s*', ' and ', regex=True)
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    abbreviations = {
        r'\broad\b': 'rd', r'\bstreet\b': 'st', r'\bavenue\b': 'ave',
        r'\bboulevard\b': 'blvd', r'\bdrive\b': 'dr', r'\blane\b': 'ln',
        r'\bsuite\b': 'ste', r'\bapartment\b': 'apt', r'\bbuilding\b': 'bldg',
        r'\bfloor\b': 'fl', r'\bsquare\b': 'sq', r'\bhighway\b': 'hwy',
        r'\bjunction\b': 'jct', r'\bplaza\b': 'plz', r'\bparkway\b': 'pkwy',
        r'\broute\b': 'rt',
    }
    for pattern, replacement in abbreviations.items():
        s = s.str.replace(pattern, replacement, regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def extract_postal_series(series: pd.Series) -> pd.Series:
    """Extract postal code digits (length 5-7)."""
    return series.fillna('').astype(str).str.extract(r'(\b\d{5,7}\b)', expand=False).fillna('')

def _distinctive_tokens(text: str, stopwords: Set[str], limit: int) -> List[str]:
    tokens = []
    seen = set()
    for token in text.split():
        if len(token) >= 3 and token not in stopwords and token not in seen and not token.isdigit():
            tokens.append(token)
            seen.add(token)
            if len(tokens) == limit:
                break
    return tokens

def _add_limited_posting(index, oversized_keys, key, entity_id, max_frequency: int) -> None:
    if key in oversized_keys:
        return
    posting = index.get(key)
    if posting is None:
        index[key] = array('I', [entity_id])
    elif len(posting) >= max_frequency:
        del index[key]
        oversized_keys.add(key)
    else:
        posting.append(entity_id)

def _ensure_normalized_columns(frame: pd.DataFrame) -> None:
    if 'norm_name' not in frame:
        frame['norm_name'] = fast_normalize_series(frame['business_name'])
    if 'norm_address' not in frame:
        frame['norm_address'] = fast_address_series(frame['business_address'])
    if 'norm_country' not in frame:
        frame['norm_country'] = fast_country_series(frame['country'])
    if 'postal_code' not in frame:
        frame['postal_code'] = extract_postal_series(frame['business_address'])
    frame.drop(
        columns=['business_name', 'business_address', 'country'],
        errors='ignore',
        inplace=True,
    )

def generate_candidates_for_sources(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_candidates_per_s1: int = 10,
    max_s1_entities: int = None,
    max_block_frequency: int = 200,
    max_postal_frequency: int = 5000
) -> Dict[str, List[str]]:
    print("   [Blocking] Normalizing uncached source columns...", flush=True)
    df_s1_sub = (
        df_s1
        if max_s1_entities is None or len(df_s1) <= max_s1_entities
        else df_s1.iloc[:max_s1_entities].copy()
    )
    _ensure_normalized_columns(df_s1_sub)
    _ensure_normalized_columns(df_s2)
    _ensure_normalized_columns(df_s3)

    print(
        f"   [Blocking] Source records: S1={len(df_s1):,}, S2={len(df_s2):,}, S3={len(df_s3):,}",
        flush=True,
    )

    indexes = {
        'name_token': {},
        'name_prefix': {},
        'address_token': {},
        'postal': {},
        'address_number': {},
    }
    oversized = {name: set() for name in indexes}
    print("   [Blocking] Building country-scoped, frequency-limited indexes...", flush=True)
    target_ids = np.concatenate((
        df_s2['entity_id'].to_numpy(dtype=object, copy=False),
        df_s3['entity_id'].to_numpy(dtype=object, copy=False),
    ))
    target_names = np.concatenate((
        df_s2['norm_name'].to_numpy(dtype=object, copy=False),
        df_s3['norm_name'].to_numpy(dtype=object, copy=False),
    ))
    target_offset = 0
    for target_frame in (df_s2, df_s3):
        rows = zip(
            target_frame['entity_id'], target_frame['norm_name'], target_frame['norm_address'],
            target_frame['norm_country'], target_frame['postal_code'],
        )
        for target_index, (entity_id, name, address, country, postal) in enumerate(
            rows, start=target_offset
        ):
            country_key = country
            if name:
                for token in _distinctive_tokens(name, NAME_STOPWORDS, 2):
                    _add_limited_posting(
                        indexes['name_token'], oversized['name_token'],
                        (country_key, token), target_index, max_block_frequency,
                    )
                if len(name) >= 4:
                    _add_limited_posting(
                        indexes['name_prefix'], oversized['name_prefix'],
                        (country_key, name[:5]), target_index, max_block_frequency,
                    )
            for token in _distinctive_tokens(address, ADDRESS_STOPWORDS, 2):
                _add_limited_posting(
                    indexes['address_token'], oversized['address_token'],
                    (country_key, token), target_index, max_block_frequency,
                )
            if postal:
                _add_limited_posting(
                    indexes['postal'], oversized['postal'],
                    (country_key, postal), target_index, max_postal_frequency,
                )
            numbers = [number for number in re.findall(r'\b\d{2,4}\b', address) if number != postal]
            for number in dict.fromkeys(numbers[:2]):
                _add_limited_posting(
                    indexes['address_number'], oversized['address_number'],
                    (country_key, number), target_index, max_block_frequency,
                )
        target_offset += len(target_frame)

    for name, index in indexes.items():
        print(
            f"   [Blocking] {name}: {len(index):,} indexed keys, "
            f"{len(oversized[name]):,} over-frequency keys skipped",
            flush=True,
        )

    print(f"   [Blocking] Querying candidate index for {len(df_s1_sub):,} Source 1 entities...", flush=True)
    candidates: Dict[str, List[str]] = {}
    rule_hits = {name: 0 for name in indexes}
    candidate_hits_before_dedup = 0
    unique_candidates_before_cap = 0
    max_unique_candidates = 0
    final_candidate_pairs = 0

    for s1_id, name, address, country, postal in zip(
        df_s1_sub['entity_id'], df_s1_sub['norm_name'], df_s1_sub['norm_address'],
        df_s1_sub['norm_country'], df_s1_sub['postal_code'],
    ):
        scores = defaultdict(int)
        postings = []
        if name:
            for token in _distinctive_tokens(name, NAME_STOPWORDS, 2):
                postings.append(('name_token', indexes['name_token'].get((country, token), ()), 5))
            if len(name) >= 4:
                postings.append(('name_prefix', indexes['name_prefix'].get((country, name[:5]), ()), 3))
        for token in _distinctive_tokens(address, ADDRESS_STOPWORDS, 2):
            postings.append(('address_token', indexes['address_token'].get((country, token), ()), 4))
        if postal:
            postings.append(('postal', indexes['postal'].get((country, postal), ()), 6))
        numbers = [number for number in re.findall(r'\b\d{2,4}\b', address) if number != postal]
        for number in dict.fromkeys(numbers[:2]):
            postings.append(('address_number', indexes['address_number'].get((country, number), ()), 2))

        for rule, posting, weight in postings:
            hit_count = len(posting)
            candidate_hits_before_dedup += hit_count
            rule_hits[rule] += hit_count
            for candidate_id in posting:
                scores[candidate_id] += weight

        for candidate_id in scores:
            if target_names[candidate_id] == name:
                scores[candidate_id] += 8

        unique_count = len(scores)
        unique_candidates_before_cap += unique_count
        max_unique_candidates = max(max_unique_candidates, unique_count)
        selected = sorted(
            scores,
            key=lambda candidate_id: (-scores[candidate_id], target_ids[candidate_id]),
        )[:max_candidates_per_s1]
        candidate_list = [target_ids[candidate_id] for candidate_id in selected]
        candidates[s1_id] = candidate_list
        final_candidate_pairs += len(candidate_list)

    # Explicitly capped entities remain in the output with no candidates.
    if len(df_s1) > len(df_s1_sub):
        for s1_id in df_s1['entity_id'].iloc[len(df_s1_sub):]:
            candidates[s1_id] = []

    print("   [Blocking] Candidate statistics:", flush=True)
    print(f"      S1/S2/S3 records: {len(df_s1):,}/{len(df_s2):,}/{len(df_s3):,}", flush=True)
    print(f"      Candidate hits before deduplication: {candidate_hits_before_dedup:,}", flush=True)
    print(f"      Unique candidates before per-S1 cap: {unique_candidates_before_cap:,}", flush=True)
    for rule, hits in rule_hits.items():
        print(f"      {rule} posting hits: {hits:,}", flush=True)
    average_candidates = final_candidate_pairs / len(df_s1) if len(df_s1) else 0.0
    print(f"      Final candidate pairs: {final_candidate_pairs:,}", flush=True)
    print(f"      Average/max candidates per S1: {average_candidates:.2f}/{max((len(v) for v in candidates.values()), default=0):,}", flush=True)

    del indexes, oversized, target_ids, target_names
    gc.collect()

    return candidates
