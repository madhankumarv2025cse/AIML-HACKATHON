from typing import Dict, List, Set

def calculate_entity_f05(gt_ids: Set[str], pred_ids: Set[str]) -> float:
    """
    Calculate entity-level F0.5 for a single Source 1 entity.
    
    Formula: F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    
    Edge cases:
    - If GT is empty and Pred is empty -> 1.0
    - If GT is empty and Pred is non-empty -> 0.0
    - If GT is non-empty and Pred is empty -> 0.0
    """
    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 and len(pred_ids) > 0:
        return 0.0
    if len(gt_ids) > 0 and len(pred_ids) == 0:
        return 0.0

    tp = len(gt_ids.intersection(pred_ids))
    fp = len(pred_ids - gt_ids)
    fn = len(gt_ids - pred_ids)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return float(f05)

def evaluate_predictions(
    ground_truth_dict: Dict[str, Set[str]],
    predictions_dict: Dict[str, Set[str]]
) -> float:
    """
    Calculate entity-level Macro F0.5 across all Source 1 entities in ground truth.
    """
    scores = []
    for s1_id, gt_set in ground_truth_dict.items():
        pred_set = predictions_dict.get(s1_id, set())
        score = calculate_entity_f05(gt_set, pred_set)
        scores.append(score)

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
