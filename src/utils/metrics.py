import numpy as np
import torch


def compute_iou_binary(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Compute IoU between two binary masks (H, W) on CPU as float.
    If both masks are empty (union=0), returns 1.0.
    """
    pred = pred_mask.bool()
    gt = gt_mask.bool()
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    if union == 0:
        return 1.0
    return inter / union


def average_precision_binary_scores(gt_flat_np: np.ndarray, scores_flat_np: np.ndarray) -> float:
    """
    Compute Average Precision (AP) for binary labels (0/1) and continuous scores in [0,1].
    Implementation without sklearn: sort by score desc, sum precision at each positive over total positives.
    Returns np.nan if there are no positive labels.
    """
    assert gt_flat_np.ndim == 1 and scores_flat_np.ndim == 1
    assert gt_flat_np.shape[0] == scores_flat_np.shape[0]
    P = int(gt_flat_np.sum())
    if P == 0:
        return np.nan  # undefined AP when no positives
    order = np.argsort(-scores_flat_np)
    gt_sorted = gt_flat_np[order]
    tp_cum = np.cumsum(gt_sorted)
    idxs = np.arange(1, gt_sorted.shape[0] + 1)
    precision = tp_cum / idxs
    # AP = sum precision at ranks where gt==1 divided by P
    ap = (precision[gt_sorted == 1].sum()) / P
    return float(ap)


def batch_miou_ap_from_logits(pred_logits_b_q_h_w: torch.Tensor, gt_masks_list: list, threshold: float = 0.5):
    """
    Compute per-batch mIoU and AP by using the best query per pixel (max over queries).
    - pred_logits_b_q_h_w: tensor [B, Q, H, W]
    - gt_masks_list: list of tensors per sample with shape [G, H, W] (can be G=1). We union them to a single GT mask.
    Returns: (miou_mean, ap_mean, valid_ap_count, batch_count)
    """
    with torch.no_grad():
        B = pred_logits_b_q_h_w.shape[0]
        probs = torch.sigmoid(pred_logits_b_q_h_w)
        best_probs = probs.max(dim=1).values  # [B, H, W]
        miou_sum = 0.0
        ap_sum = 0.0
        ap_valid = 0
        for b in range(B):
            pred_prob = best_probs[b].detach().cpu()
            pred_bin = (pred_prob >= threshold).to(torch.uint8)
            gt_masks = gt_masks_list[b]
            if isinstance(gt_masks, torch.Tensor) and gt_masks.ndim == 3 and gt_masks.shape[0] > 0:
                gt_union = gt_masks.max(dim=0).values  # union over instances
            else:
                gt_union = torch.zeros_like(pred_prob)
            gt_bin = (gt_union > 0.5).to(torch.uint8)
            miou_sum += compute_iou_binary(pred_bin, gt_bin)
            # AP
            gt_flat = gt_bin.flatten().numpy().astype(np.uint8)
            score_flat = pred_prob.flatten().numpy().astype(np.float32)
            ap = average_precision_binary_scores(gt_flat, score_flat)
            if not np.isnan(ap):
                ap_sum += ap
                ap_valid += 1
        miou_mean = miou_sum / max(1, B)
        ap_mean = (ap_sum / ap_valid) if ap_valid > 0 else float('nan')
        return miou_mean, ap_mean, ap_valid, B
