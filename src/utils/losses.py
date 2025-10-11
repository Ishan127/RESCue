import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def compute_pairwise_cost(pred_logits_q_hw: torch.Tensor, gt_mask_hw: torch.Tensor):
    """(This function remains unchanged)"""
    Q, HW = pred_logits_q_hw.shape
    G = gt_mask_hw.shape[0]
    cost = torch.zeros((Q, G), device=pred_logits_q_hw.device)
    for i in range(G):
        tgt = gt_mask_hw[i].unsqueeze(0).expand(Q, -1)
        bce = F.binary_cross_entropy_with_logits(pred_logits_q_hw, tgt, reduction='none').mean(dim=1)
        pred_prob = torch.sigmoid(pred_logits_q_hw)
        inter = (pred_prob * tgt).sum(dim=1)
        union = (pred_prob + tgt - pred_prob * tgt).sum(dim=1) + 1e-6
        iou = inter / union
        cost[:, i] = bce - 0.8 * iou
    return cost.cpu().detach().numpy()

def hungarian_loss_for_sample(pred_logits_q_hw: torch.Tensor, gt_masks_g_hw: torch.Tensor, no_object_cost=0.2):
    """
    MODIFIED to return a dictionary containing the loss AND the matching indices.
    """
    Q, HW = pred_logits_q_hw.shape
    G = gt_masks_g_hw.shape[0]
    device = pred_logits_q_hw.device

    # Handle case with no ground truth objects
    if G == 0:
        loss_noobj = F.binary_cross_entropy_with_logits(pred_logits_q_hw, torch.zeros_like(pred_logits_q_hw), reduction='mean')
        # Return empty indices
        return {
            "loss": loss_noobj,
            "indices": (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
        }

    # Perform the matching
    cost = compute_pairwise_cost(pred_logits_q_hw, gt_masks_g_hw)
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Calculate loss for matched pairs
    matched_q = torch.tensor(row_ind, dtype=torch.long, device=device)
    matched_g = torch.tensor(col_ind, dtype=torch.long, device=device)
    matched_loss = 0.0
    for mq, mg in zip(matched_q.tolist(), matched_g.tolist()):
        tgt = gt_masks_g_hw[mg].unsqueeze(0)
        pred = pred_logits_q_hw[mq].unsqueeze(0)
        bce = F.binary_cross_entropy_with_logits(pred, tgt, reduction='mean')
        p = torch.sigmoid(pred)
        inter = (p * tgt).sum()
        union = p.sum() + tgt.sum()
        dice = 1 - (2 * inter + 1e-6) / (union + 1e-6)
        matched_loss = matched_loss + (bce + dice)
    matched_loss = matched_loss / max(1, len(matched_q))

    # Calculate loss for unmatched queries
    matched_mask = torch.zeros(Q, dtype=torch.bool, device=device)
    matched_mask[matched_q] = True
    noobj_loss = torch.tensor(0.0, device=device)
    if matched_mask.sum() < Q:
        unmatched_idxs = (~matched_mask).nonzero(as_tuple=False).squeeze(1)
        noobj_preds = pred_logits_q_hw[unmatched_idxs]
        noobj_loss = F.binary_cross_entropy_with_logits(noobj_preds, torch.zeros_like(noobj_preds), reduction='mean')
    
    total_loss = matched_loss + 0.5 * noobj_loss

    # <<< FIX APPLIED HERE: Return a dictionary with both loss and indices >>>
    return {
        "loss": total_loss,
        "indices": (row_ind, col_ind)
    }
