import torch
import torch.nn.functional as F

def masked_bce_with_posweight(pred, target, valid_mask, eps=1e-6):
    """
    pred, target: (B,H,W) floats in [0,1]
    valid_mask:   (B,H,W) floats {0,1}
    Computes BCE on valid pixels only, with dynamic pos_weight.
    """
    # valid pixels
    v = valid_mask > 0.5
    p = pred[v]
    t = target[v]

    # avoid empty valid set
    if p.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    # dynamic pos_weight = neg/pos
    pos = t.sum()
    neg = (1 - t).sum()
    pos_weight = (neg + eps) / (pos + eps)

    # BCEWithLogits wants logits, so convert prob->logit safely
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    logits = torch.log(p / (1 - p))

    return F.binary_cross_entropy_with_logits(logits, t, pos_weight=pos_weight)

def masked_dice_loss(pred, target, valid_mask, eps=1.0):
    pred = pred * valid_mask
    target = target * valid_mask

    inter = (pred * target).sum(dim=(1,2))
    denom = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()

def bce_dice_loss(pred, target, valid_mask):
    bce = masked_bce_with_posweight(pred, target, valid_mask)
    dice = masked_dice_loss(pred, target, valid_mask)
    return 0.5 * bce + 0.5 * dice
