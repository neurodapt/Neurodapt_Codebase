import torch.nn.functional as F
import torch


def boundary_bce_loss(logits, targets, valid_mask):

    #simplest implemnentation of BCE loss with logits, only considering valid tokens (not padding)
    return F.binary_cross_entropy_with_logits(
        logits[valid_mask],
        targets[valid_mask].float()
    )

def weighted_boundary_bce_loss(logits,targets,valid_mask, pos_weight=5.0):
    return F.binary_cross_entropy_with_logits(
        logits[valid_mask],
        targets[valid_mask].float(),
        pos_weight=torch.tensor(pos_weight,device=logits.device)
    )

