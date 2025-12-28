import torch

def auc_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return torch.square(1 - (pos_out - neg_out)).sum()

def hinge_auc_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()

def log_rank_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()

def pull_loss(pos_out, neg_out, neg_targets):
    """
    PULL Loss = L_E' (Expected Graph Loss) + L_C (Correction Loss)
    
    L_C: Standard BCE where Pos->1, Neg->0
    L_E': Weighted BCE where Pos->1, Neg->Pseudo-Label (if pseudo-positive) or 0
    
    pos_out: Logits for positive edges
    neg_out: Logits for negative edges
    neg_targets: Probabilities [0,1] for negative edges. 
                 0 for 'Relatively Unconnected', Score for 'Pseudo-Positive'
    """
    import torch.nn.functional as F
    
    # L_C Correction Loss
    # Positive -> 1, Negative -> 0
    lc_pos = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    lc_neg = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
    loss_c = lc_pos + lc_neg
    
    # L_E' Expected Graph Loss
    # Positive -> 1, Negative -> neg_targets
    le_pos = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    le_neg = F.binary_cross_entropy_with_logits(neg_out, neg_targets)
    loss_e = le_pos + le_neg
    
    return loss_c + loss_e