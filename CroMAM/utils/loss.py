import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss
    """
    def __init__(self, gamma=2, alpha=0.5, device=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = torch.Tensor([alpha, 1-alpha]).to(device)
        self.reduction = reduction

    def forward(self, inputs, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, reduction=self.reduction)
        return loss


class FlexLoss:
    def __init__(self, device=torch.device('cpu'), outcome=None):
        # self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        if outcome in ['idh', '1p19q']:
            # LGG
            self.criterion = FocalLoss(gamma=2, alpha=0.75, device=device).to(device)
        else:
            # GBM
            self.criterion = FocalLoss(gamma=2, alpha=0.5, device=device).to(device)

    def calculate(self, pred, target):
        return self.criterion(pred, target.long().view(-1))
