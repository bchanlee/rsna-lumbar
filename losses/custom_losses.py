import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleWeightedLogLoss(nn.BCEWithLogitsLoss):

    def forward(self, p, t, w):
        return F.binary_cross_entropy_with_logits(p.float(), t.float(), weight=w.unsqueeze(1))