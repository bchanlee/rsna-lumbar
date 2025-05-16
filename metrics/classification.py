import numpy as np
import torch
import torchmetrics as tm

from sklearn import metrics as skm 


def _roc_auc_score(t, p):
    try:
        try:
            return torch.tensor(skm.roc_auc_score(t, p))
        except Exception as e:
            return torch.tensor(skm.roc_auc_score((t > 0).astype("int"), p))
    except Exception as e:
        print(e)
        return torch.tensor(0.5)


class _BaseMetric(tm.Metric):
    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.cfg = cfg 

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, p, t):
        self.p.append(p.float())
        self.t.append(t.float())

    def compute(self):
        raise NotImplementedError


class _ScoreBased(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy()  # (N,) or (N,C)
        t = torch.cat(self.t, dim=0).cpu().numpy()  # (N,) or (N,C)
        if p.ndim == 1:
            # Binary classification
            return {f"{self.name}_mean": self.metric_func(t, p)}
        metrics_dict = {}
        for c in range(p.shape[1]):
            # Depends on whether it is multilabel or multiclass
            # If multiclass using CE loss, p.shape[1] = num_classes and t.shape[1] = 1
            tmp_gt = t == c if t.shape[1] != p.shape[1] else t[:, c]
            metrics_dict[f"{self.name}{c}"] = self.metric_func(tmp_gt, p[:, c])
        metrics_dict[f"{self.name}_mean"] = np.mean([v for v in metrics_dict.values()])
        return metrics_dict

class AUROC(_ScoreBased):

    name = "auc"
    def metric_func(self, t, p): 
        return _roc_auc_score(t, p)