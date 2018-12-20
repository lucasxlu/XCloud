import torch.nn as nn
import torch.nn.functional as F


class CRLoss(nn.Module):
    """
    CRLoss definition
    """

    def __init__(self, cls_w=0.5, reg_w=0.5):
        super(CRLoss, self).__init__()

        self.cls_w = cls_w
        self.reg_w = reg_w

        self.class_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()

    def forward(self, cls_pred, cls_gt, score_pred, score_gt):
        class_loss = self.class_criterion(cls_pred, cls_gt)
        regression_loss = self.regression_criterion(score_pred, score_gt)

        cr_loss = self.cls_w * class_loss + self.reg_w * regression_loss

        return cr_loss


def log_cosh_loss(input, target, size_average=None, reduce=None, reduction='elementwise_mean'):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='elementwise_mean'):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    return torch._C._nn.smooth_l1_loss(input, target, reduction)
