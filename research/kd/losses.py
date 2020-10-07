import torch
import torch.nn as nn
from torch.nn import functional as F


class LabelSmoothingLoss(nn.Module):
    """
    vanilla Label Smoothing Loss
    """
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network, arXiv'15
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """

    def __init__(self, alpha, temperature):
        super(KDLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, teacher_outputs, student_outputs, gt_labels):
        kd_loss = nn.KLDivLoss()(F.log_softmax(student_outputs / self.temperature, dim=1),
                                 F.softmax(teacher_outputs / self.temperature, dim=1)) * (
                          self.alpha * self.temperature * self.temperature) + \
                  F.cross_entropy(student_outputs, gt_labels) * (1. - self.alpha)

        return kd_loss


class SelfTfKDLoss(nn.Module):
    """
    Revisiting Knowledge Distillation via Label Smoothing Regularization, CVPR'20
    adopt pretrained model as teacher model to supervise itself in KD training procedure
    """

    def __init__(self, alpha, temperature, multiplier=1):
        super(SelfTfKDLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.multiplier = multiplier  # multiple is 1.0 in most of cases, some cases are 10 or 50

    def forward(self, teacher_outputs, student_outputs, gt_labels):
        """
        loss function for self training: Tf-KD_{self}
        """
        loss_ce = F.cross_entropy(student_outputs, gt_labels)
        d_kl = nn.KLDivLoss()(F.log_softmax(student_outputs / self.temperature, dim=1),
                              F.softmax(teacher_outputs / self.temperature, dim=1)) * (
                       self.temperature * self.temperature) * self.multiplier
        kd_loss = (1. - self.alpha) * loss_ce + self.alpha * d_kl

        return kd_loss


class RegularizedTfKDLoss(nn.Module):
    """
    Revisiting Knowledge Distillation via Label Smoothing Regularization, CVPR'20
    adopt a virtual teacher's (a virtual model with 100% accuracy) output as additional KD supervision
    """

    def __init__(self, alpha, temperature, multiplier=1):
        super(RegularizedTfKDLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.multiplier = multiplier  # multiple is 1.0 in most of cases, some cases are 10 or 50

    def forward(self, outputs, gt_labels):
        """
        loss function for mannually-designed regularization: Tf-KD_{reg}
        """
        correct_prob = 0.99  # the probability for correct class in u(k)
        loss_ce = F.cross_entropy(outputs, gt_labels)
        K = outputs.size(1)

        teacher_soft = torch.ones_like(outputs).cuda()
        teacher_soft = teacher_soft * (1 - correct_prob) / (K - 1)  # p^d(k)
        for i in range(outputs.shape[0]):
            teacher_soft[i, gt_labels[i]] = correct_prob
        loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1),
                                        F.softmax(teacher_soft / self.temperature, dim=1)) * self.multiplier

        kd_loss = (1. - self.alpha) * loss_ce + self.alpha * loss_soft_regu

        return kd_loss
