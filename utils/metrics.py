import torch
import torch.nn as nn
import numpy as np

class Metrics:
    def __init__(self):
        # For global IoU
        self.TP_sum = 0.
        self.FP_sum = 0.
        self.FN_sum = 0.

        self.global_TP_sum = 0.
        self.global_FP_sum = 0.
        self.global_FN_sum = 0.

        self.outputs = None
        self.targets = None

        self.Softmax = nn.Softmax(dim = 1)

    def update_input(self, outputs, targets):
        outputs = self.Softmax(outputs)
        self.outputs = torch.argmax(outputs, axis=1).cpu()
        self.targets = targets.cpu()

        self.outputs = np.asarray(self.outputs)
        self.targets = np.asarray(self.targets)

        self.TP = (self.outputs == self.targets) & (self.targets != 0)
        self.FP = (self.outputs != 0) & (self.targets == 0)
        self.FN = (self.targets != 0) & (self.outputs == 0)

        self.TP_sum = self.TP.sum()
        self.FP_sum = self.FP.sum()
        self.FN_sum = self.FN.sum()

        self.global_TP_sum += self.TP_sum
        self.global_FP_sum += self.FP_sum
        self.global_FN_sum += self.FN_sum


    # def iou(self):
    #     if self.outputs is None or self.targets is None:
    #         raise Exception("Metrics Data is None!")

    #     outputs = np.asarray(self.outputs)
    #     targets = np.asarray(self.targets)

    #     inter = (outputs == targets) & (targets != 0)
    #     union = (outputs != 0) | (targets != 0)
    #     inter_sum = np.sum(inter, (1, 2))
    #     union_sum = np.sum(union, (1, 2)) + np.spacing(1)

    #     self.global_inter += inter_sum.sum()
    #     self.global_union += union_sum.sum()

    #     return (inter_sum / union_sum).mean()

    def batch_iou(self):
        iou = self.TP_sum / (self.TP_sum + self.FN_sum + self.FP_sum + np.spacing(1))
        return iou

    def batch_recall(self):
        recall = self.TP_sum / (self.TP_sum + self.FN_sum + np.spacing(1))
        return recall

    def batch_precision(self):
        prec = self.TP_sum / (self.TP_sum + self.FP_sum + np.spacing(1))
        return prec

    def global_iou(self):
        iou = self.global_TP_sum / (self.global_TP_sum + self.global_FN_sum + self.global_FP_sum + np.spacing(1))
        return iou

    def global_recall(self):
        recall = self.global_TP_sum / (self.global_TP_sum + self.global_FN_sum + np.spacing(1))
        return recall

    def global_precision(self):
        prec = self.global_TP_sum / (self.global_TP_sum + self.global_FP_sum + np.spacing(1))
        return prec

    def get_metrics(self, metrics_list):
        metrics_dict = {}
        for metric in metrics_list:
            method = getattr(self, metric)
            metrics_dict[metric] = method()
        return metrics_dict