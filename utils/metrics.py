import torch
import numpy as np

class Metrics:
    def __init__(self):
        # For global IoU
        self.global_inter = 0
        self.global_union = 0

        self.outputs = None
        self.targets = None

    def update_input(self, outputs, targets):
        self.outputs = torch.argmax(outputs, axis=1)
        self.targets = targets

    # def pixel_accuracy(self):
    #     outputs = np.asarray(self.outputs)
    #     targets = np.asarray(self.targets)

    #     pixel_labeled = np.sum(targets > 0, (1, 2))
    #     pixel_correct = np.sum(((outputs == targets) * (targets > 0)), (1, 2))
        
    #     return (pixel_correct / pixel_labeled)

    def iou(self):
        if self.outputs is None or self.targets is None:
            raise Exception("Metrics Data is None!")

        outputs = np.asarray(self.outputs)
        targets = np.asarray(self.targets)

        inter = (outputs == targets) & (targets != 0)
        union = (outputs != 0) | (targets != 0)
        inter_sum = np.sum(inter, (1, 2))
        union_sum = np.sum(union, (1, 2))

        self.global_inter += inter_sum
        self.global_union += union_sum

        if union_sum == 0:  # union_sum = 0
            return 1.
        else:
            return (inter_sum / union_sum)

    def global_iou(self):
        if self.global_union == 0:
            return 1.
        else:
            return (self.global_inter / self.global_union)

    def metrics_all(self, metrics_list):
        metrics_dict = {}
        for metric in metrics_list:
            method = getattr(self, metric)
            metrics_dict[metric] = method()
        return metrics_dict

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.initialized = False
#         self.val = None
#         self.avg = None
#         self.sum = None
#         self.count = None

#     def initialize(self, val, weight):
#         self.val = val
#         self.avg = val
#         self.sum = np.multiply(val, weight)
#         self.count = weight
#         self.initialized = True

#     def update(self, val, weight=1):
#         if not self.initialized:
#             self.initialize(val, weight)
#         else:
#             self.add(val, weight)

#     def add(self, val, weight):
#         self.val = val
#         self.sum = np.add(self.sum, np.multiply(val, weight))
#         self.count = self.count + weight
#         self.avg = self.sum / self.count

#     @property
#     def value(self):
#         return self.val

#     @property
#     def average(self):
#         return np.round(self.avg, 5)


# def inter_over_union(output, target, num_classes):  
#     output, target = np.asarray(output), np.asarray(target)
#     output = output * (target > 0)

#     # True positive
#     intersection = output * (output == target)

#     area_inter, _ = np.histogram(intersection, bins=num_classes, range=(1,num_classes))
#     area_pred, _ = np.histogram(output, bins=num_classes, range=(1,num_classes))
#     area_lab, _ = np.histogram(target, bins=num_classes, range=(1, num_classes))

#     area_union = area_pred + area_lab - area_inter
#     return area_inter, area_union

# def pixel_accuracy(output, target):
#     output = np.asarray(output)
#     target = np.asarray(target)

#     pixel_labeled = np.sum(target > 0)
#     pixel_correct = np.sum(output == target) * (target > 0)
#     return pixel_correct, pixel_labeled


# def eval_metrics(output, target, num_classes):
#     '''
#     Parameters: output and target are torch.Tensor()
#     '''
#     correct, labeled = batch_pix_accuracy(output.data, target)
#     inter, uinon = batch_intersection_union(output.data, target, num_classes)
#     return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]


# def batch_pix_accuracy(output, target, num_classes):
#     _, predict = torch.max(output, 1)
#     return

# def batch_intersection_union():
#     return
