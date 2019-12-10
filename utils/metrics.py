import torch
import numpy as np

class Metrics:
    def __init__(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets

    def pixel_accuracy(self):
        outputs = np.asarray(self.outputs)
        targets = np.asarray(self.targets)

        pixel_labeled = np.sum(targets > 0, (1, 2, 3))
        pixel_correct = np.sum(((outputs == targets) * (targets > 0)), (1, 2, 3))
        
        return (pixel_correct / pixel_labeled)

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
