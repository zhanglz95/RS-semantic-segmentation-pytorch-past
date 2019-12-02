import torch
import numpy as np


def inter_over_union(output, target, num_classes):
    # Assume pixel whose value equals to -1 is ignored
    output, target = np.asarray(output), np.asarray(target)
    # Ignore the index equals to -1.
    # Essential the iou only calculate in the area which we care.
    # Only a subset of output and target
    output = output * (target > 0)

    # True positive
    intersection = output * (output == target)

    area_inter, _ = np.histogram(intersection, bins=num_classes, range=(1,num_classes))
    area_pred, _ = np.histogram(output, bins=num_classes, range=(1,num_classes))
    area_lab, _ = np.histogram(target, bins=num_classes, range=(1, num_classes))

    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union

def pixel_accuracy(output, target):
    output = np.asarray(output)
    target = np.asarray(target)

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum(output == target) * (target > 0)
    return pixel_correct, pixel_labeled


def eval_metrics(output, target, num_classes):
    '''
    Parameters: output and target are torch.Tensor()
    '''
    correct, labeled = batch_pix_accuracy(output.data, target)
    inter, uinon = batch_intersection_union(output.data, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]


def batch_pix_accuracy():
    pass

def batch_intersection_union():
    pass
