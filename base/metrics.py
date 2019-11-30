import torch
import numpy as np


def inter_over_union(output, target, num_classes):
    # Assume pixel whose value equals to -1 is ignored
    output, target = np.asarray(output)+1, np.asarray(target)+1
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