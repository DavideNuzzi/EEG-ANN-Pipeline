import torch
from .metrics import ClassificationMetric


class BinaryPrecision(ClassificationMetric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0  # True positives
        self.fp = 0  # False positives

    def update(self, true_labels, pred_labels):
        # Assuming binary labels are 0 and 1
        self.tp += ((pred_labels == 1) & (true_labels == 1)).sum().item()
        self.fp += ((pred_labels == 1) & (true_labels == 0)).sum().item()

    def result(self):
        # Calculate precision as true positives / (true positives + false positives)
        precision = self.tp / (self.tp + self.fp + 1e-12)  # Avoid division by zero
        return precision
    

class MulticlassPrecision(ClassificationMetric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.class_tp = torch.zeros(self.num_classes)
        self.class_fp = torch.zeros(self.num_classes)

    def update(self, true_labels, pred_labels):
        # Update true positives and false positives for each class
        for cls in range(self.num_classes):
            self.class_tp[cls] += ((pred_labels == cls) & (true_labels == cls)).sum().item()
            self.class_fp[cls] += ((pred_labels == cls) & (true_labels != cls)).sum().item()

    def result(self):
        # Calculate precision per class and return the macro average
        class_precision = self.class_tp / (self.class_tp + self.class_fp + 1e-12)  # Avoid division by zero
        precision = class_precision.mean().item()
        return precision