import torch
from .metrics import ClassificationMetric


class BinaryRecall(ClassificationMetric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0  # True positives
        self.fn = 0  # False negatives

    def update(self, true_labels, pred_labels):
        # Assuming binary labels are 0 and 1
        self.tp += ((pred_labels == 1) & (true_labels == 1)).sum().item()
        self.fn += ((pred_labels == 0) & (true_labels == 1)).sum().item()

    def result(self):
        # Calculate recall as true positives / (true positives + false negatives)
        recall = self.tp / (self.tp + self.fn + 1e-12)  # Avoid division by zero
        return recall
    

class MulticlassRecall(ClassificationMetric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.class_tp = torch.zeros(self.num_classes)
        self.class_fn = torch.zeros(self.num_classes)

    def update(self, true_labels, pred_labels):
        # Update true positives and false negatives for each class
        for cls in range(self.num_classes):
            self.class_tp[cls] += ((pred_labels == cls) & (true_labels == cls)).sum().item()
            self.class_fn[cls] += ((pred_labels != cls) & (true_labels == cls)).sum().item()

    def result(self):
        # Calculate recall per class and return the macro average
        class_recall = self.class_tp / (self.class_tp + self.class_fn + 1e-12)  # Avoid division by zero
        recall = class_recall.mean().item()
        return recall