import torch
import torch.nn as nn
from .metrics import ClassificationMetric


class Accuracy(ClassificationMetric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0  # Correct predictions
        self.total = 0    # Total predictions

    def update(self, true_labels, pred_labels):
        # Update correct and total based on multiclass labels
        self.correct += (pred_labels == true_labels).sum().item()
        self.total += true_labels.size(0)

    def result(self):
        # Calculate accuracy as correct / total
        accuracy = self.correct / (self.total + 1e-12)  # Avoid division by zero
        return accuracy
    

class BinaryBalancedAccuracy(ClassificationMetric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0  # True positives
        self.tn = 0  # True negatives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives

    def update(self, true_labels, pred_labels):
        # Assuming binary labels are 0 and 1
        self.tp += ((pred_labels == 1) & (true_labels == 1)).sum().item()
        self.tn += ((pred_labels == 0) & (true_labels == 0)).sum().item()
        self.fp += ((pred_labels == 1) & (true_labels == 0)).sum().item()
        self.fn += ((pred_labels == 0) & (true_labels == 1)).sum().item()

    def result(self):
        # Calculate balanced accuracy as the average of recall for both classes
        sensitivity = self.tp / (self.tp + self.fn + 1e-12)  # True positive rate (recall)
        specificity = self.tn / (self.tn + self.fp + 1e-12)  # True negative rate
        balanced_accuracy = (sensitivity + specificity) / 2
        return balanced_accuracy
    

class BalancedAccuracy(ClassificationMetric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # Initialize true positives and total counts for each class
        self.class_correct = torch.zeros(self.num_classes)
        self.class_total = torch.zeros(self.num_classes)

    def update(self, true_labels, pred_labels):
        # Update true positives and totals for each class
        for cls in range(self.num_classes):
            self.class_correct[cls] += ((pred_labels == cls) & (true_labels == cls)).sum().item()
            self.class_total[cls] += (true_labels == cls).sum().item()

    def result(self):
        # Calculate per-class recall
        class_recall = self.class_correct / (self.class_total + 1e-12)  # Avoid division by zero
        balanced_accuracy = class_recall.mean().item()
        return balanced_accuracy