from abc import ABC, abstractmethod

class Metric(ABC):

    @abstractmethod
    def update(self, *args):
        pass

    @abstractmethod
    def result(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class ClassificationMetric(Metric, ABC):

    @abstractmethod
    def update(self, true_labels, pred_labels):
        pass

    @abstractmethod
    def result(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class MeanMetric(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_sum = 0.0  # Sum of all values across batches
        self.total_count = 0   # Total number of elements across batches

    def update(self, values):
        # Update with the sum of values in the current batch
        self.total_sum += values.sum().item()
        # Update with the count of values in the current batch
        self.total_count += values.numel()  # Number of elements in the batch

    def result(self):
        # Calculate the mean as the total sum divided by the total count
        return self.total_sum / (self.total_count + 1e-12)  # Avoid division by zero