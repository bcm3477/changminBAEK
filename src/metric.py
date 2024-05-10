from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes= 200, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=False)
        self.add_state('true_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds: 모델 예측 (BxC Tensor)
        # target: 실제 레이블 (Bx Tensor)
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape, "Predictions and targets should have the same shape"

        for cls in range(preds.max() + 1):
            true_positives = torch.sum((preds == cls) & (target == cls))
            false_positives = torch.sum((preds == cls) & (target != cls))
            false_negatives = torch.sum((preds != cls) & (target == cls))

            self.true_positives += true_positives
            self.false_positives += false_positives
            self.false_negatives += false_negatives

    def compute(self):
        precision = self.true_positives.float() / (self.true_positives + self.false_positives + 1e-6)
        recall = self.true_positives.float() / (self.true_positives + self.false_negatives + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return f1
    
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, "Shape of preds and target do not match"

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)


        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
