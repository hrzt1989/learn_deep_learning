from torch import nn
from torch.nn import functional
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, y_hat, y, mask):
        loss_vec = functional.binary_cross_entropy_with_logits(y_hat.view(y_hat.shape[0], -1), y.float(), reduction="none", weight=mask.float()).mean(dim = 1)
        return (loss_vec * mask.shape[1] / mask.float().sum(dim = 1)).mean()