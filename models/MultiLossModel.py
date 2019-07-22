import torch
import torch.nn as nn
class MultiLossModel(nn.Module):
    def __init__(self, nb_outputs):
        super().__init__()
        self.nb_outputs = nb_outputs
        self.weights = nn.Parameter(torch.zeros(nb_outputs), requires_grad=True)

    def forward(self, losses):
        precisions = torch.exp(-1 * self.weights)
        weighted_losses = losses * precisions
        weighted_losses = weighted_losses.sum()
        weighted_losses = weighted_losses

        return weighted_losses

