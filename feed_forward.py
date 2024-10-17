import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, input_size: int) -> None:
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_embeddings: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.sigmoid(self.linear(batch_embeddings))
