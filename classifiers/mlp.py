from typing import Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size: int, hidden_size_0: int, hidden_size_1: int, dropout: float) -> None:
        super(MLP, self).__init__()
        self.scaler_mean = nn.Parameter(torch.zeros(input_size, dtype=torch.float), requires_grad=False)
        self.scaler_std = nn.Parameter(torch.ones(input_size, dtype=torch.float), requires_grad=False)
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size_0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_0, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def fit_scaler(self, X_train: torch.Tensor) -> None:
        self.scaler_mean.data = torch.mean(X_train, dim=0)
        self.scaler_std.data = torch.std(X_train, dim=0)

    def forward(self, batch_embeddings: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_scaled_embeddings = (batch_embeddings - self.scaler_mean) / self.scaler_std
        return self.sigmoid(self.linear(batch_scaled_embeddings)), self.linear[0](batch_scaled_embeddings)
