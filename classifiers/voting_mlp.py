from typing import List

import torch
import torch.nn as nn


class VotingMLP(nn.Module):

    def __init__(self, input_size: int) -> None:
        super(VotingMLP, self).__init__()
        self.scaler_mean = nn.Parameter(torch.zeros(input_size, dtype=torch.float), requires_grad=False)
        self.scaler_std = nn.Parameter(torch.ones(input_size, dtype=torch.float), requires_grad=False)
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def fit_scaler(self, X_train: List[torch.Tensor]) -> None:
        X_train_tensor = torch.vstack(X_train)
        self.scaler_mean.data = torch.mean(X_train_tensor, dim=0)
        self.scaler_std.data = torch.std(X_train_tensor, dim=0)

    def forward(self, batch_embeddings: torch.Tensor, batch_num_windows: torch.Tensor) -> torch.Tensor:
        batch_scaled_embeddings = (batch_embeddings - self.scaler_mean) / self.scaler_std
        batch_windows_outputs = self.linear_stack(batch_scaled_embeddings)
        batch_size, num_windows, _ = batch_windows_outputs.shape
        mask = (
            torch.arange(num_windows).repeat(batch_size, 1).to(batch_windows_outputs.device) <
            torch.unsqueeze(batch_num_windows, -1)
        ).unsqueeze(2)
        batch_audio_output = torch.sum(batch_windows_outputs * mask, dim=1) / torch.sum(mask, dim=1)
        return self.sigmoid(batch_audio_output)
