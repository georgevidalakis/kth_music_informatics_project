from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2) -> None:
        super(LSTMModel, self).__init__()

        self.scaler_mean = nn.Parameter(torch.zeros(input_size, dtype=torch.float), requires_grad=False)
        self.scaler_std = nn.Parameter(torch.ones(input_size, dtype=torch.float), requires_grad=False)

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def fit_scaler(self, X_train: List[torch.Tensor]) -> None:
        X_train_tensor = torch.vstack(X_train)
        self.scaler_mean.data = torch.mean(X_train_tensor, dim=0)
        self.scaler_std.data = torch.std(X_train_tensor, dim=0)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # LSTM expects input of shape (batch_size, seq_length, input_size)
        x, _ = self.lstm(x)  # (batch_size, seq_length, hidden_size)

        # Take the output from the last time step
        x = x[:, -1, :]  # (batch_size, hidden_size)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class DeepLSTMModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 3,
            dropout: float = 0.3,
            bidirectional: bool = False,
            ) -> None:
        super(DeepLSTMModel, self).__init__()

        self.scaler_mean = nn.Parameter(torch.zeros(input_size, dtype=torch.float), requires_grad=False)
        self.scaler_std = nn.Parameter(torch.ones(input_size, dtype=torch.float), requires_grad=False)
        # LSTM layers (stacked)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Bidirectional doubles the hidden size, so we handle that here
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Fully connected layers (deeper)
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # Activation functions and dropout layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:

    #     # LSTM layer
    #     lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size * num_directions)

    #     # Get output of the last time step (many-to-one prediction)
    #     lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size * num_directions)

    #     # Fully connected layers
    #     x = self.relu(self.fc1(lstm_out))
    #     x = self.dropout(x)  # Add dropout for regularization
    #     x = self.relu(self.fc2(x))
    #     x = self.dropout(x)
    #     x = self.sigmoid(self.fc3(x))  # Final output with Sigmoid activation for binary classification
    #     return x

    def forward(self, batch_embeddings: torch.Tensor, batch_num_windows: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # batch_embeddings: shape (batch_size, max_seq_len, input_size)
        # batch_num_windows: shape (batch_size,), indicates valid sequence lengths for each batch element

        # Packing the padded sequence for efficient LSTM processing
        packed_input = pack_padded_sequence(
            batch_embeddings, batch_num_windows.cpu(), batch_first=True, enforce_sorted=False
        )
        # LSTM layer (packed input)
        packed_output, (hn, cn) = self.lstm(packed_input)

        # Unpacking the padded output
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Get output of the last time step for each seqhapeuence in the batch (based on batch_num_windows)
        # Extract the hidden state corresponding to the actual last time step for each sequence
        lstm_out_last = torch.stack([lstm_out[i, batch_num_windows[i] - 1, :] for i in range(lstm_out.size(0))])

        # Fully connected layers
        x = self.relu(self.fc1(lstm_out_last))
        x = self.dropout(x)  # Add dropout for regularization
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))  # Final output with Sigmoid activation for binary classification

        return x

    def fit_scaler(self, X_train: List[torch.Tensor]) -> None:
        X_train_tensor = torch.vstack(X_train)
        self.scaler_mean.data = torch.mean(X_train_tensor, dim=0)
        self.scaler_std.data = torch.std(X_train_tensor, dim=0)
