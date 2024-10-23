import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2) -> None:
        super(LSTMModel, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM expects input of shape (batch_size, seq_length, input_size)
        x, _ = self.lstm(x)  # (batch_size, seq_length, hidden_size)

        # Take the output from the last time step
        x = x[:, -1, :]  # (batch_size, hidden_size)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class DeepLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3, bidirectional: bool = False) -> None:
        super(DeepLSTMModel, self).__init__()

        # LSTM layers (stacked)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            bidirectional=bidirectional, 
            batch_first=True
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM layer
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size * num_directions)

        # Get output of the last time step (many-to-one prediction)
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size * num_directions)

        # Fully connected layers
        x = self.relu(self.fc1(lstm_out))
        x = self.dropout(x)  # Add dropout for regularization
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))  # Final output with Sigmoid activation for binary classification
        return x

