from typing import Tuple

import torch
import numpy as np
from torch import nn


class TransformerwithMLP(nn.Module):
    def __init__(self, input_size: int, num_heads: int, dim_ffn: int) -> None:
        super(TransformerwithMLP, self).__init__()
        self.nhead = num_heads
        self.ffn_dim = dim_ffn
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=self.nhead, batch_first=True, dim_feedforward=self.ffn_dim, norm_first=True,
            dropout=0.1,
        )
        self.linear = nn.Sequential(
            nn.Linear(input_size, 1)
        )
        self.linear2 = nn.Sequential(nn.Linear(input_size, input_size))
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(input_size)
        self.positional_encodings = self.generate_sinusoidal_positional_encodings(100, input_size)

    def generate_sinusoidal_positional_encodings(self, max_len: int, d_model: int):
        """Generates sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        return pe

    def forward(self, batch_embeddings: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # batch_embeddings shape: (Batch, Seq_Length, 512)
        # transformer_output = self.layernorm(self.transformer_layer(batch_embeddings))  # (Batch, Seq_Length, 512)
        # pos_enc = self.positional_encodings[:, :batch_embeddings.shape[1], :].to(batch_embeddings.device)
        # batch_embeddings += pos_enc
        transformer_output = self.layernorm(self.transformer_layer(self.linear2(batch_embeddings)))
        # print("Transformer output", transformer_output.shape)
        transformer_output_flat = transformer_output.mean(dim=1)  # (Batch, 512)
        # transformer_cls=transformer_output[:,0,:]
        return self.sigmoid(self.linear(transformer_output_flat)), transformer_output_flat
