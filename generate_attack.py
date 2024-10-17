from typing import List
from collections.abc import Iterable

import laion_clap
import numpy as np
import torch
import torch.nn as nn

from embeddings_computing import quantize_array, load_clap_music_model
from constants import CLAP_SR


def get_audio_windows_batch_data_generator(
        audio_data: torch.Tensor, window_size: int, hop_size: int, max_batch_size: int
        ) -> Iterable[torch.Tensor]:
    current_audio_windows_batch_data_list: List[torch.Tensor] = []
    for window_start_idx in range(0, len(audio_data), hop_size):
        window_end_idx = window_start_idx + window_size
        audio_window_data = audio_data[window_start_idx:window_end_idx]
        zero_pad_size = window_size - len(audio_window_data)
        padding = nn.ZeroPad1d((0, zero_pad_size))
        audio_window_data = padding(audio_window_data)
        current_audio_windows_batch_data_list.append(audio_window_data)
        if len(current_audio_windows_batch_data_list) == max_batch_size:
            yield torch.vstack(current_audio_windows_batch_data_list)
            current_audio_windows_batch_data_list = []
    if current_audio_windows_batch_data_list:
        yield torch.vstack(current_audio_windows_batch_data_list)


def get_adversarial_noise(audio_data: np.ndarray, target_label: int, clap_model: laion_clap.hook.CLAP_Module, classifier: nn.Module) -> np.ndarray:
    audio_data = quantize_array(audio_data)
    audio_data_tensor = torch.tensor(audio_data, requires_grad=True)
    clap_model = load_clap_music_model(use_cuda=True)
    audio_windows_batch_embeddings_list: List[torch.Tensor] = []
    for audio_windows_batch_data in get_audio_windows_batch_data_generator(audio_data_tensor, window_size=int(10 * CLAP_SR), hop_size=int(10 * CLAP_SR), max_batch_size=4):
        audio_windows_batch_embeddings_list.append(
            clap_model.get_audio_embedding_from_data(
                x=audio_windows_batch_data,
                use_tensor=True,
            )
        )
    audio_windows_embeddings = torch.vstack(audio_windows_batch_embeddings_list)
    audio_embedding = torch.mean(audio_windows_embeddings, dim=0)
    criterion = nn.BCELoss()
    y_pred = classifier(audio_embedding)
    loss = criterion(y_pred, torch.tensor([target_label]))
    print(loss)
    loss.backward()
    audio_data = audio_data_tensor.detach().cpu().numpy() + 0.01 * audio_data_tensor.grad.detach().cpu().numpy()
