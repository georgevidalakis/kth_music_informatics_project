import math
from typing import List
from collections.abc import Iterable

import laion_clap  # type: ignore
import numpy as np
import torch
import torch.nn as nn

from embeddings_computing import quantize_array, load_clap_music_model
from constants import CLAP_SR, Label
from embeddings_computing import load_audio_data_for_clap
from feed_forward import FeedForward


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


def get_signal_power(audio_data: torch.Tensor) -> float:
    return torch.mean(audio_data ** 2).item()


def get_snr(pure_audio_data: torch.Tensor, adversarial_audio_data: torch.Tensor) -> float:
    noise_audio_data = pure_audio_data - adversarial_audio_data
    return 10 * math.log10(get_signal_power(pure_audio_data) / get_signal_power(noise_audio_data))


def get_adversarial_audio_data(
        audio_data: np.ndarray, target_label: int, clap_model: laion_clap.hook.CLAP_Module, classifier: nn.Module,
        learning_rate: float, max_iter: int,
        ) -> np.ndarray:
    initial_audio_data_tensor = torch.from_numpy(audio_data)
    audio_data = quantize_array(audio_data)
    audio_data_tensor = torch.tensor(audio_data, requires_grad=True, device='cuda:0')
    target_label_tensor = torch.tensor([target_label], dtype=torch.float, device='cuda:0')
    for iter_idx in range(max_iter):
        audio_windows_batch_embeddings_list: List[torch.Tensor] = []
        audio_windows_batch_data_generator = get_audio_windows_batch_data_generator(
            audio_data_tensor, window_size=int(10 * CLAP_SR), hop_size=int(10 * CLAP_SR), max_batch_size=4
        )
        for audio_windows_batch_data in audio_windows_batch_data_generator:
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
        loss = criterion(y_pred, target_label_tensor)
        snr = get_snr(initial_audio_data_tensor, audio_data_tensor.detach().cpu())
        print(f'Iter {iter_idx}, y_pred {y_pred.item()}, loss {loss}, snr {snr}')
        loss.backward()
        assert audio_data_tensor.grad is not None
        audio_data_tensor = audio_data_tensor.detach() - learning_rate * audio_data_tensor.grad
        audio_data_tensor.requires_grad = True
        # reset gradients
        clap_model.zero_grad()
        classifier.zero_grad()
    return audio_data_tensor.detach().cpu().numpy()


if __name__ == '__main__':
    audio_data = load_audio_data_for_clap(audio_file_path='example.mp3')
    clap_model = load_clap_music_model(use_cuda=True)
    classifier = FeedForward(input_size=512).to('cuda:0')
    classifier.load_state_dict(torch.load('classifier_checkpoints/feed_forward.pt'))
    adversarial_audio_data = get_adversarial_audio_data(
        audio_data=audio_data,
        target_label=Label.HUMAN.value,
        clap_model=clap_model,
        classifier=classifier,
        learning_rate=0.001,
        max_iter=50,
    )
    # save adversarial audio as audio file
    import soundfile as sf
    sf.write('adversarial_example.wav', data=adversarial_audio_data, samplerate=CLAP_SR)
