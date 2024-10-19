from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Tuple, List, Optional

import torch
import laion_clap  # type: ignore
import numpy as np
import torch.nn as nn
import soundfile as sf  # type: ignore

from constants import CLAP_SR, Label
from feed_forward import FeedForward
from embeddings_computing import load_audio_data_for_clap
from embeddings_computing import quantize_array, load_clap_music_model


class AudioWindowsDataIterator:
    def __init__(self, audio_data: torch.Tensor, window_size: int, hop_size: int) -> None:
        self.audio_data = audio_data
        self.window_size = window_size
        self.hop_size = hop_size

    def zero_pad(self, audio_window_data: torch.Tensor) -> torch.Tensor:
        zero_pad_size = self.window_size - len(audio_window_data)
        return nn.ZeroPad1d((0, zero_pad_size))(audio_window_data)

    def __iter__(self) -> AudioWindowsDataIterator:
        self.current_window_start_idx = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self.current_window_start_idx >= len(self.audio_data):
            raise StopIteration
        window_end_idx = self.current_window_start_idx + self.window_size
        audio_window_data = self.audio_data[self.current_window_start_idx:window_end_idx]
        self.current_window_start_idx += self.hop_size
        return self.zero_pad(audio_window_data)


def get_audio_windows_batch_data_generator(
        audio_data: torch.Tensor, window_size: int, hop_size: int, max_batch_size: int
        ) -> Iterable[torch.Tensor]:
    current_audio_windows_batch_data_list: List[torch.Tensor] = []
    for audio_window_data in AudioWindowsDataIterator(audio_data, window_size, hop_size):
        current_audio_windows_batch_data_list.append(audio_window_data)
        if len(current_audio_windows_batch_data_list) == max_batch_size:
            yield torch.vstack(current_audio_windows_batch_data_list)
            current_audio_windows_batch_data_list = []
    if current_audio_windows_batch_data_list:
        yield torch.vstack(current_audio_windows_batch_data_list)


class AudioWindowsEmbeddingsExtractor:
    def __init__(
            self,
            window_size: int,
            hop_size: int,
            max_batch_size: int,
            clap_model: laion_clap.hook.CLAP_Module,
            ) -> None:
        self.window_size = window_size
        self.hop_size = hop_size
        self.max_batch_size = max_batch_size
        self.clap_model = clap_model

    def __call__(self, audio_data: torch.Tensor) -> torch.Tensor:
        return torch.vstack([
            clap_model.get_audio_embedding_from_data(
                x=audio_windows_batch_data,
                use_tensor=True,
            )
            for audio_windows_batch_data in get_audio_windows_batch_data_generator(
                audio_data, self.window_size, self.hop_size, self.max_batch_size
            )
        ])


def get_signal_power(audio_data: torch.Tensor) -> float:
    return torch.mean(audio_data ** 2).item()


def get_snr(pure_audio_data: torch.Tensor, adversarial_audio_data: torch.Tensor) -> float:
    noise_audio_data = pure_audio_data - adversarial_audio_data
    return 10 * math.log10(get_signal_power(pure_audio_data) / get_signal_power(noise_audio_data))


def limit_snr(
        pure_audio_data: torch.Tensor,
        adversarial_audio_data: torch.Tensor,
        min_snr: Optional[float],
        ) -> Tuple[torch.Tensor, float]:
    snr = get_snr(pure_audio_data, adversarial_audio_data)
    noise_audio_data = pure_audio_data - adversarial_audio_data
    if min_snr is not None and snr < min_snr:
        noise_audio_data *= 10. ** ((snr - min_snr) / 20)
        adversarial_audio_data = pure_audio_data + noise_audio_data
        snr = get_snr(pure_audio_data, adversarial_audio_data)
    return adversarial_audio_data, snr


def get_adversarial_audio_data(
        audio_data: np.ndarray,
        target_label: int,
        window_size: int,
        hop_size: int,
        max_batch_size: int,
        min_snr: Optional[float],
        clap_model: laion_clap.hook.CLAP_Module,
        classifier: nn.Module,
        learning_rate: float,
        max_iter: int,
        ) -> np.ndarray:
    pure_audio_data = torch.from_numpy(audio_data)
    audio_data = quantize_array(audio_data)
    adversarial_audio_data = torch.tensor(audio_data, requires_grad=True, device='cuda:0')
    target_label_tensor = torch.tensor([target_label], dtype=torch.float, device='cuda:0')
    audio_windows_embeddings_extractor = AudioWindowsEmbeddingsExtractor(
        window_size, hop_size, max_batch_size, clap_model
    )
    clap_model.eval()
    classifier.eval()
    criterion = nn.BCELoss()
    for iter_idx in range(max_iter):
        audio_windows_embeddings = audio_windows_embeddings_extractor(adversarial_audio_data)
        audio_embedding = torch.mean(audio_windows_embeddings, dim=0)
        y_pred = classifier(audio_embedding)
        loss = criterion(y_pred, target_label_tensor)
        loss.backward()
        assert adversarial_audio_data.grad is not None
        adversarial_audio_data = adversarial_audio_data.detach() - learning_rate * adversarial_audio_data.grad
        adversarial_audio_data, snr = limit_snr(pure_audio_data, adversarial_audio_data.detach().cpu(), min_snr=min_snr)
        print(f'Iter {iter_idx}, y_pred {y_pred.item():.5f}, loss {loss:.5f}, snr {snr:.5f}')
        adversarial_audio_data.requires_grad = True
        # reset gradients
        clap_model.zero_grad()
        classifier.zero_grad()
    return adversarial_audio_data.detach().cpu().numpy()


if __name__ == '__main__':
    audio_data = load_audio_data_for_clap(audio_file_path='example.mp3')
    clap_model = load_clap_music_model(use_cuda=True)
    classifier = FeedForward(input_size=512).to('cuda:0')
    classifier.load_state_dict(torch.load('classifier_checkpoints/feed_forward.pt'))
    adversarial_audio_data = get_adversarial_audio_data(
        audio_data=audio_data,
        target_label=Label.HUMAN.value,
        window_size=int(10 * CLAP_SR),
        hop_size=int(10 * CLAP_SR),
        max_batch_size=4,
        min_snr=65.,
        clap_model=clap_model,
        classifier=classifier,
        learning_rate=0.001,
        max_iter=1000,
    )
    # save adversarial audio as audio file
    sf.write('adversarial_example.wav', data=adversarial_audio_data, samplerate=CLAP_SR)
