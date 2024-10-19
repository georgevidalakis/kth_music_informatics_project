from typing import List
from collections.abc import Iterable

import torch
import librosa
import laion_clap  # type: ignore
import numpy as np

from constants import CLAP_SR


# quantization
def int16_to_float32(x) -> np.ndarray:
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x) -> np.ndarray:
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def quantize_array(a: np.ndarray) -> np.ndarray:
    return int16_to_float32(float32_to_int16(a))


def load_clap_music_model(use_cuda: bool) -> laion_clap.hook.CLAP_Module:
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Please use CPU instead.')
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device, amodel='HTSAT-base')
    model.load_ckpt(ckpt='clap_checkpoints/music_audioset_epoch_15_esc_90.14.pt')
    return model


def load_audio_data_for_clap(audio_file_path: str) -> np.ndarray:
    original_audio_data, original_sr = librosa.load(audio_file_path)
    # print(original_audio_data[:10],original_audio_data[-10:] ,original_audio_data.shape,original_audio_data.dtype)
    # np.save('original_audio_data.npy', original_audio_data)
    # exit()
    return librosa.resample(original_audio_data, orig_sr=original_sr, target_sr=CLAP_SR, fix=False)


def get_audio_windows_data(audio_data: np.ndarray, window_size: int, hop_size: int, do_quantize: bool) -> np.ndarray:
    audio_windows_data_list: List[np.ndarray] = []
    for window_start_idx in range(0, len(audio_data), hop_size):
        window_end_idx = window_start_idx + window_size
        audio_window_data = audio_data[window_start_idx:window_end_idx]
        zero_pad_size = window_size - len(audio_window_data)
        audio_window_data = np.pad(audio_window_data, pad_width=(0, zero_pad_size))
        if do_quantize:
            audio_window_data = quantize_array(audio_window_data)
        audio_windows_data_list.append(audio_window_data)
    return np.vstack(audio_windows_data_list)


# generator with appropriate type hinting
def get_audio_windows_batch_data_generator(
        audio_data: np.ndarray, window_size: int, hop_size: int, do_quantize: bool, max_batch_size: int
        ) -> Iterable[np.ndarray]:
    current_audio_windows_batch_data_list: List[np.ndarray] = []
    for window_start_idx in range(0, len(audio_data), hop_size):
        window_end_idx = window_start_idx + window_size
        audio_window_data = audio_data[window_start_idx:window_end_idx]
        zero_pad_size = window_size - len(audio_window_data)
        audio_window_data = np.pad(audio_window_data, pad_width=(0, zero_pad_size))
        if do_quantize:
            audio_window_data = quantize_array(audio_window_data)
        current_audio_windows_batch_data_list.append(audio_window_data)
        if len(current_audio_windows_batch_data_list) == max_batch_size:
            yield np.vstack(current_audio_windows_batch_data_list)
            current_audio_windows_batch_data_list = []
    if current_audio_windows_batch_data_list:
        yield np.vstack(current_audio_windows_batch_data_list)


if __name__ == '__main__':
    audio_data = load_audio_data_for_clap(audio_file_path='example.mp3')
    audio_windows_data = get_audio_windows_data(
        audio_data, window_size=int(10 * CLAP_SR), hop_size=int(10 * CLAP_SR), do_quantize=True
    )
    model = load_clap_music_model(use_cuda=True)
    audios_embeddings = model.get_audio_embedding_from_data(
        x=audio_windows_data,
        use_tensor=False,
    )
