import os

import laion_clap  # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore

from embeddings_computing import load_audio_data_for_clap, get_audio_windows_batch_data_generator, load_clap_music_model
from constants import (
    CLAP_SR, AI_AUDIO_DIR_PATH, HUMAN_AUDIO_DIR_PATH, AI_EMBEDDINGS_DIR_PATH, HUMAN_EMBEDDINGS_DIR_PATH
)


def extract_embeddings(source_dir: str, target_dir: str, model: laion_clap.hook.CLAP_Module) -> None:
    files_names_prefixes = [
        '.'.join(file_name.split('.')[:-1])
        for file_name in os.listdir(source_dir)
    ]
    for file_name_prefix in tqdm(files_names_prefixes, desc=f'Extracting embeddings from {source_dir}'):
        source_file_path = os.path.join(source_dir, f'{file_name_prefix}.mp3')
        target_file_path = os.path.join(target_dir, f'{file_name_prefix}.npy')
        if os.path.exists(target_file_path):
            continue
        audio_data = load_audio_data_for_clap(source_file_path)
        audio_windows_batch_data_generator = get_audio_windows_batch_data_generator(
            audio_data, window_size=int(10 * CLAP_SR), hop_size=int(10 * CLAP_SR), do_quantize=True, max_batch_size=4
        )
        audio_windows_batches_embeddings_list = [
            model.get_audio_embedding_from_data(
                x=audio_windows_batch_data,
                use_tensor=False,
            )
            for audio_windows_batch_data in audio_windows_batch_data_generator
        ]
        audio_embeddings = np.vstack(audio_windows_batches_embeddings_list)
        np.save(target_file_path, audio_embeddings)


if __name__ == '__main__':
    model = load_clap_music_model(use_cuda=False)
    extract_embeddings(source_dir=AI_AUDIO_DIR_PATH, target_dir=AI_EMBEDDINGS_DIR_PATH, model=model)
    extract_embeddings(source_dir=HUMAN_AUDIO_DIR_PATH, target_dir=HUMAN_EMBEDDINGS_DIR_PATH, model=model)
