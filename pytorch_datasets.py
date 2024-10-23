import os
from typing import Tuple, List

import torch
import numpy as np
from torch.utils.data import Dataset

from dataset_splits_models import LabeledAudioFilePath
from constants import Label, AI_EMBEDDINGS_DIR_PATH, HUMAN_EMBEDDINGS_DIR_PATH


def load_audio_embeddings(audio_embeddings_file_path: str) -> torch.Tensor:
    return torch.tensor(np.load(audio_embeddings_file_path))


class MeanAudioDataset(Dataset):
    def get_embedding_file_path(self, labeled_audio_file_path: LabeledAudioFilePath) -> str:
        audio_file_name = os.path.basename(labeled_audio_file_path.audio_file_path)
        audio_file_name_prefix = os.path.splitext(audio_file_name)[0]
        if labeled_audio_file_path.label == Label.AI.value:
            embedding_dir_path = AI_EMBEDDINGS_DIR_PATH
        else:
            embedding_dir_path = HUMAN_EMBEDDINGS_DIR_PATH
        return os.path.join(embedding_dir_path, f'{audio_file_name_prefix}.npy')

    def __init__(self, labeled_audio_file_paths: List[LabeledAudioFilePath], device: torch.device) -> None:
        self.audios_embeddings = torch.vstack([
            torch.mean(load_audio_embeddings(self.get_embedding_file_path(labeled_audio_file_path)), dim=0)
            for labeled_audio_file_path in labeled_audio_file_paths
        ]).to(device)
        self.labels = torch.tensor(
            [labeled_audio_file_path.label for labeled_audio_file_path in labeled_audio_file_paths],
            dtype=torch.float,
            device=device,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.audios_embeddings[idx], self.labels[idx].view(1)


class IdentityAudioDataset(Dataset):
    def get_embedding_file_path(self, labeled_audio_file_path: LabeledAudioFilePath) -> str:
        audio_file_name = os.path.basename(labeled_audio_file_path.audio_file_path)
        audio_file_name_prefix = os.path.splitext(audio_file_name)[0]
        if labeled_audio_file_path.label == Label.AI.value:
            embedding_dir_path = AI_EMBEDDINGS_DIR_PATH
        else:
            embedding_dir_path = HUMAN_EMBEDDINGS_DIR_PATH
        return os.path.join(embedding_dir_path, f'{audio_file_name_prefix}.npy')

    def __init__(self, labeled_audio_file_paths: List[LabeledAudioFilePath], device: torch.device) -> None:
        self.audios_embeddings = [
            load_audio_embeddings(self.get_embedding_file_path(labeled_audio_file_path)).to(device)
            for labeled_audio_file_path in labeled_audio_file_paths
        ]
        self.labels = torch.tensor(
            [labeled_audio_file_path.label for labeled_audio_file_path in labeled_audio_file_paths],
            dtype=torch.float,
            device=device,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.audios_embeddings[idx], self.labels[idx].view(1)
