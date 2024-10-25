from __future__ import annotations

import os
import math
import json
import time
import string
import random
from collections.abc import Iterable
from typing import Tuple, List, Optional

import torch
import laion_clap  # type: ignore
import numpy as np
import torch.nn as nn
from tqdm import tqdm  # type: ignore
import soundfile as sf  # type: ignore

from classifiers.mlp import MLP
from utils import seed_everything
from train_mlp import TrainingParams
from dataset_splits_models import DatasetSplits
from embeddings_computing import load_clap_music_model
from embeddings_computing import load_audio_data_for_clap
from adversarial_models import (
    AdversarialInterationResult, AdversarialResult, AdversarialAttacker, AdversarialExperimentParams,
    AdversarialExperiment,
)
from constants import (
    CLAP_SR, Label, ADVERSARIAL_AUDIO_DIR_PATH, DATASET_SPLITS_FILE_PATH, ADVERSARIAL_EXPERIMENTS_DIR_PATH,
    CLASSIFIERS_CHECKPOINTS_DIR_PATH, CLAP_EMBEDDING_SIZE, MAX_ADVERSARIAL_AUDIO_DURATION_SECS,
)


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
            self.clap_model.get_audio_embedding_from_data(
                x=audio_windows_batch_data,
                use_tensor=True,
            )
            for audio_windows_batch_data in get_audio_windows_batch_data_generator(
                audio_data, self.window_size, self.hop_size, self.max_batch_size
            )
        ])


def get_signal_power(audio_data: torch.Tensor) -> float:
    return torch.mean(audio_data ** 2).item()


def get_noise_audio_data(pure_audio_data: torch.Tensor, adversarial_audio_data: torch.Tensor) -> torch.Tensor:
    return pure_audio_data - adversarial_audio_data


def get_snr(pure_audio_data: torch.Tensor, noise_audio_data: torch.Tensor) -> float:
    return 10 * math.log10(get_signal_power(pure_audio_data) / get_signal_power(noise_audio_data))


def limit_snr(
        pure_audio_data: torch.Tensor,
        adversarial_audio_data: torch.Tensor,
        min_snr: Optional[float],
        ) -> Tuple[torch.Tensor, float]:
    noise_audio_data = get_noise_audio_data(pure_audio_data, adversarial_audio_data)
    snr = get_snr(pure_audio_data, noise_audio_data)
    if min_snr is not None and snr < min_snr:
        noise_audio_data *= 10. ** ((snr - min_snr) / 20)
        adversarial_audio_data = pure_audio_data + noise_audio_data
        snr = min_snr
    return adversarial_audio_data, snr


class SNRProjector:
    def __init__(self, min_snr: Optional[float]) -> None:
        self.min_snr = min_snr

    def __call__(
            self, pure_audio_data: torch.Tensor, adversarial_audio_data: torch.Tensor
            ) -> Tuple[torch.Tensor, float]:
        return limit_snr(pure_audio_data, adversarial_audio_data, self.min_snr)


def get_target_pred_confidence(y_pred: torch.Tensor, target_label: int) -> float:
    if target_label:
        return y_pred.item()
    return 1 - y_pred.item()


class PGDAdversarialAttacker(AdversarialAttacker):
    def __init__(
            self,
            audio_windows_embeddings_extractor: AudioWindowsEmbeddingsExtractor,
            classifier: nn.Module,
            max_iter: int,
            required_target_pred_confidence: float,
            learning_rate: float,
            snr_projector: SNRProjector,
            ) -> None:
        self.audio_windows_embeddings_extractor = audio_windows_embeddings_extractor
        self.classifier = classifier
        self.max_iter = max_iter
        self.required_target_pred_confidence = required_target_pred_confidence
        self.learning_rate = learning_rate
        self.snr_projector = snr_projector
        self.criterion = nn.BCELoss()

    def __call__(
            self,
            audio_file_path: str,
            target_label: int,
            ) -> Tuple[AdversarialResult, np.ndarray]:
        audio_data = load_audio_data_for_clap(audio_file_path)
        start_timestamp = time.time()
        pure_audio_data = torch.tensor(audio_data, device='cuda:0')
        adversarial_audio_data = torch.tensor(audio_data, requires_grad=True, device='cuda:0')
        target_label_tensor = torch.tensor([target_label], dtype=torch.float, device='cuda:0')
        self.audio_windows_embeddings_extractor.clap_model.eval()
        self.classifier.eval()
        init_target_pred_confidence = None
        max_target_pred_confidence = float('-inf')
        argmax_adversarial_audio_data = None
        snr = float('inf')
        adversarial_iterations_results: List[AdversarialInterationResult] = []
        for iter_idx in range(self.max_iter):
            audio_windows_embeddings = self.audio_windows_embeddings_extractor(adversarial_audio_data)
            audio_embedding = torch.mean(audio_windows_embeddings, dim=0)
            y_pred = self.classifier(audio_embedding)[0]
            loss = self.criterion(y_pred, target_label_tensor)
            target_pred_confidence = get_target_pred_confidence(y_pred, target_label)
            adversarial_iterations_results.append(AdversarialInterationResult(
                target_pred_confidence=target_pred_confidence,
                snr=snr,
                duration_secs_since_attack_start=time.time() - start_timestamp,
            ))
            print(f'Iter {iter_idx}:')
            print(adversarial_iterations_results[-1].model_dump_json(indent=4))
            if init_target_pred_confidence is None:
                init_target_pred_confidence = target_pred_confidence
            if target_pred_confidence > max_target_pred_confidence:
                max_target_pred_confidence = target_pred_confidence
                argmax_adversarial_audio_data = adversarial_audio_data.detach().cpu().numpy()
                if target_pred_confidence >= self.required_target_pred_confidence:
                    break
            loss.backward()
            assert adversarial_audio_data.grad is not None
            adversarial_audio_data = adversarial_audio_data.detach() - self.learning_rate * adversarial_audio_data.grad
            adversarial_audio_data, snr = self.snr_projector(pure_audio_data, adversarial_audio_data.detach())
            adversarial_audio_data.requires_grad = True
        assert argmax_adversarial_audio_data is not None
        return (
            AdversarialResult(
                audio_file_path=audio_file_path,
                iterations_results=adversarial_iterations_results,
            ),
            argmax_adversarial_audio_data,
        )


def is_adversarial_experiment_covered_check(adversarial_experiment_params: AdversarialExperimentParams) -> bool:
    adversarial_experiments_files_names = os.listdir(ADVERSARIAL_EXPERIMENTS_DIR_PATH)
    for other_adversarial_experiment_file_name in adversarial_experiments_files_names:
        other_adversarial_experiment_file_path = os.path.join(
            ADVERSARIAL_EXPERIMENTS_DIR_PATH, other_adversarial_experiment_file_name
        )
        with open(other_adversarial_experiment_file_path) as f:
            other_adversarial_experiment = AdversarialExperiment.model_validate_json(f.read())
        other_adversarial_experiment_params = other_adversarial_experiment.params
        if (other_adversarial_experiment_params.window_size == adversarial_experiment_params.window_size and
                other_adversarial_experiment_params.hop_size == adversarial_experiment_params.hop_size and
                other_adversarial_experiment_params.min_snr == adversarial_experiment_params.min_snr and
                other_adversarial_experiment_params.max_iter >= adversarial_experiment_params.max_iter and
                other_adversarial_experiment_params.required_target_pred_confidence >= adversarial_experiment_params.required_target_pred_confidence and
                other_adversarial_experiment_params.learning_rate == adversarial_experiment_params.learning_rate):
            return True
    return False


def should_skip_audio_file_check(audio_file_path: str) -> bool:
    return len(load_audio_data_for_clap(audio_file_path)) > CLAP_SR * MAX_ADVERSARIAL_AUDIO_DURATION_SECS


def run_adversarial_experiment(
        adversarial_experiment_params: AdversarialExperimentParams,
        audio_files_paths: List[str],
        clap_model: laion_clap.hook.CLAP_Module,
        classifier: MLP,
        ) -> Optional[AdversarialExperiment]:
    if is_adversarial_experiment_covered_check(adversarial_experiment_params):
        print('Skipping adversarial experiment that already exists with params:')
        print(adversarial_experiment_params.model_dump_json(indent=4))
        print()
        return None
    print('Running adversarial experiment with params:')
    print(adversarial_experiment_params.model_dump_json(indent=4))
    print()
    audio_windows_embeddings_extractor = AudioWindowsEmbeddingsExtractor(
        window_size=adversarial_experiment_params.window_size,
        hop_size=adversarial_experiment_params.hop_size,
        max_batch_size=4,
        clap_model=clap_model,
    )
    snr_projector = SNRProjector(min_snr=adversarial_experiment_params.min_snr)
    pgd_adversarial_attacker = PGDAdversarialAttacker(
        audio_windows_embeddings_extractor=audio_windows_embeddings_extractor,
        classifier=classifier,
        max_iter=adversarial_experiment_params.max_iter,
        required_target_pred_confidence=adversarial_experiment_params.required_target_pred_confidence,
        learning_rate=adversarial_experiment_params.learning_rate,
        snr_projector=snr_projector,
    )
    adversarial_results: List[AdversarialResult] = []
    for audio_file_path in tqdm(audio_files_paths, desc='Generating adversarial attacks'):
        audio_file_name = os.path.basename(audio_file_path)
        print()
        print(f'Processing {audio_file_name}')
        if should_skip_audio_file_check(audio_file_path):
            print('Skipping!')
            continue
        adversarial_result, argmax_adversarial_audio_data = pgd_adversarial_attacker(
            audio_file_path=audio_file_path, target_label=Label.HUMAN.value
        )
        adversarial_results.append(adversarial_result)
        # print(adversarial_result.model_dump_json(
        #     indent=4, exclude={'audio_file_path'}
        # ))
        # adversarial_audio_file_path = os.path.join(ADVERSARIAL_AUDIO_DIR_PATH, audio_file_name)
        # sf.write(adversarial_audio_file_path, data=argmax_adversarial_audio_data, samplerate=CLAP_SR)
        print()
    return AdversarialExperiment(
        params=adversarial_experiment_params,
        results=adversarial_results,
    )


def get_random_hex_string(length: int) -> str:
    return ''.join(random.choices(string.hexdigits, k=length))


def main() -> None:
    seed_everything(42)

    with open(DATASET_SPLITS_FILE_PATH) as f:
        dataset_splits = DatasetSplits.model_validate_json(f.read())

    audio_files_paths = [
        labeled_audio_file_path.audio_file_path
        for labeled_audio_file_path in dataset_splits.test
        if labeled_audio_file_path.label == Label.AI.value
    ]

    clap_model = load_clap_music_model(use_cuda=True)

    classifier_checkpoint_dir_path = os.path.join(CLASSIFIERS_CHECKPOINTS_DIR_PATH, 'mlp')
    classifier_training_params_file_path = os.path.join(classifier_checkpoint_dir_path, 'training_params.json')
    classifier_checkpoint_file_path = os.path.join(classifier_checkpoint_dir_path, 'classifier.pt')
    with open(classifier_training_params_file_path) as f:
        training_params = TrainingParams.model_validate_json(f.read())
    classifier = MLP(
        input_size=CLAP_EMBEDDING_SIZE,
        hidden_size_0=training_params.hidden_size_0,
        hidden_size_1=training_params.hidden_size_1,
        dropout=training_params.dropout,
    ).to('cuda:0')
    classifier.load_state_dict(torch.load(classifier_checkpoint_file_path, weights_only=True))

    window_size = int(10 * CLAP_SR)
    hop_size = int(10 * CLAP_SR)
    max_iter = 50
    required_target_pred_confidence = 0.9
    min_snr_values = [None, 50., 60.]
    learning_rate_values = [1e-6, 1e-5, 1e-4]

    for min_snr in min_snr_values:
        for learning_rate in learning_rate_values:
            adversarial_experiment_params = AdversarialExperimentParams(
                window_size=window_size,
                hop_size=hop_size,
                min_snr=min_snr,
                max_iter=max_iter,
                required_target_pred_confidence=required_target_pred_confidence,
                learning_rate=learning_rate,
            )
            adversarial_experiment = run_adversarial_experiment(
                adversarial_experiment_params=adversarial_experiment_params,
                audio_files_paths=audio_files_paths,
                clap_model=clap_model,
                classifier=classifier,
            )
            if adversarial_experiment is not None:
                adversarial_experiment_file_name = f'{get_random_hex_string(16)}_{int(time.time())}.json'
                adversarial_experiment_file_path = os.path.join(
                    ADVERSARIAL_EXPERIMENTS_DIR_PATH, adversarial_experiment_file_name
                )
                assert not os.path.exists(adversarial_experiment_file_path)
                with open(adversarial_experiment_file_path, 'w') as f:
                    json.dump(adversarial_experiment.model_dump(), f, indent=4)
                print(f'Saved adversarial experiment to {adversarial_experiment_file_path}')


if __name__ == '__main__':
    main()
