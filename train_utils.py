from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

import torch
import optuna
from torch import nn
from pydantic import BaseModel
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

from classifiers.mlp import MLP
from constants import CLAP_EMBEDDING_SIZE
from classifiers.lstm import DeepLSTMModel
from classifiers.voting_mlp import VotingMLP
from pytorch_datasets import LabeledAudioFilePath
from classifiers.transformer import TransformerwithMLP
from pytorch_datasets import MeanAudioDataset, IdentityAudioDataset


class TrainingParams(BaseModel):
    pass


class TrainingUtils(ABC):
    model_kind_name: str
    does_model_return_embeddings: bool
    num_epochs: int
    num_hyperparameter_tuning_trials: int

    @staticmethod
    @abstractmethod
    def get_dataset(split: List[LabeledAudioFilePath], device: torch.device) -> Dataset:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_non_train_dataloader(dataset: Dataset) -> DataLoader:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_training_objects(
            training_params: TrainingParams, train_dataset: Dataset, device: torch.device
            ) -> Tuple[DataLoader, nn.Module, Optimizer, Optional[LRScheduler]]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_training_params_for_trial(trial: optuna.trial.Trial) -> TrainingParams:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_training_params_from_study(study: optuna.study.Study) -> TrainingParams:
        raise NotImplementedError


class MLPTrainingUtils(TrainingUtils):
    model_kind_name = 'mlp'
    does_model_return_embeddings = True
    num_epochs = 20
    num_hyperparameter_tuning_trials = 5

    class MLPTrainingParams(TrainingParams):
        do_fit_scaler: bool
        hidden_size_0: int
        hidden_size_1: int
        dropout: float
        train_batch_size: int
        learning_rate: float
        weight_decay: float

    @staticmethod
    def get_dataset(split: List[LabeledAudioFilePath], device: torch.device) -> Dataset:
        return MeanAudioDataset(split, device=device)

    @staticmethod
    def get_non_train_dataloader(dataset: Dataset) -> DataLoader:
        assert isinstance(dataset, MeanAudioDataset)
        return DataLoader(dataset, batch_size=32, shuffle=False)

    @staticmethod
    def get_training_objects(
            training_params: TrainingParams, train_dataset: Dataset, device: torch.device,
            ) -> Tuple[DataLoader, nn.Module, Optimizer, Optional[LRScheduler]]:
        assert (
            isinstance(training_params, MLPTrainingUtils.MLPTrainingParams) and
            isinstance(train_dataset, MeanAudioDataset)
        )
        train_dataloader = DataLoader(train_dataset, batch_size=training_params.train_batch_size, shuffle=True)
        model = MLP(
            input_size=CLAP_EMBEDDING_SIZE,
            hidden_size_0=training_params.hidden_size_0,
            hidden_size_1=training_params.hidden_size_1,
            dropout=training_params.dropout,
        )
        if training_params.do_fit_scaler:
            model.fit_scaler(train_dataset.audios_embeddings)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay
        )
        return train_dataloader, model, optimizer, None

    @staticmethod
    def get_training_params_for_trial(trial: optuna.trial.Trial) -> TrainingParams:
        do_fit_scaler = trial.suggest_categorical('do_fit_scaler', [True, False])
        hidden_size_0_exp = trial.suggest_int('hidden_size_0_exp', 5, 7)
        hidden_size_0 = 2 ** hidden_size_0_exp
        hidden_size_1_exp = trial.suggest_int('hidden_size_1_exp', 4, 6)
        hidden_size_1 = 2 ** hidden_size_1_exp
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        train_batch_size_exp = trial.suggest_int('train_batch_size_exp', 4, 6)
        train_batch_size = 2 ** train_batch_size_exp
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        return MLPTrainingUtils.MLPTrainingParams(
            do_fit_scaler=do_fit_scaler,
            hidden_size_0=hidden_size_0,
            hidden_size_1=hidden_size_1,
            dropout=dropout,
            train_batch_size=train_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    @staticmethod
    def get_training_params_from_study(study: optuna.study.Study) -> TrainingParams:
        return MLPTrainingUtils.MLPTrainingParams(
            do_fit_scaler=study.best_trial.params['do_fit_scaler'],
            hidden_size_0=2 ** study.best_trial.params['hidden_size_0_exp'],
            hidden_size_1=2 ** study.best_trial.params['hidden_size_1_exp'],
            dropout=study.best_trial.params['dropout'],
            train_batch_size=2 ** study.best_trial.params['train_batch_size_exp'],
            learning_rate=study.best_trial.params['learning_rate'],
            weight_decay=study.best_trial.params['weight_decay'],
        )


def zero_pad(batch_samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_windows = max(audio_embeddings.shape[0] for audio_embeddings, _ in batch_samples)
    batch_audio_embeddings_list: List[torch.Tensor] = []
    batch_num_windows_list: List[torch.Tensor] = []
    batch_labels_list: List[torch.Tensor] = []
    for audio_embeddings, label in batch_samples:
        num_windows = audio_embeddings.shape[0]
        padding = torch.zeros((max_num_windows - num_windows, audio_embeddings.shape[1])).to(audio_embeddings.device)
        audio_embeddings = torch.vstack((audio_embeddings, padding))
        batch_audio_embeddings_list.append(audio_embeddings)
        batch_num_windows_list.append(torch.tensor(num_windows).to(audio_embeddings.device))
        batch_labels_list.append(label)
    return torch.stack(batch_audio_embeddings_list), torch.stack(batch_num_windows_list), torch.stack(batch_labels_list)


class VotingMLPTrainingUtils(TrainingUtils):
    model_kind_name = 'voting_mlp'
    does_model_return_embeddings = False
    num_epochs = 25
    num_hyperparameter_tuning_trials = 100

    class VotingMLPTrainingParams(TrainingParams):
        do_fit_scaler: bool
        train_batch_size: int
        learning_rate: float
        weight_decay: float

    @staticmethod
    def get_dataset(split: List[LabeledAudioFilePath], device: torch.device) -> Dataset:
        return IdentityAudioDataset(split, device=device)

    @staticmethod
    def get_non_train_dataloader(dataset: Dataset) -> DataLoader:
        assert isinstance(dataset, IdentityAudioDataset)
        return DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=zero_pad)

    @staticmethod
    def get_training_objects(
            training_params: TrainingParams, train_dataset: Dataset, device: torch.device,
            ) -> Tuple[DataLoader, nn.Module, Optimizer, Optional[LRScheduler]]:
        assert (
            isinstance(training_params, VotingMLPTrainingUtils.VotingMLPTrainingParams) and
            isinstance(train_dataset, IdentityAudioDataset)
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=training_params.train_batch_size, shuffle=True, collate_fn=zero_pad
        )
        model = VotingMLP(input_size=CLAP_EMBEDDING_SIZE)
        if training_params.do_fit_scaler:
            model.fit_scaler(train_dataset.audios_embeddings)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay
        )
        return train_dataloader, model, optimizer, None

    @staticmethod
    def get_training_params_for_trial(trial: optuna.trial.Trial) -> TrainingParams:
        do_fit_scaler = trial.suggest_categorical('do_fit_scaler', [True, False])
        train_batch_size_exp = trial.suggest_int('train_batch_size_exp', 4, 6)
        train_batch_size = 2 ** train_batch_size_exp
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        return VotingMLPTrainingUtils.VotingMLPTrainingParams(
            do_fit_scaler=do_fit_scaler,
            train_batch_size=train_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    @staticmethod
    def get_training_params_from_study(study: optuna.study.Study) -> TrainingParams:
        return VotingMLPTrainingUtils.VotingMLPTrainingParams(
            do_fit_scaler=study.best_trial.params['do_fit_scaler'],
            train_batch_size=2 ** study.best_trial.params['train_batch_size_exp'],
            learning_rate=study.best_trial.params['learning_rate'],
            weight_decay=study.best_trial.params['weight_decay'],
        )


class LSTMTrainingUtils(TrainingUtils):
    model_kind_name = 'lstm'
    does_model_return_embeddings = False
    num_epochs = 25
    num_hyperparameter_tuning_trials = 100

    class LSTMTrainingParams(TrainingParams):
        do_fit_scaler: bool
        train_batch_size: int
        learning_rate: float
        weight_decay: float
        hidden_size: int
        num_layers: int
        dropout: float
        bidirectional: bool

    @staticmethod
    def get_dataset(split: List[LabeledAudioFilePath], device: torch.device) -> Dataset:
        return IdentityAudioDataset(split, device=device)

    @staticmethod
    def get_non_train_dataloader(dataset: Dataset) -> DataLoader:
        assert isinstance(dataset, IdentityAudioDataset)
        return DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=zero_pad)

    @staticmethod
    def get_training_objects(
            training_params: TrainingParams, train_dataset: Dataset, device: torch.device,
            ) -> Tuple[DataLoader, nn.Module, Optimizer, Optional[LRScheduler]]:
        assert (
            isinstance(training_params, LSTMTrainingUtils.LSTMTrainingParams) and
            isinstance(train_dataset, IdentityAudioDataset)
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=training_params.train_batch_size, shuffle=True, collate_fn=zero_pad
        )
        model = DeepLSTMModel(
            input_size=CLAP_EMBEDDING_SIZE,
            hidden_size=training_params.hidden_size,
            num_layers=training_params.num_layers,
            dropout=training_params.dropout,
            bidirectional=training_params.bidirectional,
        )
        if training_params.do_fit_scaler:
            model.fit_scaler(train_dataset.audios_embeddings)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay
        )
        return train_dataloader, model, optimizer, None

    @staticmethod
    def get_training_params_for_trial(trial: optuna.trial.Trial) -> TrainingParams:
        do_fit_scaler = trial.suggest_categorical('do_fit_scaler', [True, False])
        train_batch_size_exp = trial.suggest_int('train_batch_size_exp', 4, 6)
        train_batch_size = 2 ** train_batch_size_exp
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        hidden_size = 2 ** trial.suggest_int('hidden_size_exp', 6, 8)
        num_layers = trial.suggest_int('num_layers', 2, 3)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        bidirectional = trial.suggest_categorical('bidirectional', [True, False])
        return LSTMTrainingUtils.LSTMTrainingParams(
            do_fit_scaler=do_fit_scaler,
            train_batch_size=train_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    @staticmethod
    def get_training_params_from_study(study: optuna.study.Study) -> TrainingParams:
        return LSTMTrainingUtils.LSTMTrainingParams(
            do_fit_scaler=study.best_trial.params['do_fit_scaler'],
            train_batch_size=2 ** study.best_trial.params['train_batch_size_exp'],
            learning_rate=study.best_trial.params['learning_rate'],
            weight_decay=study.best_trial.params['weight_decay'],
            hidden_size=2 ** study.best_trial.params['hidden_size_exp'],
            num_layers=study.best_trial.params['num_layers'],
            dropout=study.best_trial.params['dropout'],
            bidirectional=study.best_trial.params['bidirectional'],
        )


class TransformerTrainingUtils(TrainingUtils):
    model_kind_name = 'transformer'
    does_model_return_embeddings = True
    num_epochs = 25
    num_hyperparameter_tuning_trials = 100

    class TransformerTrainingParams(TrainingParams):
        train_batch_size: int
        learning_rate: float
        weight_decay: float
        T_max: int
        eta_min: float
        dim_ffn: int
        num_heads: int

    @staticmethod
    def get_dataset(split: List[LabeledAudioFilePath], device: torch.device) -> Dataset:
        return IdentityAudioDataset(split, device=device)

    @staticmethod
    def get_non_train_dataloader(dataset: Dataset) -> DataLoader:
        assert isinstance(dataset, IdentityAudioDataset)
        return DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=zero_pad)

    @staticmethod
    def get_training_objects(
            training_params: TrainingParams, train_dataset: Dataset, device: torch.device,
            ) -> Tuple[DataLoader, nn.Module, Optimizer, Optional[LRScheduler]]:
        assert (
            isinstance(training_params, TransformerTrainingUtils.TransformerTrainingParams) and
            isinstance(train_dataset, IdentityAudioDataset)
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=training_params.train_batch_size, shuffle=True, collate_fn=zero_pad
        )
        model = TransformerwithMLP(
            input_size=CLAP_EMBEDDING_SIZE, num_heads=training_params.num_heads, dim_ffn=training_params.dim_ffn
        )
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=training_params.T_max, eta_min=training_params.eta_min)
        return train_dataloader, model, optimizer, scheduler

    @staticmethod
    def get_training_params_for_trial(trial: optuna.trial.Trial) -> TrainingParams:
        train_batch_size = 2 ** trial.suggest_int('train_batch_size_exp', 4, 6)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        T_max = trial.suggest_int('T_max', 10, 50)
        eta_min = trial.suggest_float('eta_min', 1e-6, 1e-3, log=True)
        dim_ffn = 2 ** trial.suggest_int('dim_ffn_exp', 7, 10)
        num_heads = 2 ** trial.suggest_int('num_heads_exp', 0, 3)
        return TransformerTrainingUtils.TransformerTrainingParams(
            train_batch_size=train_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            T_max=T_max,
            eta_min=eta_min,
            dim_ffn=dim_ffn,
            num_heads=num_heads,
        )

    @staticmethod
    def get_training_params_from_study(study: optuna.study.Study) -> TrainingParams:
        return TransformerTrainingUtils.TransformerTrainingParams(
            train_batch_size=2 ** study.best_trial.params['train_batch_size_exp'],
            learning_rate=study.best_trial.params['learning_rate'],
            weight_decay=study.best_trial.params['weight_decay'],
            T_max=study.best_trial.params['T_max'],
            eta_min=study.best_trial.params['eta_min'],
            dim_ffn=2 ** study.best_trial.params['dim_ffn_exp'],
            num_heads=2 ** study.best_trial.params['num_heads_exp'],
        )
