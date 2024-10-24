import os
from typing import Tuple, List

import torch
import optuna
import numpy as np
import torch.nn as nn
from pydantic import BaseModel
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (  # type: ignore
    roc_auc_score, roc_curve, f1_score, ConfusionMatrixDisplay, confusion_matrix, accuracy_score
)

from classifiers.mlp import MLP
from utils import seed_everything
from dataset_splits_models import DatasetSplits, LabeledAudioFilePath
from pytorch_datasets import MeanAudioDataset
from constants import (
    NUM_MLP_EPOCHS, TRAINING_METRICS_DIR_PATH, DATASET_SPLITS_FILE_PATH, CLAP_EMBEDDING_SIZE, Metric,
    OPTIMIZATION_METRIC, CLASSIFIERS_CHECKPOINTS_DIR_PATH, NUM_MLP_HYPERPARAMETER_TUNING_TRIALS,
)


def train_epoch(
        model: nn.Module, train_dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer
        ) -> None:
    model.train()
    for X_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch)[0]
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()


def get_y_and_y_pred_and_loss(model: nn.Module, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, float]:
    y_list: List[float] = []
    y_pred_list: List[float] = []
    loss_sum = 0.
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        for batch_embeddings, batch_labels in dataloader:
            batch_pred_confidences = model(batch_embeddings)[0]
            batch_pred_labels = torch.round(batch_pred_confidences)
            batch_loss = criterion(batch_pred_labels, batch_labels).item()
            y_list += batch_labels.cpu().squeeze(1).tolist()
            y_pred_list += batch_pred_labels.cpu().squeeze(1).tolist()
            loss_sum += batch_loss
    return np.array(y_list), np.array(y_pred_list), float(loss_sum / len(dataloader))


def save_confusion_matrix_plot(y_test: np.ndarray, y_pred_test: np.ndarray) -> None:
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AI', 'Human'])
    disp.plot()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'mlp_confusion_matrix.png'))
    plt.close()


def save_losses_plot(train_losses: List[float], val_losses: List[float]) -> None:
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'mlp_training_losses.png'))
    plt.close()


def save_accuracies_plot(train_accuracies: List[float], val_accuracies: List[float]) -> None:
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'mlp_training_accuracies.png'))
    plt.close()


def save_roc_curve_plot(
        y_train: np.ndarray, y_pred_train: np.ndarray, y_val: np.ndarray, y_pred_val: np.ndarray
        ) -> None:
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)
    plt.plot(fpr_train, tpr_train, label='Train ROC curve')
    plt.plot(fpr_val, tpr_val, label='Validation ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'mlp_train_val_roc_curve.png'))
    plt.close()


def get_model_embeddings(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    embeddings_list: List[np.ndarray] = []
    for X_batch, _ in dataloader:
        batch_embeddings = model(X_batch)[1]
        embeddings_list.append(batch_embeddings.detach().cpu().numpy())
    return np.vstack(embeddings_list)


def save_training_results(
        test_embeddings: np.ndarray,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: List[float],
        val_accuracies: List[float],
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_val: np.ndarray,
        y_pred_val: np.ndarray
        ) -> None:
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'embeddings.npy'), test_embeddings)
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'val_losses.npy'), np.array(val_losses))
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'train_accuracies.npy'), np.array(train_accuracies))
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'val_accuracies.npy'), np.array(val_accuracies))
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_pred_train.npy'), y_pred_train)
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_pred_val.npy'), y_pred_val)


class TrainingParams(BaseModel):
    do_fit_scaler: bool
    hidden_size_0: int
    hidden_size_1: int
    dropout: float
    train_batch_size: int
    learning_rate: float
    weight_decay: float


def eval_model_training_params(
        training_params: TrainingParams,
        optimization_metric: Metric,
        device: torch.device,
        train_dataset: MeanAudioDataset,
        val_dataset: MeanAudioDataset,
        ) -> float:
    train_dataloader = DataLoader(train_dataset, batch_size=training_params.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MLP(
        input_size=CLAP_EMBEDDING_SIZE,
        hidden_size_0=training_params.hidden_size_0,
        hidden_size_1=training_params.hidden_size_1,
        dropout=training_params.dropout,
    )
    if training_params.do_fit_scaler:
        model.fit_scaler(train_dataset.audios_embeddings)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay
    )

    max_val_accuracy = float('-inf')
    min_val_loss = float('inf')

    for _ in range(NUM_MLP_EPOCHS):
        train_epoch(model, train_dataloader, criterion, optimizer)
        y_val, y_pred_val, val_loss = get_y_and_y_pred_and_loss(model, val_dataloader)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        max_val_accuracy = max(max_val_accuracy, val_accuracy)
        min_val_loss = min(min_val_loss, val_loss)

    metrics_values = {
        Metric.ACCURACY: max_val_accuracy,
        Metric.LOSS: min_val_loss,
    }
    return metrics_values[optimization_metric]


def get_train_and_val_splits(
        train_val_splits: List[List[LabeledAudioFilePath]],
        val_split_idx: int,
        ) -> Tuple[List[LabeledAudioFilePath], List[LabeledAudioFilePath]]:
    val_split = train_val_splits[val_split_idx]
    train_splits = train_val_splits[:val_split_idx] + train_val_splits[val_split_idx + 1:]
    train_split = [
        labeled_audio_file_path
        for train_split in train_splits
        for labeled_audio_file_path in train_split
    ]
    return train_split, val_split


def get_splits_metric_values(
        training_params: TrainingParams,
        optimization_metric: Metric,
        device: torch.device,
        train_val_splits: List[List[LabeledAudioFilePath]],
        ) -> List[float]:
    splits_metric_values: List[float] = []
    for val_split_idx in range(len(train_val_splits)):
        train_split, val_split = get_train_and_val_splits(train_val_splits, val_split_idx)
        train_dataset = MeanAudioDataset(train_split, device=device)
        val_dataset = MeanAudioDataset(val_split, device=device)
        splits_metric_values.append(eval_model_training_params(
            training_params,
            optimization_metric,
            device,
            train_dataset,
            val_dataset,
        ))
    return splits_metric_values


def objective(
        trial: optuna.trial.Trial,
        optimization_metric: Metric,
        device: torch.device,
        train_val_splits: List[List[LabeledAudioFilePath]],
        ) -> float:
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

    training_params = TrainingParams(
        do_fit_scaler=do_fit_scaler,
        hidden_size_0=hidden_size_0,
        hidden_size_1=hidden_size_1,
        dropout=dropout,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    splits_metric_values = get_splits_metric_values(
        training_params,
        optimization_metric,
        device,
        train_val_splits,
    )

    return float(np.mean(splits_metric_values))


def tune_hyperparameters(
        optimization_metric: Metric,
        device: torch.device,
        train_val_splits: List[List[LabeledAudioFilePath]],
        ) -> Tuple[float, TrainingParams]:
    metrics_directions = {
        Metric.ACCURACY: 'maximize',
        Metric.LOSS: 'minimize',
    }

    study = optuna.create_study(direction=metrics_directions[optimization_metric])

    study.optimize(
        lambda trial: objective(
            trial,
            optimization_metric,
            device,
            train_val_splits,
        ),
        n_trials=NUM_MLP_HYPERPARAMETER_TUNING_TRIALS,
    )

    best_training_params = TrainingParams(
        do_fit_scaler=study.best_trial.params['do_fit_scaler'],
        hidden_size_0=2 ** study.best_trial.params['hidden_size_0_exp'],
        hidden_size_1=2 ** study.best_trial.params['hidden_size_1_exp'],
        dropout=study.best_trial.params['dropout'],
        train_batch_size=2 ** study.best_trial.params['train_batch_size_exp'],
        learning_rate=study.best_trial.params['learning_rate'],
        weight_decay=study.best_trial.params['weight_decay'],
    )
    return study.best_value, best_training_params


def save_training_params(training_params: TrainingParams, file_path: str) -> None:
    with open(file_path, 'w') as f:
        f.write(training_params.model_dump_json(indent=4))


def compute_metrics(
        training_params: TrainingParams,
        optimization_metric: Metric,
        device: torch.device,
        train_split: List[LabeledAudioFilePath],
        val_split: List[LabeledAudioFilePath],
        test_split: List[LabeledAudioFilePath],
        classifier_checkpoint_file_path: str,
        ) -> None:
    train_dataset = MeanAudioDataset(train_split, device=device)
    val_dataset = MeanAudioDataset(val_split, device=device)
    test_dataset = MeanAudioDataset(test_split, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=training_params.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MLP(
        input_size=CLAP_EMBEDDING_SIZE,
        hidden_size_0=training_params.hidden_size_0,
        hidden_size_1=training_params.hidden_size_1,
        dropout=training_params.dropout,
    )
    if training_params.do_fit_scaler:
        model.fit_scaler(train_dataset.audios_embeddings)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay
    )

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []

    for epoch in range(NUM_MLP_EPOCHS):
        train_epoch(model, train_dataloader, criterion, optimizer)

        y_train, y_pred_train, train_loss = get_y_and_y_pred_and_loss(model, train_dataloader)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        y_val, y_pred_val, val_loss = get_y_and_y_pred_and_loss(model, val_dataloader)
        val_accuracy = accuracy_score(y_val, y_pred_val)

        do_save_classifier = (
            (optimization_metric == Metric.ACCURACY and (not val_accuracies or val_accuracy > max(val_accuracies))) or
            (optimization_metric == Metric.LOSS and (not val_losses or val_loss < min(val_losses)))
        )

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        print(
            f'Epoch: {epoch}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, ' +
            f'Train F1 Score: {f1_score(y_train, y_pred_train):.4f}, ' +
            f'Train AUC: {roc_auc_score(y_train, y_pred_train):.4f}'
        )

        if do_save_classifier:
            torch.save(model.state_dict(), classifier_checkpoint_file_path)
            print('Model saved.')

    model.load_state_dict(torch.load(classifier_checkpoint_file_path, weights_only=True))

    y_test, y_pred_test, _ = get_y_and_y_pred_and_loss(model, test_dataloader)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print()
    print(
        f'Testing: Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {f1_score(y_test, y_pred_test):.4f}, ' +
        f'Test AUC: {roc_auc_score(y_test, y_pred_test):.4f}'
    )
    print()

    torch.save(model.state_dict(), classifier_checkpoint_file_path)

    save_confusion_matrix_plot(y_test, y_pred_test)
    save_losses_plot(train_losses, val_losses)
    save_accuracies_plot(train_accuracies, val_accuracies)
    save_roc_curve_plot(y_train, y_pred_train, y_val, y_pred_val)

    test_embeddings = get_model_embeddings(model, test_dataloader)

    save_training_results(
        test_embeddings,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        y_train,
        y_pred_train,
        y_val,
        y_pred_val,
    )


def main() -> None:
    seed_everything(42)

    classifier_checkpoint_dir_path = os.path.join(CLASSIFIERS_CHECKPOINTS_DIR_PATH, 'mlp')
    classifier_training_params_file_path = os.path.join(classifier_checkpoint_dir_path, 'training_params.json')
    classifier_checkpoint_file_path = os.path.join(classifier_checkpoint_dir_path, 'classifier.pt')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(DATASET_SPLITS_FILE_PATH) as f:
        dataset_splits = DatasetSplits.model_validate_json(f.read())

    print('Hyperparameter tuning:')
    best_metric_value, best_training_params = tune_hyperparameters(
        optimization_metric=OPTIMIZATION_METRIC,
        device=device,
        train_val_splits=dataset_splits.train_val_splits,
    )

    print()
    print(f'Best k-fold {OPTIMIZATION_METRIC.value}: {best_metric_value}')
    print(f'Best training params: {best_training_params}')
    print()

    save_training_params(best_training_params, classifier_training_params_file_path)

    splits_metric_values = get_splits_metric_values(
        best_training_params,
        OPTIMIZATION_METRIC,
        device,
        dataset_splits.train_val_splits,
    )

    print(f'Value of {OPTIMIZATION_METRIC.value} in splits for best training params: {splits_metric_values}')
    print()

    if OPTIMIZATION_METRIC == Metric.ACCURACY:
        best_val_idx = int(np.argmax(splits_metric_values))
    elif OPTIMIZATION_METRIC == Metric.LOSS:
        best_val_idx = int(np.argmin(splits_metric_values))
    else:
        raise ValueError('Unknown optimization metric')

    best_train_split, best_val_split = get_train_and_val_splits(
        dataset_splits.train_val_splits,
        best_val_idx,
    )

    print('Training using best training params and train/val split:')
    compute_metrics(
        best_training_params,
        OPTIMIZATION_METRIC,
        device,
        best_train_split,
        best_val_split,
        dataset_splits.test,
        classifier_checkpoint_file_path,
    )


if __name__ == '__main__':
    main()
