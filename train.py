import os
from typing import Tuple, List

import torch
import optuna
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score  # type: ignore

from utils import seed_everything
from dataset_splits_models import DatasetSplits, LabeledAudioFilePath
from train_utils import (
    TrainingParams, TrainingUtils, MLPTrainingUtils, VotingMLPTrainingUtils, LSTMTrainingUtils, TransformerTrainingUtils
)
from constants import (
    DATASET_SPLITS_FILE_PATH, Metric, OPTIMIZATION_METRIC, CLASSIFIERS_CHECKPOINTS_DIR_PATH, TRAINING_METRICS_DIR_PATH
)
from train_funcs import (
    train_epoch, get_y_and_y_pred_and_loss, get_train_and_val_splits, save_confusion_matrix_plot, save_losses_plot,
    save_accuracies_plot, save_roc_curve_plot, get_model_embeddings, save_training_results,
)


def eval_model_training_params(
        training_params: TrainingParams,
        optimization_metric: Metric,
        device: torch.device,
        train_dataset: Dataset,
        val_dataset: Dataset,
        training_utils: TrainingUtils,
        ) -> float:

    train_dataloader, model, optimizer, scheduler = training_utils.get_training_objects(
        training_params, train_dataset, device
    )
    val_dataloader = training_utils.get_non_train_dataloader(val_dataset)
    criterion = nn.BCELoss()

    max_val_accuracy = float('-inf')
    min_val_loss = float('inf')

    for _ in range(training_utils.num_epochs):
        train_epoch(
            model, train_dataloader, criterion, optimizer, scheduler, training_utils.does_model_return_embeddings
        )
        y_val, y_pred_val, val_loss = get_y_and_y_pred_and_loss(
            model, val_dataloader, training_utils.does_model_return_embeddings
        )
        val_accuracy = accuracy_score(y_val, y_pred_val)
        max_val_accuracy = max(max_val_accuracy, val_accuracy)
        min_val_loss = min(min_val_loss, val_loss)

    metrics_values = {
        Metric.ACCURACY: max_val_accuracy,
        Metric.LOSS: min_val_loss,
    }
    return metrics_values[optimization_metric]


def get_splits_metric_values(
        training_params: TrainingParams,
        optimization_metric: Metric,
        device: torch.device,
        train_val_splits: List[List[LabeledAudioFilePath]],
        training_utils: TrainingUtils,
        ) -> List[float]:
    splits_metric_values: List[float] = []
    for val_split_idx in range(len(train_val_splits)):
        train_split, val_split = get_train_and_val_splits(train_val_splits, val_split_idx)
        train_dataset = training_utils.get_dataset(train_split, device)
        val_dataset = training_utils.get_dataset(val_split, device)
        splits_metric_values.append(eval_model_training_params(
            training_params,
            optimization_metric,
            device,
            train_dataset,
            val_dataset,
            training_utils,
        ))
    return splits_metric_values


def objective(
        trial: optuna.trial.Trial,
        optimization_metric: Metric,
        device: torch.device,
        train_val_splits: List[List[LabeledAudioFilePath]],
        training_utils: TrainingUtils,
        ) -> float:
    training_params = training_utils.get_training_params_for_trial(trial)
    splits_metric_values = get_splits_metric_values(
        training_params,
        optimization_metric,
        device,
        train_val_splits,
        training_utils,
    )
    return float(np.mean(splits_metric_values))


def tune_hyperparameters(
        optimization_metric: Metric,
        device: torch.device,
        train_val_splits: List[List[LabeledAudioFilePath]],
        training_utils: TrainingUtils,
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
            training_utils,
        ),
        n_trials=training_utils.num_hyperparameter_tuning_trials,
    )

    best_training_params = training_utils.get_training_params_from_study(study)
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
        training_utils: TrainingUtils,
        ) -> None:
    train_dataset = training_utils.get_dataset(train_split, device)
    val_dataset = training_utils.get_dataset(val_split, device)
    test_dataset = training_utils.get_dataset(test_split, device)

    train_dataloader, model, optimizer, scheduler = training_utils.get_training_objects(
        training_params, train_dataset, device
    )
    val_dataloader = training_utils.get_non_train_dataloader(val_dataset)
    test_dataloader = training_utils.get_non_train_dataloader(test_dataset)
    criterion = nn.BCELoss()

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []

    for epoch in range(training_utils.num_epochs):
        train_epoch(model, train_dataloader, criterion, optimizer, scheduler, training_utils.does_model_return_embeddings)

        y_train, y_pred_train, train_loss = get_y_and_y_pred_and_loss(
            model, train_dataloader, training_utils.does_model_return_embeddings
        )
        train_accuracy = accuracy_score(y_train, y_pred_train)
        y_val, y_pred_val, val_loss = get_y_and_y_pred_and_loss(
            model, val_dataloader, training_utils.does_model_return_embeddings
        )
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

    y_test, y_pred_test, _ = get_y_and_y_pred_and_loss(
        model, test_dataloader, training_utils.does_model_return_embeddings
    )
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print()
    print(
        f'Testing: Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {f1_score(y_test, y_pred_test):.4f}, ' +
        f'Test AUC: {roc_auc_score(y_test, y_pred_test):.4f}'
    )
    print()

    torch.save(model.state_dict(), classifier_checkpoint_file_path)

    classifier_training_metrics_dir_path = os.path.join(TRAINING_METRICS_DIR_PATH, training_utils.model_kind_name)

    save_confusion_matrix_plot(y_test, y_pred_test, classifier_training_metrics_dir_path)
    save_losses_plot(train_losses, val_losses, classifier_training_metrics_dir_path)
    save_accuracies_plot(train_accuracies, val_accuracies, classifier_training_metrics_dir_path)
    save_roc_curve_plot(y_train, y_pred_train, y_val, y_pred_val, classifier_training_metrics_dir_path)

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
        classifier_training_metrics_dir_path,
    )


def main() -> None:
    seed_everything(42)

    training_utils = LSTMTrainingUtils()

    classifier_checkpoint_dir_path = os.path.join(CLASSIFIERS_CHECKPOINTS_DIR_PATH, training_utils.model_kind_name)
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
        training_utils=training_utils,
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
        training_utils,
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
        training_utils,
    )


if __name__ == '__main__':
    main()
