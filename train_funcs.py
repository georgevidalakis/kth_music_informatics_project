import os
from typing import Tuple, List, Optional

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay, confusion_matrix  # type: ignore

from dataset_splits_models import LabeledAudioFilePath


# sequence / non-sequence specific

def train_epoch(
        model: nn.Module, train_dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
        scheduler: Optional[LRScheduler], does_model_return_embeddings: bool,
        ) -> None:
    model.train()
    for dataset_element in train_dataloader:
        optimizer.zero_grad()
        model_result = model(*dataset_element[:2])
        y_pred = model_result[0] if does_model_return_embeddings else model_result
        y_batch = dataset_element[-1]
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()


def get_y_and_y_pred_and_loss(
        model: nn.Module, dataloader: DataLoader,
        does_model_return_embeddings: bool,
        ) -> Tuple[np.ndarray, np.ndarray, float]:
    y_list: List[float] = []
    y_pred_list: List[float] = []
    loss_sum = 0.
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        for dataset_element in dataloader:
            model_result = model(*dataset_element[:2])
            batch_pred_confidences = model_result[0] if does_model_return_embeddings else model_result
            batch_labels = dataset_element[-1]
            batch_pred_labels = torch.round(batch_pred_confidences)
            batch_loss = criterion(batch_pred_labels, batch_labels).item()
            y_list += batch_labels.cpu().squeeze(1).tolist()
            y_pred_list += batch_pred_labels.cpu().squeeze(1).tolist()
            loss_sum += batch_loss
    return np.array(y_list), np.array(y_pred_list), float(loss_sum / len(dataloader))


def get_model_embeddings(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    embeddings_list: List[np.ndarray] = []
    for dataset_element in dataloader:
        batch_embeddings = model(*dataset_element[:2])[1]
        embeddings_list.append(batch_embeddings.detach().cpu().numpy())
    return np.vstack(embeddings_list)


# general

def save_confusion_matrix_plot(y_test: np.ndarray, y_pred_test: np.ndarray, dir_path: str) -> None:
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AI', 'Human'])
    disp.plot()
    plt.savefig(os.path.join(dir_path, 'confusion_matrix.png'))
    plt.close()


def save_losses_plot(train_losses: List[float], val_losses: List[float], dir_path: str) -> None:
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'training_losses.png'))
    plt.close()


def save_accuracies_plot(train_accuracies: List[float], val_accuracies: List[float], dir_path: str) -> None:
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'training_accuracies.png'))
    plt.close()


def save_roc_curve_plot(
        y_train: np.ndarray, y_pred_train: np.ndarray, y_val: np.ndarray, y_pred_val: np.ndarray, dir_path: str
        ) -> None:
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)
    plt.plot(fpr_train, tpr_train, label='Train ROC curve')
    plt.plot(fpr_val, tpr_val, label='Validation ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'train_val_roc_curve.png'))
    plt.close()


def save_training_results(
        test_embeddings: np.ndarray,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: List[float],
        val_accuracies: List[float],
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_val: np.ndarray,
        y_pred_val: np.ndarray,
        dir_path: str,
        ) -> None:
    np.save(os.path.join(dir_path, 'embeddings.npy'), test_embeddings)
    np.save(os.path.join(dir_path, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(dir_path, 'val_losses.npy'), np.array(val_losses))
    np.save(os.path.join(dir_path, 'train_accuracies.npy'), np.array(train_accuracies))
    np.save(os.path.join(dir_path, 'val_accuracies.npy'), np.array(val_accuracies))
    np.save(os.path.join(dir_path, 'y_train.npy'), y_train)
    np.save(os.path.join(dir_path, 'y_pred_train.npy'), y_pred_train)
    np.save(os.path.join(dir_path, 'y_val.npy'), y_val)
    np.save(os.path.join(dir_path, 'y_pred_val.npy'), y_pred_val)


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
