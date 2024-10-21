import os
import json
from typing import Tuple, List, Callable

import tqdm  # type: ignore
import torch
import numpy as np
import torch.nn as nn
from pydantic import BaseModel
from classifiers.mlp import MLP
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, ConfusionMatrixDisplay, confusion_matrix  # type: ignore
from torch.utils.data import Dataset, DataLoader

from utils import seed_everything
from constants import (
    Label, NUM_MLP_EPOCHS, TRAINING_METRICS_DIR_PATH, DATASET_SPLITS_FILE_PATH, AI_EMBEDDINGS_DIR_PATH,
    HUMAN_EMBEDDINGS_DIR_PATH, CLAP_EMBEDDING_SIZE
)


def load_audio_embeddings(audio_embeddings_file_path: str) -> torch.Tensor:
    return torch.tensor(np.load(audio_embeddings_file_path))


def get_first_embedding(audio_embeddings: np.ndarray) -> np.ndarray:
    return audio_embeddings[0]


def get_mean_embedding(audio_embeddings: np.ndarray) -> np.ndarray:
    return np.mean(audio_embeddings, axis=0)


def get_mean_and_std_embedding(audio_embeddings: np.ndarray) -> np.ndarray:
    mean_embedding = np.mean(audio_embeddings, axis=0)
    std_embedding = np.std(audio_embeddings, axis=0)
    return np.hstack((mean_embedding, std_embedding))


class LabeledAudioFilePath(BaseModel):
    audio_file_path: str
    label: int


class AudioDataset(Dataset):
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


def main() -> None:
    seed_everything(42)

    with open(DATASET_SPLITS_FILE_PATH) as f:
        dataset_splits = json.load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = AudioDataset(
        labeled_audio_file_paths=[
            LabeledAudioFilePath.model_validate(labeled_audio_file_path_dict)
            for labeled_audio_file_path_dict in dataset_splits['train']
        ],
        device=device,
    )
    val_dataset = AudioDataset(
        labeled_audio_file_paths=[
            LabeledAudioFilePath.model_validate(labeled_audio_file_path_dict)
            for labeled_audio_file_path_dict in dataset_splits['val']
        ],
        device=device,
    )
    test_dataset = AudioDataset(
        labeled_audio_file_paths=[
            LabeledAudioFilePath.model_validate(labeled_audio_file_path_dict)
            for labeled_audio_file_path_dict in dataset_splits['test']
        ],
        device=device,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MLP(input_size=CLAP_EMBEDDING_SIZE)
    model.fit_scaler(train_dataset.audios_embeddings)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(NUM_MLP_EPOCHS):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred, _= model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_dataloader))

        model.eval()
        y_train_list = []
        y_pred_train_list = []
        with torch.no_grad():
            correct_cnt = 0
            total_cnt = 0
            for X_batch, y_batch in train_dataloader:
                y_pred = torch.round(model(X_batch)[0])
                y_train_list.extend(y_batch.cpu().numpy())
                y_pred_train_list.extend(y_pred.cpu().numpy())
                correct_cnt += (y_pred == y_batch).sum().item()
                total_cnt += len(y_batch)

        train_accuracy = correct_cnt / total_cnt
        train_accuracies.append(train_accuracy)
        y_train = np.array(y_train_list)
        y_pred_train = np.array(y_pred_train_list)

        epoch_val_loss = 0
        y_val_list = []
        y_pred_val_list = []
        with torch.no_grad():
            correct_cnt = 0
            total_cnt = 0
            for X_batch, y_batch in val_dataloader:
                y_pred = torch.round(model(X_batch)[0])
                loss = criterion(y_pred, y_batch)
                epoch_val_loss += loss.item()
                correct_cnt += (y_pred == y_batch).sum().item()
                total_cnt += len(y_batch)
                y_val_list.extend(y_batch.cpu().numpy())
                y_pred_val_list.extend(y_pred.cpu().numpy())

        y_val = np.array(y_val_list)
        y_pred_val = np.array(y_pred_val_list)
        val_losses.append(epoch_val_loss / len(val_dataloader))

        val_accuracy = correct_cnt / total_cnt
        val_accuracies.append(val_accuracy)
        print(f'Epoch: {epoch},Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f} Train F1 Score: {f1_score(y_train, y_pred_train):.4f} Train AUC: {roc_auc_score(y_train, y_pred_train):.4f}')

    # testing
    model.eval()
    y_test_list = []
    y_pred_test_list = []
    with torch.no_grad():
        correct_cnt = 0
        total_cnt = 0
        for X_batch, y_batch in test_dataloader:
            y_pred = torch.round(model(X_batch)[0])
            correct_cnt += (y_pred == y_batch).sum().item()
            total_cnt += len(y_batch)
            y_test_list.extend(y_batch.cpu().numpy())
            y_pred_test_list.extend(y_pred.cpu().numpy())

    y_test = np.array(y_test_list)
    y_pred_test = np.array(y_pred_test_list)

    accuracy = correct_cnt / total_cnt
    print(f'\nTesting: Test Accuracy: {accuracy:.4f} Test F1 Score: {f1_score(y_test, y_pred_test):.4f} Test AUC: {roc_auc_score(y_test, y_pred_test):.4f}')
    torch.save(model.state_dict(), 'classifier_checkpoints/mlp.pt')

    # Save the losses for plotting
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AI', 'Human'])
    disp.plot()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'mlp_confusion_matrix.png'))
    plt.close()

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'mlp_training_losses.png'))
    plt.close()

    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'mlp_training_accuracies.png'))
    plt.close()

    # plot train and validation roc curves
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

    # print the emebeddings of the  2nd value returned by the model for test instances
    embed = []
    for X_batch, y_batch in test_dataloader:
        y_pred, embeddings = model(X_batch)
        embed.append(embeddings.detach().cpu().numpy())

    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'embeddings.npy'), np.vstack(embed))

    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'val_losses.npy'), np.array(val_losses))

    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'train_accuracies.npy'), np.array(train_accuracies))
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'val_accuracies.npy'), np.array(val_accuracies))

    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_pred_train.npy'), y_pred_train)

    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_pred_val.npy'), y_pred_val)



if __name__ == '__main__':
    main()
