import os
from typing import Tuple, List

import torch
import numpy as np
import torch.nn as nn
from classifiers.mlp import MLP
from classifiers.voting_mlp import VotingMLP
from classifiers.lstm_models import *
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, ConfusionMatrixDisplay, confusion_matrix  # type: ignore
from torch.utils.data import DataLoader

from utils import seed_everything
from dataset_splits_models import DatasetSplits
from pytorch_datasets import MeanAudioDataset, IdentityAudioDataset
from constants import NUM_MLP_EPOCHS, TRAINING_METRICS_DIR_PATH, DATASET_SPLITS_FILE_PATH, CLAP_EMBEDDING_SIZE


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


def get_y_and_y_pred_and_loss(model: nn.Module, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, float]:
    y_list: List[float] = []
    y_pred_list: List[float] = []
    loss_sum = 0.
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        for batch_embeddings, batch_num_windows, batch_labels in dataloader:
            batch_pred_confidences = model(batch_embeddings, batch_num_windows)
            batch_pred_labels = torch.round(batch_pred_confidences)
            batch_loss = criterion(batch_pred_labels, batch_labels).item()
            y_list += batch_labels.cpu().squeeze(1).tolist()
            y_pred_list += batch_pred_labels.cpu().squeeze(1).tolist()
            loss_sum += batch_loss
    return np.array(y_list), np.array(y_pred_list), float(loss_sum / len(dataloader))


def main() -> None:
    seed_everything(42)

    with open(DATASET_SPLITS_FILE_PATH) as f:
        dataset_splits = DatasetSplits.model_validate_json(f.read())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # train_dataset = MeanAudioDataset(dataset_splits.train, device=device)
    # val_dataset = MeanAudioDataset(dataset_splits.val, device=device)
    # test_dataset = MeanAudioDataset(dataset_splits.test, device=device)

    train_dataset = IdentityAudioDataset(dataset_splits.train, device=device)
    val_dataset = IdentityAudioDataset(dataset_splits.val, device=device)
    test_dataset = IdentityAudioDataset(dataset_splits.test, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=zero_pad)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=zero_pad)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=zero_pad)

    # model = MLP(input_size=CLAP_EMBEDDING_SIZE)
    # model = VotingMLP(input_size=CLAP_EMBEDDING_SIZE)
    # model.fit_scaler(train_dataset.audios_embeddings)
    # model.to(device)

    # Initialize the LSTM model
    model = DeepLSTMModel(input_size=CLAP_EMBEDDING_SIZE)  # input_size is the number of features in the embeddings
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
        for batch_embeddings, batch_num_windows, batch_labels in train_dataloader:
            optimizer.zero_grad()
            # print(batch_embeddings.shape, batch_num_windows.shape)
            batch_pred_labels = model(batch_embeddings, batch_num_windows)
            loss = criterion(batch_pred_labels, batch_labels)
            loss.backward()
            optimizer.step()

        y_train, y_pred_train, train_loss = get_y_and_y_pred_and_loss(model, train_dataloader)
        train_accuracy = np.mean(y_train == y_pred_train)
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)

        y_val, y_pred_val, val_loss = get_y_and_y_pred_and_loss(model, val_dataloader)
        val_accuracy = np.mean(y_val == y_pred_val)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        print(f'Epoch: {epoch},Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f} Train F1 Score: {f1_score(y_train, y_pred_train):.4f} Train AUC: {roc_auc_score(y_train, y_pred_train):.4f}')

    # testing
    y_test, y_pred_test, _ = get_y_and_y_pred_and_loss(model, test_dataloader)
    test_accuracy = np.mean(y_test == y_pred_test)
    print(f'\nTesting: Test Accuracy: {test_accuracy:.4f} Test F1 Score: {f1_score(y_test, y_pred_test):.4f} Test AUC: {roc_auc_score(y_test, y_pred_test):.4f}')

    torch.save(model.state_dict(), 'classifier_checkpoints/deeplstm_v1.pt')

    # Save the losses for plotting
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AI', 'Human'])
    disp.plot()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'lstm_confusion_matrix.png'))
    plt.close()

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'lstm_training_losses.png'))
    plt.close()

    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'lstm_training_accuracies.png'))
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
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'lstm_train_val_roc_curve.png'))
    plt.close()

    # np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'train_losses.npy'), np.array(train_losses))
    # np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'val_losses.npy'), np.array(val_losses))

    # np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'train_accuracies.npy'), np.array(train_accuracies))
    # np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'val_accuracies.npy'), np.array(val_accuracies))

    # np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_train.npy'), y_train)
    # np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_pred_train.npy'), y_pred_train)

    # np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_val.npy'), y_val)
    # np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_pred_val.npy'), y_pred_val)


if __name__ == '__main__':
    main()
