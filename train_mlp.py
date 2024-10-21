import os
from collections import Counter
from typing import Tuple, List, Callable

import numpy as np
import tqdm  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, ConfusionMatrixDisplay, confusion_matrix  # type: ignore
import torch
from torch.utils.data import TensorDataset, DataLoader
from classifiers.mlp import MLP
import torch.nn as nn

from constants import (
    AI_EMBEDDINGS_DIR_PATH, HUMAN_EMBEDDINGS_DIR_PATH, SPLIT_STRATEGY, SplitStrategy, Label, NUM_MLP_EPOCHS, TRAINING_METRICS_DIR_PATH
)
from utils import seed_everything


def load_audio_embeddings(audio_embeddings_file_path: str) -> np.ndarray:
    return np.load(audio_embeddings_file_path)


def get_first_embedding(audio_embeddings: np.ndarray) -> np.ndarray:
    return audio_embeddings[0]


def get_mean_embedding(audio_embeddings: np.ndarray) -> np.ndarray:
    return np.mean(audio_embeddings, axis=0)


def get_mean_and_std_embedding(audio_embeddings: np.ndarray) -> np.ndarray:
    mean_embedding = np.mean(audio_embeddings, axis=0)
    std_embedding = np.std(audio_embeddings, axis=0)
    return np.hstack((mean_embedding, std_embedding))


def get_X(
        embeddings_dir_path: str, audio_embedding_aggregation_func: Callable[[np.ndarray], np.ndarray]
        ) -> Tuple[np.ndarray, List[str]]:
    X_list: List[np.ndarray] = []
    embeddings_files_names: List[str] = os.listdir(embeddings_dir_path)
    for audio_embeddings_file_name in tqdm.tqdm(embeddings_files_names, desc=f'Loading embeddings from {embeddings_dir_path}'):
        audio_embeddings_file_path = os.path.join(embeddings_dir_path, audio_embeddings_file_name)
        audio_embeddings = load_audio_embeddings(audio_embeddings_file_path)
        X_list.append(audio_embedding_aggregation_func(audio_embeddings))
    return np.vstack(X_list), embeddings_files_names


def get_author_from_file_name(file_name: str) -> str:
    return file_name.split('-')[1].strip()


def train_test_split_authors(authors: List[str], test_size: int) -> Tuple[List[str], List[str]]:
    # each author should appear only in train or test set
    # len(test_authors) / len(authors) == test_size
    authors_cnts = Counter(authors)
    author_cnt_rand_float_tuples = [
        (author, author_cnt, np.random.rand())
        for author, author_cnt in authors_cnts.items()
    ]
    author_cnt_rand_float_tuples.sort(key=lambda e: (e[1], e[2]), reverse=True)
    train_authors: List[str] = []
    test_authors: List[str] = []
    for author, author_cnt, _ in author_cnt_rand_float_tuples:
        if len(test_authors) * len(authors) < test_size * (len(train_authors) + len(test_authors)):
            test_authors += [author] * author_cnt
        else:
            train_authors += [author] * author_cnt
    return train_authors, test_authors


def main() -> None:
    seed_everything(42)

    audio_embedding_aggregation_func = get_mean_embedding

    X_ai, _ = get_X(AI_EMBEDDINGS_DIR_PATH, audio_embedding_aggregation_func)
    X_human, human_embeddings_files_names = get_X(HUMAN_EMBEDDINGS_DIR_PATH, audio_embedding_aggregation_func)
    if SPLIT_STRATEGY == SplitStrategy.AUTHORS_IGNORED:
        X = np.vstack((X_ai, X_human))
        y = np.array([Label.AI.value] * len(X_ai) + [Label.HUMAN.value] * len(X_human))

        X_train_val_adv_val, X_test, y_train_val_adv_val, y_test = train_test_split(X, y, test_size=200, stratify=y)
        X_train, X_val_adv_val, y_train, y_val_adv_val = train_test_split(
            X_train_val_adv_val, y_train_val_adv_val, test_size=200, stratify=y_train_val_adv_val
        )
        X_val, X_adv_val, y_val, y_adv_val = train_test_split(
            X_val_adv_val, y_val_adv_val, test_size=100, stratify=y_val_adv_val
        )
    elif SPLIT_STRATEGY == SplitStrategy.AUTHORS_CONSIDERED:
        human_files_authors = list(map(get_author_from_file_name, human_embeddings_files_names))

        X_ai_train = X_ai[:-200]
        X_ai_val = X_ai[-200:-150]
        X_ai_adv_val = X_ai[-150:-100]
        X_ai_test = X_ai[-100:]

        human_train_val_adv_val_authors, human_test_authors = train_test_split_authors(human_files_authors, test_size=100)
        human_train_authors, human_val_adv_val_authors = train_test_split_authors(human_train_val_adv_val_authors, test_size=100)
        human_val_authors, human_adv_val_authors = train_test_split_authors(human_val_adv_val_authors, test_size=50)
        human_train_authors_set = set(human_train_authors)
        human_val_authors_set = set(human_val_authors)
        human_adv_val_authors_set = set(human_adv_val_authors)
        human_test_authors_set = set(human_test_authors)

        if len(human_train_authors_set) + len(human_val_authors_set) + len(human_adv_val_authors_set) + len(human_test_authors_set) != len(set(human_files_authors)):
            raise RuntimeError('Problem with human authors splitting')

        X_human_train_list: List[np.ndarray] = []
        X_human_val_list: List[np.ndarray] = []
        X_human_adv_val_list: List[np.ndarray] = []
        X_human_test_list: List[np.ndarray] = []
        for human_file_idx, human_file_author in enumerate(human_files_authors):
            if human_file_author in human_train_authors_set:
                X_human_train_list.append(X_human[human_file_idx])
            elif human_file_author in human_val_authors_set:
                X_human_val_list.append(X_human[human_file_idx])
            elif human_file_author in human_adv_val_authors_set:
                X_human_adv_val_list.append(X_human[human_file_idx])
            elif human_file_author in human_test_authors_set:
                X_human_test_list.append(X_human[human_file_idx])
            else:
                raise RuntimeError(f'Unexpected human file author: {human_file_author}')

        X_train = np.vstack((X_ai_train, X_human_train_list))
        X_val = np.vstack((X_ai_val, X_human_val_list))
        X_adv_val = np.vstack((X_ai_adv_val, X_human_adv_val_list))
        X_test = np.vstack((X_ai_test, X_human_test_list))
        y_train = np.array([Label.AI.value] * len(X_ai_train) + [Label.HUMAN.value] * len(X_human_train_list))
        y_adv_val = np.array([Label.AI.value] * len(X_ai_adv_val) + [Label.HUMAN.value] * len(X_human_adv_val_list))
        y_val = np.array([Label.AI.value] * len(X_ai_val) + [Label.HUMAN.value] * len(X_human_val_list))
        y_test = np.array([Label.AI.value] * len(X_ai_test) + [Label.HUMAN.value] * len(X_human_test_list))
    else:
        raise RuntimeError(f'Unexpected split strategy: {SPLIT_STRATEGY}')

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.tensor(y_train, dtype=torch.float).view(-1, 1))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.tensor(y_val, dtype=torch.float).view(-1, 1))
    val_adv_val_dataset = TensorDataset(torch.from_numpy(X_adv_val), torch.tensor(y_adv_val, dtype=torch.float).view(-1, 1))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.tensor(y_test, dtype=torch.float).view(-1, 1))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_adv_val_dataloader = DataLoader(val_adv_val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MLP(input_size=X_train.shape[1])
    model.fit_scaler(torch.from_numpy(X_train))
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
                y_train_list.extend(y_batch.numpy())
                y_pred_train_list.extend(y_pred.numpy())
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
                y_val_list.extend(y_batch.numpy())
                y_pred_val_list.extend(y_pred.numpy())

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
            y_test_list.extend(y_batch.numpy())
            y_pred_test_list.extend(y_pred.numpy())

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
