import os
import random
from collections import Counter
from typing import Tuple, List, Callable

import numpy as np
import tqdm  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import torch
from torch.utils.data import TensorDataset, DataLoader
from feed_forward import FeedForward
import torch.nn as nn

import torch.nn.utils.rnn as rnn_utils
from lstm_model import *
from cnn_1d import *

from constants import (
    AI_EMBEDDINGS_DIR_PATH, HUMAN_EMBEDDINGS_DIR_PATH, SPLIT_STRATEGY, SplitStrategy, Label, NUM_FEED_FORWARD_EPOCHS
)


def load_audio_embeddings(audio_embeddings_file_path: str) -> np.ndarray:
    return np.load(audio_embeddings_file_path)


def get_first_embedding(audio_embeddings: np.ndarray) -> np.ndarray:
    return audio_embeddings[0]


def get_seq_embeddings(audio_embeddings: np.ndarray) -> np.ndarray:
    return audio_embeddings


def get_mean_embedding(audio_embeddings: np.ndarray) -> np.ndarray:
    return np.mean(audio_embeddings, axis=0)


def get_mean_and_std_embedding(audio_embeddings: np.ndarray) -> np.ndarray:
    mean_embedding = np.mean(audio_embeddings, axis=0)
    std_embedding = np.std(audio_embeddings, axis=0)
    return np.hstack((mean_embedding, std_embedding))


def pad_sequences(sequences: List[np.ndarray]) -> torch.Tensor:
    # Convert list of numpy arrays to list of tensors
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    
    # Pad sequences to the same length
    padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=True)
    
    return padded_sequences


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

# Ensure your data is 3D for LSTM: (batch_size, seq_length, input_size)
def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    """This function reshapes the input data to 3D, which is required by the LSTM."""
    # Reshape the input to be (batch_size, seq_length, input_size).
    # Here, assuming that seq_length can be 1 (single time step) and input_size = number of features.
    return np.expand_dims(X, axis=1)  # Shape (batch_size, 1, input_size)

# Ensure your data is 3D for CNN: (batch_size, 1, input_size)
def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """This function reshapes the input data to 3D, required by CNN."""
    # Reshape the input to be (batch_size, 1, input_size)
    # CNN needs an additional 'channel' dimension, which is 1 in this case (single channel)
    return np.expand_dims(X, axis=1)  # Shape (batch_size, 1, input_size)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    audio_embedding_aggregation_func = get_seq_embeddings

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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_adv_val = scaler.transform(X_adv_val)
    X_test = scaler.transform(X_test)

    # Reshape for LSTM: (batch_size, seq_length, input_size)
    X_train = reshape_for_lstm(X_train)
    X_val = reshape_for_lstm(X_val)
    X_adv_val = reshape_for_lstm(X_adv_val)
    X_test = reshape_for_lstm(X_test)

    X_train = pad_sequences(X_train)
    X_val = pad_sequences(X_val)
    X_adv_val = pad_sequences(X_adv_val)
    X_test = pad_sequences(X_test)

    train_dataset = TensorDataset(X_train, torch.tensor(y_train, dtype=torch.float).view(-1, 1))
    val_dataset = TensorDataset(X_val, torch.tensor(y_val, dtype=torch.float).view(-1, 1))
    val_adv_val_dataset = TensorDataset(X_adv_val, torch.tensor(y_adv_val, dtype=torch.float).view(-1, 1))
    test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.float).view(-1, 1))


    # Reshape for CNN: (batch_size, 1, input_size) -> Adding the channel dimension
    # X_train = reshape_for_cnn(X_train)
    # X_val = reshape_for_cnn(X_val)
    # X_adv_val = reshape_for_cnn(X_adv_val)
    # X_test = reshape_for_cnn(X_test)

    # train_dataset = TensorDataset(torch.from_numpy(X_train), torch.tensor(y_train, dtype=torch.float).view(-1, 1))
    # val_dataset = TensorDataset(torch.from_numpy(X_val), torch.tensor(y_val, dtype=torch.float).view(-1, 1))
    # val_adv_val_dataset = TensorDataset(torch.from_numpy(X_adv_val), torch.tensor(y_adv_val, dtype=torch.float).view(-1, 1))
    # test_dataset = TensorDataset(torch.from_numpy(X_test), torch.tensor(y_test, dtype=torch.float).view(-1, 1))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_adv_val_dataloader = DataLoader(val_adv_val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the LSTM model
    model = DeepLSTMModel(input_size=X_train.shape[2])  # input_size is the number of features in the embeddings
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # model = FeedForward(input_size=X_train.shape[1])
    # criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize the CNN model
    # model = CNN1D(input_size=X_train.shape[2])  # input_size is the number of features in your embeddings
    # criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(NUM_FEED_FORWARD_EPOCHS):
        model.train()
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct_cnt = 0
            total_cnt = 0
            for X_batch, y_batch in val_dataloader:
                y_pred = torch.round(model(X_batch))
                correct_cnt += (y_pred == y_batch).sum().item()
                total_cnt += len(y_batch)

        accuracy = correct_cnt / total_cnt
        print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}')

    # testing
    model.eval()
    with torch.no_grad():
        correct_cnt = 0
        total_cnt = 0
        for X_batch, y_batch in test_dataloader:
            y_pred = torch.round(model(X_batch))
            correct_cnt += (y_pred == y_batch).sum().item()
            total_cnt += len(y_batch)

    accuracy = correct_cnt / total_cnt
    print(f'Test Accuracy: {accuracy:.4f}')

    torch.save(model.state_dict(), 'classifier_checkpoints/deeplstm__seq_epoch20.pt')
