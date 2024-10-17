import os
from collections import Counter
from typing import Tuple, List, Callable

import numpy as np
from tqdm import tqdm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.naive_bayes import GaussianNB  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from constants import AI_EMBEDDINGS_DIR_PATH, HUMAN_EMBEDDINGS_DIR_PATH, SPLIT_STRATEGY, SplitStrategy, Label


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
    for audio_embeddings_file_name in tqdm(embeddings_files_names, desc=f'Loading embeddings from {embeddings_dir_path}'):
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


if __name__ == '__main__':
    np.random.seed(42)

    audio_embedding_aggregation_func = get_mean_embedding

    X_ai, _ = get_X(AI_EMBEDDINGS_DIR_PATH, audio_embedding_aggregation_func)
    X_human, human_embeddings_files_names = get_X(HUMAN_EMBEDDINGS_DIR_PATH, audio_embedding_aggregation_func)
    if SPLIT_STRATEGY == SplitStrategy.AUTHORS_IGNORED:
        X = np.vstack((X_ai, X_human))
        y = np.array([Label.AI] * len(X_ai) + [Label.HUMAN] * len(X_human))

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=200, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=200, stratify=y_train_val
        )
    elif SPLIT_STRATEGY == SplitStrategy.AUTHORS_CONSIDERED:
        human_files_authors = list(map(get_author_from_file_name, human_embeddings_files_names))

        X_ai_train = X_ai[:-200]
        X_ai_val = X_ai[-200:-100]
        X_ai_test = X_ai[-100:]

        human_train_val_authors, human_test_authors = train_test_split_authors(human_files_authors, test_size=100)
        human_train_authors, human_val_authors = train_test_split_authors(human_train_val_authors, test_size=100)
        human_train_authors_set = set(human_train_authors)
        human_val_authors_set = set(human_val_authors)
        human_test_authors_set = set(human_test_authors)

        if len(human_train_authors_set) + len(human_val_authors_set) + len(human_test_authors_set) != len(set(human_files_authors)):
            raise RuntimeError('Problem with human authors splitting')

        X_human_train_list: List[np.ndarray] = []
        X_human_val_list: List[np.ndarray] = []
        X_human_test_list: List[np.ndarray] = []
        for human_file_idx, human_file_author in enumerate(human_files_authors):
            if human_file_author in human_train_authors_set:
                X_human_train_list.append(X_human[human_file_idx])
            elif human_file_author in human_val_authors_set:
                X_human_val_list.append(X_human[human_file_idx])
            elif human_file_author in human_test_authors_set:
                X_human_test_list.append(X_human[human_file_idx])
            else:
                raise RuntimeError(f'Unexpected human file author: {human_file_author}')

        X_train = np.vstack((X_ai_train, X_human_train_list))
        X_val = np.vstack((X_ai_val, X_human_val_list))
        X_test = np.vstack((X_ai_test, X_human_test_list))
        y_train = np.array([Label.AI] * len(X_ai_train) + [Label.HUMAN] * len(X_human_train_list))
        y_val = np.array([Label.AI] * len(X_ai_val) + [Label.HUMAN] * len(X_human_val_list))
        y_test = np.array([Label.AI] * len(X_ai_test) + [Label.HUMAN] * len(X_human_test_list))
    else:
        raise RuntimeError(f'Unexpected split strategy: {SPLIT_STRATEGY}')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Initialize t-SNE with 3 components
    tsne = TSNE(n_components=3)
    X_train_3d = tsne.fit_transform(X_train)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for 'ai' and 'human' labels in 3D
    ax.scatter(X_train_3d[y_train == 'ai', 0], X_train_3d[y_train == 'ai', 1], X_train_3d[y_train == 'ai', 2], c='red', label='ai')
    ax.scatter(X_train_3d[y_train == 'human', 0], X_train_3d[y_train == 'human', 1], X_train_3d[y_train == 'human', 2], c='green', label='human')

    # Add labels and legend
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()

    # Show the plot
    plt.show()

    # tsne = TSNE(n_components=2)
    # X_train_2d = tsne.fit_transform(X_train)

    # plt.figure()
    # plt.scatter(X_train_2d[y_train == 'ai', 0], X_train_2d[y_train == 'ai', 1], c='red', label='ai')
    # plt.scatter(X_train_2d[y_train == 'human', 0], X_train_2d[y_train == 'human', 1], c='green', label='human')
    # plt.legend()
    # plt.show()

    print()
    log_reg_clf = LogisticRegression().fit(X_train, y_train)
    log_reg_val_accuracy = log_reg_clf.score(X_val, y_val)
    log_reg_test_accuracy = log_reg_clf.score(X_test, y_test)
    print(f'Logistic regression validation accuracy: {log_reg_val_accuracy}')
    print(f'Logistic regression test accuracy: {log_reg_test_accuracy}')

    print()
    knn_clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    knn_val_accuracy = knn_clf.score(X_val, y_val)
    knn_test_accuracy = knn_clf.score(X_test, y_test)
    print(f'KNN validation accuracy: {knn_val_accuracy}')
    print(f'KNN test accuracy: {knn_test_accuracy}')

    print()
    svc_clf = SVC().fit(X_train, y_train)
    svc_val_accuracy = svc_clf.score(X_val, y_val)
    svc_test_accuracy = svc_clf.score(X_test, y_test)
    print(f'SVC validation accuracy: {svc_val_accuracy}')
    print(f'SVC test accuracy: {svc_test_accuracy}')

    print()
    nb_clf = GaussianNB().fit(X_train, y_train)
    nb_val_accuracy = nb_clf.score(X_val, y_val)
    nb_test_accuracy = nb_clf.score(X_test, y_test)
    print(f'Naive Bayes validation accuracy: {nb_val_accuracy}')
    print(f'Naive Bayes test accuracy: {nb_test_accuracy}')

    print()
    rf_clf = RandomForestClassifier().fit(X_train, y_train)
    rf_val_accuracy = rf_clf.score(X_val, y_val)
    rf_test_accuracy = rf_clf.score(X_test, y_test)
    print(f'Random forest validation accuracy: {rf_val_accuracy}')
    print(f'Random forest test accuracy: {rf_test_accuracy}')

    print()
    mlp_clf = MLPClassifier().fit(X_train, y_train)
    mlp_val_accuracy = mlp_clf.score(X_val, y_val)
    mlp_test_accuracy = mlp_clf.score(X_test, y_test)
    print(f'MLP validation accuracy: {mlp_val_accuracy}')
    print(f'MLP test accuracy: {mlp_test_accuracy}')
