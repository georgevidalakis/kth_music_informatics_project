import os
import json
import random
from typing import Tuple, List, Dict

import numpy as np

from utils import seed_everything
from constants import (
    Label, SplitStrategy, SPLIT_STRATEGY, AI_AUDIO_DIR_PATH, HUMAN_AUDIO_DIR_PATH, DATASET_SPLITS_FILE_PATH
)


def check_splits_sizes(names_values: Dict[str, int], splits_sizes: List[int]) -> None:
    if sum(splits_sizes) != len(names_values):
        raise ValueError('sum(splits_sizes) != len(names_values)')


def get_values_names(names_values: Dict[str, int]) -> Dict[int, List[str]]:
    values_names: Dict[int, List[str]] = {}
    for name in names_values:
        value = names_values[name]
        if value not in values_names:
            values_names[value] = []
        values_names[value].append(name)
    return values_names


def shuffle_values_names_values(values_names: Dict[int, List[str]]) -> None:
    for value_names in values_names.values():
        random.shuffle(value_names)


def shuffle_splits_names(splits_names: List[List[str]]) -> None:
    for split_names in splits_names:
        random.shuffle(split_names)


def get_similar_splits(names_values: Dict[str, int], splits_sizes: List[int]) -> List[List[str]]:
    check_splits_sizes(names_values, splits_sizes)
    values_names = get_values_names(names_values)
    shuffle_values_names_values(values_names)
    splits_names: List[List[str]] = [[] for _ in splits_sizes]
    num_elements = len(names_values)
    for value, value_names in values_names.items():
        num_value_names = len(value_names)
        cur_value_name_idx = 0
        for split_names, split_size in zip(splits_names, splits_sizes):
            num_split_value_names = split_size * num_value_names // num_elements
            split_names += value_names[cur_value_name_idx:cur_value_name_idx + num_split_value_names]
            cur_value_name_idx += num_split_value_names
        if cur_value_name_idx < num_value_names:
            raise ValueError(f'Value "{value}" could not be split')
    return splits_names


def get_different_splits(names_values: Dict[str, int], splits_sizes: List[int]) -> List[List[str]]:
    check_splits_sizes(names_values, splits_sizes)
    values_names = get_values_names(names_values)
    shuffle_values_names_values(values_names)
    sorted_values = sorted(list(values_names), key=lambda value: len(values_names[value]), reverse=True)
    splits_names: List[List[str]] = [[] for _ in splits_sizes]
    splits_missing_elements_cnts = splits_sizes.copy()
    for value in sorted_values:
        value_names = values_names[value]
        argmax_split_idx = int(np.argmax(splits_missing_elements_cnts))
        splits_names[argmax_split_idx] += value_names
        splits_missing_elements_cnts[argmax_split_idx] -= len(value_names)
        if splits_missing_elements_cnts[argmax_split_idx] < 0:
            raise ValueError(f'Value "{value}" could not be split')
    return splits_names


def get_labeled_splits(
        splits_names: List[List[str]], names: List[str], values: List[int]
        ) -> List[Tuple[List[str], List[int]]]:
    labeled_splits: List[Tuple[List[str], List[int]]] = []
    names_values = {name: value for name, value in zip(names, values)}
    for split_names in splits_names:
        split_values = [names_values[name] for name in split_names]
        labeled_splits.append((split_names, split_values))
    return labeled_splits


def split_audio_files_paths_ignoring_authors(
        audio_files_paths: List[str], labels: List[int], splits_sizes: List[int]
        ) -> List[Tuple[List[str], List[int]]]:
    audio_files_paths_labels = {path: label for path, label in zip(audio_files_paths, labels)}
    splits_audio_files_paths = get_similar_splits(audio_files_paths_labels, splits_sizes)
    shuffle_splits_names(splits_audio_files_paths)
    return get_labeled_splits(splits_audio_files_paths, audio_files_paths, labels)


def get_author_from_file_path(file_path: str) -> str:
    return file_path.split('-')[-2].strip()


class StrToIntMapper:
    def __init__(self) -> None:
        self.str_to_int: Dict[str, int] = {}

    def __call__(self, str_value: str) -> int:
        if str_value not in self.str_to_int:
            self.str_to_int[str_value] = len(self.str_to_int)
        return self.str_to_int[str_value]


def split_audio_files_paths_considering_authors(
        audio_files_paths: List[str], labels: List[int], splits_sizes: List[int]
        ) -> List[Tuple[List[str], List[int]]]:
    ai_audio_files_paths_labels = {
        path: label
        for path, label in zip(audio_files_paths, labels)
        if label == Label.AI.value
    }
    author_to_int_mapper = StrToIntMapper()
    human_audio_files_paths_labels = {
        path: author_to_int_mapper(get_author_from_file_path(path))
        for path, label in zip(audio_files_paths, labels)
        if label == Label.HUMAN.value
    }
    num_audio_files = len(labels)
    num_ai_audio_files = sum(label == Label.AI.value for label in labels)
    num_human_audio_files = sum(label == Label.HUMAN.value for label in labels)
    ai_splits_sizes = [split_size * num_ai_audio_files // num_audio_files for split_size in splits_sizes]
    human_splits_sizes = [split_size * num_human_audio_files // num_audio_files for split_size in splits_sizes]
    if sum(ai_splits_sizes + human_splits_sizes) != num_audio_files:
        raise ValueError('sum(ai_splits_sizes + human_splits_sizes) != num_audio_files')
    ai_splits = get_similar_splits(ai_audio_files_paths_labels, ai_splits_sizes)
    human_splits = get_different_splits(human_audio_files_paths_labels, human_splits_sizes)
    splits_audio_files_paths = [ai_split + human_split for ai_split, human_split in zip(ai_splits, human_splits)]
    shuffle_splits_names(splits_audio_files_paths)
    return get_labeled_splits(splits_audio_files_paths, audio_files_paths, labels)


def get_dir_files_paths(dir_path: str) -> List[str]:
    return [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)]


def main() -> None:
    seed_everything(42)
    ai_audio_files_paths = get_dir_files_paths(AI_AUDIO_DIR_PATH)
    human_audio_files_paths = get_dir_files_paths(HUMAN_AUDIO_DIR_PATH)
    audio_files_paths = ai_audio_files_paths + human_audio_files_paths
    labels = [Label.AI.value] * len(ai_audio_files_paths) + [Label.HUMAN.value] * len(human_audio_files_paths)
    splits_sizes = [600, 100, 100, 200]
    if SPLIT_STRATEGY == SplitStrategy.AUTHORS_IGNORED:
        split_audio_files_func = split_audio_files_paths_ignoring_authors
    elif SPLIT_STRATEGY == SplitStrategy.AUTHORS_CONSIDERED:
        split_audio_files_func = split_audio_files_paths_considering_authors
    else:
        raise ValueError(f'Split strategy "{SPLIT_STRATEGY}" is not supported')
    (
        (train_audio_files_paths, train_labels),
        (val_audio_files_paths, val_labels),
        (adv_val_audio_files_paths, adv_val_labels),
        (test_audio_files_paths, test_labels),
    ) = split_audio_files_func(audio_files_paths, labels, splits_sizes)
    dataset_split = {
        'train': [
            {
                'audio_file_path': audio_files_path,
                'label': label,
            }
            for audio_files_path, label in zip(train_audio_files_paths, train_labels)
        ],
        'val': [
            {
                'audio_file_path': audio_files_path,
                'label': label,
            }
            for audio_files_path, label in zip(val_audio_files_paths, val_labels)
        ],
        'adv_val': [
            {
                'audio_file_path': audio_files_path,
                'label': label,
            }
            for audio_files_path, label in zip(adv_val_audio_files_paths, adv_val_labels)
        ],
        'test': [
            {
                'audio_file_path': audio_files_path,
                'label': label,
            }
            for audio_files_path, label in zip(test_audio_files_paths, test_labels)
        ],
    }
    with open(DATASET_SPLITS_FILE_PATH, 'w') as f:
        json.dump(dataset_split, f, indent=4)


if __name__ == '__main__':
    main()
