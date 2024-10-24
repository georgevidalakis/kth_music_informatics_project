import os
from enum import Enum


class Label(Enum):
    AI = 0
    HUMAN = 1


class SplitStrategy(str, Enum):
    AUTHORS_IGNORED = 'authors_ignored'
    AUTHORS_CONSIDERED = 'authors_considered'


class Metric(str, Enum):
    ACCURACY = 'accuracy'
    LOSS = 'loss'


CLAP_SR = 48_000
PARENT_DIR_PATH = os.path.dirname(os.path.abspath(os.getcwd()))

NUM_CROSS_VALIDATION_FOLDS = 5
TRAIN_VAL_SIZE = 700
ADV_VAL_SIZE = 100
TEST_SIZE = 200
DATASET_SPLITS_FILE_PATH = 'dataset_splits.json'

TRAINING_METRICS_DIR_PATH = 'training_metrics'

AUDIO_DIR_PATH = 'rolling_udio'
AI_AUDIO_DIR_PATH = os.path.join(AUDIO_DIR_PATH, 'udio_500')
HUMAN_AUDIO_DIR_PATH = os.path.join(AUDIO_DIR_PATH, 'rolling_500')

EMBEDDINGS_DIR_PATH = 'clap_embeddings'
AI_EMBEDDINGS_DIR_PATH = os.path.join(EMBEDDINGS_DIR_PATH, 'udio_500')
HUMAN_EMBEDDINGS_DIR_PATH = os.path.join(EMBEDDINGS_DIR_PATH, 'rolling_500')

CLASSIFIERS_CHECKPOINTS_DIR_PATH = 'classifiers_checkpoints'

ADVERSARIAL_AUDIO_DIR_PATH = 'adversarial_audio'

SPLIT_STRATEGY = SplitStrategy.AUTHORS_CONSIDERED

OPTIMIZATION_METRIC = Metric.ACCURACY

NUM_MLP_EPOCHS = 50
NUM_MLP_HYPERPARAMETER_TUNING_TRIALS = 20

CLAP_EMBEDDING_SIZE = 512

ADVERSARIAL_EXPERIMENTS_DIR_PATH = 'adversarial_experiments'
