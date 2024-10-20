import os
from enum import Enum


class Label(Enum):
    AI = 0
    HUMAN = 1


class SplitStrategy(str, Enum):
    AUTHORS_IGNORED = 'authors_ignored'
    AUTHORS_CONSIDERED = 'authors_considered'


CLAP_SR = 48_000
PARENT_DIR_PATH =  os.path.dirname(os.path.abspath(os.getcwd()))

AUDIO_DIR_PATH = 'rolling_udio'
AI_AUDIO_DIR_PATH = os.path.join(AUDIO_DIR_PATH, 'udio_500')
HUMAN_AUDIO_DIR_PATH = os.path.join(AUDIO_DIR_PATH, 'rolling_500')

EMBEDDINGS_DIR_PATH = 'clap_embeddings'
AI_EMBEDDINGS_DIR_PATH = os.path.join(EMBEDDINGS_DIR_PATH, 'udio_500')
HUMAN_EMBEDDINGS_DIR_PATH = os.path.join(EMBEDDINGS_DIR_PATH, 'rolling_500')

ADVERSARIAL_AUDIO_DIR_PATH = 'adversarial_audio'

SPLIT_STRATEGY = SplitStrategy.AUTHORS_CONSIDERED

NUM_FEED_FORWARD_EPOCHS = 20
