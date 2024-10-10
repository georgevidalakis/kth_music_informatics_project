import os
from enum import Enum


class SplitStrategy(str, Enum):
    AUTHORS_IGNORED = 'authors_ignored'
    AUTHORS_CONSIDERED = 'authors_considered'


CLAP_SR = 48_000

AUDIO_DIR_PATH = 'rolling_udio'
AI_AUDIO_DIR_PATH = os.path.join(AUDIO_DIR_PATH, 'udio_500')
HUMAN_AUDIO_DIR_PATH = os.path.join(AUDIO_DIR_PATH, 'rolling_500')

EMBEDDINGS_DIR_PATH = 'clap_embeddings'
AI_EMBEDDINGS_DIR_PATH = os.path.join(EMBEDDINGS_DIR_PATH, 'udio_500')
HUMAN_EMBEDDINGS_DIR_PATH = os.path.join(EMBEDDINGS_DIR_PATH, 'rolling_500')

SPLIT_STRATEGY = SplitStrategy.AUTHORS_CONSIDERED
