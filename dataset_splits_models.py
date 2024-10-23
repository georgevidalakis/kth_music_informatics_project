from typing import List

from pydantic import BaseModel


class LabeledAudioFilePath(BaseModel):
    audio_file_path: str
    label: int


class DatasetSplits(BaseModel):
    train: List[LabeledAudioFilePath]
    val: List[LabeledAudioFilePath]
    adv_val: List[LabeledAudioFilePath]
    test: List[LabeledAudioFilePath]
