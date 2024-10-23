from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

import numpy as np
from pydantic import BaseModel


class AdversarialInterationResult(BaseModel):
    target_pred_confidence: float
    snr: float
    duration_secs_since_attack_start: float


class AdversarialResult(BaseModel):
    audio_file_path: str
    iterations_results: List[AdversarialInterationResult]


class AdversarialAttacker(ABC):
    @abstractmethod
    def __call__(self, audio_file_path: str, target_label: int) -> Tuple[AdversarialResult, np.ndarray]:
        raise NotImplementedError


class AdversarialExperimentParams(BaseModel):
    window_size: int
    hop_size: int
    min_snr: Optional[float]
    max_iter: int
    required_target_pred_confidence: float
    learning_rate: float


class AdversarialExperiment(BaseModel):
    params: AdversarialExperimentParams
    results: List[AdversarialResult]
