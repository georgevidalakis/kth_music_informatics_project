from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel


class AdversarialResult(BaseModel):
    audio_file_path: str
    init_target_pred_confidence: float
    num_iter: int
    max_target_pred_confidence: float
    argmax_adversarial_audio_data: np.ndarray
    argmax_snr: float

    class Config:
        arbitrary_types_allowed = True


class AdversarialAttacker(ABC):
    @abstractmethod
    def __call__(self, audio_file_path: str, target_label: int) -> AdversarialResult:
        raise NotImplementedError
