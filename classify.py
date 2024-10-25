import os
import argparse

import torch
import numpy as np
import soundfile as sf
from embeddings_computing import load_clap_music_model


from classifiers.mlp import MLP
from train_mlp import TrainingParams
from generate_attack import AudioWindowsEmbeddingsExtractor
from constants import CLAP_EMBEDDING_SIZE, CLASSIFIERS_CHECKPOINTS_DIR_PATH, CLAP_SR


def load_audio_data_for_clap(audio_file_path: str) -> np.ndarray:
    original_audio_data, _ = sf.read(audio_file_path)
    return original_audio_data


def demo() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    args = parser.parse_args()

    audio_file_path = os.path.join('demo', args.input)
    audio_data = torch.tensor(load_audio_data_for_clap(audio_file_path), dtype=torch.float)
    clap_model = load_clap_music_model(use_cuda=True)
    audio_windows_embeddings_extractor = AudioWindowsEmbeddingsExtractor(
        window_size=int(10 * CLAP_SR),
        hop_size=int(10 * CLAP_SR),
        max_batch_size=4,
        clap_model=clap_model,
    )
    audio_windows_embeddings = audio_windows_embeddings_extractor(audio_data)
    audio_embedding = torch.mean(audio_windows_embeddings, dim=0)

    classifier_checkpoint_dir_path = os.path.join(CLASSIFIERS_CHECKPOINTS_DIR_PATH, 'mlp')
    classifier_training_params_file_path = os.path.join(classifier_checkpoint_dir_path, 'training_params.json')
    classifier_checkpoint_file_path = os.path.join(classifier_checkpoint_dir_path, 'classifier.pt')
    with open(classifier_training_params_file_path) as f:
        training_params = TrainingParams.model_validate_json(f.read())
    classifier = MLP(
        input_size=CLAP_EMBEDDING_SIZE,
        hidden_size_0=training_params.hidden_size_0,
        hidden_size_1=training_params.hidden_size_1,
        dropout=training_params.dropout,
    ).to('cuda:0')
    classifier.load_state_dict(torch.load(classifier_checkpoint_file_path, weights_only=True))

    confidence = classifier(audio_embedding.unsqueeze(0))[0].item()
    if confidence > 0.5:
        print(f'Human with {100 * confidence:.2f}% confidence')
    else:
        print(f'AI with {100 * (1 - confidence):.2f}% confidence')




if __name__ == '__main__':
    demo()

