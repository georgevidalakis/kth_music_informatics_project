from typing import List
import librosa
import laion_clap
import numpy as np
import pickle
import os


CLAP_SR = 48_000


# quantization
def int16_to_float32(x) -> np.ndarray:
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x) -> np.ndarray:
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def quantize_array(a: np.ndarray) -> np.ndarray:
    return int16_to_float32(float32_to_int16(a))


def load_clap_music_model() -> laion_clap.hook.CLAP_Module:
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(ckpt='clap_checkpoints/music_audioset_epoch_15_esc_90.14.pt')
    return model


def load_audio_data_for_clap(audio_file_path: str) -> np.ndarray:
    original_audio_data, original_sr = librosa.load(audio_file_path)
    return librosa.resample(original_audio_data, orig_sr=original_sr, target_sr=CLAP_SR, fix=False)


def get_audio_windows_data(audio_data: np.ndarray, window_size: int, hop_size: int, do_quantize: bool) -> np.ndarray:
    audio_windows_data_list: List[np.ndarray] = []
    for window_start_idx in range(0, len(audio_data), hop_size):
        window_end_idx = window_start_idx + window_size
        audio_window_data = audio_data[window_start_idx:window_end_idx]
        zero_pad_size = window_size - len(audio_window_data)
        audio_window_data = np.pad(audio_window_data, pad_width=(0, zero_pad_size))
        if do_quantize:
            audio_window_data = quantize_array(audio_window_data)
        audio_windows_data_list.append(audio_window_data)
    return np.vstack(audio_windows_data_list)


# Save embeddings as npz
def save_embeddings_npz(embeddings: np.ndarray, output_file_path: str):
    np.savez(output_file_path, embeddings=embeddings)

# Save embeddings as pickle
def save_embeddings_pickle(embeddings: np.ndarray, output_file_path: str):
    with open(output_file_path, 'wb') as f:
        pickle.dump(embeddings, f)

# Main function to save embeddings with choice of method (npz or pickle)
def save_embeddings(embeddings: np.ndarray, output_file_path: str, method: str = 'npz'):
    """
    Save embeddings to file using specified method ('npz' or 'pickle').
    """
    if method == 'npz':
        save_embeddings_npz(embeddings, output_file_path + '.npz')
        print(f"Embeddings saved as NPZ: {output_file_path}.npz")
    elif method == 'pickle':
        save_embeddings_pickle(embeddings, output_file_path + '.pkl')
        print(f"Embeddings saved as Pickle: {output_file_path}.pkl")
    else:
        raise ValueError("Invalid method. Use 'npz' or 'pickle'.")


# Process all audio files in a folder and save embeddings
def process_audio_folder(audio_folder: str, embeddings_output_folder: str, save_method: str = 'npz'):
    model = load_clap_music_model()
    
    # Ensure the output folder exists
    if not os.path.exists(embeddings_output_folder):
        os.makedirs(embeddings_output_folder)

    # Iterate through all audio files in the folder
    for audio_file_name in os.listdir(audio_folder):
        if audio_file_name.endswith(('.mp3', '.wav', '.flac')):  # Add more formats if needed
            audio_file_path = os.path.join(audio_folder, audio_file_name)
            
            print(f"Processing: {audio_file_name}")
            
            # Load and process audio data
            audio_data = load_audio_data_for_clap(audio_file_path=audio_file_path)
            audio_windows_data = get_audio_windows_data(
                audio_data, window_size=int(10 * CLAP_SR), hop_size=int(10 * CLAP_SR), do_quantize=True
            )
            
            # Get embeddings
            audio_embeddings = model.get_audio_embedding_from_data(x=audio_windows_data, use_tensor=False)

            # Save the embeddings
            embedding_file_name = os.path.splitext(audio_file_name)[0]
            embedding_file_path = os.path.join(embeddings_output_folder, embedding_file_name)

            save_embeddings(audio_embeddings, embedding_file_path, method=save_method)

            
            print(f"Saved embeddings to: {embedding_file_path}")

# Example usage:
# Folder with audio files to process
audio_folder = './data'

# Folder to save embeddings
embeddings_output_folder = './embeddings'

# Process the folder and save embeddings
process_audio_folder(audio_folder=audio_folder, embeddings_output_folder=embeddings_output_folder, save_method='npz')