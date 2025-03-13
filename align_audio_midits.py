import numpy as np
import librosa
import os
from dotenv import load_dotenv

# Sample rate and processing window settings
sr = 16000  # Sample rate (16 kHz)
window_size = 1024  # 1024 samples per 64 ms window
chunk_size = 156  # Number of time steps per 10s chunk

# Load environment variables
load_dotenv()

# Path to MAESTRO dataset
MAESTRO_PATH = os.getenv("MAESTRO_PATH")

# Paths for processed audio and MIDI time series
AUDIO_DIR = os.path.join(MAESTRO_PATH, "wav_16k")
MIDI_DIR = os.path.join(MAESTRO_PATH, "time_series")

# Output directories
OUTPUT_MIDI_DIR = os.path.join(MAESTRO_PATH, "midi_chunks")
OUTPUT_AUDIO_DIR = os.path.join(MAESTRO_PATH, "audio_chunks")

# Ensure output directories exist
os.makedirs(OUTPUT_MIDI_DIR, exist_ok=True)
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

MAX_FILES = 500  # Maximum number of files to process

def process_file(audio_file, file_count, total_files):
    """Process a single audio file and corresponding MIDI time series."""
    file_name = os.path.basename(audio_file).replace(".wav", "")  # Get base filename
    midi_file = os.path.join(MIDI_DIR, f"{file_name}_time_series.csv")

    if not os.path.exists(midi_file):
        print(f"Skipping {file_name}: No matching MIDI time series found.")
        return
    
    # Load MIDI time series
    midi_time_series = np.loadtxt(midi_file, delimiter=",")
    num_time_steps = midi_time_series.shape[0]
    print(f"Processing {file_name}: MIDI shape {midi_time_series.shape}")

    # Load and reshape audio
    y, _ = librosa.load(audio_file, sr=sr)
    expected_length = num_time_steps * window_size  # Expected samples

    # Ensure correct audio length
    if len(y) < expected_length:
        y = np.pad(y, (0, expected_length - len(y)), mode='constant')
    else:
        y = y[:expected_length]

    # Reshape audio into time-series format
    audio_time_series = y.reshape(num_time_steps, window_size)

    # Compute number of chunks
    num_chunks = num_time_steps // chunk_size
    if num_chunks == 0:
        print(f"Skipping {file_name}: Not enough time steps for chunks.")
        return

    # Save chunks
    for i in range(num_chunks):
        midi_chunk = midi_time_series[i * chunk_size:(i + 1) * chunk_size, :]
        audio_chunk = audio_time_series[i * chunk_size:(i + 1) * chunk_size, :]

        midi_chunk_path = os.path.join(OUTPUT_MIDI_DIR, f"{file_name}_chunk_{i}.npy")
        audio_chunk_path = os.path.join(OUTPUT_AUDIO_DIR, f"{file_name}_chunk_{i}.npy")

        np.save(midi_chunk_path, midi_chunk)
        np.save(audio_chunk_path, audio_chunk)

        # print(f"Saved {midi_chunk_path} and {audio_chunk_path}")

    print(f"Finished processing file {file_count}/{total_files} {file_name}")

def process_all_files():
    """Process up to 500 audio and MIDI files in the dataset."""
    if not os.path.exists(AUDIO_DIR) or not os.path.exists(MIDI_DIR):
        print("Error: Audio or MIDI time series directory not found.")
        return

    audio_files = [os.path.join(root, file)
                   for root, _, files in os.walk(AUDIO_DIR)
                   for file in files if file.endswith(".wav")]

    total_files = len(audio_files)
    num_to_process = min(total_files, MAX_FILES)  # Limit to 500 files

    print(f"Found {total_files} audio files. Processing up to {num_to_process} files.")

    file_count = 1
    for audio_file in audio_files[:num_to_process]:  # Slice to only process 500
        process_file(audio_file, file_count, num_to_process)
        file_count += 1

if __name__ == "__main__":
    if not MAESTRO_PATH:
        print("Error: MAESTRO_PATH environment variable is not set.")
    else:
        process_all_files()
        print("Processing complete!")


if __name__ == "__main__":
    if not MAESTRO_PATH:
        print("Error: MAESTRO_PATH environment variable is not set.")
    else:
        process_all_files()
        print("All files processed successfully!")
