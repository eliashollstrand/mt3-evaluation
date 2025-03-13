import numpy as np
import librosa
import os

# Sample rate and processing window settings
sr = 16000  # Sample rate (16 kHz)
window_size = 1024  # 1024 samples per 64 ms window
chunk_size = 156  # Number of time steps per 10s chunk

# Get script directory
repo_root = os.path.dirname(os.path.abspath(__file__))

# File paths
midi_time_series_file = os.path.join(repo_root, "test", "time_series", "Chamber2_time_series.csv")
audio_file = os.path.join(repo_root, "test", "mono", "Chamber2_mono.wav")

# Output directories
output_dir_midi = os.path.join(repo_root, "test", "midi_chunks")
output_dir_audio = os.path.join(repo_root, "test", "audio_chunks")

# Ensure output directories exist
os.makedirs(output_dir_midi, exist_ok=True)
os.makedirs(output_dir_audio, exist_ok=True)

# Load MIDI time series (expected shape: [num_time_steps, 131])
midi_time_series = np.loadtxt(midi_time_series_file, delimiter=",")
num_time_steps = midi_time_series.shape[0]

print(f"Loaded MIDI time series with shape: {midi_time_series.shape}")

# Load raw audio
y, _ = librosa.load(audio_file, sr=sr)

# Ensure audio length matches expected number of windows
expected_length = num_time_steps * window_size  # Total expected samples
if len(y) < expected_length:
    y = np.pad(y, (0, expected_length - len(y)), mode='constant')
else:
    y = y[:expected_length]

# Reshape audio into (num_time_steps, 1024)
audio_time_series = y.reshape(num_time_steps, window_size)

print(f"Reshaped audio into time-series format with shape: {audio_time_series.shape}")

# Compute number of chunks
num_chunks = num_time_steps // chunk_size

if num_chunks == 0:
    print("Error: Not enough time steps to create a chunk.")
else:
    print(f"Number of 10-second chunks: {num_chunks}")

# Split and save chunks
for i in range(num_chunks):
    midi_chunk = midi_time_series[i * chunk_size:(i + 1) * chunk_size, :]
    audio_chunk = audio_time_series[i * chunk_size:(i + 1) * chunk_size, :]

    midi_chunk_path = os.path.join(output_dir_midi, f"midi_chunk_{i}.npy")
    audio_chunk_path = os.path.join(output_dir_audio, f"audio_chunk_{i}.npy")

    np.save(midi_chunk_path, midi_chunk)
    np.save(audio_chunk_path, audio_chunk)

    print(f"Saved {midi_chunk_path} and {audio_chunk_path}")

print("Audio and MIDI alignment complete.")
