import numpy as np
import librosa
import librosa.display
import os  # Import the os module for handling file paths

# File paths
midi_time_series_file = "/home/leo/kth/kexjobb/test/time_series/Chamber2_time_series.csv"
audio_file = "/home/leo/kth/kexjobb/test/mono/Chamber2_mono.wav"
sr = 16000  # Sample rate (16 kHz)

# Output directory for saving chunks
output_dir = "/home/leo/kth/kexjobb/test/ready_data/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Print to confirm directory is created
print(f"Saving chunks to: {output_dir}")

# Load MIDI time series
midi_time_series = np.loadtxt(midi_time_series_file, delimiter=",")

# Check shape of the midi_time_series to see if it has enough time steps
print(f"Shape of midi_time_series: {midi_time_series.shape}")

# Check number of time steps in the midi_time_series
print(f"Number of time steps in midi_time_series: {midi_time_series.shape[0]}")

# Load audio and extract features (Mel spectrogram)
y, _ = librosa.load(audio_file, sr=sr)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=1024)

# Convert spectrogram to time steps matching MIDI (64ms per step)
mel_spec = mel_spec[:, :midi_time_series.shape[0]]  # Trim or pad to match

# Define chunk size (adjusted to match available time steps)
# Define chunk size (156 time steps for 9.984s)
chunk_size = 156
num_chunks = midi_time_series.shape[0] // chunk_size

# Print number of chunks to be processed
print(f"Number of chunks: {num_chunks}")

# If num_chunks is zero, we need to check the MIDI data
if num_chunks == 0:
    print("The number of time steps in the MIDI time series is too small to create chunks.")
    print("Consider adjusting the chunk_size or check the MIDI time series data.")

# Split into chunks and save them to the specified directory
for i in range(num_chunks):
    midi_chunk = midi_time_series[i * chunk_size:(i + 1) * chunk_size, :]
    audio_chunk = mel_spec[:, i * chunk_size:(i + 1) * chunk_size]

    # Save chunks (e.g., as .npy for training)
    midi_chunk_path = os.path.join(output_dir, f"midi_chunk_{i}.npy")
    audio_chunk_path = os.path.join(output_dir, f"audio_chunk_{i}.npy")
    
    print(f"Saving {midi_chunk_path}")
    print(f"Saving {audio_chunk_path}")

    np.save(midi_chunk_path, midi_chunk)
    np.save(audio_chunk_path, audio_chunk)

print(f"MIDI and audio aligned into 10-second chunks and saved to {output_dir}")




