import os
import tensorflow as tf
import numpy as np
import librosa
import pretty_midi
import glob
import matplotlib.pyplot as plt

# Ensure MAESTRO_PATH is set
MAESTRO_PATH = os.environ.get("MAESTRO_PATH")
if not MAESTRO_PATH:
    raise EnvironmentError("MAESTRO_PATH environment variable not set.")

MODEL_PATH = os.path.join(MAESTRO_PATH, "models")
MODEL_NAME = "blstm_model_piano_small_overfit.h5"

# Hyperparameters
SAMPLE_RATE = 16000
FRAME_SIZE = 1024
HOP_LENGTH = 256
N_MELS = 128
SEQ_LENGTH = 128  # Length of input sequence for BLSTM
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# MIDI parameters
MIN_MIDI = 21  # A0
MAX_MIDI = 108 # C8
NUM_MIDI_BINS = MAX_MIDI - MIN_MIDI + 1

def load_audio_and_midi(audio_path, midi_path):
    """Loads audio and MIDI data."""
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    return audio, midi_data

def preprocess_audio(audio):
    """Preprocesses audio into mel spectrogram."""
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_db = np.transpose(mel_spectrogram_db)  # Time-major
    return mel_spectrogram_db

def midi_to_pianoroll(midi_data, length):
    """Converts MIDI data to pianoroll."""
    pianoroll = midi_data.get_piano_roll(fs=SAMPLE_RATE / HOP_LENGTH, times=np.arange(0, length / SAMPLE_RATE, HOP_LENGTH / SAMPLE_RATE))
    pianoroll = pianoroll[MIN_MIDI:MAX_MIDI+1, :]
    pianoroll = np.transpose(pianoroll) # Time-major
    pianoroll = np.clip(pianoroll, 0, 1) # Ensure binary
    return pianoroll

def create_dataset(audio_paths, midi_paths):
    """Creates a TensorFlow dataset."""
    def generator():
        for audio_path, midi_path in zip(audio_paths, midi_paths):
            try:
                audio, midi_data = load_audio_and_midi(audio_path, midi_path)
                mel_spectrogram = preprocess_audio(audio)
                pianoroll = midi_to_pianoroll(midi_data, len(audio))

                # Pad or truncate to match lengths
                min_len = min(mel_spectrogram.shape[0], pianoroll.shape[0])
                mel_spectrogram = mel_spectrogram[:min_len]
                pianoroll = pianoroll[:min_len]

                # Create sequences
                for i in range(0, min_len - SEQ_LENGTH, HOP_LENGTH):
                  mel_seq = mel_spectrogram[i:i + SEQ_LENGTH]
                  pianoroll_seq = pianoroll[i:i + SEQ_LENGTH]
                  yield mel_seq, pianoroll_seq
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=(
            tf.TensorSpec(shape=(SEQ_LENGTH, N_MELS), dtype=tf.float32),
            tf.TensorSpec(shape=(SEQ_LENGTH, NUM_MIDI_BINS), dtype=tf.float32),
        )
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# Get file paths
audio_paths = sorted(glob.glob(os.path.join(MAESTRO_PATH, "**/*.wav"), recursive=True))
midi_paths = sorted(glob.glob(os.path.join(MAESTRO_PATH, "**/*.midi"), recursive=True))




# # Create dataset
# dataset = create_dataset(audio_paths, midi_paths)

# # Load or create the model
# model_path = os.path.join(MODEL_PATH, MODEL_NAME)

# if os.path.exists(model_path):
#     print(f"Loading saved model from {model_path}")
#     model = tf.keras.models.load_model(model_path)
# else:
#     print("Creating and training a new model.")
#     model = tf.keras.Sequential([
#         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True), input_shape=(SEQ_LENGTH, N_MELS)),
#         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
#         tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(NUM_MIDI_BINS, activation='sigmoid'))
#     ])

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])

#     model.fit(dataset, epochs=EPOCHS, steps_per_epoch=len(audio_paths) // BATCH_SIZE)

#     # Save the model
#     model.save(model_path)



# Overfitting a small batch
print("Overfitting a small batch...")

# Select a small batch (e.g., first 5 samples)
small_audio_paths = audio_paths[:5]
small_midi_paths = midi_paths[:5]

# Create a small dataset
small_dataset = create_dataset(small_audio_paths, small_midi_paths)

# Take only one batch from the dataset
small_batch = next(iter(small_dataset))

# Load or create the model (as before)
model_path = os.path.join(MODEL_PATH, MODEL_NAME)

if os.path.exists(model_path):
    print(f"Loading saved model from {model_path}")
    model = tf.keras.models.load_model(model_path)
else:
    print("Creating and training a new model.")
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True), input_shape=(SEQ_LENGTH, N_MELS)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(NUM_MIDI_BINS, activation='sigmoid'))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train on the small batch for many epochs
model.fit(tf.data.Dataset.from_tensors(small_batch).repeat(), epochs=1000, steps_per_epoch=1)

print("Small batch overfitting complete.")





def transcribe_audio(audio_path, model, seq_length, hop_length, sample_rate, min_midi, num_midi_bins):
    """Transcribes audio to MIDI using the trained model."""
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    mel_spectrogram = preprocess_audio(audio)  # Assuming preprocess_audio is defined

    predictions = []
    for i in range(0, mel_spectrogram.shape[0] - seq_length, hop_length):
        mel_seq = mel_spectrogram[i:i + seq_length]
        mel_seq = np.expand_dims(mel_seq, axis=0)  # Add batch dimension
        prediction = model.predict(mel_seq)
        predictions.append(prediction[0])  # Remove batch dimension
    predictions = np.concatenate(predictions, axis=0)

    # Pad predictions to match original length
    if mel_spectrogram.shape[0] > predictions.shape[0]:
        padding_length = mel_spectrogram.shape[0] - predictions.shape[0]
        padding = np.zeros((padding_length, num_midi_bins))
        predictions = np.concatenate((predictions, padding), axis=0)
    elif mel_spectrogram.shape[0] < predictions.shape[0]:
        predictions = predictions[:mel_spectrogram.shape[0]]

    return predictions

def pianoroll_to_midi(pianoroll, output_midi_path, hop_length, sample_rate, min_midi):
    """Converts pianoroll to MIDI with improved note onset/offset detection."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Piano
    time_per_frame = hop_length / sample_rate

    # Threshold and smoothing for note detection
    threshold = 0.5
    smoothed_pianoroll = np.copy(pianoroll)

    # Simple smoothing (you can use more advanced techniques)
    for i in range(1, pianoroll.shape[0] - 1):
        smoothed_pianoroll[i] = (pianoroll[i - 1] + pianoroll[i] + pianoroll[i + 1]) / 3.0

    active_notes = {}  # Track active notes and their start times

    for t, frame in enumerate(smoothed_pianoroll):
        for note_idx, velocity in enumerate(frame):
            note_number = note_idx + min_midi

            if velocity > threshold and note_number not in active_notes:
                # Note onset
                active_notes[note_number] = t * time_per_frame
            elif velocity <= threshold and note_number in active_notes:
                # Note offset
                start_time = active_notes.pop(note_number)
                end_time = t * time_per_frame
                note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=end_time)
                instrument.notes.append(note)

    # Handle remaining active notes at the end
    for note_number, start_time in active_notes.items():
        end_time = pianoroll.shape[0] * time_per_frame
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=end_time)
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(output_midi_path)

# Example usage (Replace with your actual audio path)
example_audio_path = audio_paths[0] # Use the first audio file from the dataset for testing.
print(f"Transcribing {example_audio_path}")
transcription = transcribe_audio(example_audio_path, model, SEQ_LENGTH, HOP_LENGTH, SAMPLE_RATE, MIN_MIDI, NUM_MIDI_BINS)

output_midi_path = "improved_output.midi"
pianoroll_to_midi(transcription, output_midi_path, HOP_LENGTH, SAMPLE_RATE, MIN_MIDI)
print(f"Transcription saved to {output_midi_path}")

# After loading a sample audio and MIDI
audio, midi_data = load_audio_and_midi(audio_paths[0], midi_paths[0])
mel_spectrogram = preprocess_audio(audio)
pianoroll = midi_to_pianoroll(midi_data, len(audio))

# Visualize mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH), ref=np.max), sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# Visualize pianoroll
plt.figure(figsize=(10, 4))
plt.imshow(pianoroll.T, aspect='auto', origin='lower', cmap='gray_r')
plt.title('Pianoroll')
plt.xlabel('Time')
plt.ylabel('MIDI Pitch')
plt.tight_layout()
plt.show()