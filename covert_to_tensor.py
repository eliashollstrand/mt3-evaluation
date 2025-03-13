import os
import glob
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MAESTRO_PATH = os.getenv("MAESTRO_PATH")

if not MAESTRO_PATH:
    raise ValueError("Error: MAESTRO_PATH environment variable is not set.")

# Define input directories for MIDI and audio chunks
midi_dir = os.path.join(MAESTRO_PATH, "midi_chunks")
audio_dir = os.path.join(MAESTRO_PATH, "audio_chunks")

# Define the output directory for TFRecord chunks
tfrecord_output_dir = os.path.join(MAESTRO_PATH, "tfrecord_chunks")

# Ensure the output directory exists
os.makedirs(tfrecord_output_dir, exist_ok=True)

# Get all .npy files dynamically for MIDI and audio chunks
midi_files = sorted(glob.glob(os.path.join(midi_dir, "*.npy")))
audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.npy")))

# Ensure equal number of MIDI and audio files
total_records = min(len(midi_files), len(audio_files))

if total_records == 0:
    raise ValueError("No matching MIDI and audio chunks found.")

print(f"Total records to store: {total_records}")

# Batch size for TFRecords (storing 16 samples per file)
batch_size = 16
num_batches = total_records // batch_size

# Function to create a TF Example
def serialize_example(midi_array, audio_array):
    feature = {
        "midi": tf.train.Feature(float_list=tf.train.FloatList(value=midi_array.flatten())),
        "audio": tf.train.Feature(float_list=tf.train.FloatList(value=audio_array.flatten()))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Process and save TFRecords in batches
for batch_idx in range(num_batches):
    batch_midi = []
    batch_audio = []

    for i in range(batch_size):
        idx = batch_idx * batch_size + i
        midi_data = np.load(midi_files[idx]).astype(np.float32)  # Ensure float32
        audio_data = np.load(audio_files[idx]).astype(np.float32)

        batch_midi.append(midi_data.flatten())  # Flatten before storing
        batch_audio.append(audio_data.flatten())

    tfrecord_path = os.path.join(tfrecord_output_dir, f"batch_{batch_idx}.tfrecord")

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for midi_sample, audio_sample in zip(batch_midi, batch_audio):
            tf_example = serialize_example(np.array(midi_sample), np.array(audio_sample))
            writer.write(tf_example)

    print(f"Saved TFRecord batch {batch_idx + 1}/{num_batches}: {tfrecord_path}")

print("All chunks have been processed successfully!")
