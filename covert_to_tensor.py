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

# Function to create a TF Example
def serialize_example(midi_array, audio_array):
    feature = {
        "midi": tf.train.Feature(float_list=tf.train.FloatList(value=midi_array.flatten())),
        "audio": tf.train.Feature(float_list=tf.train.FloatList(value=audio_array.flatten()))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Process and save TFRecords
for record_num, (midi_path, audio_path) in enumerate(zip(midi_files, audio_files), start=1):
    file_name = os.path.basename(midi_path).replace(".npy", "")
    print(f"Processing {record_num}/{total_records}: {file_name}...")

    # Load the MIDI and audio chunk files
    midi_data = np.load(midi_path)
    audio_data = np.load(audio_path)

    # Define the output TFRecord file path
    tfrecord_path = os.path.join(tfrecord_output_dir, f"{file_name}.tfrecord")

    # Write to TFRecord
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        tf_example = serialize_example(midi_data, audio_data)
        writer.write(tf_example)

    print(f"Saved TFRecord {record_num}/{total_records}: {tfrecord_path}")

print("All chunks have been processed successfully!")
