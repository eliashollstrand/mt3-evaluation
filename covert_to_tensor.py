import os
import glob
import numpy as np
import tensorflow as tf

# Get the absolute path of the current script's directory
repo_root = os.path.dirname(os.path.abspath(__file__))

# Define input directories for MIDI and audio chunks
midi_dir = os.path.join(repo_root, "test", "midi_chunks")
audio_dir = os.path.join(repo_root, "test", "audio_chunks")

# Define the output directory for TFRecord chunks
tfrecord_output_dir = os.path.join(repo_root, "test", "tfrecord_chunks")

# Ensure the output directory exists
os.makedirs(tfrecord_output_dir, exist_ok=True)

# Get all .npy files dynamically for MIDI and audio chunks
midi_files = sorted(glob.glob(os.path.join(midi_dir, "midi_chunk_*.npy")))
audio_files = sorted(glob.glob(os.path.join(audio_dir, "audio_chunk_*.npy")))

print(f"Found {len(midi_files)} MIDI files and {len(audio_files)} audio files")

# Function to create a TF Example
def serialize_example(midi_array, audio_array):
    feature = {
        "midi": tf.train.Feature(float_list=tf.train.FloatList(value=midi_array.flatten())),
        "audio": tf.train.Feature(float_list=tf.train.FloatList(value=audio_array.flatten()))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Loop through and load each file
for midi_path, audio_path in zip(midi_files, audio_files):
    print(f"Processing {midi_path} and {audio_path}...")

    # Load the MIDI and audio chunk files
    midi_data = np.load(midi_path)
    audio_data = np.load(audio_path)

    # Convert them to TensorFlow tensors
    midi_tensor = tf.convert_to_tensor(midi_data, dtype=tf.float32)
    audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)

    # Define the output TFRecord file paths
    midi_record_name = os.path.basename(midi_path).replace(".npy", ".tfrecord")
    audio_record_name = os.path.basename(audio_path).replace(".npy", ".tfrecord")
    midi_tfrecord_path = os.path.join(tfrecord_output_dir, midi_record_name)
    audio_tfrecord_path = os.path.join(tfrecord_output_dir, audio_record_name)

    # Write MIDI to TFRecord
    with tf.io.TFRecordWriter(midi_tfrecord_path) as writer:
        tf_example = serialize_example(midi_tensor.numpy(), audio_tensor.numpy())
        writer.write(tf_example)
    print(f"MIDI TFRecord saved to: {midi_tfrecord_path}")

    # Write audio to TFRecord
    with tf.io.TFRecordWriter(audio_tfrecord_path) as writer:
        tf_example = serialize_example(midi_tensor.numpy(), audio_tensor.numpy())
        writer.write(tf_example)
    print(f"Audio TFRecord saved to: {audio_tfrecord_path}")

