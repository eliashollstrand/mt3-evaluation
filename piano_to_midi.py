# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Directory containing your TFRecord files (test/tfrecord_chunks)
# repo_root = os.path.dirname(os.path.abspath(__file__))
# tfrecord_path = os.path.join(repo_root, "test", "tfrecord_chunks")

# # Define a parser for the TFRecord examples
# def _parse_function(example_proto):
#     feature_description = {
#         'audio': tf.io.FixedLenFeature([], tf.string),
#         'midi': tf.io.FixedLenFeature([], tf.string)
#     }
#     # Parse the input tf.Example proto using the dictionary above.
#     parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
#     # Decode the serialized tensors
#     audio = tf.io.parse_tensor(parsed_features['audio'], out_type=tf.float32)
#     midi = tf.io.parse_tensor(parsed_features['midi'], out_type=tf.float32)
    
#     # Reshape tensors to the expected dimensions (as described in the report: 156 time-steps)
#     audio = tf.reshape(audio, [156, 1024])
#     midi = tf.reshape(midi, [156, 131])
    
#     return audio, midi

# # Create a dataset from the TFRecord files
# tfrecord_pattern = os.path.join(tfrecord_path, "*.tfrecord")
# files = tf.data.Dataset.list_files(tfrecord_pattern)
# dataset = tf.data.TFRecordDataset(files)
# dataset = dataset.map(_parse_function)
# dataset = dataset.batch(16)  # Using a batch size similar to the report

# # Define the BLSTM model (as in your original file and Ebert’s report)
# model = Sequential()
# model.add(layers.Bidirectional(layers.LSTM(1024, return_sequences=True, dropout=0.2), merge_mode='sum'))
# model.add(layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=0.2), merge_mode='sum'))
# model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2), merge_mode='sum'))
# model.add(layers.Bidirectional(layers.LSTM(131, return_sequences=True, dropout=0.2), merge_mode='sum'))

# model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

# # Train the model on the dataset from TFRecords
# model.fit(dataset, epochs=250)

# # Save the model
# model.save("blstm_model")

# # Load the model
# loaded_model = tf.keras.models.load_model("blstm_model")

# # Evaluate the model
# loaded_model.evaluate(dataset)

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory containing TFRecord files (now from MAESTRO_PATH)
MAESTRO_PATH = os.getenv("MAESTRO_PATH")
if not MAESTRO_PATH:
    raise ValueError("Error: MAESTRO_PATH environment variable is not set.")

MODEL_PATH = os.path.join(MAESTRO_PATH, "blstm_model")

tfrecord_path = os.path.join(MAESTRO_PATH, "tfrecord_chunks")

# Ensure TFRecord directory exists
if not os.path.exists(tfrecord_path):
    raise ValueError(f"Error: TFRecord directory not found at {tfrecord_path}")

MODEL_NAME = "blstm_model_piano.h5"

# Define a parser for the TFRecord examples
def _parse_function(example_proto):
    feature_description = {
        "audio": tf.io.FixedLenFeature([156 * 1024], tf.float32),  # Expecting flattened audio
        "midi": tf.io.FixedLenFeature([156 * 131], tf.float32)  # Expecting flattened MIDI
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Reshape parsed tensors
    audio = tf.reshape(parsed_features["audio"], [156, 1024])
    midi = tf.reshape(parsed_features["midi"], [156, 131])

    return audio, midi


# Create a dataset from TFRecord files
tfrecord_pattern = os.path.join(tfrecord_path, "*.tfrecord")
files = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=True)

# Ensure there are TFRecords available
num_files = sum(1 for _ in files)
if num_files == 0:
    raise ValueError(f"Error: No TFRecord files found in {tfrecord_path}")

print(f"Found {num_files} TFRecord files.")

# Load and preprocess dataset
dataset = tf.data.TFRecordDataset(files)
dataset = dataset.map(_parse_function)
dataset = dataset.batch(16).shuffle(100)  # Shuffle for better training

# Split dataset into training and validation (90% train, 10% validation)
train_size = int(0.9 * num_files)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

print(f"Training on {train_size} records, validating on {num_files - train_size} records.")

# Define the BLSTM model (based on Ebert’s report)
model = Sequential([
    # layers.Bidirectional(layers.LSTM(1024, return_sequences=True, dropout=0.2), merge_mode='sum'),
    layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=0.2), merge_mode='sum'),
    layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2), merge_mode='sum'),
    layers.Bidirectional(layers.LSTM(131, return_sequences=True, dropout=0.2), merge_mode='sum')
])

# Compile model
model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=1, steps_per_epoch=10, validation_steps=10)

# Save the model
model.save(os.path.join(MAESTRO_PATH, MODEL_NAME))

print("Training complete and model saved.")