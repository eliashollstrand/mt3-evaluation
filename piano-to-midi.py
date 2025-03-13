import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory containing your TFRecord files (test/tfrecord_chunks)
repo_root = os.path.dirname(os.path.abspath(__file__))
tfrecord_path = os.path.join(repo_root, "test", "tfrecord_chunks")

# Define a parser for the TFRecord examples
def _parse_function(example_proto):
    feature_description = {
        'audio': tf.io.FixedLenFeature([], tf.string),
        'midi': tf.io.FixedLenFeature([], tf.string)
    }
    # Parse the input tf.Example proto using the dictionary above.
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode the serialized tensors
    audio = tf.io.parse_tensor(parsed_features['audio'], out_type=tf.float32)
    midi = tf.io.parse_tensor(parsed_features['midi'], out_type=tf.float32)
    
    # Reshape tensors to the expected dimensions (as described in the report: 156 time-steps)
    audio = tf.reshape(audio, [156, 1024])
    midi = tf.reshape(midi, [156, 131])
    
    return audio, midi

# Create a dataset from the TFRecord files
tfrecord_pattern = os.path.join(tfrecord_path, "*.tfrecord")
files = tf.data.Dataset.list_files(tfrecord_pattern)
dataset = tf.data.TFRecordDataset(files)
dataset = dataset.map(_parse_function)
dataset = dataset.batch(16)  # Using a batch size similar to the report

# Define the BLSTM model (as in your original file and Ebert’s report)
model = Sequential()
model.add(layers.Bidirectional(layers.LSTM(1024, return_sequences=True, dropout=0.2), merge_mode='sum'))
model.add(layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=0.2), merge_mode='sum'))
model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2), merge_mode='sum'))
model.add(layers.Bidirectional(layers.LSTM(131, return_sequences=True, dropout=0.2), merge_mode='sum'))

model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

# Train the model on the dataset from TFRecords
model.fit(dataset, epochs=250)

# Save the model
model.save("blstm_model")

# Load the model
loaded_model = tf.keras.models.load_model("blstm_model")

# Evaluate the model
loaded_model.evaluate(dataset)

