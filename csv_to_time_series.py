import numpy as np
import csv
import os
from dotenv import load_dotenv

# Parameters
window_size_ms = 64  # Each time step represents 64 ms
sample_rate = 16000  # 16 kHz audio
samples_per_window = 1024  # 1024 samples per 64 ms
num_piano_keys = 128  # MIDI has 128 keys
num_pedals = 3  # Sustain, soft, sostenuto
total_features = num_piano_keys + num_pedals  # 131 total features

# Load environment variables
load_dotenv()

# Get the path to the Maestro dataset
MAESTRO_PATH = os.getenv("MAESTRO_PATH")

# Define input and output directories
CSV_INPUT_DIR = os.path.join(MAESTRO_PATH, "csv_midi")
TIME_SERIES_OUTPUT_DIR = os.path.join(MAESTRO_PATH, "time_series")

# Ensure the output directory exists
os.makedirs(TIME_SERIES_OUTPUT_DIR, exist_ok=True)

# Maximum number of files to process
MAX_FILES = 500

def convert_csv_to_time_series(csv_file_path, output_time_series_file):
    """Converts a single MIDI CSV file to time-series format."""
    
    # Step 1: Read MIDI events from CSV
    midi_events = []
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            time = float(row[0])  # Event time in ms
            event_type = row[1]  # 'note_on' or 'note_off' or 'control_change'
            note = int(row[2])  # MIDI note number (0-127)
            velocity = int(row[3])  # MIDI velocity (0-127)

            # Normalize velocity to [0,1]
            velocity = velocity / 127.0  

            # Append the event
            midi_events.append((time, event_type, note, velocity))

    # Step 2: Convert MIDI events to time-series format
    # Determine the total number of time steps
    if not midi_events:
        print(f"Skipping empty file: {csv_file_path}")
        return
    
    max_time = max(event[0] for event in midi_events)  # Last event timestamp
    num_time_steps = int(np.ceil(max_time / window_size_ms))  # Number of 64ms windows

    # Initialize a time-series array (num_time_steps x 131)
    midi_time_series = np.zeros((num_time_steps, total_features))

    # Step 3: Assign MIDI events to the correct time step
    for time, event_type, note, velocity in midi_events:
        time_step = min(int(time // window_size_ms), num_time_steps - 1)

        # Update note velocity in the time-series matrix
        if event_type == "note_on":
            midi_time_series[time_step, note] = velocity  # Store normalized velocity
        elif event_type == "note_off":
            midi_time_series[time_step, note] = 0  # Reset velocity when key is released

        # Handle pedal events 
        if event_type == "control_change":
            if note == 64:  # Sustain pedal (CC 64)
                midi_time_series[time_step, 128] = velocity  # Store sustain pedal value
            elif note == 66:  # Sostenuto pedal (CC 66)
                midi_time_series[time_step, 129] = velocity
            elif note == 67:  # Soft pedal (CC 67)
                midi_time_series[time_step, 130] = velocity

    # Step 4: Save the time-series data as a new CSV file
    np.savetxt(output_time_series_file, midi_time_series, delimiter=",")

    print(f"Time-series MIDI data saved to {output_time_series_file}")

# Calculate total number of CSV files to convert
csv_files = [f for f in os.listdir(CSV_INPUT_DIR) if f.endswith(".csv")]
total_num_files = min(len(csv_files), MAX_FILES)
print(f"Found {len(csv_files)} CSV files in {CSV_INPUT_DIR}. Processing up to {MAX_FILES} files.")

# Process up to 500 CSV files
file_count = 0
for csv_file in csv_files:
    if file_count >= MAX_FILES:
        print("Reached maximum file limit (500). Stopping...")
        break

    file_count += 1
    print(f"Processing file {file_count}/{total_num_files}: {csv_file}")
    csv_path = os.path.join(CSV_INPUT_DIR, csv_file)
    time_series_path = os.path.join(TIME_SERIES_OUTPUT_DIR, f"{os.path.splitext(csv_file)[0]}_time_series.csv")
    convert_csv_to_time_series(csv_path, time_series_path)

print("Processing complete!")
