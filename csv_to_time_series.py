import numpy as np
import csv
import os

# Parameters
window_size_ms = 64  # Each time step represents 64 ms
sample_rate = 16000  # 16 kHz audio
samples_per_window = 1024  # 1024 samples per 64 ms
num_piano_keys = 128  # MIDI has 128 keys
num_pedals = 3  # Sustain, soft, sostenuto
total_features = num_piano_keys + num_pedals  # 131 total features

# Load MIDI CSV
# csv_file_path = '/home/leo/kth/kexjobb/test/csv_midi/Chamber2_csv.csv'  # Replace with your CSV file path

# output file path
# output_time_series_file = "/home/leo/kth/kexjobb/test/time_series/Chamber2_time_series.csv"

# Get the absolute path of the current script's directory
repo_root = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the repository
csv_file_path = os.path.join(repo_root, "test", "csv_midi", "Chamber2_csv.csv")
output_time_series_file = os.path.join(repo_root, "test", "time_series", "Chamber2_time_series.csv")

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
max_time = max(event[0] for event in midi_events)  # Find the last event's timestamp
num_time_steps = int(np.ceil(max_time / window_size_ms))  # Number of 64ms windows

print(f'Number of time steps: {num_time_steps}')
print(max_time)

# Initialize a time-series array (num_time_steps x 131)
midi_time_series = np.zeros((num_time_steps, total_features))

# Step 3: Assign MIDI events to the correct time step
for time, event_type, note, velocity in midi_events:
    time_step = int(time // window_size_ms)  # Find the corresponding time step

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

# Print first 5 time steps
print(midi_time_series[:1])  

# Step 4: Save the time-series data as a new CSV file
np.savetxt(output_time_series_file, midi_time_series, delimiter=",")

print(f"Time-series MIDI data saved to {output_time_series_file}")
