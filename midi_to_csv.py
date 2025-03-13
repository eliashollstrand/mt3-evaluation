import mido
import csv
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the path to the Maestro dataset
MAESTRO_PATH = os.getenv("MAESTRO_PATH")

# Define output directory for CSV files
CSV_OUTPUT_DIR = os.path.join(MAESTRO_PATH, "csv_midi")
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)  # Ensure the directory exists

def convert_midi_to_csv(midi_path, csv_path):
    """Converts a single MIDI file to CSV format."""
    try:
        midi_file = mido.MidiFile(midi_path)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Time', 'Type', 'Note', 'Velocity'])  # CSV Header
            
            time = 0  # Track accumulated time
            for track in midi_file.tracks:
                for msg in track:
                    time += msg.time  # Accumulate delta time
                    if msg.type in ['note_on', 'note_off']:  # Filter MIDI messages
                        writer.writerow([time, msg.type, msg.note, msg.velocity])

        print(f"Converted: {midi_path} -> {csv_path}")
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")

# Recursively process all MIDI files in MAESTRO_PATH
if MAESTRO_PATH:
    print(f"Converting MIDI files in {MAESTRO_PATH} to CSV...")
    for root, _, files in os.walk(MAESTRO_PATH):
        for file_name in files:
            if file_name.lower().endswith(('.midi', '.mid')):  # Check for MIDI files
                midi_path = os.path.join(root, file_name)
                csv_file_name = f"{os.path.splitext(file_name)[0]}.csv"
                csv_path = os.path.join(CSV_OUTPUT_DIR, csv_file_name)
                convert_midi_to_csv(midi_path, csv_path)
    print("Conversion complete.")
else:
    print("Error: MAESTRO_PATH environment variable not set.")

