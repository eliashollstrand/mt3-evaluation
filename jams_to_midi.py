from dotenv import load_dotenv
import os
import json
import pretty_midi

# Load environment variables
load_dotenv()

# GuitarSet paths
GUITARSET_PATH = os.getenv("GUITARSET_PATH")
GUITARSET_MIC_PATH = os.path.join(GUITARSET_PATH, "audio_mono-mic")
GUITARSET_ANNOTATIONS_PATH = os.path.join(GUITARSET_PATH, "annotation")
GUITARSET_MIDI_PATH = os.path.join(GUITARSET_PATH, "midi")


def jams_to_midi(jams_file, midi_file):
    """
    Convert a JAMS file (loaded as JSON) to a MIDI file.
    """
    # Load the JAMS file as JSON
    with open(jams_file, "r", encoding="us-ascii") as f:
        # jam_data = json.load(f)
        try:
            with open(jams_file, "r", encoding="utf-8") as f:
                jam_data = json.load(f)
        except UnicodeDecodeError as e:
            return

    # Create a new MIDI file
    midi = pretty_midi.PrettyMIDI()

    # Create an instrument
    instrument = pretty_midi.Instrument(program=0)

    # Extract note annotations
    for annotation in jam_data["annotations"]:
        if annotation["namespace"] == "note_midi":  # Ensure it's a note-based annotation
            for note in annotation["data"]:
                start_time = note["time"]
                duration = note["duration"]
                pitch = int(note["value"])  # MIDI pitch number
                
                # Handle confidence (set to default if None)
                confidence = note.get("confidence", 1)
                if confidence is None:
                    confidence = 1  # Default to max confidence if missing
                
                velocity = int(confidence * 127)  # Scale confidence to MIDI velocity
                
                # Create a MIDI note
                midi_note = pretty_midi.Note(
                    velocity=velocity, pitch=pitch,
                    start=start_time, end=start_time + duration
                )
                instrument.notes.append(midi_note)

    # Add the instrument to the MIDI object
    midi.instruments.append(instrument)

    # Write the MIDI file
    midi.write(midi_file)
    print(f"Converted {jams_file} → {midi_file}")


def convert_jams_to_midi():
    """ 
    Convert all JAMS files in the GuitarSet dataset to MIDI files.
    """
    for filename in os.listdir(GUITARSET_ANNOTATIONS_PATH):
        # check if already converted
        if os.path.exists(os.path.join(GUITARSET_MIDI_PATH, filename.replace(".jams", "_mic.midi"))):
            print(f"Already converted {filename}")
            continue
        if filename.endswith(".jams"):
            jams_file = os.path.join(GUITARSET_ANNOTATIONS_PATH, filename)
            midi_file = os.path.join(GUITARSET_MIDI_PATH, filename.replace(".jams", "_mic.midi"))
            jams_to_midi(jams_file, midi_file)

# convert_jams_to_midi()