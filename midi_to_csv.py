import mido
import csv
import os

# Path to input MIDI file
# midi_file_path = '/home/leo/kth/kexjobb/test/midi/Chamber2.midi'  # Make sure this path points to your MIDI file

# Output CSV file path
# csv_file_path = '/home/leo/kth/kexjobb/test/csv_midi/Chamber2_csv.csv'

## Testing if midi info is written to the file
# test_csv_path = '/home/leo/kth/kexjobb/test/csv_midi/test_output.csv'

# Get the directory of the script
repo_root = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the repository root
midi_file_path = os.path.join(repo_root, "test", "midi", "Chamber2.midi")
csv_file_path = os.path.join(repo_root, "test", "csv_midi", "Chamber2_csv.csv")
test_csv_path = os.path.join(repo_root, "test", "csv_midi", "test_output.csv")

# Load the MIDI file
midi_file = mido.MidiFile(midi_file_path)


# Open the CSV file for writing
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the header row
    writer.writerow(['Time', 'Type', 'Note', 'Velocity'])
    
    # Iterate through each track and its messages in the MIDI file
    for track in midi_file.tracks:
        time = 0
        for msg in track:
            time += msg.time  # Accumulate the time delta
            
            # Print the message to see what's being processed
            print(f"Message: {msg}")
            
            # Write 'note_on' and 'note_off' events to CSV, filtering out types like control change that contains metadata
            if msg.type == 'note_on' or msg.type == 'note_off':
                writer.writerow([time, msg.type, msg.note, msg.velocity])

    # Explicitly flush the writer to ensure data is written to the file
    csv_file.flush()


'''
## Testing if midi info is written to the file
with open(test_csv_path, 'w', newline='', encoding='utf-8') as test_cv:
    writer=csv.writer(test_cv)
    writer.writerow(['Test', 'Data'])
    writer.writerow([1, 2])
    writer.writerow([3, 4])

print(f"Message Type: {msg.type}")
'''
print(f"MIDI file converted to CSV: {csv_file_path}")





#input: '/home/leo/kth/kexjobb/test/midi/DearYou.midi'
#output: '/home/leo/kth/kexjobb/test/csv_midi/DearYouCSV.csv'
