import os
import librosa
import soundfile as sf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the MAESTRO dataset path from environment variables
MAESTRO_PATH = os.getenv("MAESTRO_PATH")

# Directory to save the downsampled audio
OUTPUT_DIR = os.path.join(MAESTRO_PATH, "wav_16k")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target sample rate
TARGET_SR = 16000  # 16kHz

# Total number of files to be processed
total_files = 0

def downsample_wav(input_path, output_path, target_sr=TARGET_SR):
    """Loads a WAV file, downsamples to 16kHz, converts to mono, and saves it."""
    try:
        # Load audio (converts to mono automatically if mono=True is set)
        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

        # Save the new file
        sf.write(output_path, audio, target_sr)
        print(f"Processed: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_all_wav_files(maestro_path):
    """Find and downsample all WAV files only in the selected year folders."""
    file_count = 0
    allowed_folders = {str(year) for year in range(2004, 2019)}  # Set of allowed years

    for root, _, files in os.walk(maestro_path):
        # Get the folder name at the first level
        relative_root = os.path.relpath(root, maestro_path)
        first_level_folder = relative_root.split(os.sep)[0]  # Get top-level folder

        # Skip directories not in the allowed list
        if first_level_folder not in allowed_folders:
            continue

        for file in files:
            if file.endswith(".wav"):
                file_count += 1
                print(f"Processing file {file_count}/{total_files}: {file}")
                input_path = os.path.join(root, file)
                
                # Create relative output path inside wav_16k
                relative_path = os.path.relpath(input_path, maestro_path)
                output_path = os.path.join(OUTPUT_DIR, relative_path)

                # Ensure the subdirectories exist 
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Process the WAV file
                downsample_wav(input_path, output_path)


if __name__ == "__main__":
    if not MAESTRO_PATH:
        print("Error: MAESTRO_PATH environment variable is not set.")
    else:
        # Count wav files in MAESTRO_PATH
        total_files = sum(1 for root, _, files in os.walk(MAESTRO_PATH) for file in files if file.endswith(".wav"))

        print(f"Processing all {total_files} WAV files in: {MAESTRO_PATH}")
        process_all_wav_files(MAESTRO_PATH)
        print("All files processed successfully!")
