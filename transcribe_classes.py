import os
import webvtt

#create a function to convert vtt to text
def read_vtt(file_path):
    captions = webvtt.read(file_path)
    text = " ".join([cap.text for cap in captions])
    return text

# Create the ClassTranscriptions directory if it doesn't exist
transcriptions_dir = "ClassTranscriptions"
if not os.path.exists(transcriptions_dir):
    os.makedirs(transcriptions_dir)

# List all VTT files in the VTTFiles directory
VTT_dir = "ClassVTTFiles"
VTT_files = [f for f in os.listdir(VTT_dir) if f.endswith('.vtt')]

total_files = len(VTT_files)

# Transcribe each episode
for index, VTT_file in enumerate(VTT_files, start=1):
    title = os.path.splitext(VTT_file)[0]

    # Check if transcription file already exists
    txt_filename = title + ".txt"
    txt_path = os.path.join(transcriptions_dir, txt_filename)
    if os.path.exists(txt_path):
        print(
            f'Skipping "{title}" - {index}/{total_files} (transcription already exists)')
        continue

    print(f'Transcribing "{title}" - {index}/{total_files}')

    VTT_path = os.path.join(VTT_dir, VTT_file)
    result = read_vtt(VTT_path)

    # Save the transcription to a .txt file
    with open(txt_path, 'w') as txt_file:
        txt_file.write(result)

print("All classes transcribed and saved in the ClassTranscriptions directory!")
