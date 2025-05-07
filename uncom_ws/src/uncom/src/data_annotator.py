import os
import json
import argparse
from uncom_utils.audio import AudioTranscriber, separate_audio
import torch


def process_videos_in_folder(folder_path):
    transcriber = AudioTranscriber()    
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            audio_path = separate_audio(video_path)
            analysis_result = transcriber.transcribe(audio_path)
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(folder_path, json_filename)
            
            with open(json_path, "w") as json_file:
                json.dump(analysis_result, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files in a folder and create JSON files with audio analysis.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing video files")
    args = parser.parse_args()
    
    process_videos_in_folder(args.folder_path) 