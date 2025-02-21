import argparse
import os
import shutil
import tempfile
from pathlib import Path

import torch

from uncom.audio import AudioTranscriber, separate_audio
from uncom.image import (
    PointingDetector,
    ObjectDetector,
    Segmenter,
    annotate_action,
    annotate_image,
    extract_frame,
    load_image,
    pointed_result_index,
)
from uncom.text import CommandExtractor

import webrtcvad
import pyaudio
import numpy as np
import cv2
from scipy.io import wavfile
import threading 
import queue

# This is an extension of the regular understand.py script that makes it run in real time. It uses both whisper to 
# detect that a user has started to speak and media pipe to detect that the person is withih the robot's field of view.  

stream_audio = True
video_recording = False
stream_frames = queue.Queue()

def audio_stream(sample_rate=16000):
    global stream_frames
    global stream_audio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")
    while stream_audio:
        data = stream.read(1024)
        stream_frames.put(np.frombuffer(data, dtype=np.int16))

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

# Function to save audio frames to a file
def save_audio(filename, sample_rate, audio):
    wavfile.write(filename, sample_rate, audio)

def video_stream(output_dir):
    global video_recording
    cap, out = None, None
    print("\n\n\n\n\n\n\nNOT RECORDING\n\n\n\n\n\n\n")
    while not video_recording: pass

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_dir / "input.avi", fourcc, 20.0, (640, 480))

    print("\n\n\n\n\n\n\n RECORDING\n\n\n\n\n\n\n")
    while video_recording:
        ret, frame = cap.read()
        if ret:
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def understand(output_dir, real_time=False, video_path=None , device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Torch on CPU does not support half precision
    torch_dtype = torch.float32 if device == "cpu" else "auto"

    print("Device:", device)

    # Load big models
    transcriber = AudioTranscriber(device=device, torch_dtype=torch_dtype)
    command_extractor = CommandExtractor(device=device, torch_dtype=torch_dtype)

    is_tmp = output_dir is None

    # Create a temporary directory if not specified otherwise clean the given directory
    if is_tmp:
        output_dir = tempfile.mkdtemp()
        output_dir = Path(output_dir)
    else:
        output_dir = Path(output_dir)
        shutil.rmtree(output_dir)
        output_dir.mkdir()

    print("Output directory:", output_dir)
    print("Input path:", video_path)


    if real_time: 
        vad = webrtcvad.Vad()
        vad.set_mode(1)  # Aggressive mode
        full_text = ""
        all_audio_frames = []

        speech_detected = False
        max_silence = 1000
        silence = 0 
 
        video_thread = threading.Thread(target=video_stream,args=(output_dir))
        video_thread.start()
        speech_frames = []
        audio_thread = threading.Thread(target=audio_stream, args=())
        audio_thread.start()

        while True: 
            while True:
                print(1)
                sample_rate = 16000
                frame_duration = 10  # Frame duration in milliseconds
                frame_size = int(sample_rate * frame_duration / 1000)  # Frame size in samples
                stream_frames
                try:
                    audio = np.concatenate(list(stream_frames.queue))
                except:
                    audio = []
                frames = [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]
                for frame in frames:
                    print(2)
                    if len(frame) == frame_size and vad.is_speech(frame.tobytes(), sample_rate):
                        global video_recording
                        video_recording, speech_detected = True, True
                        silence = 0 
                    elif speech_detected:
                        print(3)
                        silence += 10
                    if speech_detected:
                        print(4)
                        speech_frames.append(frame)
                
                if speech_detected and silence > max_silence: 
                    print(5)
                    global stream_audio
                    stream_audio = False
                    break 
                elif not speech_detected:
                    frames = queue.Queue()
            video_recording = False

            if speech_frames:
                print(6)
                # Save audio frames to a file
                audio_data = np.concatenate(speech_frames)
                all_audio_frames.append(audio_data)
                save_audio(output_dir / "temp.wav", sample_rate, audio_data)
                print(7)
                # Convert speech to text using Whisper
                result = transcriber.transcribe(output_dir / "temp.wav")
                text = result["text"]
                full_text += text
                if command_extractor.command_check(full_text):
                    break
                print(8)
        video_path = output_dir / "input.avi"
    # Copy the file to the temp dir
    tmp_video_path = shutil.copy(video_path, output_dir)
    # Extract audio from the video
    tmp_audio_path = separate_audio(tmp_video_path)
    print("Separated audio to", tmp_audio_path)

    # Transcribe the audio
    transcription = transcriber.transcribe(tmp_audio_path)
    print("Transcription:", transcription)

    # Extract the command
    command = command_extractor.extract(transcription)
    print("Command:", command)
    command_path = output_dir / "command.json"
    command.save(command_path)
    print(f"Saved command to {command_path}")

    # Extract relevant frames from the video
    object_frame_path = extract_frame(tmp_video_path, command.object.timestamp[1])
    target_frame_path = extract_frame(tmp_video_path, command.target.timestamp[1])

    print(f"Extracted {command.object.timestamp[1]}s frame from {object_frame_path}")
    print(f"Extracted {command.target.timestamp[1]}s frame from {target_frame_path}")

    del command_extractor
    del transcriber 
    torch.cuda.empty_cache()

    object_detector = ObjectDetector(device=device, torch_dtype=torch_dtype)
    segmenter = Segmenter(device=device, torch_dtype=torch_dtype)
    hand_detector = PointingDetector()

    # Load images of the extracted frames
    object_image = load_image(object_frame_path)
    target_image = load_image(target_frame_path)

    # Detect objects in the corresponding frames
    object_results = object_detector.detect(object_image, command.object.text)
    target_results = object_detector.detect(target_image, command.target.text)

    print(f"Detected {len(object_results)} object instances of '{command.object.text}'")
    print(f"Detected {len(target_results)} target instances of '{command.target.text}'")

    #TODO: Here is the moment to handle concrete vs non-concrete targets

    # If there are multiple objects detected, detect the pointing direction and choose the most likely one
    if len(object_results) > 1:
        object_pointing_vec = hand_detector.detect(object_frame_path)
        print(f"Detected object pointing {object_pointing_vec}")
        pointed_object_idx = pointed_result_index(object_results, object_pointing_vec)
    else:
        pointed_object_idx = object_results[0]

    if len(target_results) > 1:
        target_pointing_vec = hand_detector.detect(target_frame_path)
        print(f"Detected target pointing {target_pointing_vec}")
        pointed_target_idx = pointed_result_index(target_results, target_pointing_vec)
    else:
        pointed_target_idx = target_results[0]

    # Segment only the relevant (pointed at) objects in the corresponding frames
    # Yes... quite a strange destruction expression
    [object_results[pointed_object_idx]] = segmenter.segment(
        object_image, [object_results[pointed_object_idx]]
    )
    [target_results[pointed_target_idx]] = segmenter.segment(
        target_image, [target_results[pointed_target_idx]]
    )

    print(f"Segmented object '{command.object.text}'")
    print(f"Segmented target '{command.target.text}'")

    # Annotate object image
    annotated_object_image = annotate_image(
        object_image, object_results, object_pointing_vec, emph_idx=pointed_object_idx
    )
    annotated_object_image_path = output_dir / "annotated_object.png"
    annotated_object_image.save(annotated_object_image_path)
    print(f"Saved annotated object image to {annotated_object_image_path}")

    # Annotate target image
    annotated_target_image = annotate_image(
        target_image, target_results, target_pointing_vec, emph_idx=pointed_target_idx
    )
    annotated_target_image_path = output_dir / "annotated_target.png"
    annotated_target_image.save(annotated_target_image_path)
    print(f"Saved annotated target image to {annotated_target_image_path}")

    # Produce a complete annotated action image
    caption = f"{command.object.text} - {command.action.text} - {command.target.text}"
    annotated_action = annotate_action(
        annotated_object_image, annotated_target_image, caption
    )
    annotated_action_path = output_dir / "annotated_action.png"
    annotated_action.save(annotated_action_path)
    print(f"Saved annotated action image to {annotated_action_path}")

    # Clean up the temp dir if was used
    if is_tmp:
        shutil.rmtree(output_dir)\

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("real_time")
    parser.add_argument("video")
    parser.add_argument(
        "-o", "--output-dir", default=None, help="output directory path"
    )
    parser.add_argument("--device", default="auto", help="device to use")
    args = parser.parse_args()

    understand(args.output_dir, args.real_time, args.video, args.device)