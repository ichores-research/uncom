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


def understand(video_path, output_dir, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Torch on CPU does not support half precision
    torch_dtype = torch.float32 if device == "cpu" else "auto"

    print("Device:", device)

    # Load big models
    transcriber = AudioTranscriber(device=device, torch_dtype=torch_dtype)
    command_extractor = CommandExtractor(device=device, torch_dtype=torch_dtype)
    object_detector = ObjectDetector(device=device, torch_dtype=torch_dtype)
    segmenter = Segmenter(device=device, torch_dtype=torch_dtype)
    hand_detector = PointingDetector()

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

    # Load images of the extracted frames
    object_image = load_image(object_frame_path)
    target_image = load_image(target_frame_path)

    # Detect objects in the corresponding frames
    object_results = object_detector.detect(object_image, command.object.text)
    target_results = object_detector.detect(target_image, command.target.text)

    print(f"Detected {len(object_results)} object instances of '{command.object.text}'")
    print(f"Detected {len(target_results)} target instances of '{command.target.text}'")

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
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument(
        "-o", "--output-dir", default=None, help="output directory path"
    )
    parser.add_argument("--device", default="auto", help="device to use")
    args = parser.parse_args()

    understand(args.video, args.output_dir, args.device)
