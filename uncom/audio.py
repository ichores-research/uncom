import subprocess
from typing import Dict
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def separate_audio(video_path):
    """
    Extract audio from a video file. Uses ffmpeg.
    """
    audio_path = video_path.replace(".mp4", ".mp3")
    # Extract audio from video using ffmpeg
    command = [
        "ffmpeg",
        "-i",
        video_path,  # input video file
        audio_path,  # output image file
    ]
    # Run the FFmpeg command
    subprocess.run(
        command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    return audio_path


class AudioTranscriber:
    """
    Speech-to-text with timestamps.
    """
    def __init__(self, device="cuda", torch_dtype="auto") -> None:
        # Whisper model
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, use_safetensors=True, device_map=device, torch_dtype=torch_dtype
        )
        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True
        )

    def transcribe(self, audio_path: str) -> Dict:
        return self.pipe(audio_path, return_timestamps="word", generate_kwargs={"language": "english"})
