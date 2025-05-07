#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int16MultiArray
import wave
import numpy as np
import soundfile as sf
import resampy
import argparse
import subprocess

def generate_audio(text, path):
    command = ['espeak-ng',
               text,
               '-w',
               path
    ]
    process = subprocess.run(command) 
    # process.wait()

def parse_args():
    parser = argparse.ArgumentParser(description="Process file paths.")
    parser.add_argument('--text', type=str, required=True, help='Speech to be generated')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the generated speech')
    return parser.parse_args()

def preprocess_audio(file_path, target_sample_rate=48000):
    data, sample_rate = sf.read(file_path, dtype='int16')
    if sample_rate != target_sample_rate:
        data = resampy.resample(data, sample_rate, target_sample_rate)
    data = data.astype(np.int16)
    return data

def publish_audio_data(file_path, sample_rate=48000, frame_size=3072, silence_duration=7):
    pub = rospy.Publisher('/audio_frames', Int16MultiArray, queue_size=10)
    rospy.init_node('simulation_audio_publisher', anonymous=False)
    rate = rospy.Rate(10)  # 10 Hz

    audio_data = preprocess_audio(file_path)
    # Add 1 second worth of silence at the end (48,000 samples for 48kHz sample rate)
    silence_samples = silence_duration * sample_rate
    silence_frames = np.zeros(silence_samples, dtype=np.int16)
    audio_data = np.concatenate((audio_data, silence_frames))

    num_frames = len(audio_data) // frame_size

    for i in range(num_frames):
        if rospy.is_shutdown():
            break
        start_idx = i * frame_size
        end_idx = start_idx + frame_size
        frame_data = audio_data[start_idx:end_idx]

        audio_msg = Int16MultiArray()
        audio_msg.data = frame_data.tolist()
        pub.publish(audio_msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        args = parse_args()
        speech = str(args.text)
        audio_file_path = str(args.file_path)
        generate_audio(speech, audio_file_path)
        publish_audio_data(audio_file_path)
    except rospy.ROSInterruptException:
        pass
