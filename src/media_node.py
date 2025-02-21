#!/usr/bin/env python3

import rospy
import threading
from std_msgs.msg import Bool, String, Int16MultiArray, Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pathlib import Path
from scipy.io import wavfile
import numpy as np
import cv2
from queue import Queue
import webrtcvad

class MediaNode:
    def __init__(self, output_dir, fps, frame_size):
        # Initialize the ROS Node
        rospy.init_node('media_node', anonymous=True)

        # Initialize a CvBridge object to convert ROS images to OpenCV images
        self.bridge = CvBridge()
        # Subscribe to video and audio recording control topics (Bool)
        self.video_record_sub = rospy.Subscriber('/video_recording', Bool, self.video_recording_callback)
        self.audio_record_sub = rospy.Subscriber('/audio_recording', Bool, self.audio_recording_callback)

        # Subscribe to image and audio topics
        self.image_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, self.image_callback)
        
        # Subscribe to audio topic with Int16MultiArray to receive raw audio data
        self.audio_sub = rospy.Subscriber('/audio_frames', Int16MultiArray, self.audio_callback)

        self.video_save_sub = rospy.Subscriber('/save_video', String, self.save_video)
        self.audio_save_sub = rospy.Subscriber('/save_audio', String, self.save_audio)

        # Receive requests to clear buffered audio frames
        self.clear_audio_sub = rospy.Subscriber('/clear_video', Empty, self.clear_audio_callback)
        # Receive requests to clear buffered video frames
        self.clear_video_sub = rospy.Subscriber('/clear_audio', Empty, self.clear_video_callback)        

        # Publishes when speech is detected
        self.speech_detected_pub = rospy.Publisher('/speech_detected', Bool, queue_size=10)

        # Subscribe to output directory topic (String)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1) # agressiveness from 1-9 

        # Class variables to store recording status and output directory

        self.video_record = False  # Video recording statushelp fleshing out the backstory for this clan's arrival in Brazil?

        self.audio_record = False  # Audio recording status;
        self.output_dir = Path()  # Output directory as a Path object;
        self.audio_frames = Queue()  # List to store audio frames;
        self.audio_buffer = Queue()
        self.video_frames = Queue()  # List to store video frames;
        self.framerate = 30.0  # camera framerate;
        self.dim = (640, 480)  # camera image frame dimensions;
        self.sample_rate = 44100  # microphone sampling rate;
        self.speech_detected = False  # stores whether speech has already been detected or not.

    def video_recording_callback(self, msg):
        """
        Callback for the video recording control topic.
        Updates self.video_record based on the received boolean.
        """
        self.video_record = msg.data
        rospy.loginfo(f"Video recording status updated: {self.video_record}")

    def audio_recording_callback(self, msg):
        """
        Callback for the audio recording control topic.
        Updates self.audio_record based on the received boolean.
        """
        self.audio_record = msg.data
        rospy.loginfo(f"Audio recording status updated: {self.audio_record}")

    def image_callback(self, msg):
        """
        Callback for the image topic.
        Converts the ROS Image message to an OpenCV image.
        """
        try:
            # Convert the ROS Image message to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rospy.loginfo("Received Image!")
            # If video recording is active, call the video_stream method
            if self.video_record:
                self.video_frames.put(cv_image)
        except CvBridgeError as e:
            rospy.logerr("Error converting image: %s", str(e))

    def audio_callback(self, msg: Int16MultiArray):
        rospy.loginfo("Received Audio Data (Int16MultiArray)")
        print("Oie", msg.data)
        
        try:
            # Convert the list of integers to a numpy array
            audio_data = np.array(msg.data, dtype=np.int16)
            self.audio_buffer.put(audio_data)
            self.publish_speech_detected(self.detect_speech())
            
            if self.audio_record:
                self.audio_frames.put(audio_data)
                rospy.loginfo(f"Audio frame received, total frames: {self.audio_frames.qsize()}")
        except Exception as e:
            rospy.logerr(f"Failed to convert audio data: {e}")

    def clear_audio_callback(self,msg):
        """
        Helper fucntion that clears all stored audio_frames
        """
        self.audio_frames = Queue()

    def clear_video_callback(self,msg):
        """
        Helper fucntion that clears all stored video_frames
        """
        self.video_frames = Queue()

    def publish_speech_detected(self, status):
        """
        Publish the speech detection status to the /speech_detected topic.
        :param status: Bool - True if speech was detected, False otherwise
        """
        msg = Bool()
        msg.data = status
        self.speech_detected_pub.publish(msg)
        rospy.loginfo(f"Published speech detection status: {status}")

    def save_video(self, msg):
        """
        Helper function that saves currently stored video frames
        """
        path = msg.data
        if not Path(path).parent.exists():
            rospy.logerr(f"Directory for {path} does not exist!")
            return
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path, fourcc, self.framerate, self.dim)
        
        # Dequeue video frames and write to video file
        for frame in list(self.video_frames.queue):
            out.write(frame)
        
        out.release()

    def save_audio(self, msg):
        """
        Helper function that saves currently stored audio frames
        """
        # Concatenate the audio frames into one array
        path = msg.data
        audio = np.concatenate(list(self.audio_frames.queue))
        # Save the audio as a WAV file
        wavfile.write(path, self.sample_rate, audio)
        rospy.loginfo(f"Audio saved to {path}")

    def detect_speech(self):
        """
        Helper method that detects speech in the frames stored in the audio buffer.
        """
        frame_duration = 10  # Frame duration in milliseconds
        frame_size = int(self.sample_rate * frame_duration / 1000)  # Frame size in samples
        frames = list(self.audio_buffer.queue)
        audio = np.concatenate(frames)
        frames = [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]
        for frame in frames:
            if len(frame) == frame_size:
                if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                    return True
                self.audio_buffer = Queue()
        return False

    def run(self):
        """
        Run the ROS node.
        """
        rospy.spin()


if __name__ == '__main__':
    # Instantiate the MediaNode class and run the node
    node = MediaNode(output_dir=Path('/tmp'), fps=30, frame_size=(640, 480))
    node.run()
