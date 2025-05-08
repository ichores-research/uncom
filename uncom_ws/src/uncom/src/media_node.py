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
    def __init__(self):
        # Initialize the ROS Node
        rospy.init_node('media_node', anonymous=True)

        # Class variables to store recording status and output directory

        self.video_record = False  # Video recording statushelp fleshing out the backstory for this clan's arrival in Brazil?
        self.audio_record = False  # Audio recording status
        self.output_dir = Path()  # Output directory as a Path object
        self.audio_frames = Queue()  # List to store audio frames
        self.audio_buffer = Queue()
        self.video_frames = Queue()  # List to store video frames
        self.speech_detected = False
        self.vad_status = True
        self.depth_frame = None
        
        # Initialize a CvBridge object to convert ROS images to OpenCV images
        self.bridge = CvBridge()
        # Subscribe to video and audio recording control topics (Bool)
        self.video_record_sub = rospy.Subscriber('/video_recording', Bool, self.video_recording_callback)
        self.audio_record_sub = rospy.Subscriber('/audio_recording', Bool, self.audio_recording_callback)

        self.vad_status_sub = rospy.Subscriber('/perform_vad', Bool, self.perform_vad_callback)

        # Subscribe to image topic #default: /xtion/rgb/image_raw
        camera_topic = rospy.get_param("camera_topic")
        self.image_sub = rospy.Subscriber(camera_topic, Image, self.image_callback)

        # Subscribe to audio topic with Int16MultiArray to receive raw audio data # default /audio_frames
        audio_topic = rospy.get_param("audio_topic")
        self.audio_sub = rospy.Subscriber(audio_topic, Int16MultiArray, self.audio_callback)

        self.video_save_sub = rospy.Subscriber('/save_video', String, self.save_video)
        self.audio_save_sub = rospy.Subscriber('/save_audio', String, self.save_audio)

        self.clear_audio_sub = rospy.Subscriber('/clear_video', Empty, self.clear_audio_callback)
        self.clear_video_sub = rospy.Subscriber('/clear_audio', Empty, self.clear_video_callback)        

        ## publisher responsible for broadcasting that human speech was detected.
        self.speech_detected_pub = rospy.Publisher('/speech_detected', Bool, queue_size=10)


        ## voice activity detector 
        self.vad = webrtcvad.Vad()


        #ROS parameters used by the node.

        self.framerate = rospy.get_param("camera_fps")
        self.dim = (rospy.get_param("camera_image_width"), rospy.get_param("camera_image_height"))
        self.sample_rate = rospy.get_param("audio_rate")
        self.vad.set_mode(rospy.get_param("vad_aggressiveness")) # agressiveness from 1-9

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


    def perform_vad_callback(self, msg):
        """
        Callback for the video recording control topic.
        Updates self.video_record based on the received boolean.
        """
        self.vad_status = msg.data
        rospy.loginfo(f"VAD status updated: {self.vad_status}")


    def image_callback(self, msg):
        """
        Callback for the image topic.
        Converts the ROS Image message to an OpenCV image.
        """
        try:
            # Convert the ROS Image message to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # rospy.loginfo("Received Image!")
            # If video recording is active, call the video_stream method
            if self.video_record:
                self.video_frames.put(cv_image)
        except CvBridgeError as e:
            rospy.logerr("Error converting image: %s", str(e))

    def audio_callback(self, msg: Int16MultiArray):
        # rospy.loginfo("Received Audio Data (Int16MultiArray)")
        
        try:
            audio_data = np.array(msg.data, dtype=np.int16)
            self.audio_buffer.put(audio_data)
            speech_detected = self.detect_speech()

            self.publish_speech_detected(speech_detected)
            
            if self.audio_record:
               self.audio_frames.put(audio_data)
               rospy.loginfo(f"Audio frame received, total frames: {self.audio_frames.qsize()}")
        except Exception as e:
            rospy.logerr(f"Failed to convert audio data: {e}")


    def clear_audio_callback(self,msg):
        self.audio_frames = Queue()

    def clear_video_callback(self,msg):
        self.video_frames = Queue()

    def publish_speech_detected(self, status):
        """
        Publish the speech detection status to the /speech_detected topic.
        :param status: Bool - True if speech was detected, False otherwise
        """
        msg = Bool()
        msg.data = status
        self.speech_detected_pub.publish(msg)
        # rospy.loginfo(f"Published speech detection status: {status}")

    def save_video(self, msg):
        path = msg.data
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(path, fourcc, self.framerate, list(self.video_frames.queue)[0].shape[:2][::-1])
        if not out.isOpened():
            print("Failed to open video writer")
            return
        for frame in list(self.video_frames.queue):
            out.write(frame)
        out.release()


    def save_audio(self, msg):
        # Concatenate the audio frames into one array
        try: 
            path = msg.data
            audio = np.concatenate(list(self.audio_frames.queue))
            # Save the audio as a WAV file
            wavfile.write(path, self.sample_rate, audio)
            rospy.loginfo(f"Audio saved to {path}")
        except Exception as e: 
            rospy.logerr(f"Audio saving failed due to {e}")
            
    def detect_speech(self):
        if not self.vad_status: return False
        frame_duration = 10  # Frame duration in milliseconds
        frame_size = int(self.sample_rate * frame_duration / 1000)  # Frame size in samples
        frames = list(self.audio_buffer.queue)
        audio = np.concatenate(frames)
        frames = [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]
        for frame in frames:
            if len(frame) == frame_size:
                frame_as_bytes = frame.tobytes()
                if self.vad.is_speech(frame_as_bytes, self.sample_rate):
                    self.audio_buffer = Queue()
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
    node = MediaNode()
    node.run()
