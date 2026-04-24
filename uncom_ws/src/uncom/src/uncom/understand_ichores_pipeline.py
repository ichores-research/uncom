#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import cv2
import rospy
from std_msgs.msg import Bool, String, Empty
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from actionlib import SimpleActionClient
import json
import paho.mqtt.client as mqtt
from threading import Thread
from time import time, sleep
import tf 
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Twist
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
from ast import literal_eval
import tf2_ros
import tf2_geometry_msgs
from object_detector_msgs.msg import Detections
from uncom.ycb_objects import get_ycb_objects_info

from uncom.object_detection import (detect_objects, 
                                    get_object_pose)

from uncom.pick_and_place import (prepare_robot,
                            reset_planning_scene,
                            pick_object,
                            open_gripper,
                            close_gripper,
                            place_object,
                            move_to_pose)
understood = []


def _collapse_tasks(tasks):
    """Group sequential tasks by shared destination into confirmation batches.

    Tasks with the same (action, target_text, target_center) are collapsed
    into one group — the robot will enumerate each object then confirm once.
    Tasks with different destinations produce separate groups.

    Each group is a dict:
        {
            "objects":    [(obj_text, obj_center), ...],
            "action":     str,
            "tgt_text":   str,
            "tgt_center": list | None,
            "tasks":      [task, ...]   # original task tuples for execution
        }
    """
    groups = []
    for task in tasks:
        if task[0] != "OK":
            continue
        obj_text  = task[1] if len(task) > 1 else ""
        action    = task[2] if len(task) > 2 else ""
        tgt_text  = task[3] if len(task) > 3 else ""
        obj_center = task[4] if len(task) > 4 else None
        tgt_center = task[5] if len(task) > 5 else None

        # Try to merge into the last group if destination matches.
        # Keyed on (action, tgt_text) only — tgt_center is float-derived
        # and may differ slightly between detections of the same object.
        if (groups
                and groups[-1]["action"] == action
                and groups[-1]["tgt_text"] == tgt_text):
            groups[-1]["objects"].append((obj_text, obj_center))
            groups[-1]["tasks"].append(task)
        else:
            groups.append({
                "objects":    [(obj_text, obj_center)],
                "action":     action,
                "tgt_text":   tgt_text,
                "tgt_center": tgt_center,
                "tasks":      [task],
            })
    return groups


def on_message(client, userdata, message):
    global understood
    print(f"Received message {message}")

    try:
        parsed = literal_eval(literal_eval(message.payload.decode("utf-8")))
        # understand() returns a list of result tuples: [["OK",...], ...]
        # check_agree() returns a flat list: ["1"] or ["0"]
        if parsed and isinstance(parsed[0], (list, tuple)):
            understood = parsed       # multi-task
        else:
            understood = [parsed]     # single result or check_agree
    except Exception as e:
        rospy.logerr(f"Failed to understand due to: {e}")
        understood = []

    print(f"I understood {understood}")


def loop_client(client):
    client.loop_forever()


client = mqtt.Client()
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("inference/response")


class UnderstandingNode:

    def __init__(self, output_dir=None, device = 'auto', mqtt_client = client):
        # Initialize the ROS Node
        rospy.init_node('understanding_node')

        self.finished_startup = False
        self.mqtt_client = mqtt_client


        # Obtain rosparams and assign them to node variables

        self.output_dir = Path(rospy.get_param("output_directory"))
        self.video_filename = rospy.get_param("temporary_video_filename")
        self.audio_filename = rospy.get_param("temporary_audio_filename")
        self.device = rospy.get_param("inference_device")
        self.simulation = rospy.get_param("simulation")
        self.img_heigth = rospy.get_param("img_height")
        self.img_width = rospy.get_param("img_width")
        self.robot_model = rospy.get_param("robot_model")
        depth_topic = rospy.get_param("depth_topic")

        self.objects_info = get_ycb_objects_info("ycb_ichores")

        
        self.silence_time = 0
        self.max_silence = 500  
        self.wait_confirmation = False

        self.goal_pose = Pose()
        self.depth_frame = None
        self.saved_depth_frame = self.depth_frame
        self.roi_frame = None
        self.yolo_img = None
        self.yolo_detections = None

        self.arm_plan_tf = 'base_footprint' if self.robot_model=='krakow' else 'base_link'
        self.depth_camera_tf = 'xtion_rgb_optical_frame' 
        self.shoulder_tf = 'arm_1_link' if self.robot_model=='krakow' else 'arm_right_1_link'

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cv_bridge = CvBridge()

        
        arm_2_cam_tf_available = False
        while not arm_2_cam_tf_available:
            try:
                # This blocks until the transform is available or the timeout is reached
                self.tf_buffer.can_transform(self.arm_plan_tf, self.depth_camera_tf, rospy.Time(), rospy.Duration(1.0))
                arm_2_cam_tf_available = True

            except tf2_ros.TransformException as ex:
                rospy.logwarn(f"Could not transform: {ex}")

        arm_2_shoulder_tf_available = False
        while not arm_2_shoulder_tf_available:
            try:
                # This blocks until the transform is available or the timeout is reached
                self.tf_buffer.can_transform(self.arm_plan_tf, self.shoulder_tf, rospy.Time(), rospy.Duration(1.0))
                arm_2_shoulder_tf_available = True

            except tf2_ros.TransformException as ex:
                rospy.logwarn(f"Could not transform: {ex}")        

        self.transform_broadcaster = tf2_ros.TransformBroadcaster()

        self.object_tf = TransformStamped()
        self.object_tf.header.frame_id = self.depth_camera_tf
        self.object_tf.child_frame_id = "object"
        self.object_tf.transform.rotation.w = 1

        self.object_pointing_tf = TransformStamped()
        self.object_pointing_tf.header.frame_id = self.arm_plan_tf
        self.object_pointing_tf.child_frame_id = "pointing_object"
        self.object_pointing_tf.transform.rotation.w = 1

        self.target_tf = TransformStamped()
        self.target_tf.header.frame_id  = self.depth_camera_tf
        self.target_tf.child_frame_id = "target"
        self.target_tf.transform.rotation.w = 1

        self.target_pointing_tf = TransformStamped()
        self.target_pointing_tf.header.frame_id = self.arm_plan_tf
        self.target_pointing_tf.child_frame_id = "pointing_target"
        self.target_pointing_tf.transform.rotation.w = 1

        self.camera_info = rospy.wait_for_message('/xtion/depth_registered/camera_info', CameraInfo)

        self.publishable_tfs = [self.object_tf, self.object_pointing_tf, self.target_tf, self.target_pointing_tf]
        self.stored_command = []
        self.pending_groups = []
        self.current_group_idx = 0   # list of result tuples, one per task


        # Subscribe to registered_depth topic # default = '/xtion/depth_registered/image_raw'
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback)
        self.roi_sub = rospy.Subscriber('/pose_estimator/image_with_roi', Image, self.roi_callback, queue_size=1, tcp_nodelay=True)
        self.yolo_sub = rospy.Subscriber('/yolov5/detections', Detections, self.yolo_callback, queue_size=1)
        
        # Publishers for topics
        
        self.video_record_pub = rospy.Publisher('/video_recording', Bool, queue_size=10)
        self.audio_record_pub = rospy.Publisher('/audio_recording', Bool, queue_size=10)
        self.vad_status_pub = rospy.Publisher('/perform_vad', Bool, queue_size=10)
        self.move_base_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)

        self.save_video_pub = rospy.Publisher('/save_video', String, queue_size=10)
        self.save_audio_pub = rospy.Publisher('/save_audio', String, queue_size=10)
    
        self.clear_video_pub = rospy.Publisher('/clear_video', Empty, queue_size=10)
        self.clear_audio_pub = rospy.Publisher('/clear_audio', Empty, queue_size=10)

        self.set_listening_status = rospy.Publisher('/listening_mode', Bool, queue_size=10)
        self.set_wrist_publishing_status = rospy.Publisher('/wrist_streaming_mode', Bool, queue_size=10)

        self.speech_detected = False
        self.speech_detected_sub = rospy.Subscriber('/speech_detected', Bool, self.speech_detected_callback)


        timer = rospy.Timer(rospy.Duration(0.01), self.tf_callback)
        timer

        # Move to prepare pose before anything starts so camera frame is consistent.
        # Block listening so robot movement noise doesn't trigger VAD.
        rospy.loginfo("Initial prepare_robot call...")
        self.publish_vad_status(False)
        prepare_robot()
        rospy.loginfo("Robot ready.")
        self.publish_vad_status(True)

        self.finished_startup = True

    def tf_callback(self, event):
        for transf in self.publishable_tfs:
            if transf is not None:
                t = transf.transform.translation
                if any(np.isnan(v) for v in [t.x, t.y, t.z]):
                    continue  # don't broadcast invalid transforms
                msg = TransformStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = transf.header.frame_id
                msg.child_frame_id = transf.child_frame_id
                msg.transform = transf.transform
                self.transform_broadcaster.sendTransform(msg)

    def speech_detected_callback(self, msg):
        if self.finished_startup:
            detection_status = msg.data
            if detection_status:
                self.speech_detected = True
                self.publish_video_recording(True)
                self.publish_audio_recording(True)
                self.silence_time = 0
            
            else:
                if self.speech_detected:
                    self.silence_time+=10
                    if self.silence_time >= self.max_silence:
                        self.speech_detected = False
                        self.silence_time = 0
                        self.publish_audio_recording(False)
                        self.publish_video_recording(False)
                        rospy.sleep(0.5)
                        self.publish_save_video(str(self.output_dir / self.video_filename))
                        self.publish_save_audio(str(self.output_dir / self.audio_filename))
                        global understood

                        if not self.wait_confirmation:
                            understood = []
                            rospy.loginfo("SENDING UNDERSTAND REQUEST!")
                            self.mqtt_understand_request()
                            rospy.loginfo("REQUEST SENT!")
                            t0 = time()
                            timeout = 1000
                            self.publish_vad_status(False)
                            
                            while not understood: 
                                # print("Thinking", (time()-t0),"%")
                                if time()-t0>timeout:
                                    rospy.loginfo("Timeout, thinking took too long!")
                                    understood = ["ambiguous"]
                                    break 

                            if understood[0][0] == "OK":
                                self.publish_vad_status(False)
                                self.publish_clear_video()
                                self.publish_clear_audio()
                                self.stored_command = understood

                                # Collapse tasks and execute directly.
                                groups = _collapse_tasks(self.stored_command)
                                rospy.loginfo(f"{len(self.stored_command)} task(s) collapsed into {len(groups)} group(s).")
                                self.pending_groups = groups
                                self.current_group_idx = 0
                                for i, group in enumerate(groups):
                                    self.current_group_idx = i
                                    rospy.loginfo(f"Executing group {i+1}/{len(groups)}")
                                    self._execute_current_group()
                                self.pending_groups = []
                                self.current_group_idx = 0
                                self.stored_command = []
                                rospy.sleep(2.0)
                                self.publish_vad_status(True)

                            elif understood[0][0] == "ambiguous":
                                self.publish_vad_status(False)
                                self.publish_clear_video()
                                self.publish_clear_audio()
                                self.publish_video_recording(False)
                                self.publish_audio_recording(False)
                                self.silence_time = 0 
                                self.speech_detected = False
                                self.request_repeat()
                                self.stored_command = []
                                rospy.sleep(6.0)
                                self.publish_vad_status(True)

                            elif understood[0][0] == "incomplete": # Command is still incomplete, immediately resume recording video and audio
                                self.publish_vad_status(False)
                                self.request_continue()
                                rospy.sleep(2.0)
                                self.publish_vad_status(True)
                                self.publish_video_recording(True)
                                self.publish_audio_recording(True)
                                self.silence_time = 0 
                                self.speech_detected = True

                            else:
                                self.publish_video_recording(False)
                                self.publish_audio_recording(False)
                                self.silence_time = 0 
                                self.speech_detected = False
                                rospy.logerr("Error, unexpected result for the understanding operation.")
                                self.stored_command = []
                                
                        else:
                            understood = []
                            self.mqtt_agree_request()
                            t0_agree = time()
                            while not understood:
                                if time() - t0_agree > 15:
                                    rospy.logwarn("Confirmation timeout (15s), treating as decline.")
                                    understood = [["0"]]
                                    break
                            self.publish_clear_video()
                            self.publish_clear_audio()
                            self.publish_video_recording(False)
                            self.publish_audio_recording(False)
                            self.silence_time = 0 
                            self.speech_detected = False
                            self.publish_vad_status(False)
                            self.wait_confirmation = False
                            # understood is [[val]] from check_agree wrapping
                            _agree_val = understood[0][0] if isinstance(understood[0], (list,tuple)) else understood[0]
                            consent = bool(int(_agree_val)) if _agree_val in ['0', '1'] else False
                            
                            if consent:
                                self.publish_vad_status(False)
                                self.tiago_talk("OK, I will start!")
                                self._execute_current_group()
                            else:
                                self.publish_vad_status(False)
                                self.pending_groups = []
                                self.current_group_idx = 0
                                self.stored_command = []
                                self.request_repeat()
                                rospy.sleep(6.0)
                                rospy.loginfo("CANCEL TASK")
                            self.publish_vad_status(True)

    def depth_callback(self, msg):
        """
        Callback for the registered depth point cloud topic.
        Just stores the lates depth frame. 
        """
        try:
            self.depth_frame = msg
            self.depth_frame = self.cv_bridge.imgmsg_to_cv2(self.depth_frame, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr("Error reading the depth frame: %s", str(e))
            self.depth_frame = None

    def roi_callback(self, msg):
        """Store the latest GDRNet image_with_roi frame."""
        try:
            self.yolo_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rospy.loginfo("roi_callback: stored new yolo_img.")
        except CvBridgeError as e:
            rospy.logerr("Error reading roi frame: %s", str(e))
            self.yolo_img = None

    def yolo_callback(self, msg):
        """Store the latest YOLO detections for bbox-based matching."""
        self.yolo_detections = msg.detections
            
    # def change_parent(self, input_tf, new_parent):
    #         try:
    #             t = self.tf_buffer.lookup_transform(new_parent, 
    #                                                 input_tf.child_frame_id, 
    #                                                 rospy.Time(0), 
    #                                                 rospy.Duration(1.0)) # Added a small wait for stability
            
    #         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
    #             rospy.logerr(f"Failed to lookup transform from {new_parent} to {input_tf.child_frame_id}: {e}")
    #             return None

    #         t.header.stamp = rospy.Time.now()            
    #         return t

    def change_parent(self, input_tf, new_parent):
        try:
            # DIAGNOSTIC
            now = rospy.Time.now()
            rospy.logwarn(f"=== DIAGNOSTIC change_parent ===")
            rospy.logwarn(f"child_frame_id: {input_tf.child_frame_id}")
            rospy.logwarn(f"input_tf.header.stamp: {input_tf.header.stamp.to_sec()}")
            rospy.logwarn(f"rospy.Time.now(): {now.to_sec()}")
            
            # What does the buffer actually have for these frames?
            try:
                latest_obj = self.tf_buffer.lookup_transform(
                    input_tf.header.frame_id,
                    input_tf.child_frame_id,
                    rospy.Time(0)
                )
                rospy.logwarn(f"Latest '{input_tf.child_frame_id}' stamp in buffer: {latest_obj.header.stamp.to_sec()}")
            except Exception as e:
                rospy.logwarn(f"Can't even find '{input_tf.child_frame_id}' in buffer at all: {e}")

            try:
                latest_base = self.tf_buffer.lookup_transform(
                    new_parent,
                    input_tf.header.frame_id,
                    rospy.Time(0)
                )
                rospy.logwarn(f"Latest '{new_parent}' stamp in buffer: {latest_base.header.stamp.to_sec()}")
            except Exception as e:
                rospy.logwarn(f"Can't find '{new_parent}' in buffer: {e}")

            # Use Time(0) for the child (our freshly published frame) but
            # also accept Time(0) for the parent — this gives the most recent
            # available transform and avoids stale-stamp extrapolation errors.
            output_tf = self.tf_buffer.lookup_transform(
                new_parent,
                input_tf.child_frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)   # wait up to 1s for transform to appear
            )
            output_tf.header.stamp = rospy.Time.now()
            return output_tf

        except Exception as e:
            rospy.logerr(f"Failed to re-parent transform: {e}")
            return None

    def set_object_tf(self, center, input_tf):
        x, y = center
        # Get the depth value at the pixel

        depth_image = self.saved_depth_frame.astype(np.float32)

        depth = depth_image[y, x]
        if np.isnan(depth) or depth <= 0:
            rospy.logwarn(f"Invalid depth at ({x}, {y}): {depth}")
            return False

        # Camera intrinsic parameters
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        # Convert pixel coordinates to 3D coordinates
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        input_tf.transform.translation.x = X
        input_tf.transform.translation.y = Y
        input_tf.transform.translation.z = Z
        input_tf.header.stamp = rospy.Time.now()  # stamp must be set for TF buffer lookup
        return True

    def tf_to_pose(self, transform):
        pose = Pose()
        pose.position.x = transform.transform.translation.x
        pose.position.y = transform.transform.translation.y
        pose.position.z = transform.transform.translation.z
        pose.orientation.x = transform.transform.rotation.x
        pose.orientation.y = transform.transform.rotation.y
        pose.orientation.z = transform.transform.rotation.z
        pose.orientation.w = transform.transform.rotation.w
        return pose

    def set_pointing_tf(self, input_tf, pointing_tf):
        try:
            object_to_base = self.tf_buffer.lookup_transform(
                self.arm_plan_tf,         
                input_tf.child_frame_id,  
                rospy.Time(0), 
                rospy.Duration(1.0)
            ) 
            
            shoulder = self.tf_buffer.lookup_transform(
                self.arm_plan_tf, 
                self.shoulder_tf, 
                rospy.Time(0), 
                rospy.Duration(1.0)
            )

            s = shoulder.transform.translation
            o = object_to_base.transform.translation

            pointing_tf.transform.translation.x = s.x + 0.5 * (o.x - s.x)
            pointing_tf.transform.translation.y = s.y + 0.95 * (o.y - s.y)
            pointing_tf.transform.translation.z = s.z + 0.5 * (o.z - s.z)

            rospy.logwarn(f"POINTING DEBUG: shoulder=({s.x:.3f},{s.y:.3f},{s.z:.3f}) "
                         f"object=({o.x:.3f},{o.y:.3f},{o.z:.3f}) "
                         f"goal=({pointing_tf.transform.translation.x:.3f},"
                         f"{pointing_tf.transform.translation.y:.3f},"
                         f"{pointing_tf.transform.translation.z:.3f})")

            pointing_tf.transform.rotation.x = 0
            pointing_tf.transform.rotation.y = -0.3827
            pointing_tf.transform.rotation.z = 0
            pointing_tf.transform.rotation.w = 0.9239

            pointing_tf.header.frame_id = self.arm_plan_tf
            pointing_tf.header.stamp = rospy.Time.now()

        except Exception as e:
            rospy.logerr(f"Pointing alignment failed: {e}")

    def publish_video_recording(self, status):
        """
        Publish the video recording status to the /video_recording topic.
        :param status: Bool - True if video recording is enabled, False otherwise
        """
        msg = Bool()
        msg.data = status
        self.video_record_pub.publish(msg)
        rospy.loginfo(f"Published video recording status: {status}")

    def publish_audio_recording(self, status):
        """
        Publish the audio recording status to the /audio_recording topic.
        :param status: Bool - True if audio recording is enabled, False otherwise
        """
        msg = Bool()
        msg.data = status
        self.audio_record_pub.publish(msg)
        rospy.loginfo(f"Published audio recording status: {status}")

    def publish_vad_status(self, status):
        """
        Publish the video recording status to the /video_recording topic.
        :param status: Bool - True if video recording is enabled, False otherwise
        """
        if status and not self.simulation:
            Thread(target=self.tiago_talk, args=("Listening",), daemon=True).start()
            rospy.sleep(2.0)
        msg = Bool()
        msg.data = status
        self.vad_status_pub.publish(msg)
        # self.set_listening_status.publish(msg)  #Only uncoment is you want the robot to actually stop sending audio packages!
        rospy.loginfo(f"Published VAD status: {status}")

    def publish_save_video(self, file_path):
        """
        Publish the file path to save the video to the /save_video_topic.
        :param file_path: String - Path to the video file to save
        """
        msg = String()
        msg.data = file_path
        self.save_video_pub.publish(msg)
        rospy.loginfo(f"Published video save path: {file_path}")

    def publish_save_audio(self, file_path):
        """
        Publish the file path to save the audio to the /save_audio_topic.
        :param file_path: String - Path to the audio file to save
        """
        msg = String()
        msg.data = file_path
        self.save_audio_pub.publish(msg)
        rospy.loginfo(f"Published audio save path: {file_path}")

    def publish_clear_audio(self):
        """
        Publish a request to clear currently buffered audio frames.
        """
        self.clear_audio_pub.publish(Empty())

    def publish_clear_video(self):
        """
        Publish a request to clear currently buffered video frames.
        """
        self.clear_video_pub.publish(Empty())

    def request_continue(self):
        self.tiago_talk("Please, go on.")

    def request_repeat(self):
        self.publish_clear_audio()
        self.publish_clear_video()
        self.publish_video_recording(False)
        self.publish_audio_recording(False)
        self.tiago_talk("Sorry, I am not able to understand your command, could you please repeat it again?")

    def tts_connection(self):
        tts_client = SimpleActionClient('/tts', TtsAction)
        tts_client.wait_for_server()
        return tts_client

    def tiago_talk(self, speech):
        if self.robot_model=='krakow':
            try:
                rospy.loginfo("Requesting speech action")
                tts_client = self.tts_connection()
                goal = TtsGoal()
                goal.rawtext.text = speech
                goal.rawtext.lang_id = 'en_GB'
                tts_client.send_goal_and_wait(goal)
                rospy.loginfo("Finished")
            except rospy.ROSInterruptException:
                rospy.logerr("Abruptly finished!")
        else:
            self.tts_request_pub.publish(String(data=str(speech)))

    def _ask_next_group(self):
        """Point-and-ask for the current pending group, then wait for confirmation."""
        if self.current_group_idx >= len(self.pending_groups):
            # All groups confirmed and executed
            self.pending_groups = []
            self.current_group_idx = 0
            self.stored_command = []
            rospy.sleep(2.0)
            self.publish_vad_status(True)
            return

        # Block listening so robot movement noise doesn't trigger VAD
        self.publish_vad_status(False)
        rospy.loginfo("Moving to prepare pose before pointing...")
        prepare_robot()

        group = self.pending_groups[self.current_group_idx]
        self.point_and_ask(
            objects=group["objects"],
            action=group["action"],
            target=group["tgt_text"],
            tgt_center=group["tgt_center"],
        )
        rospy.sleep(2.0)
        self.publish_vad_status(True)
        self.wait_confirmation = True

    def _execute_current_group(self):
        """Execute all tasks in the current pending group, then move to next."""
        group = self.pending_groups[self.current_group_idx]
        rospy.loginfo(f"Executing group {self.current_group_idx + 1}/{len(self.pending_groups)}: "
                      f"{len(group['tasks'])} task(s) -> '{group['tgt_text']}'")

        for task in group["tasks"]:
            if task[0] != "OK":
                continue
            obj_text  = task[1] if len(task) > 1 else ""
            tgt_text  = task[3] if len(task) > 3 else ""
            obj_center = task[4] if len(task) > 4 else None
            tgt_center = task[5] if len(task) > 5 else None

            if obj_text and tgt_text:
                self.execute_pick_place(obj_center, tgt_center)
                rospy.sleep(1.0)
                prepare_robot()
            elif obj_text and not tgt_text:
                self.execute_pick_only(obj_center)
            elif not obj_text and tgt_text:
                self.execute_place_only(tgt_center)
                rospy.sleep(1.0)
                prepare_robot()
            rospy.sleep(1.0)

        self.current_group_idx += 1

    def mqtt_understand_request(self):
        file_paths = [str(self.output_dir / self.audio_filename), str(self.output_dir / self.video_filename), 'understand']
        self.mqtt_client.publish("inference/request", json.dumps(file_paths))

    def mqtt_agree_request(self):
        file_paths = [str(self.output_dir / self.audio_filename), str(self.output_dir / self.video_filename), 'check_agree']
        self.mqtt_client.publish("inference/request", json.dumps(file_paths))

    def move_base(self, speed_lin, speed_rot, duration):
        t0 = time()
        movement_speed = Twist()
        movement_speed.linear.x = speed_lin
        movement_speed.angular.z = speed_rot
        while time()-t0 < duration:
            self.move_base_pub.publish(movement_speed)
        # Enforce robot stop.
        movement_speed.linear.x = 0.0
        movement_speed.angular.z = 0.0
        self.move_base_pub.publish(movement_speed)

    def move_arm(self, goal):
        move_to_pose(goal)

    def point_and_ask(self, objects, action, target, tgt_center):
        """Point at each object in sequence, then at the target, and ask for confirmation.

        Args:
            objects:    list of (obj_text, obj_center) — one entry per object to pick.
            action:     verb string e.g. "pick", "place".
            target:     destination noun string.
            tgt_center: 3D center of the destination.
        """
        self.saved_depth_frame = self.depth_frame
        if self.saved_depth_frame is None:
            rospy.logwarn("point_and_ask: no depth frame available, skipping pointing.")
            return

        # --- Point at each object and name it ----------------------------
        if not self.simulation:
            self.tiago_talk(f"Do you want me to {action}")
        rospy.sleep(1.0)

        for idx, (obj_text, obj_center) in enumerate(objects):
            if obj_center is not None:
                if not self.set_object_tf(obj_center, self.object_tf):
                    rospy.logwarn(f"Invalid depth for object {idx}, skipping point.")
                    continue
                obj_tf = self.change_parent(self.object_tf, "base_footprint")
                if obj_tf is None:
                    rospy.logwarn(f"change_parent failed for object {idx}, skipping point.")
                    continue
                self.set_pointing_tf(obj_tf, self.object_pointing_tf)
                rospy.sleep(0.5)
                self.move_arm(self.tf_to_pose(self.object_pointing_tf))
                if self.robot_model != "krakow":
                    rospy.sleep(4.0)

            label = obj_text if obj_text and obj_text not in ["this", "that", "it"] else ""
            if not self.simulation:
                if idx < len(objects) - 1:
                    self.tiago_talk(f"this {label},".strip().rstrip(",") + ",")
                else:
                    self.tiago_talk(f"this {label}".strip())
            rospy.sleep(0.5)

        # --- Point at target and ask -------------------------------------
        if tgt_center is not None:
            if not self.set_object_tf(tgt_center, self.target_tf):
                rospy.logwarn("Invalid depth for target, skipping point.")
            else:
                tgt_tf = self.change_parent(self.target_tf, "base_footprint")
                if tgt_tf is None:
                    rospy.logwarn("change_parent failed for target, skipping point.")
                else:
                    self.set_pointing_tf(tgt_tf, self.target_pointing_tf)
                    rospy.sleep(0.5)
                    self.move_arm(self.tf_to_pose(self.target_pointing_tf))
            if self.robot_model != "krakow":
                rospy.sleep(4.0)

        tgt_label = target if target and target not in ["this", "that", "here", "there"] else ""
        if not self.simulation:
            if tgt_center is not None and (target or tgt_label):
                if len(objects) > 1:
                    self.tiago_talk(f"and put them in this {tgt_label}?".strip().rstrip("?") + "?")
                else:
                    self.tiago_talk(f"and put it in this {tgt_label}?".strip().rstrip("?") + "?")
            else:
                self.tiago_talk("Should I proceed?")
        rospy.sleep(2.0)

    def check_inside_bbox(self, bbox, xpoint, ypoint):
        return (bbox.xmin <= xpoint <= bbox.xmax) and (bbox.ymin <= ypoint <= bbox.ymax)
         
    def match_dino_2_gdrnet(self, detections, object_center):
        if not detections:
            return None
        # YOLO bboxes have x/y swapped (xmin holds y, ymin holds x),
        # so swap the DINO center to match
        px, py = object_center[1], object_center[0]
        # First try: exact bbox containment
        for det in detections:
            if self.check_inside_bbox(det.bbox, px, py):
                return det
        # Fallback: expand each bbox by a margin and try again
        MARGIN = 30
        for det in detections:
            if (det.bbox.xmin - MARGIN <= px <= det.bbox.xmax + MARGIN and
                det.bbox.ymin - MARGIN <= py <= det.bbox.ymax + MARGIN):
                return det
        return None

    def execute_pick_only(self, object_center):
        """Pick an object and hold it. Uses GDRNet++ with depth fallback."""
        reset_planning_scene()
        preparation_success = prepare_robot()
        if not preparation_success:
            rospy.logerr("Robot failed to assume initial position, giving up.")
            return

        try:
            self.tf_buffer.lookup_transform("base_footprint", "xtion_depth_optical_frame",
                                            rospy.Time(0), rospy.Duration(4.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

        detections = detect_objects()
        to_pick_object = self.match_dino_2_gdrnet(detections, object_center)

        if to_pick_object is None:
            # Depth-based fallback
            rospy.set_param("/motion/closed_gripper_joint", 0.025)
            depth_image = self.saved_depth_frame.astype(np.float32)
            depth = depth_image[object_center[1], object_center[0]]
            fx, fy = self.camera_info.K[0], self.camera_info.K[4]
            cx, cy = self.camera_info.K[2], self.camera_info.K[5]
            pick_pose = Pose()
            pick_pose.position.x = (object_center[0] - cx) * depth / fx
            pick_pose.position.y = (object_center[1] - cy) * depth / fy
            pick_pose.position.z = depth
            open_gripper()
            rospy.sleep(1.0)
            move_to_pose(pick_pose)
            close_gripper()
        else:
            pick_object_info = self.objects_info.get(to_pick_object.name, None)
            if pick_object_info is None:
                rospy.logwarn(f"Object {to_pick_object.name} not in dataset, falling back to depth.")
                return
            pose_gdrnpp_pick = get_object_pose(to_pick_object.name)
            if pose_gdrnpp_pick is None:
                rospy.logwarn("Could not estimate object pose.")
                return
            pose_in_head = PoseStamped()
            pose_in_head.header.frame_id = "xtion_depth_optical_frame"
            pose_in_head.header.stamp = rospy.Time(0)
            pose_in_head.pose.position = pose_gdrnpp_pick.pose.position
            pose_in_head.pose.orientation = pose_gdrnpp_pick.pose.orientation
            try:
                transform = self.tf_buffer.lookup_transform(
                    "base_footprint", "xtion_depth_optical_frame", rospy.Time(0))
                pose_in_base = tf2_geometry_msgs.do_transform_pose(pose_in_head, transform)
            except Exception as e:
                rospy.logerr(f"Transform failed: {e}")
                return

            # Set gripper closure value based on object type
            obj_name = to_pick_object.name.lower()
            if "banana" in obj_name:
                rospy.set_param("/motion/closed_gripper_joint", 0.015)
            elif "mustard" in obj_name:
                rospy.set_param("/motion/closed_gripper_joint", 0.025)
            elif "apple" in obj_name:
                rospy.set_param("/motion/closed_gripper_joint", 0.03)
            elif "mug" in obj_name:
                rospy.set_param("/motion/closed_gripper_joint", 0.005)
            else:
                rospy.set_param("/motion/closed_gripper_joint", 0.025)

            for i in range(10):
                success = pick_object(index=i,
                                      mesh_path=pick_object_info["mesh_path"],
                                      grasps=pick_object_info["grasps"],
                                      pose=pose_in_base.pose)
                if success:
                    break

    def execute_place_only(self, target_center):
        """Place held object at target_center using depth-based method + placeholder mesh.
        NOTE: intentionally does NOT reset the planning scene — MoveIt needs
        the object to remain attached to the gripper from the prior pick."""
        if self.saved_depth_frame is None:
            self.saved_depth_frame = self.depth_frame
        if self.saved_depth_frame is None:
            rospy.logerr("No depth frame available for place — aborting.")
            return
        depth_image = self.saved_depth_frame.astype(np.float32)
        depth = depth_image[target_center[1], target_center[0]]
        fx, fy = self.camera_info.K[0], self.camera_info.K[4]
        cx, cy = self.camera_info.K[2], self.camera_info.K[5]

        place_stamped = PoseStamped()
        place_stamped.header.frame_id = "xtion_depth_optical_frame"
        place_stamped.header.stamp = rospy.Time(0)
        place_stamped.pose.position.x = (target_center[0] - cx) * depth / fx
        place_stamped.pose.position.y = (target_center[1] - cy) * depth / fy
        place_stamped.pose.position.z = depth
        place_stamped.pose.orientation.w = 1.0

        try:
            transform = self.tf_buffer.lookup_transform(
                "base_footprint", "xtion_depth_optical_frame",
                rospy.Time(0), rospy.Duration(2.0))
            place_pose = tf2_geometry_msgs.do_transform_pose(place_stamped, transform).pose
        except Exception as e:
            rospy.logerr("Failed to transform place pose to base_footprint: %s", e)
            return

        place_pose.position.z += 0.14
        place_object(place_pose,
                     mesh_path="/root/catkin_ws/src/uncom/data/obj_000010.ply")

    def execute_pick_place(self, object_1_center, object_2_center):
        reset_planning_scene()
        self.publish_vad_status(False)

        # ---- Capture depth + detections BEFORE any arm movement ----
        rospy.loginfo("Capturing depth frame and detections before arm movement...")
        self.depth_frame = None
        for attempt in range(1, 6):
            rospy.sleep(1.0)
            if self.depth_frame is not None:
                self.saved_depth_frame = self.depth_frame
                rospy.loginfo("Got fresh depth frame on attempt %d.", attempt)
                break
            rospy.logwarn("Depth frame not ready, attempt %d/5...", attempt)
        else:
            rospy.logerr("No depth frame received after 5 attempts — aborting pick/place.")
            return

        wait_success = True
        try:
            self.tf_buffer.lookup_transform("base_footprint", "xtion_depth_optical_frame",
                                            rospy.Time(0), rospy.Duration(4.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            wait_success = False

        print(f"wait success = {wait_success}")
        print("waiting done.")

        # Call YOLO detection and match by name
        rospy.loginfo("Calling detect_objects()...")
        gdrnet_dets = detect_objects()
        rospy.loginfo("detect_objects() returned %d detections.", len(gdrnet_dets) if gdrnet_dets else 0)

        yolo_pick = self.match_dino_2_gdrnet(gdrnet_dets, object_1_center)
        yolo_place = self.match_dino_2_gdrnet(gdrnet_dets, object_2_center)
        rospy.loginfo("matched_pick=%s matched_place=%s",
                      yolo_pick.name if yolo_pick else None,
                      yolo_place.name if yolo_place else None)

        def find_by_name(dets, name):
            if not dets or not name:
                return None
            for d in dets:
                if d.name == name:
                    return d
            return None

        to_pick_object = find_by_name(gdrnet_dets, yolo_pick.name if yolo_pick else None)
        place_destination = find_by_name(gdrnet_dets, yolo_place.name if yolo_place else None)
        rospy.loginfo("to_pick_object=%s place_destination=%s",
                      to_pick_object.name if to_pick_object else None,
                      place_destination.name if place_destination else None)

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        # ---- Pre-compute place pose while camera is unobstructed ----
        place_pose_base = None
        if place_destination is None:
            rospy.logwarn("No correspondence to GDRNet++ detections for place target, using depth-based method.")
            depth_image = self.saved_depth_frame.astype(np.float32)
            depth = depth_image[object_2_center[1], object_2_center[0]]

            place_stamped = PoseStamped()
            place_stamped.header.frame_id = "xtion_depth_optical_frame"
            place_stamped.header.stamp = rospy.Time(0)
            place_stamped.pose.position.x = (object_2_center[0] - cx) * depth / fx
            place_stamped.pose.position.y = (object_2_center[1] - cy) * depth / fy
            place_stamped.pose.position.z = depth
            place_stamped.pose.orientation.w = 1.0

            try:
                transform = self.tf_buffer.lookup_transform(
                    "base_footprint", "xtion_depth_optical_frame",
                    rospy.Time(0), rospy.Duration(2.0))
                place_pose_base = tf2_geometry_msgs.do_transform_pose(place_stamped, transform).pose
            except Exception as e:
                rospy.logerr("Failed to transform place pose to base_footprint: %s", e)
        else:
            pose_gdrnpp_place = get_object_pose(place_destination.name)
            if pose_gdrnpp_place is not None:
                place_in_head = PoseStamped()
                place_in_head.header.frame_id = "xtion_depth_optical_frame"
                place_in_head.header.stamp = rospy.Time(0)
                place_in_head.pose.position = pose_gdrnpp_place.pose.position
                place_in_head.pose.orientation = pose_gdrnpp_place.pose.orientation
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "base_footprint", "xtion_depth_optical_frame", rospy.Time(0))
                    place_pose_base = tf2_geometry_msgs.do_transform_pose(place_in_head, transform).pose
                except Exception as e:
                    rospy.logerr("Failed to transform place pose: %s", e)
            else:
                rospy.logwarn("Could not estimate pose for place destination '%s'.", place_destination.name)

        place_object_info = None
        if place_destination is not None:
            place_object_info = self.objects_info.get(place_destination.name, None)

        # ---- Now move the arm: prepare + pick ----
        preparation_success = prepare_robot()
        if not preparation_success:
            rospy.logerr("Robot failed to assume initial position, giving up.")
            return

        if to_pick_object is None:
            rospy.set_param("/motion/closed_gripper_joint", 0.025)
            depth_image = self.saved_depth_frame.astype(np.float32)
            depth = depth_image[object_1_center[1], object_1_center[0]]

            pick_pose_stamped = PoseStamped()
            pick_pose_stamped.header.frame_id = "xtion_depth_optical_frame"
            pick_pose_stamped.header.stamp = rospy.Time(0)
            pick_pose_stamped.pose.position.x = (object_1_center[0] - cx) * depth / fx
            pick_pose_stamped.pose.position.y = (object_1_center[1] - cy) * depth / fy
            pick_pose_stamped.pose.position.z = depth
            pick_pose_stamped.pose.orientation.w = 1.0

            try:
                transform = self.tf_buffer.lookup_transform(
                    "base_footprint", "xtion_depth_optical_frame",
                    rospy.Time(0), rospy.Duration(2.0))
                pick_pose_base = tf2_geometry_msgs.do_transform_pose(pick_pose_stamped, transform)
            except Exception as e:
                rospy.logerr("Failed to transform pick pose to base_footprint: %s", e)
                return

            open_gripper()
            rospy.sleep(1.0)
            move_to_pose(pick_pose_base.pose)
            close_gripper()

        else:
            pick_object_info = self.objects_info.get(to_pick_object.name, None)

            if pick_object_info is None:
                print(f"Object {to_pick_object.name} not found in dataset.")
                return

            pose_gdrnpp_pick = get_object_pose(to_pick_object.name)
            if pose_gdrnpp_pick is None:
                print("Could not estimate object pose.")
                return

            pose_in_head = PoseStamped()
            pose_in_head.header.frame_id = "xtion_depth_optical_frame"
            pose_in_head.header.stamp = rospy.Time(0)
            pose_in_head.pose.position = pose_gdrnpp_pick.pose.position
            pose_in_head.pose.orientation = pose_gdrnpp_pick.pose.orientation

            try:
                transform = self.tf_buffer.lookup_transform("base_footprint", "xtion_depth_optical_frame", rospy.Time(0))
                pose_in_base = tf2_geometry_msgs.do_transform_pose(pose_in_head, transform)
            except Exception as e:
                print(f"Transform of the pose to base footprint failed due to: {e}")
                return

            # Set gripper closure value based on object type
            obj_name = to_pick_object.name.lower()
            if "banana" in obj_name:
                rospy.set_param("/motion/closed_gripper_joint", 0.015)
            elif "mustard" in obj_name:
                rospy.set_param("/motion/closed_gripper_joint", 0.025)
            elif "apple" in obj_name:
                rospy.set_param("/motion/closed_gripper_joint", 0.03)
            elif "mug" in obj_name:
                rospy.set_param("/motion/closed_gripper_joint", 0.005)
            else:
                rospy.set_param("/motion/closed_gripper_joint", 0.025)

            for i in range(10):
                success = pick_object(index=i,
                                      mesh_path=pick_object_info["mesh_path"],
                                      grasps=pick_object_info["grasps"],
                                      pose=pose_in_base.pose)
                if success:
                    break

        # ---- Place using pre-computed pose ----
        if place_pose_base is not None:
            place_pose_base.position.z += 0.10
            mesh_path = place_object_info["mesh_path"] if place_object_info else "/root/catkin_ws/src/uncom/data/obj_000010.ply"
            place_object(place_pose_base, mesh_path=mesh_path)
        else:
            rospy.logerr("Place pose could not be pre-computed — skipping place.")


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    # Instantiate the UnderstandingNode class and run the node
    sleep(3.0)
    client_thread = Thread(target=loop_client,args=[client])
    client_thread.start()

    node = UnderstandingNode()
    # node.publish_vad_status(False)
    # node.go_home()
    # if node.robot_model != 'krakow':
    #     rospy.sleep(10.0)
    # node.publish_vad_status(True)
    node.run()