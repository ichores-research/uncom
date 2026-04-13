#!/usr/bin/env python3

from pathlib import Path
import numpy as np
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
from sensor_msgs.msg import CameraInfo, Image, JointState
from ast import literal_eval
import moveit_commander
import tf2_ros

understood = []


def on_message(client, userdata, message):
    global understood
    try:
        understood = literal_eval(literal_eval(message.payload.decode("utf-8")))
    except Exception as e: 
        rospy.logerr(f"Failed to understand due to: {e}")
        understood = []


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
        self.mqtt_client = mqtt_client
        # Publishers for topics
        
        self.video_record_pub = rospy.Publisher('/video_recording', Bool, queue_size=10)
        self.audio_record_pub = rospy.Publisher('/audio_recording', Bool, queue_size=10)
        self.vad_status_pub = rospy.Publisher('/perform_vad', Bool, queue_size=10)

        self.save_video_pub = rospy.Publisher('/save_video', String, queue_size=10)
        self.save_audio_pub = rospy.Publisher('/save_audio', String, queue_size=10)
    
        self.clear_video_pub = rospy.Publisher('/clear_video', Empty, queue_size=10)
        self.clear_audio_pub = rospy.Publisher('/clear_audio', Empty, queue_size=10)

        self.set_obst_detect_mode = rospy.Publisher('/bridge/set_obstacle_detection_mode', Bool, queue_size=10)
        self.set_arm_pub = rospy.Publisher('/bridge/select_arm', String, queue_size=10)

        self.speech_detected = False
        self.speech_detected_sub = rospy.Subscriber('/speech_detected', Bool, self.speech_detected_callback)

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

        self.tts_request_pub = rospy.Publisher('/bridge/tts', String, queue_size=10) if self.robot_model!='krakow' else None
        self.arm_joint_request = rospy.Publisher('/set_joints', String, queue_size=10)
        self.move_base_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)

        self.joint_states = []

        # Subscribe to registered_depth topic # default = '/xtion/depth_registered/image_raw'
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback)
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)
        
        self.silence_time = 0
        self.max_silence = 500  
        self.wait_confirmation = False

        self.goal_pose = Pose()
        self.depth_frame = None
        self.saved_depth_frame = self.depth_frame

        self.arm_plan_tf = 'base_footprint' if self.robot_model=='krakow' else 'base_link'
        self.depth_camera_tf = 'xtion_rgb_optical_frame' 
        self.shoulder_tf = 'arm_1_link' if self.robot_model=='krakow' else 'arm_right_1_link'

        #for controlling Krakow's TIAGo
        self.robot = None
        self.scene = None
        self.group_arm_torso = None
        self.group_gripper = None
        
        #for controlling Prague's TIAGo
        self.arm_controller = None
        self.gripper_controller = None

        if self.robot_model == 'krakow':
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_commander.PlanningSceneInterface()
            self.group_arm_torso = moveit_commander.MoveGroupCommander("arm_torso") 

            self.group_gripper = moveit_commander.MoveGroupCommander("gripper")

            self.group_arm_torso.set_planner_id("SBLkConfigDefault")
            self.group_arm_torso.set_pose_reference_frame(self.arm_plan_tf)

        else:
            self.arm_controller = rospy.Publisher('/target_pose', Pose, queue_size=10)
            self.gripper_controller = rospy.Publisher('/target_gripper', String, queue_size=10)

       ## listen to TIAGo's tf tree
        self.tf_listener = tf.TransformListener()

        self.tf_listener.waitForTransform(self.arm_plan_tf, self.depth_camera_tf, rospy.Time(), rospy.Duration(5.0))
        self.tf_listener.waitForTransform(self.arm_plan_tf, self.shoulder_tf, rospy.Time(), rospy.Duration(5.0))

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

        timer = rospy.Timer(rospy.Duration(0.1), self.tf_callback)
        timer 

    def tf_callback(self, event):
        for transf in [self.object_tf, self.object_pointing_tf, self.target_tf, self.target_pointing_tf]:
            if transf is not None:
                transf.header.stamp = rospy.Time.now()
                self.transform_broadcaster.sendTransform(transf)

    def joint_states_callback(self, msg):
        self.joint_states = msg.position

    def speech_detected_callback(self, msg):
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
                        timeout = 100
                        self.publish_vad_status(False)
                        
                        while not understood: 
                            # print("Thinking", (time()-t0),"%")
                            if time()-t0>timeout:
                                rospy.loginfo("Timeout, thinking took too long!")
                                understood = ["ambiguous"]
                                break 

                        if understood[0] == "OK":
                            self.publish_vad_status(False)    
                            self.publish_clear_video()
                            self.publish_clear_audio() 
                            self.point_and_ask(understood[1], understood[2], understood[3], understood[4], understood[5],)
                            rospy.sleep(2.0)
                            self.publish_vad_status(True)
                            self.wait_confirmation = True

                        elif understood[0] == "ambiguous":
                            self.publish_vad_status(False)
                            self.publish_clear_video()
                            self.publish_clear_audio()
                            self.publish_video_recording(False)
                            self.publish_audio_recording(False)
                            self.silence_time = 0 
                            self.speech_detected = False
                            self.request_repeat()
                            rospy.sleep(6.0)
                            self.publish_vad_status(True)

                        elif understood[0] == "incomplete": # Command is still incomplete, immediately resume recording video and audio
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
                            rospy.logerr("Error, unnexpected result for the understanding operation.")
                    else:
                        understood = []
                        self.mqtt_agree_request()
                        while not understood:
                            pass
                        self.publish_clear_video()
                        self.publish_clear_audio()
                        self.publish_video_recording(False)
                        self.publish_audio_recording(False)
                        self.silence_time = 0 
                        self.speech_detected = False
                        self.publish_vad_status(False)
                        self.wait_confirmation = False
                        
                        if bool(understood[0]):
                            self.publish_vad_status(False)
                            self.tiago_talk("OK, I will start!")
                            # self.execute_pick_place() # TODO: FINISH CORRECTING PICK & PLACE
                            rospy.sleep(3.0)
                            self.publish_vad_status(True)
                            
                        else:
                            self.publish_vad_status(False)
                            self.request_repeat()
                            rospy.sleep(6.0)
                            rospy.loginfo ("CANCEL TASK")
                        self.publish_vad_status(True)

    def go_home(self):
        self.joint_command({"torso_lift_joint": 0.1499477988896708, 
                            "arm_left_1_joint":-1.0999970646935833, 
                            "arm_left_2_joint":1.4678949728052277,
                            "arm_left_3_joint":2.7139684702563316, 
                            "arm_left_4_joint":1.7095026751384435, 
                            "arm_left_5_joint":-1.5709468137969365,
                            "arm_left_6_joint":1.3898194804244532, 
                            "arm_left_7_joint":0.00013088278376799378}) 
        
        self.joint_command({"arm_right_1_joint":-1.0999153649440856,
                            "arm_right_2_joint":1.468119695531148, 
                            "arm_right_3_joint":2.713946763172928, 
                            "arm_right_4_joint":1.7095291050241606,
                            "arm_right_5_joint":-1.5706577118487064, 
                            "arm_right_6_joint":1.3897482611250676, 
                            "arm_right_7_joint":-0.00016294827082547966}) 

    def go_top_grasp(self):
        self.joint_command({"torso_lift_joint": 0.3,
                            "arm_left_1_joint":-1.0999970646935833, 
                            "arm_left_2_joint":1.4678949728052277,
                            "arm_left_3_joint":2.7139684702563316, 
                            "arm_left_4_joint":1.7095026751384435, 
                            "arm_left_5_joint":-1.5709468137969365,
                            "arm_left_6_joint":1.3898194804244532, 
                            "arm_left_7_joint":0.00013088278376799378}) 
        
        self.joint_command({"torso_lift_joint":0.3,
                            "arm_right_1_joint":0.11809687638497293, 
                            "arm_right_2_joint":-0.9500183338909938, 
                            "arm_right_3_joint":1.3132196466348618, 
                            "arm_right_4_joint":1.4600851804356563, 
                            "arm_right_5_joint":-2.071999018919099, 
                            "arm_right_6_joint":0.042998151819657195, 
                            "arm_right_7_joint":0.38959791372275704}) 


    def joint_command(self, command):
        self.arm_joint_request.publish(String(data=str(command)))

    def depth_callback(self, msg):
        """
        Callback for the registered depth point cloud topic.
        Just stores the lates depth frame. 
        """
        try:
            self.depth_frame = msg
        except CvBridgeError as e:
            rospy.logerr("Error reading the depth frame: %s", str(e))
            self.depth_frame = None
            
    def change_parent(self, input_tf, new_parent):
        try:
            # Get the transform from the new parent to the child frame
            (trans, rot) = self.tf_listener.lookupTransform(new_parent, input_tf.child_frame_id, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Failed to lookup transform from new parent to child.")
            return None

        # Create a new TransformStamped with the new parent as the frame_id
        new_tf = TransformStamped()
        new_tf.header.stamp = rospy.Time.now()
        new_tf.header.frame_id = new_parent
        new_tf.child_frame_id = input_tf.child_frame_id
        new_tf.transform.translation.x = trans[0]
        new_tf.transform.translation.y = trans[1]
        new_tf.transform.translation.z = trans[2]
        new_tf.transform.rotation.x = rot[0]
        new_tf.transform.rotation.y = rot[1]
        new_tf.transform.rotation.z = rot[2]
        new_tf.transform.rotation.w = rot[3]
        
        return new_tf

    def set_object_tf(self, center, input_tf):
        x, y = center
        # Get the depth value at the pixel

        depth_image = np.frombuffer(self.saved_depth_frame.data, dtype=np.float32).reshape(self.saved_depth_frame.height, self.saved_depth_frame.width)

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
            object_to_base_tf, object_to_base_rot = self.tf_listener.lookupTransform(self.arm_plan_tf, input_tf.child_frame_id, rospy.Time(0)) 
            shoulder_to_base_tf, shoulder_to_base_rot = self.tf_listener.lookupTransform(self.arm_plan_tf, self.shoulder_tf, rospy.Time(0))

            # Calculate the point 3/4 of the way between the shoulder and the input transform
            # pointing_tf.transform.translation.x = shoulder_to_base_tf[0] + .75 * (object_to_base_tf[0]+input_tf.transform.translation.z   - shoulder_to_base_tf[0])
            # pointing_tf.transform.translation.y = shoulder_to_base_tf[1] + .75 * (object_to_base_tf[1]-input_tf.transform.translation.x  - shoulder_to_base_tf[1])
            # pointing_tf.transform.translation.z = shoulder_to_base_tf[2] + .75 * (object_to_base_tf[2]-input_tf.transform.translation.y  - shoulder_to_base_tf[2])

            pointing_tf.transform.translation.x = shoulder_to_base_tf[0] + .5 * (object_to_base_tf[0]  - shoulder_to_base_tf[0])
            pointing_tf.transform.translation.y = shoulder_to_base_tf[1] + .95 * (object_to_base_tf[1]  - shoulder_to_base_tf[1])
            pointing_tf.transform.translation.z = shoulder_to_base_tf[2] + .5 * (object_to_base_tf[2]  - shoulder_to_base_tf[2])

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Error in lookupTransform: {e}")

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
        msg = Bool()
        msg.data = status
        self.vad_status_pub.publish(msg)
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
        if self.robot_model == 'krakow':
            self.group_arm_torso.set_pose_target(goal)
            self.group_arm_torso.set_planning_time(50.0)
            self.group_arm_torso.set_start_state_to_current_state()
            self.group_arm_torso.set_max_velocity_scaling_factor(1.0)
            plan = self.group_arm_torso.plan()
            if not plan:
                rospy.logerr("No plan found")
                return
            self.group_arm_torso.go(wait=True)
        else: 
            self.arm_controller.publish(goal)

    def set_gripper_joint_state(self, left_gripper, right_gripper, velocity=1.0):
        if self.robot_model == 'krakow':
            self.group_gripper.set_start_state_to_current_state()
            self.group_gripper.set_joint_value_target("gripper_left_finger_joint", left_gripper)
            self.group_gripper.set_joint_value_target("gripper_right_finger_joint", right_gripper)
            self.group_gripper.set_planning_time(1.0)        
            self.group_gripper.set_max_velocity_scaling_factor(velocity)
            plan = self.group_gripper.plan()
            if not plan:
                rospy.logerr("No plan found")
                return
            self.group_gripper.go(wait=True)
            
        else:
            self.gripper_controller.publish(String(data=str(left_gripper)+str(right_gripper)))

    def open_gripper(self, velocity=1.0):
        if self.robot_model == 'krakow':
            self.set_gripper_joint_state(left_gripper=0.04, right_gripper=0.04, velocity=velocity)

        else: 
            self.set_gripper_joint_state(left_gripper=0, right_gripper=0)

    def close_gripper(self, velocity=1.0):
        if self.robot_model == 'krakow':
            self.set_gripper_joint_state(left_gripper=0.00125, right_gripper=0.00125, velocity=velocity)
        else:
            self.set_gripper_joint_state(left_gripper=1, right_gripper=1)

    def point_and_ask(self, object, action, target , object_1_center, object_2_center):  # TODO: To be implemented, robot needs to repeat what it understood while pointing at objects.
        self.saved_depth_frame = self.depth_frame
        self.set_arm_pub.publish(String(data="right"))
        
        obj_valid = self.set_object_tf(object_1_center, self.object_tf)
        rospy.sleep(0.5)
        tgt_valid = self.set_object_tf(object_2_center, self.target_tf)
        rospy.sleep(0.5)

        if obj_valid:
            self.object_tf = self.change_parent(self.object_tf, "map")
            if self.object_tf is not None:
                self.set_pointing_tf(self.object_tf, self.object_pointing_tf)
        rospy.sleep(0.75)

        if tgt_valid:
            self.target_tf = self.change_parent(self.target_tf, "map")
            if self.target_tf is not None:
                self.set_pointing_tf(self.target_tf, self.target_pointing_tf)
        rospy.sleep(0.5)

        if not self.simulation:
            self.tiago_talk(f"Would you like me to {action}")

        rospy.sleep(2.0)

        self.set_obst_detect_mode.publish(Bool(data=True))

        if not obj_valid:
            rospy.logwarn("Invalid depth for object, skipping point.")
        else:
            self.move_arm(self.tf_to_pose(self.object_pointing_tf))
        if self.robot_model!="krakow":
            rospy.sleep(5.0)
        
        if not self.simulation:
            if object in ["this", "that"]:
                object = ""
            self.tiago_talk(f" this {object}")
        rospy.sleep(.5)
        self.set_arm_pub.publish(String(data="left"))
        rospy.sleep(.5)

        if not tgt_valid:
            rospy.logwarn("Invalid depth for target, skipping point.")
        else:
            self.move_arm(self.tf_to_pose(self.target_pointing_tf))

        if self.robot_model!="krakow":
            rospy.sleep(5.0)

        if target in ["this", "that"]:
            target = ''
        if not self.simulation:
            self.tiago_talk(f"at this {target}?")
        sleep(2)
        
        self.saved_depth_frame = None
        #TODO 1: IMPLEMENT WAITING FOR HUMAN CONFIRMATION 
        
        #TODO 2: IMPLEMENT ACTUAL TASK EXECUTION 

    def execute_pick_place(self):
        #move to top grap pose
        rospy.loginfo("Moving to top grasp.")
        self.go_top_grasp()
        rospy.sleep(10)
        rospy.loginfo("Approaching table")
        self.move_base(0.15, 0.0, 2.0)

        self.set_arm_pub.publish(String(data="right"))

        self.set_obst_detect_mode.publish(Bool(data=False))

        # Prepare for pre-grasp
        rospy.loginfo("Move to pre-grasp ")
        # self.open_gripper(1.0)
        self.object_tf.transform.translation.z += 0.35
        
        self.target_tf.transform.translation.z += 0.35
        rospy.sleep(0.5)

        self.object_tf = self.change_parent(self.object_tf, self.arm_plan_tf)

        self.object_tf.transform.rotation.x=0
        self.object_tf.transform.rotation.y=0
        self.object_tf.transform.rotation.z=0
        self.object_tf.transform.rotation.w=1

        rospy.logerr(str([self.object_tf.transform.translation.x,
                     self.object_tf.transform.translation.y,
                     self.object_tf.transform.translation.z]))

        self.target_tf = self.change_parent(self.target_tf, self.arm_plan_tf)

        self.object_tf.transform.translation.x += 0.02
        self.object_tf.transform.translation.y -= 0.01
        self.target_tf.transform.rotation.x=0
        self.target_tf.transform.rotation.y=0
        self.target_tf.transform.rotation.z=0
        self.target_tf.transform.rotation.w=1
        

        rospy.sleep(.5)
        
        pick_pose = self.tf_to_pose(self.object_tf)
        self.move_arm(pick_pose)
        if  self.robot_model!='krakow':
            rospy.sleep(7.0)

        #adjust to gripper for pick
        rospy.loginfo("Adjusting gripper pose")
        
        self.joint_command({"torso_lift_joint":self.joint_states[20]+0.1,
                            "arm_right_1_joint":self.joint_states[7], 
                            "arm_right_2_joint":self.joint_states[8],
                            "arm_right_3_joint":self.joint_states[9],
                            "arm_right_4_joint":self.joint_states[10], 
                            "arm_right_5_joint": -2.0,
                            "arm_right_6_joint": 1.38, 
                            "arm_right_7_joint": 0.0})

        if  self.robot_model!='krakow':
            rospy.sleep(5.0)

        rospy.loginfo("Lowering torso")
        #lower
        
        self.joint_command({"torso_lift_joint": self.joint_states[20]-0.05,
                            "arm_right_1_joint":self.joint_states[7], 
                            "arm_right_2_joint":self.joint_states[8],
                            "arm_right_3_joint":self.joint_states[9],
                            "arm_right_4_joint":self.joint_states[10], 
                            "arm_right_5_joint": -2.0,
                            "arm_right_6_joint": 1.38, 
                            "arm_right_7_joint": 0.0})

        if  self.robot_model!='krakow':
            rospy.sleep(5.0)

        rospy.loginfo("Closing gripper")
        #close gripper
        self.close_gripper(1.0)
        rospy.sleep(3.0)

        self.joint_command({"torso_lift_joint":self.joint_states[20]+0.15,
                            "arm_right_1_joint":self.joint_states[7], 
                            "arm_right_2_joint":self.joint_states[8],
                            "arm_right_3_joint":self.joint_states[9],
                            "arm_right_4_joint":self.joint_states[10], 
                            "arm_right_5_joint": self.joint_states[11],
                            "arm_right_6_joint": self.joint_states[12], 
                            "arm_right_7_joint": self.joint_states[13]})

        if  self.robot_model!='krakow':
            rospy.sleep(5.0)


        # move to drop position
        rospy.loginfo("Moving to target location")
        drop_pose = self.tf_to_pose(self.target_tf)
        self.move_arm(drop_pose)

        if  self.robot_model!='krakow':
            rospy.sleep(7.0)
        
        # adjest gripper if needed
        rospy.loginfo("Adjusting gripper again")
        
        self.joint_command({"torso_lift_joint":self.joint_states[20],
                            "arm_right_1_joint":self.joint_states[7], 
                            "arm_right_2_joint":self.joint_states[8],
                            "arm_right_3_joint":self.joint_states[9],
                            "arm_right_4_joint":self.joint_states[10], 
                            "arm_right_5_joint": -2.0,
                            "arm_right_6_joint": 1.38, 
                            "arm_right_7_joint": 1.8})
        #open gripper
        if  self.robot_model!='krakow':
            rospy.sleep(7.0)

        rospy.loginfo("Opening gripper")
        self.open_gripper(1.0)
        rospy.sleep(3.0)
        if  self.robot_model!='krakow':
            rospy.sleep(7.0)
        self.set_obst_detect_mode.publish(Bool(data=True))
        
    def run(self):
        # Run the ROS node
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

