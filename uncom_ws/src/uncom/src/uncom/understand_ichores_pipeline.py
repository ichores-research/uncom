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
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from ast import literal_eval
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from uncom.ycb_objects import get_ycb_objects_info

from uncom.object_detection import (detect_objects, 
                                    get_object_pose)

from uncom.pick_and_place import (prepare_robot,
                            reset_planning_scene,
                            pick_object,
                            place_object,
                            object_pose_tf_publisher)
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


        self.speech_detected = False
        self.speech_detected_sub = rospy.Subscriber('/speech_detected', Bool, self.speech_detected_callback)

        self.objects_info = get_ycb_objects_info("ycb_ichores")

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


        # Subscribe to registered_depth topic # default = '/xtion/depth_registered/image_raw'
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback)
        
        self.silence_time = 0
        self.max_silence = 500  
        self.wait_confirmation = False

        self.goal_pose = Pose()
        self.depth_frame = None
        self.saved_depth_frame = self.depth_frame

        self.arm_plan_tf = 'base_footprint' if self.robot_model=='krakow' else 'base_link'
        self.depth_camera_tf = 'xtion_rgb_optical_frame' 
        self.shoulder_tf = 'arm_1_link' if self.robot_model=='krakow' else 'arm_right_1_link'

       ## listen to TIAGo's tf tree
        self.tf_listener = tf.TransformListener()
        self.cv_bridge = CvBridge()

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

        self.publishable_tfs = [self.object_tf, self.object_pointing_tf, self.target_tf, self.target_pointing_tf]

        

        timer = rospy.Timer(rospy.Duration(0.1), self.tf_callback)
        timer 

    def tf_callback(self, event):
        for transf in self.publishable_tfs:
            if transf is not None:
                transf.header.stamp = rospy.Time.now()
                self.transform_broadcaster.sendTransform(transf)

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

    def move_arm(self, goal): pass

    def point_and_ask(self, object, action, target , object_1_center, object_2_center):  # TODO: To be implemented, robot needs to repeat what it understood while pointing at objects.
        self.saved_depth_frame = self.depth_frame
        self.set_object_tf(object_1_center, self.object_tf)
        rospy.sleep(0.5)
        self.set_object_tf(object_2_center, self.target_tf)    
        rospy.sleep(0.5)
        self.object_tf = self.change_parent(self.object_tf, "map")
        self.target_tf = self.change_parent(self.target_tf, "map")
        
        self.set_pointing_tf(self.object_tf, self.object_pointing_tf)
        rospy.sleep(0.75)

        self.set_pointing_tf(self.target_tf, self.target_pointing_tf)
        rospy.sleep(0.5)

        if not self.simulation:
            self.tiago_talk(f"Would you like me to {action}")
        
        rospy.sleep(2.0)
        
        self.set_obst_detect_mode.publish(Bool(data=True))

        self.move_arm(self.tf_to_pose(self.object_pointing_tf))
        if self.robot_model!="krakow":
            rospy.sleep(5.0)
        
        if not self.simulation:
            if object in ["this", "that"]:
                object = ""
            self.tiago_talk(f" this {object}")
        rospy.sleep(.5)
        rospy.sleep(.5)

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
        """
        Test the pick and place functionality.
        Picks an apple from the table in front of the robot.
        1. Prepares the robot
        2. Detects objects on the table
        3. Picks the apple
        4. Reports success or failure
        5. Retries up to 10 times if picking fails
        6. Prints the result
        """
        print("Hello, starting pick up test")
        
        # First prepare the robot
        # move_2_prepare_pose = bool(input("Insert 1 if you want to move the arm to the prepare pose, 0 otherwise. "))
        # if move_2_prepare_pose:
        preparation_success = prepare_robot()
        
        # print(f"Preparation submit success {preparation_success}") # TODO: This currently prints "None" and claims the preparation was unsuccesful
        # if not preparation_success:
        #     print("Robot preparation failed.")
        #     return

        # input("Press enter to continue with transform:")
        
        listener = tf.TransformListener()
        print(f"{listener}")
        wait_success = listener.waitForTransform("xtion_depth_optical_frame", "base_footprint", rospy.Time(), rospy.Duration(4.0))
        print(f"wait success = {wait_success}")
        print("waiting done.")

        # input("Press enter to detect objections:")

        detections = detect_objects()
        if len(detections) == 0:
            print("No objects detected.")
            return

        print(f"Detected {detections}")

        detection = detections[0]
        rospy.loginfo(f"Working with detection = {detection.name}")
        if "banana" in detection.name:
            rospy.set_param("/motion/closed_gripper_joint", 0.015)
        elif "mustard" in detection.name:
            rospy.set_param("/motion/closed_gripper_joint", 0.025)
        elif "apple" in detection.name:
            rospy.set_param("/motion/closed_gripper_joint", 0.03)
        elif "mug" in detection.name:
            rospy.set_param("/motion/closed_gripper_joint", 0.005)
        else:
            rospy.set_param("/motion/closed_gripper_joint", 0.025)
            
        pose_gdrnpp = get_object_pose(detection.name)

        print(pose_gdrnpp)
        
        if pose_gdrnpp is  None:
            print("Could not estimate object pose.")
            return

        pose_in_head = PoseStamped() #parsing to pose stamped
        pose_in_head.header.frame_id = "xtion_depth_optical_frame"
        pose_in_head.header.stamp = rospy.Time(0)  # latest available

        pose_in_head.pose.position = pose_gdrnpp.pose.position
        pose_in_head.pose.orientation = pose_gdrnpp.pose.orientation

        #print("Detected ", detections[0].name)
        #print(f"At position :{round( pose_in_head.pose.position.x,2)}, {round(pose_in_head.pose.position.y,2)}, {round(pose_in_head.pose.position.z,2)}")
            
        try:
            pose_in_base = listener.transformPose("base_footprint", pose_in_head)
            print("Transformed pose:")
            print("Position:", pose_in_base.pose.position)
            print("Orientation:", pose_in_base.pose.orientation)
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Transform of the pose to base footprint failed.")
            return
            
        print("Attempting to pick...")

        self.pick_object = TransformStamped()
        self.pick_object.header.frame_id = "base_footprint"
        self.pick_object.child_frame_id = detections[0].name
        
        self.pick_object.transform.translation.x = pose_in_base.pose.position.x
        self.pick_object.transform.translation.y = pose_in_base.pose.position.y
        self.pick_object.transform.translation.z = pose_in_base.pose.position.z
        
        self.pick_object.transform.rotation.x = pose_in_base.pose.orientation.x
        self.pick_object.transform.rotation.y = pose_in_base.pose.orientation.y
        self.pick_object.transform.rotation.z = pose_in_base.pose.orientation.z
        self.pick_object.transform.rotation.w = pose_in_base.pose.orientation.w

        self.pose_pick_object = TransformStamped()
        self.pose_pick_object.header.frame_id = f"pose_{detections[0].name}"
        self.pose_pick_object.child_frame_id = detections[0].name
        
        self.pose_pick_object.transform.translation.x = pose_in_base.pose.position.x
        self.pose_pick_object.transform.translation.y = pose_in_base.pose.position.y
        self.pose_pick_object.transform.translation.z = pose_in_base.pose.position.z
        
        self.pose_pick_object.transform.rotation.x = pose_in_base.pose.orientation.x
        self.pose_pick_object.transform.rotation.y = pose_in_base.pose.orientation.y
        self.pose_pick_object.transform.rotation.z = pose_in_base.pose.orientation.z
        self.pose_pick_object.transform.rotation.w = pose_in_base.pose.orientation.w   

        self.publishable_tfs += [self.pick_object, self.pose_pick_object]     
        pose_in_base.pose.position.z += 0.06

        pick_success = False
        count = 10
        
        object_info = self.objects_info.get(detections[0].name, None)
        if object_info is None:
            print(f"Object {detection.name} not found in dataset.")
            return

        print (f"\n\n\n\nShape of the grasp array is: {object_info['grasps'].shape}\n\n\n\n")
        
        filtered_grasps = object_info["grasps"] #np.array([grasp for grasp in object_info["grasps"] if grasp[0][11]<0])
        pick_counter = 0 
        while not pick_success or pick_counter < filtered_grasps.shape[0]:
            pose_in_base.header.stamp = rospy.Time(0) 
            print("\tAttempts left ", count)
            print(f"Attempting grasp index {pick_counter}")
            # index = int(input("Enter the grasp you want to try: "))
            
            pick_success = pick_object(
                pick_counter, 
                mesh_path=object_info["mesh_path"],
                grasps = object_info["grasps"],
                pose= pose_in_base.pose  
                )
            reset_planning_scene()
            count -= 1
            pick_counter += 1
            if count == 0:
                break
                
            input(f"Press enter to try again: ")

        message = f"Picked!" if pick_success else f"Failed to pick"
        print(message)
        
        place_pose = pose_in_base.pose
        place_pose.pose.position.x-=.3
        result = place_object(pose = place_pose.pose, mesh_path = object_info["mesh_path"])
        
        message = f"Placed!" if result else f"Failed to place"
        print(message)

        return        
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
    node.execute_pick_place()

