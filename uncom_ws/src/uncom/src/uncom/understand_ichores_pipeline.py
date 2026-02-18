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
from cv_bridge import CvBridge
from ast import literal_eval
import tf2_ros
import tf2_geometry_msgs
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


def on_message(client, userdata, message):
    global understood
    print(f"Received message {message}")

    try:
        understood = literal_eval(literal_eval(message.payload.decode("utf-8")))
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
        self.mqtt_client = mqtt_client
        # Publishers for topics
        
        self.video_record_pub = rospy.Publisher('/video_recording', Bool, queue_size=10)
        self.audio_record_pub = rospy.Publisher('/audio_recording', Bool, queue_size=10)
        self.vad_status_pub = rospy.Publisher('/perform_vad', Bool, queue_size=10)

        self.save_video_pub = rospy.Publisher('/save_video', String, queue_size=10)
        self.save_audio_pub = rospy.Publisher('/save_audio', String, queue_size=10)
    
        self.clear_video_pub = rospy.Publisher('/clear_video', Empty, queue_size=10)
        self.clear_audio_pub = rospy.Publisher('/clear_audio', Empty, queue_size=10)

        self.set_listening_status = rospy.Publisher('/listening_mode', Bool, queue_size=10)
        self.set_wrist_publishing_status = rospy.Publisher('/wrist_streaming_mode', Bool, queue_size=10)

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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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
        self.stored_command = []
        
        self.objects_info = get_ycb_objects_info("ycb_ichores")

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
                        timeout = 1000
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
                            self.stored_command = understood
                            self.point_and_ask(understood[1], understood[2], understood[3], understood[4], understood[5])
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
                            self.stored_command = []
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
                            rospy.logerr("Error, unexpected result for the understanding operation.")
                            self.stored_command = []
                            
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
                            self.execute_pick_place(self.stored_command[4], self.stored_command[5])

                            rospy.sleep(3.0)
                            self.publish_vad_status(True)
                            
                        else:
                            self.publish_vad_status(False)
                            self.stored_command=[]
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
            target_to_source_tf = self.tf_buffer.lookup_transform(new_parent, 
                                                                input_tf.header.frame_id, 
                                                                rospy.Time(0), 
                                                                rospy.Duration(1.0))
            
            output_tf = tf2_geometry_msgs.do_transform_transform(input_tf, target_to_source_tf)
            output_tf.header.frame_id = new_parent
            output_tf.header.stamp = rospy.Time.now()
            
            return output_tf

        except Exception as e:
            rospy.logerr(f"Failed to re-parent transform: {e}")
            return None

    def set_object_tf(self, center, input_tf):
        x, y = center
        # Get the depth value at the pixel
        
        depth_image = np.frombuffer(self.saved_depth_frame.data, dtype=np.float32).reshape(self.saved_depth_frame.shape[0], self.saved_depth_frame.shape[1])

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
        msg = Bool()
        msg.data = status
        self.vad_status_pub.publish(msg)
        self.set_listening_status.publish(msg)
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

    def move_arm(self, goal):
        move_to_pose(goal)

    def point_and_ask(self, object, action, target , object_1_center, object_2_center):  # TODO: To be implemented, robot needs to repeat what it understood while pointing at objects.
        self.saved_depth_frame = self.depth_frame
        self.set_object_tf(object_1_center, self.object_tf)
        rospy.sleep(0.5)
        self.set_object_tf(object_2_center, self.target_tf)    
        rospy.sleep(0.5)
        print("Debug 1")
        self.object_tf = self.change_parent(self.object_tf, "base_footprint")
        print("Debug 2")

        self.target_tf = self.change_parent(self.target_tf, "base_footprint")
        
        self.set_pointing_tf(self.object_tf, self.object_pointing_tf)
        rospy.sleep(0.75)
        self.set_pointing_tf(self.target_tf, self.target_pointing_tf)
        rospy.sleep(0.5)
        print("Debug 3")

        if not self.simulation:
            self.tiago_talk(f"Would you like me to {action}")
        
        rospy.sleep(2.0)
        
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
        
        rospy.sleep(2.0)        
        self.saved_depth_frame = None

    def check_inside_bbox(self, bbox, xpoint, ypoint):
        return (bbox.xmin <= xpoint <= bbox.xmax) and (bbox.ymin <= ypoint <= bbox.ymax)
         
    def match_dino_2_gdrnet(self, detections, object_center):
        for object in detections:
            if self.check_inside_bbox(object.bbox, object_center[0], object_center[1]):
                return object
        return

    def execute_pick_place(self, object_1_center, object_2_center):      
        
        preparation_success = prepare_robot()
        if not preparation_success:
            rospy.logerr("Robot failed to assume initial position, giving up.")
            return 

        wait_success = True
        try:
            self.tf_buffer.lookup_transform("base_footprint", "xtion_depth_optical_frame", 
                                            rospy.Time(0), rospy.Duration(4.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            wait_success = False
               
        print(f"wait success = {wait_success}")
        print("waiting done.")

        detections = detect_objects()

        to_pick_object = self.match_dino_2_gdrnet(detections, object_1_center)
        place_destination = self.match_dino_2_gdrnet(detections, object_2_center)
        pose_gdrnpp_place = None

        if to_pick_object is None:
            depth_image = np.frombuffer(self.saved_depth_frame.data, dtype=np.float32).reshape(self.saved_depth_frame.height, self.saved_depth_frame.width)
            depth = depth_image[object_1_center[1], object_1_center[0]]
            
            fx = self.camera_info.K[0]
            fy = self.camera_info.K[4]
            cx = self.camera_info.K[2]
            cy = self.camera_info.K[5]
            
            X = (object_1_center[0] - cx) * depth / fx
            Y = (object_1_center[1] - cy) * depth / fy
            Z = depth

            pick_pose = Pose()
            pick_pose.position.x = X
            pick_pose.position.y = Y
            pick_pose.position.z = Z            
            open_gripper()
            rospy.sleep(1.0)
            move_to_pose(pick_pose)
            close_gripper()
            
        else:
            pick_object_info = self.objects_info.get(to_pick_object.name, None)
        
            if pick_object_info is None:
                print(f"Object {to_pick_object.name} not found in dataset.")
                return

            else:
                pose_gdrnpp_pick = get_object_pose(to_pick_object.name)
                if pose_gdrnpp_pick is  None:
                    print("Could not estimate object pose.")
                    return

                pose_in_head = PoseStamped() #parsing to pose stamped
                pose_in_head.header.frame_id = "xtion_depth_optical_frame"
                pose_in_head.header.stamp = rospy.Time(0)  # latest available

                pose_in_head.pose.position = pose_gdrnpp_pick.pose.position
                pose_in_head.pose.orientation = pose_gdrnpp_pick.pose.orientation

                try:
                    transform = self.tf_buffer.lookup_transform("base_footprint", "xtion_depth_optical_frame", rospy.Time(0))
                    pose_in_base = tf2_geometry_msgs.do_transform_pose(pose_in_head, transform)
                    # pose_in_base = self.tf_listener.transformPose("base_footprint", pose_in_head)

                except Exception as e:
                    print(f"Transform of the pose to base footprint failed due to: {e}")
                    return

                for i in range (10):
                    success = pick_object(index=i, 
                                        mesh_path = pick_object_info["mesh_path"],
                                        grasps = pick_object_info["grasps"],
                                        pose = pose_in_base)
                    if success:
                        break 

        if place_destination is None:
            rospy.logwarn("No correspondence to GDRNet++ detections, might be empty space, fall back to depth-based method.")
            depth_image = np.frombuffer(self.saved_depth_frame.data, dtype=np.float32).reshape(self.saved_depth_frame.height, self.saved_depth_frame.width)
            depth = depth_image[object_2_center[1], object_2_center[0]]
            fx = self.camera_info.K[0]
            fy = self.camera_info.K[4]
            cx = self.camera_info.K[2]
            cy = self.camera_info.K[5]
            
            # Convert pixel coordinates to 3D coordinates
            X = (object_2_center[0] - cx) * depth / fx
            Y = (object_2_center[1] - cy) * depth / fy
            Z = depth

            place_pose = Pose()

            place_pose.position.x = X
            place_pose.position.y = Y
            place_pose.position.z = Z 
            place_object(place_pose, mesh_path=f"/root/catkin_ws/src/uncom/data/datasets/ycb_ichores/models/obj_{int(11):06d}.ply")

        else: 
            pose_gdrnpp_place = get_object_pose(place_destination.name)
            place_object_info = self.objects_info.get(place_destination.name, None)
            pose_in_head = PoseStamped() #parsing to pose stamped
            pose_in_head.header.frame_id = "xtion_depth_optical_frame"
            pose_in_head.header.stamp = rospy.Time(0)  # latest available

            pose_in_head.pose.position = pose_gdrnpp_place.pose.position
            pose_in_head.pose.orientation = pose_gdrnpp_place.pose.orientation
        
            try:
                transform = self.tf_buffer.lookup_transform("base_footprint", "xtion_depth_optical_frame", rospy.Time(0))
                pose_in_base = tf2_geometry_msgs.do_transform_pose(pose_in_head, transform)
                # pose_in_base = self.tf_listener.transformPose("base_footprint", pose_in_head)
                
            except Exception as e:
                print(f"Transform of the pose to base footprint failed due to: {e}.")
                return

            if pose_in_base:
                    success = place_object(pose_in_base, 
                                           place_object_info["mesh_path"])


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

