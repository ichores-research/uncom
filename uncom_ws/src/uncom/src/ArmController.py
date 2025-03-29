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
from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from ast import literal_eval
import moveit_commander
import tf2_ros


class ArmInterface:
    def __init__(self):
        # Initialize the ROS Node
        rospy.init_node('arm_interface_node')
        # Publishers for topics

        # Subscribe to registered_depth topic
        self.depth_sub = rospy.Subscriber('/xtion/depth_registered/image_raw', Image, self.depth_callback)

        self.depth_frame = None
        self.saved_depth_frame = self.depth_frame

        self.arm_plan_tf = 'base_footprint'
        self.depth_camera_tf = 'xtion_rgb_optical_frame'
        self.shoulder_tf = 'arm_1_link'


        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_arm_torso = moveit_commander.MoveGroupCommander("arm_torso")

        self.group_arm_torso.set_planner_id("SBLkConfigDefault")
        self.group_arm_torso.set_pose_reference_frame(self.arm_plan_tf)


       ## listen to TIAGo's tf tree
        print("Looking for robot part' s location")
        self.tf_listener = tf.TransformListener()
        # waits to see if a transform is possible
        print ("Can we find a tranform between the base and the camera?")


        self.tf_listener.waitForTransform(self.arm_plan_tf, self.depth_camera_tf, rospy.Time(), rospy.Duration(5.0))
        self.tf_listener.waitForTransform(self.arm_plan_tf, self.shoulder_tf, rospy.Time(), rospy.Duration(5.0))
        
        # obtains the transform betwee TIAGo's base and the depth camera frames
        self.head_base_trans, self.head_base_rot = self.tf_listener.lookupTransform( self.arm_plan_tf, self.depth_camera_tf, rospy.Time(0))
        print (f"Found the head-base transform! It is {str(self.head_base_trans)}, {str(self.head_base_rot)}")

        self.base_arm_trans, _ = self.tf_listener.lookupTransform(self.arm_plan_tf, self.shoulder_tf, rospy.Time(0))
        print (f"Found the base-arm transform! It is {str(self.base_arm_trans)}")

        self.transform_broadcaster = tf2_ros.TransformBroadcaster()

        self.object_tf = TransformStamped()
        self.object_tf.header.frame_id = self.depth_camera_tf
        self.object_tf.child_frame_id = "object"
        self.object_tf.transform.rotation.w = 1

        self.object_pointing_tf = TransformStamped()
        self.object_pointing_tf.header.frame_id = self.arm_plan_tf
        self.object_pointing_tf.child_frame_id = "pointing_object"
        self.object_pointing_tf.transform.rotation.w = 1

        self.camera_info = rospy.wait_for_message('/xtion/depth/camera_info', CameraInfo)

        timer = rospy.Timer(rospy.Duration(0.1), self.tf_callback)

    def tf_callback(self, event):
        for transf in [self.object_tf, self.object_pointing_tf, self.target_tf, self.target_pointing_tf]:
            if transf is not None:
                transf.header.stamp = rospy.Time.now()
                self.transform_broadcaster.sendTransform(transf)

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

    def move_arm_to_pose(self, goal):

        self.group_arm_torso.set_pose_target(goal)
        self.group_arm_torso.set_planning_time(5.0)
        self.group_arm_torso.set_start_state_to_current_state()
        self.group_arm_torso.set_max_velocity_scaling_factor(1.0)

        plan = self.group_arm_torso.plan()
        if not plan:
            rospy.logerr("No plan found")
            return
        self.group_arm_torso.go(wait=True)

    def move_arm_to_cartesian(self, x, y, z, w):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = w
        self.move_arm_to_pose(pose)

    def move_arm_to_tf(self, tf_name):
        object_to_base_transform = self.tf_listener.lookupTransform(self.arm_plan_tf, tf_name, rospy.Time(0))         
        goal = self.tf_to_pose(object_to_base_transform)
        self.move_arm_to_pose(goal)

    def run(self):
        # Run the ROS node
        rospy.spin()


if __name__ == '__main__':
    # Instantiate the UnderstandingNode class and run the node
    client_thread = Thread(target=loop_client,args=[client])
    client_thread.start()

    node = UnderstandingNode()
    node.run()

