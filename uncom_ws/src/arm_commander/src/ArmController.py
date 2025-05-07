#!/usr/bin/env python3
import numpy as np
import rospy
from threading import Thread
from time import time
import tf 
from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from arm_commander.srv import MoveToCartesian, MoveToCartesianResponse, OpenCloseGripper, OpenCloseGripperResponse, SetGripperJoints, SetGripperJointsResponse

import moveit_commander
import tf2_ros

class ArmInterface:
    def __init__(self):
        """! ArmInterface Initializer method."""
        # Initialize the ROS Node
        rospy.init_node('arm_interface_node')
        # Publishers for topics

        # Subscribe to registered_depth topic
        self.depth_sub = rospy.Subscriber('/xtion/depth_registered/image_raw', Image, self.depth_callback)

        ## service for moving the arm to x,y,z location with respect to the robot's footprint
        self.move_arm_cartesian_service = rospy.Service('move_arm_cartesian', MoveToCartesian, self.cartesian_move_callback)

        ## service for fully opening the gripper
        self.open_gripper_service = rospy.Service('open_gripper', OpenCloseGripper, self.open_gripper_callback)

        ## service for fully closing the gripper
        self.close_gripper_service = rospy.Service('close_gripper', OpenCloseGripper, self.close_gripper_callback)

        ## service for opening and closing the gripper in succession
        self.open_close_gripper_service = rospy.Service('open_and_close_gripper', OpenCloseGripper, self.open_close_gripper_callback) 

        ## service for setting gripper joint state
        self.set_gripper_joint_state_service = rospy.Service('set_gripper_joint_state', SetGripperJoints, self.set_gripper_state_callback)

        ## current depth frame image of the robot; it is a numpy array object. 
        self.depth_frame = None

        ## a stored depth frame to be used as reference regardless of the robot's present depth frame. numpy.array object.
        self.saved_depth_frame = self.depth_frame

        ## name of the tf of the arm's planner
        self.arm_plan_tf = 'base_footprint'

        ## depth camera frame tf name
        self.depth_camera_tf = 'xtion_rgb_optical_frame'

        ## shoulder tf name 
        self.shoulder_tf = 'arm_1_link'

        ## arm and torso move it commander
        self.robot = moveit_commander.RobotCommander()

        ## planning scene for the robot's arm and torso
        self.scene = moveit_commander.PlanningSceneInterface()

        ## move group commander for the robot's arm and torso
        self.group_arm_torso = moveit_commander.MoveGroupCommander("arm_torso")
        self.group_arm_torso.set_planner_id("SBLkConfigDefault")
        self.group_arm_torso.set_pose_reference_frame(self.arm_plan_tf)

        ## move group commander for the robot's gripper
        self.group_gripper = moveit_commander.MoveGroupCommander("gripper")
        self.group_gripper.set_planner_id("SBLkConfigDefault")
        self.group_gripper.set_pose_reference_frame(self.arm_plan_tf)

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

        ## tf that should be used as reference for purposes of moving thre robot arm
        self.object_tf = TransformStamped()
        self.object_tf.header.frame_id = self.depth_camera_tf
        self.object_tf.child_frame_id = "object"
        self.object_tf.transform.rotation.w = 1

        ## tf that should be used as reference for purposes of pointing at something.
        self.object_pointing_tf = TransformStamped()
        self.object_pointing_tf.header.frame_id = self.arm_plan_tf
        self.object_pointing_tf.child_frame_id = "pointing_object"
        self.object_pointing_tf.transform.rotation.w = 1

        ## obtains the characteristics of the robot's camera.
        self.camera_info = rospy.wait_for_message('/xtion/depth/camera_info', CameraInfo)

        ## Timer responsible for publishing the custom tfs created by this node every 0.1s
        timer = rospy.Timer(rospy.Duration(0.1), self.tf_callback)
        timer 

    def tf_callback(self, event):
        """Method responsible for continuysouly updating the tfs for the objectsand pointing position.
        @param event : event that triggers the publishing of the update tfs. """
        for transf in [self.object_tf, self.object_pointing_tf]:
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
        except Exception as e:
            rospy.logerr("Error reading the depth frame: %s", str(e))
            self.depth_frame = None

    def cartesian_move_callback(self,req):
        """!Callback for moving the robot's manipulator to a x, y, z position in respect to the robot's base footprint.
        @param req <arm_commander.srv.MoveToCartesian>: request to move the robot's arm to a desired x, y, z position with respect to the robot base's footprint.
        @return response <arm_commander.srv.MoveToCartesianResponse>: whether the arm movement was successfull or not"""
        x, y, z = req.x, req.y, req.z
        response = MoveToCartesianResponse()
        try:
            self.move_arm_to_cartesian(x, y, z, 1.0)
            response.result = True
        except Exception as e:
            rospy.logerr(f"{e}")
            response.result = False
        return response

    def open_gripper_callback(self,req):
        """!Callback for opening the robot's gripper.
        @param req <arm_commander.srv.OpenCloseGripper>: request to open the robot's gripper.
        @return response <arm_commander.srv.OpenCloseGripperResponse>: whether the opening the gripper was successfull or not."""

        response = OpenCloseGripperResponse()
        try:
            self.open_gripper(1.0)
            response.result = True
        except Exception as e:
            rospy.logerr(f"{e}")
            response.result = False
        return response

    def close_gripper_callback(self,req):
        """!Callback for closing the robot's gripper.
        @param req <arm_commander.srv.OpenCloseGripper>: request to close the robot's gripper.
        @return response <arm_commander.srv.OpenCloseGripperResponse>: whether the closing the gripper was successfull or not."""
        response = OpenCloseGripperResponse()
        try:
            self.close_gripper(1.0)
            response.result = True
        except Exception as e:
            rospy.logerr(f"{e}")
            response.result = False
        return response

    def open_close_gripper_callback(self,req):
        """!Callback for opening and closing the robot's gripper in succession.
        @param req <arm_commander.srv.OpenCloseGripper>: request to open & close the robot's gripper.
        @return response <arm_commander.srv.OpenCloseGripperResponse>: whether the opening & closing the gripper was successfull or not."""

        response = OpenCloseGripperResponse()
        try:
            self.open_gripper(1.0)
            rospy.sleep(0.1)
            self.close_gripper(1.0)
            response.result = True
        except Exception as e:
            rospy.logerr(f"{e}")
            response.result = False
        return response

    def set_gripper_state_callback(self,req):
        """!Callback for opening the robot's gripper.
        @param req <arm_commander.srv.SetGripperJoints>: request to set the joint state of the robot's gripper.
        @return response <arm_commander.srv.SetGripperJointsResponse>: whether setting the joint state of the robot's gripper was successfull or not."""

        response = SetGripperJointsResponse()
        try:
            self.set_gripper_joint_state(req.left_gripper ,req.right_gripper, 1.0)
            response.result = True
        except Exception as e:
            rospy.logerr(f"{e}")
            response.result = False
        return response


    def store_current_depth_frame(self):
        """!Method responsible for storing the current depth frame, so the arm can calculate depth with respect with a frozen frame regardless fo the current robot's position."""
        self.saved_depth_frame = self.depth_frame


    def set_object_tf(self, center, input_tf):
        """!Method responsible for setting the pose of a tf object with respect to the center """
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
        """!Helper method that generates a Pose from a given transform
        @param transform<geometry_msgs.msg.TransformStampped>: transform that will become a Pose object
        @return pose <geometry_msgs.msg.Pose>: Pose obtained from the given transform."""
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
        """!Method responsible for moving the robot arm's gripper to a given Pose wusing inverse kinematics.
        @param goal<geometry_msgs.msg.Pose>: destination to which the robot arm's manipulator should move.
        @return None"""
        self.group_arm_torso.set_pose_target(goal)
        self.group_arm_torso.set_planning_time(50.0)
        self.group_arm_torso.set_start_state_to_current_state()
        self.group_arm_torso.set_max_velocity_scaling_factor(1.0)

        plan = self.group_arm_torso.plan()
        if not plan:
            rospy.logerr("No plan found")
            return
        self.group_arm_torso.go(wait=True)

    def move_arm_to_cartesian(self, x, y, z, w):
        """!Method responsible for moving the robot arm's gripper to a given Pose wusing inverse kinematics.
        @param x <float>: x coordinate of the destination to which the robot arm's manipulator should move.
        @param y <float>: y coordinate of the destination to which the robot arm's manipulator should move.
        @param z <float>: z coordinate of the destination to which the robot arm's manipulator should move.
        @param w <float>: w orienation of the robot gripper at the destination.
        @return None"""

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
        """!Method responsible for moving the robot arm's gripper to a given tf using inverse kinematics.
        @param tf_name<str>: name of the tf to which the robot's gripper should be moved to.
        @return None"""
        object_to_base_transform = self.tf_listener.lookupTransform(self.arm_plan_tf, tf_name, rospy.Time(0))         
        goal = self.tf_to_pose(object_to_base_transform)
        self.move_arm_to_pose(goal)

    def move_arm_to_pixel(self, x, y):
        """!Method responsible for moving the robot arm's gripper to a position in the real world obtained from a pixel in the robot's image.
        @param x<int>: x coordinate of in the robot's camera image where the gripper should move to. 
        @param y<int>: y coordinate of in the robot's camera image where the gripper should move to.
        @return None"""

        self.store_current_depth_frame()  # Comment this line if you use updated images of the environment during the process instead of planning everything on the start from the initial picture 
        self.set_object_tf((x, y), self.object_tf)
        rospy.sleep(0.2)
        self.move_arm_to_tf(self.object_tf.child_frame)

    def set_gripper_joint_state(self, left_gripper, right_gripper, velocity=1.0):
        """!Method responsible for setting the gripper joint state.
        @param left_gripper<float>: joint state of the left gripper finger, should have a value larger than 0.0 and not larger than 0.04
        @param right_gripper<float>: joint state of the right gripper finger, should have a value larger than 0.0 and not larger than 0.04
        @param velocity<float>: velocity scale in which the robot will move the gripper fingers.
        @return None"""

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

    def open_gripper(self, velocity=1.0):
        """!Helper method that fully opens the robot's gripper with a given velocity scale.
        @param velocity<float>: velocity scale for opening the gripper"""
        self.set_gripper_joint_state(left_gripper=0.04, right_gripper=0.04, velocity=velocity)

    def close_gripper(self, velocity=1.0):
        """!Helper method that closes the robot's gripper with a given velocity scale.
        @param velocity<float>: velocity scale for closing the gripper"""
        self.set_gripper_joint_state(left_gripper=0.00125, right_gripper=0.00125, velocity=velocity)

    def run(self):
        """!Method responsible for keeping the node up and running."""
        # Run the ROS node
        rospy.spin()

if __name__ == '__main__':
    node = ArmInterface()
    node.run()

