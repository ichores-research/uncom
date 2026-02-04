#!/usr/bin/env python3

import numpy as np
from motion_msgs.srv import Prepare, Pick, PickRequest, PrepareRequest
from motion_msgs.srv import Pick as Place
from motion_msgs.srv import PickRequest as PlaceRequest
from motion_msgs.srv import Pick as ToPose  # added for code understandability, so ToPose requests are not misunderstood as Pick
from motion_msgs.srv import PickRequest as ToPoseRequest # added for code understandability
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse

from std_srvs.srv import SetBool, SetBoolRequest
from geometry_msgs.msg import Pose, PoseArray, Point32, PoseStamped #, Twist 
import rospy
from shape_msgs.msg import Mesh

import open3d as o3d
from shape_msgs.msg import Mesh, MeshTriangle
import tf.transformations as tft
import tf
import tf2_ros
import tf2_geometry_msgs

import os
from uncom.ycb_objects import get_ycb_objects_info
from uncom.object_detection import *
import threading


stop_publishing_tf = threading.Event()


def object_pose_tf_publisher(object_pose, object_name = "detected_object", parent_frame = "parent_tf"):
    rospy.sleep(2.0)
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(30)
    while not stop_publishing_tf.is_set() and not rospy.is_shutdown():
        br.sendTransform([object_pose.position.x, 
	                      object_pose.position.y,
	                      object_pose.position.z],
	                      [object_pose.orientation.x, 
	                       object_pose.orientation.y,
	                       object_pose.orientation.z,
	                       object_pose.orientation.w], 
	                       rospy.Time.now(), 
	                       object_name, 
	                       parent_frame)
        rate.sleep()


def transform_grasp_obj2world(grasps, pose):
    transformed_grasps = []

    # Convert object quaternion to a 4x4 transformation matrix, then extract the 3x3 rotation matrix
    obj_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    obj_transform = tft.quaternion_matrix(obj_quat)  # This gives a 4x4 matrix
    obj_R = obj_transform[:3, :3]  # Extract the 3x3 rotation part
    obj_t = np.array([pose.position.x, pose.position.y, pose.position.z])  # Translation vector

    for grasp in grasps:
        # Convert the grasp to a 4x4 matrix
        grasp_matrix = np.array(grasp).reshape(4, 4)

        # Apply rotation and translation to transform the grasp to world coordinates
        transformed_grasp_matrix = np.eye(4)
        transformed_grasp_matrix[:3, :3] = np.dot(obj_R, grasp_matrix[:3, :3])  # Rotate
        transformed_grasp_matrix[:3, 3] = np.dot(obj_R, grasp_matrix[:3, 3]) + obj_t  # Rotate and translate

        # Flatten the transformed matrix and store it
        transformed_grasps.append(transformed_grasp_matrix.flatten())

    return np.array(transformed_grasps)


def o3d_to_shape_mesh(model):
    # Read PLY file using Open3D
    mesh_msg = Mesh()

    # Convert vertices
    vertices = np.asarray(model.vertices)
    triangles = np.asarray(model.triangles)

    # Add vertices
    for vertex in vertices:
        point = Point32()
        point.x = float(vertex[0]) / 1000.0
        point.y = float(vertex[1]) / 1000.0
        point.z = float(vertex[2]) / 1000.0
        mesh_msg.vertices.append(point)

    # Add triangles
    for triangle in triangles:
        mesh_triangle = MeshTriangle()
        mesh_triangle.vertex_indices = [int(triangle[0]), 
                                      int(triangle[1]), 
                                      int(triangle[2])]
        mesh_msg.triangles.append(mesh_triangle)

    return mesh_msg


def ndarray_to_pose_array(poses):
    pose_array = PoseArray()
    align_x_to_z = tft.quaternion_from_euler(0, np.pi / 2, 0)
    for pose in poses:
        matrix = pose.reshape(4,4)
        translation = matrix[:3, 3]
        orientation = tft.quaternion_from_matrix(matrix)
        adjusted_orientation = tft.quaternion_multiply(orientation, align_x_to_z)

        p = Pose()
        p.position.x = float(translation[0])
        p.position.y = float(translation[1])
        p.position.z = float(translation[2])
        p.orientation.x = float(adjusted_orientation[0])
        p.orientation.y = float(adjusted_orientation[1])
        p.orientation.z = float(adjusted_orientation[2])
        p.orientation.w = float(adjusted_orientation[3])
        pose_array.poses.append(p)
    return pose_array


def prepare_robot():
    prepare_service = rospy.ServiceProxy('/motion/prepare', Prepare)
    rospy.wait_for_service('/motion/prepare')
    
    # Prepare the robot for picking
    try:
        prepare_service(PrepareRequest())
        return True
    except rospy.ServiceException as e:
        print(f"Motion prepare call failed: {e}")
        return False


def reset_planning_scene():
    prepare_service = rospy.ServiceProxy('/motion/reset_planning_scene', SetBool)
    rospy.wait_for_service('/motion/reset_planning_scene')
    
    # Prepare the robot for picking
    try:
        prepare_service(SetBoolRequest(data=True))
        return True
    except rospy.ServiceException as e:
        print(f"Planning scene reset call failed due to: {e}")
        return False


def pick_object(index: int, mesh_path: str, grasps: np.ndarray, pose: Pose, **kwargs):
    
    pick_service = rospy.ServiceProxy('/motion/pick', Pick)
    rospy.wait_for_service('/motion/pick')

    print(type(grasps))
    print(f"Grasps: {grasps}")
    print("Filtering grasps")
    filtered_grasps = []

    for i, grasp in enumerate(grasps):
        if i == index:
            filtered_grasps.append(grasp)

    filtered_grasps = np.array(filtered_grasps)
    print(filtered_grasps)

    try:
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh_msg = o3d_to_shape_mesh(mesh)
        except Exception as e:
            print(f"Failed to read mesh from {mesh_path}: {e}")
            return False

        grasps_transformed = transform_grasp_obj2world(filtered_grasps, pose)
        pose_array = ndarray_to_pose_array(grasps_transformed)

        # Create Pick message
        pick_req = PickRequest()
        pick_req.object_mesh = mesh_msg
        pick_req.object_pose = pose
        pick_req.grasps = pose_array
        

        # Call the pick service
        response = pick_service(pick_req)
        print(f"Pick service response: {response.success}, {response.message}")

        place_service = rospy.ServiceProxy('/motion/place', Pick)
        rospy.wait_for_service('/motion/place')    

        response = place_service(pick_req)
        print(f"Pick service response: {response.success}, {response.message}")

        return response.success
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def place_object(pose: Pose, mesh_path: str, **kwargs):

    place_service = rospy.ServiceProxy('/motion/place', Place)
    rospy.wait_for_service('/motion/place')    
    
    place_req = PlaceRequest()
    try:
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh_msg = o3d_to_shape_mesh(mesh)
            place_req.object_mesh = mesh_msg
        except Exception as e:
            rospy.logerr(f"Failed to read mesh from {mesh_path}: {e}")

        place_req.object_pose = pose
        response = place_service(place_req)
        print(f"Pick service response: {response.success}, {response.message}")
        return response.success
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    

def move_to_pose(pose:Pose):
    to_pose_service = rospy.ServiceProxy('/motion/to_pose', ToPose)
    rospy.wait_for_service('/motion/to_pose')

    to_pose_req = ToPoseRequest()
    try:
            to_pose_req.object_pose = pose
            response = to_pose_service(to_pose_req)
            print(f"To Pose service response: {response.success}, {response.message}")
            return response.success
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def open_gripper():
    open_gripper_service = rospy.ServiceProxy('/motion/open_gripper', Empty)
    rospy.wait_for_service('/motion/open_grippper')    
    gripper_req = EmptyRequest()
    open_gripper_service(gripper_req)
    return True


def close_gripper():
    close_gripper_service = rospy.ServiceProxy('/motion/close_gripper', Empty)
    rospy.wait_for_service('/motion/close_grippper')    
    gripper_req = EmptyRequest()
    close_gripper_service(gripper_req)
    return True