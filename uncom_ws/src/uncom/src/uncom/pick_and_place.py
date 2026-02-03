#!/usr/bin/env python3

import numpy as np
from motion_msgs.srv import Prepare, Pick, PickRequest, PrepareRequest
from motion_msgs.srv import Pick as Place
from motion_msgs.srv import PickRequest as PlaceRequest
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

# import time

# pub = rospy.Publisher('robot_wiggler', Twist, queue_size=10)
# vel_cmd = Twist()
# vel_cmd.linear.x = 0.05
# vel_cmd.angular.z = 0.01
# t0 = time.time()

# while time.time()-t0<0.2:
#     pub.publish (vel_cmd)
# pub.publish(Twist())

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

def pick_object_with_grasp(mesh_path: str, grasps: np.ndarray, pose: Pose, **kwargs):
    
    pick_service = rospy.ServiceProxy('/motion/pick', Pick)
    rospy.wait_for_service('/motion/pick')

    try:
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh_msg = o3d_to_shape_mesh(mesh)
        except Exception as e:
            print(f"Failed to read mesh from {mesh_path}: {e}")
            return False

        grasps_transformed = transform_grasp_obj2world(grasps, pose)
        pose_array = ndarray_to_pose_array(grasps_transformed)

        # Create Pick message
        pick_req = PickRequest()
        pick_req.object_mesh = mesh_msg
        pick_req.object_pose = pose
        pick_req.grasps = pose_array
        

        # Call the pick service
        response = pick_service(pick_req)
        print(f"Pick service response: {response.success}, {response.message}")

        return response.success
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def pick_object_by_info(object_info: dict, object_name: str = None):
    if object_info is None:
        rospy.logwarn("pick_object_by_info: object_info is None")
        return False
    
    if object_name is None:
        rospy.logwarn("pick_object_by_info: object_name is None")
        return False
    
    if "mesh_path" not in object_info or "grasps" not in object_info:
        rospy.logwarn(f"pick_object_by_info: object_info missing required keys. Got: {list(object_info.keys())}")
        return False
    
    # Get object pose from detection
    pose_gdrnpp = get_object_pose(object_name)
    if pose_gdrnpp is None:
        rospy.logwarn(f"pick_object_by_info: Could not estimate pose for {object_name}")
        return False
    
    # Transform pose to base_footprint frame
    listener = tf.TransformListener()
    try:
        listener.waitForTransform("xtion_depth_optical_frame", "base_footprint", rospy.Time(), rospy.Duration(4.0))
        
        pose_in_head = PoseStamped()
        pose_in_head.header.frame_id = "xtion_depth_optical_frame"
        pose_in_head.header.stamp = rospy.Time(0)
        pose_in_head.pose.position = pose_gdrnpp.pose.position
        pose_in_head.pose.orientation = pose_gdrnpp.pose.orientation
        
        pose_in_base = listener.transformPose("base_footprint", pose_in_head)
        
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logwarn(f"pick_object_by_info: Transform failed: {e}")
        return False
    
    # Objects are slightly incorporated in the table plane, so move them slightly higher
    pose_in_base.pose.position.z += 0.06
    
    # Start TF publisher threads for visualization (optional, but helps with debugging)
    if stop_publishing_tf.is_set():
        stop_publishing_tf.clear()
    
    arguments = (pose_in_base.pose, object_name, "base_footprint")
    tf_publisher_thread = threading.Thread(target=object_pose_tf_publisher, args=arguments)
    tf_publisher_thread.start()
    
    # Try multiple grasps automatically (up to 5 or all available)
    grasps = object_info["grasps"]
    max_attempts = min(len(grasps), 5)
    
    rospy.loginfo(f"pick_object_by_info: Attempting to pick {object_name} with {max_attempts} grasp attempts")
    
    for grasp_idx in range(max_attempts):
        rospy.loginfo(f"pick_object_by_info: Attempting to pick {object_name} with {max_attempts} grasp attempts")
        success = pick_object_with_grasp(
            mesh_path=object_info["mesh_path"],
            grasps=grasps,
            pose=pose_in_base.pose
        )
        if success:
            rospy.loginfo(f"pick_object_by_info: Successfully picked {object_name} with grasp {grasp_idx}")
            # Stop TF publisher threads
            stop_publishing_tf.set()
            if tf_publisher_thread.is_alive():
                tf_publisher_thread.join(timeout=2.0)
            return True
    
    # All grasps failed
    rospy.logwarn(f"pick_object_by_info: Failed to pick {object_name} after {max_attempts} attempts")
    # Stop TF publisher threads
    stop_publishing_tf.set()
    if tf_publisher_thread.is_alive():
        tf_publisher_thread.join(timeout=2.0)
    return False


# def pick_object(object_info: dict, object_name: str = None):
#     return pick_object_by_info(object_info, object_name=object_name)


def test_pick(objects_info):
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
    if stop_publishing_tf.is_set():
        stop_publishing_tf.clear()
    
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

    arguments = (pose_in_base.pose, detections[0].name, "base_footprint")
    tf_publisher_thread = threading.Thread(target = object_pose_tf_publisher, args = arguments)
    tf_publisher_thread.start()

    arguments2 = (pose_in_base.pose, f"pose_{detections[0].name}", "base_footprint")
    tf_publisher_thread2 = threading.Thread(target = object_pose_tf_publisher, args = arguments2)
    tf_publisher_thread2.start()

    object_info = objects_info.get(detections[0].name, None)
    if object_info is None:
        print(f"Object {detection.name} not found in dataset.")
        return
    
    # objects are slightly incorporated in the table plane,
    # so this is moving them slightly higher
    pose_in_base.pose.position.z += 0.06

    pick_success = False
    count = 10
    print (f"\n\n\n\nShape of the grasp array is: {object_info['grasps'].shape}\n\n\n\n")
    
    filtered_grasps = object_info["grasps"] #np.array([grasp for grasp in object_info["grasps"] if grasp[0][11]<0])
    pick_counter = 0 
    while not pick_success or pick_counter < filtered_grasps.shape[0]:
        pose_in_base.header.stamp = rospy.Time(0) 
        print("\tAttempts left ", count)
        print(f"Attempting grasp index {pick_counter}")
        index = int(input("Enter the grasp you want to try: "))
        
        pick_success = pick_object(
            index, 
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
    
    
if __name__=="__main__":
    rospy.init_node('pick_and_place_test_node')

    DATASET = os.environ.get("DATASET", "ycb_ichores")
    OBJECTS_INFO = get_ycb_objects_info(DATASET)
    try:
        test_pick(OBJECTS_INFO)

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
