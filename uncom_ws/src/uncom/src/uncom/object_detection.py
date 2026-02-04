#!/usr/bin/env python3
import os
import rospy
from object_detector_msgs.srv import detectron2_service_server,  estimate_poses
from sensor_msgs.msg import Image
from uncom.ycb_objects import *
import tf.transformations as tf_trans


def detect_objects(rgb=None):
    if rgb is None:
        rgb = rospy.wait_for_message(rospy.get_param('/pose_estimator/color_topic'), Image)
    rospy.wait_for_service('detect_objects')
    try:
        detect_objects_service = rospy.ServiceProxy('detect_objects', detectron2_service_server)
        response = detect_objects_service(rgb)
        return response.detections.detections
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return None


def estimate_object_pose(rgb, depth, detection):

    rospy.wait_for_service('/estimate_poses')
    
    try:
        estimate_poses_service = rospy.ServiceProxy('/estimate_poses', estimate_poses)
        response = estimate_poses_service(detection, rgb, depth)
        return response.poses
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def get_object_pose(object_name):
    print("Waiting for RGB image")
    rgb = rospy.wait_for_message(rospy.get_param('/pose_estimator/color_topic'), Image)
    print("Received RGB Image")
    print("Waiting depth Image")
    depth = rospy.wait_for_message(rospy.get_param('/pose_estimator/depth_topic'), Image)
    print ("received depth Image")
    detections = detect_objects(rgb)
    
    if detections is None or len(detections) == 0:
        return "Nothing detected"
    else:

        estimated_pose_camFrame = None

        try:
            for detection in detections:
                if detection.name == object_name:
                    print(f"Checking object {object_name}")
                    estimated_pose_camFrame = estimate_object_pose(rgb, depth, detection)[0]
                    
                    break
                
        except Exception as e:
            return "Pose estimation failed"
        return estimated_pose_camFrame


def get_object_poses():
    detections = detect_objects()
    return [get_object_pose(det.name) for det in detections]


def parse_scene_for_placing(objects_info, object_in_the_gripper=None):
    scene = {}
    obj_poses = get_object_poses()
    scene['objects on the table'] = []
    for obj in obj_poses:
        scene['objects on the table'].append(
        { 
            'name': obj.name,
            'position': [round(obj.pose.position.x , 2), 
                         round(obj.pose.position.y , 2), 
                         round(obj.pose.position.z , 2)
                        ],
            'diameter': objects_info[obj.name]['diameter']
        })
    scene['table limitations'] = TABLE_HEURISTIC
    if object_in_the_gripper is not None:
        scene['object in the gripper'] = {'name': object_in_the_gripper, 'diameter': objects_info[object_in_the_gripper]['diameter']}
    return scene