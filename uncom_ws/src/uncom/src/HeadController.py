import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from control_msgs.msg import PointHeadAction, PointHeadGoal
import actionlib

# Constants
WINDOW_NAME = "Inside of TIAGo's head"
CAMERA_FRAME = "/xtion_rgb_optical_frame"
IMAGE_TOPIC = "/xtion/rgb/image_raw"
CAMERA_INFO_TOPIC = "/xtion/rgb/camera_info"

# Global variables
camera_intrinsics = None
latest_image_stamp = None
point_head_client = None

# ROS callback for every new image received
def image_callback(img_msg):
    global latest_image_stamp
    latest_image_stamp = img_msg.header.stamp

    bridge = CvBridge()
    cv_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    cv2.imshow(WINDOW_NAME, cv_img)
    cv2.waitKey(15)

# OpenCV callback function for mouse events on a window
def on_mouse(event, u, v, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    rospy.loginfo(f"Pixel selected ({u}, {v}) Making TIAGo look to that direction")

    point_stamped = PointStamped()
    point_stamped.header.frame_id = CAMERA_FRAME
    point_stamped.header.stamp = latest_image_stamp

    # Compute normalized coordinates of the selected pixel
    x = (u - camera_intrinsics[0, 2]) / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) / camera_intrinsics[1, 1]
    Z = 1.0  # Define an arbitrary distance
    point_stamped.point.x = x * Z
    point_stamped.point.y = y * Z
    point_stamped.point.z = Z

    # Build the action goal
    goal = PointHeadGoal()
    goal.pointing_frame = CAMERA_FRAME
    goal.pointing_axis.x = 0.0
    goal.pointing_axis.y = 0.0
    goal.pointing_axis.z = 1.0
    goal.min_duration = rospy.Duration(1.0)
    goal.max_velocity = 0.25
    goal.target = point_stamped

    point_head_client.send_goal(goal)
    rospy.sleep(0.5)

# Create a ROS action client to move TIAGo's head
def create_point_head_client():
    global point_head_client
    rospy.loginfo("Creating action client to head controller ...")

    point_head_client = actionlib.SimpleActionClient("/head_controller/point_head_action", PointHeadAction)

    max_iterations = 3
    for i in range(max_iterations):
        if point_head_client.wait_for_server(rospy.Duration(2.0)):
            rospy.loginfo("Connected to point_head_action server")
            return
        else:
            rospy.logdebug("Waiting for the point_head_action server to come up")

    raise RuntimeError("Error in create_point_head_client: head controller action server not available")

def main():
    global camera_intrinsics

    rospy.init_node('look_to_point')

    # Subscribe to image and camera info topics
    rospy.Subscriber(IMAGE_TOPIC, Image, image_callback)
    camera_info_msg = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo)
    camera_intrinsics = np.array(camera_info_msg.K).reshape(3, 3)

    # Create the action client
    create_point_head_client()

    # Set up OpenCV window and mouse callback
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    rospy.spin()

if __name__ == '__main__':
    main()
