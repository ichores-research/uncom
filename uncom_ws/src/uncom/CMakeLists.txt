cmake_minimum_required(VERSION 3.5)
project(uncom)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  pal_interaction_msgs
  rospy
  std_msgs
)

## 1. ADD THIS LINE RIGHT HERE
## This tells ROS to run your setup.py
catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS actionlib pal_interaction_msgs rospy std_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

## 2. UPDATE THESE PATHS
## Ensure these point to where your executable scripts actually live
catkin_install_python(PROGRAMS
  src/uncom/media_node.py
  src/uncom/understand_ROS.py
  src/uncom/understand_ichores_pipeline.py
  src/uncom/simulation_audio.py
  src/uncom/simulation_tts.py
  src/uncom/ycb_objects.py
  src/uncom/object_detection.py
  src/uncom/pick_and_place.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## 3. ADD CONFIG AND DATA INSTALLS
## Since you have JSON and YAML files in /config and /data, 
## you MUST install them or the TIAGO won't find them during roslaunch.
install(DIRECTORY config/ data/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)