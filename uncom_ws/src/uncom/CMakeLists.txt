cmake_minimum_required(VERSION 3.5)
project(uncom)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  pal_interaction_msgs
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib pal_interaction_msgs rospy std_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

#############
## Install ##
#############

## Install Python scripts
catkin_install_python(PROGRAMS
  src/media_node.py
  src/understand_ROS.py
  src/simulation_audio.py
  src/arm_move.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)