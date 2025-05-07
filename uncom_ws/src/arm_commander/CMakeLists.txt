cmake_minimum_required(VERSION 3.5)
project(arm_commander)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  pal_interaction_msgs
  rospy
  std_msgs
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  MoveToCartesian.srv
  OpenCloseGripper.srv
  SetGripperJoints.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib pal_interaction_msgs rospy std_msgs message_runtime
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
  src/ArmController.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
