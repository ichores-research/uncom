#!/usr/bin/env python3

import rospy
import moveit_commander
import geometry_msgs.msg
import sys
import tf

def main():
    if len(sys.argv) < 7:
        rospy.loginfo(" ")
        rospy.loginfo("\tUsage:")
        rospy.loginfo(" ")
        rospy.loginfo("\trosrun your_package your_script.py x y z r p y")
        rospy.loginfo(" ")
        rospy.loginfo("\twhere the list of arguments specify the target pose of /arm_tool_link expressed in /base_footprint")
        rospy.loginfo(" ")
        return

    rospy.init_node('plan_arm_torso_ik', anonymous=False)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_arm_torso = moveit_commander.MoveGroupCommander("arm_torso")

    group_arm_torso.set_planner_id("SBLkConfigDefault")
    group_arm_torso.set_pose_reference_frame("base_footprint")


    goal_pose = geometry_msgs.msg.Pose()
    goal_pose.position.x = float(sys.argv[1])
    goal_pose.position.y = float(sys.argv[2])
    goal_pose.position.z = float(sys.argv[3])
    quat = tf.transformations.quaternion_from_euler(0,0,0)
    
    goal_pose.orientation.x = quat[0]
    goal_pose.orientation.y = quat[1] 
    goal_pose.orientation.z = quat[2] 
    goal_pose.orientation.w = quat[3] 



    group_arm_torso.set_pose_target(goal_pose)

    group_arm_torso.set_planning_time(50.0)

    group_arm_torso.set_start_state_to_current_state()
    group_arm_torso.set_max_velocity_scaling_factor(1.0)

    plan = group_arm_torso.plan()
    if not plan:
        rospy.logerr("No plan found")
        return

    start_time = rospy.Time.now()
    group_arm_torso.go(wait=True)

    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()
    rospy.spin()