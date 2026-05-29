#!/usr/bin/env python3
"""
Test script to debug record_loop with teleoperation
Uses the same setup as record.py but without dataset recording
"""

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.processor.pipeline import IdentityProcessorStep
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.utils.control_utils import init_keyboard_listener

FPS = 30
CONTROL_TIME_SEC = 60  # 1 minute test

def main():
    print("=" * 70)
    print("Testing record_loop with teleoperation (no dataset)")
    print("=" * 70)

    # Robot configurations (same as record.py)
    follower_config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id="None",
        use_degrees=True
    )

    leader_config = SOLeaderTeleopConfig(
        port="/dev/ttyACM1",
        id="None",
        use_degrees=True
    )

    # Initialize robots
    follower = SOFollower(follower_config)
    leader = SOLeader(leader_config)

    # Kinematics solvers
    follower_kinematics_solver = RobotKinematics(
        urdf_path="/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    leader_kinematics_solver = RobotKinematics(
        urdf_path="/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    # Build identity processor for observations (IK needs raw joints, not EE)
    follower_observation_passthrough = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    leader_joints_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver,
                motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.4, -0.4, 0.0], "max": [0.4, 0.4, 0.5]},
                max_ee_step_m=0.05,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect hardware
    print("\nConnecting to hardware...")
    leader.connect()
    follower.connect()
    print("✓ Connected")

    # Initialize keyboard listener
    listener, events = init_keyboard_listener()

    print("\n" + "=" * 70)
    print("✓ READY TO TEST")
    print("=" * 70)
    print("\nRunning teleoperation test for 60 seconds...")
    print("Press Ctrl+C to stop early")
    print()

    try:
        # Call record_loop without dataset (just for teleoperation)
        record_loop(
            robot=follower,
            events=events,
            fps=FPS,
            teleop=leader,
            dataset=None,  # No dataset recording
            control_time_s=CONTROL_TIME_SEC,
            single_task="test",
            display_data=False,  # No visualization
            teleop_action_processor=leader_joints_to_ee,
            robot_action_processor=ee_to_follower_joints,
            robot_observation_processor=follower_observation_passthrough,
        )

        print("\n✓ Test completed successfully!")

    finally:
        print("\nDisconnecting...")
        leader.disconnect()
        follower.disconnect()
        listener.stop()
        print("✓ Disconnected")

if __name__ == "__main__":
    main()
