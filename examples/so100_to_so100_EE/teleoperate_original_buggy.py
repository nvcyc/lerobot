#!/usr/bin/env python
"""
TEST SCRIPT: teleoperate.py with ORIGINAL BUGGY configuration
This replicates the exact configuration from original record.py that caused issues.

Differences from working teleoperate.py:
1. Leader pipeline: tuple format (like original record.py)
2. initial_guess_current_joints=True (like original record.py) ← SUSPECTED BUG
3. EE bounds: [-1, -1, -1] to [1, 1, 1] (like original record.py)
4. max_ee_step_m=0.10 (like original record.py)

Expected result: This should oscillate/drift like the original bug.
"""

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    print("\n" + "=" * 70)
    print("TEST: Teleoperate with ORIGINAL BUGGY configuration")
    print("=" * 70)
    print("Replicates exact config from original record.py that caused issues:")
    print("  - Tuple format for FK")
    print("  - initial_guess_current_joints=True ← SUSPECTED BUG")
    print("  - Large EE bounds")
    print("Expected: Oscillation/drift (reproducing the bug)")
    print("=" * 70 + "\n")

    # Initialize configs
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

    # Initialize hardware
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

    # Leader FK pipeline: TUPLE format (original buggy config)
    leader_to_ee = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver,
                motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # IK pipeline: ORIGINAL BUGGY settings
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},  # Original large bounds
                max_ee_step_m=0.10,  # Original 10cm max step
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=True,  # ← ORIGINAL BUGGY SETTING
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect hardware
    follower.connect()
    leader.connect()

    # Init rerun
    init_rerun(session_name="teleoperate_buggy_test")

    print("Starting teleop loop...")
    print("⚠️  EXPECTED: Oscillation or drift (this replicates the bug)")
    print("Press Ctrl+C to stop\n")

    loop_count = 0

    try:
        while True:
            start_loop_t = time.perf_counter()

            # Get observations
            robot_obs = follower.get_observation()
            leader_joints_obs = leader.get_action()

            # FK with TUPLE (original buggy way)
            leader_ee_act = leader_to_ee((leader_joints_obs, robot_obs))

            # IK with initial_guess_current_joints=True (original buggy setting)
            follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))

            # Send to robot
            _ = follower.send_action(follower_joints_act)

            # Visualize
            log_rerun_data(observation=leader_ee_act, action=follower_joints_act)

            # Timing
            dt_s = time.perf_counter() - start_loop_t
            sleep_time = max(1.0 / FPS - dt_s, 0.0)
            precise_sleep(sleep_time)

            # Print action values frequently to see drift
            loop_count += 1
            if loop_count % 10 == 0:  # Print every 10 iterations for faster feedback
                print(f"[Loop {loop_count:4d}] "
                      f"shoulder_pan: {follower_joints_act.get('shoulder_pan.pos', 0):7.2f}° | "
                      f"shoulder_lift: {follower_joints_act.get('shoulder_lift.pos', 0):7.2f}° | "
                      f"elbow_flex: {follower_joints_act.get('elbow_flex.pos', 0):7.2f}° | "
                      f"wrist_flex: {follower_joints_act.get('wrist_flex.pos', 0):7.2f}°")

    except KeyboardInterrupt:
        print("\n\nTest stopped by user")
    finally:
        print("\nDisconnecting...")
        follower.disconnect()
        leader.disconnect()
        print("✓ Disconnected\n")


if __name__ == "__main__":
    main()
