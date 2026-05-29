#!/usr/bin/env python
"""
TEST SCRIPT: teleoperate.py modified to use TUPLE format for FK
This tests whether passing unused observation to FK causes issues.

Differences from working teleoperate.py:
1. Leader pipeline changed from RobotAction -> tuple[RobotAction, RobotObservation]
2. Converter changed from robot_action_to_transition -> robot_action_observation_to_transition
3. Call changed from leader_to_ee(act) -> leader_to_ee((act, obs))
4. Keep initial_guess_current_joints=False (the working setting)

If this works smoothly: The tuple format is fine, bug was the IK parameter.
If this oscillates: The tuple format itself causes issues.
"""

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
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
    print("TEST: Teleoperate with TUPLE format (unused observation to FK)")
    print("=" * 70)
    print("This tests whether passing observation to FK causes issues.")
    print("Settings: initial_guess_current_joints=False (working setting)")
    print("=" * 70 + "\n")

    # Initialize the robot and teleoperator config
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

    # Initialize the robot and teleoperator
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

    # ========================================================================
    # KEY DIFFERENCE: Using TUPLE format like original buggy record.py
    # ========================================================================
    leader_to_ee = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver,
                motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_observation_to_transition,  # ← TUPLE converter
        to_output=transition_to_robot_action,
    )

    # IK pipeline (unchanged)
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.4, -0.4, 0.0], "max": [0.4, 0.4, 0.5]},
                max_ee_step_m=0.05,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=False,  # ← Keep working setting
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to hardware
    follower.connect()
    leader.connect()

    # Init rerun viewer
    init_rerun(session_name="teleoperate_tuple_test")

    print("Starting teleop loop...")
    print("Watch for oscillation or drift when arms are still.")
    print("Press Ctrl+C to stop\n")

    loop_count = 0

    try:
        while True:
            start_loop_t = time.perf_counter()

            # Get observations
            t0 = time.perf_counter()
            robot_obs = follower.get_observation()
            t_obs = time.perf_counter() - t0

            # Get leader action
            t1 = time.perf_counter()
            leader_joints_obs = leader.get_action()
            t_get_action = time.perf_counter() - t1

            # ========================================================================
            # KEY DIFFERENCE: Passing TUPLE (leader_action, follower_observation)
            # ========================================================================
            t2 = time.perf_counter()
            leader_ee_act = leader_to_ee((leader_joints_obs, robot_obs))  # ← TUPLE!
            t_teleop_proc = time.perf_counter() - t2

            # IK (unchanged)
            t3 = time.perf_counter()
            follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))
            t_robot_proc = time.perf_counter() - t3

            # Send to robot
            t4 = time.perf_counter()
            _ = follower.send_action(follower_joints_act)
            t_send = time.perf_counter() - t4

            # Visualize
            t5 = time.perf_counter()
            log_rerun_data(observation=leader_ee_act, action=follower_joints_act)
            t_visualize = time.perf_counter() - t5

            # Timing
            dt_s = time.perf_counter() - start_loop_t
            sleep_time = max(1.0 / FPS - dt_s, 0.0)
            precise_sleep(sleep_time)

            # Print action values every 30 iterations
            loop_count += 1
            if loop_count % 30 == 0:
                actual_fps = 1.0 / (dt_s + sleep_time)
                print(f"[Loop {loop_count}] "
                      f"shoulder_pan: {follower_joints_act.get('shoulder_pan.pos', 0):.2f}° | "
                      f"shoulder_lift: {follower_joints_act.get('shoulder_lift.pos', 0):.2f}° | "
                      f"elbow_flex: {follower_joints_act.get('elbow_flex.pos', 0):.2f}° | "
                      f"FPS: {actual_fps:.1f}")

    except KeyboardInterrupt:
        print("\n\nTest stopped by user")
    finally:
        print("\nDisconnecting...")
        follower.disconnect()
        leader.disconnect()
        print("✓ Disconnected\n")


if __name__ == "__main__":
    main()
