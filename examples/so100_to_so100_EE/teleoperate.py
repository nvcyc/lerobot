# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
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

try:
    import rerun  # noqa: F401
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    _rerun_available = True
except ImportError:
    _rerun_available = False

FPS = 30


def main():
    parser = argparse.ArgumentParser(
        description="SO-ARM101 leader-follower teleoperation via end-effector (EE) control.",
        epilog=(
            "Examples:\n"
            "  scripts/start.sh teleop --id orangebean\n"
            "  scripts/start.sh teleop --id orangebean --follower-port /dev/ttyACM1 --leader-port /dev/ttyACM0\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--id", default="None", help="Calibration ID saved by arm.sh calibrate (e.g. orangebean). Default: None")
    parser.add_argument("--follower-port", default="/dev/ttyACM0", help="Serial port for the follower arm. Default: /dev/ttyACM0")
    parser.add_argument("--leader-port", default="/dev/ttyACM1", help="Serial port for the leader arm.   Default: /dev/ttyACM1")
    parser.add_argument("--viz", action="store_true", help="Enable rerun visualization (requires a rerun viewer to be connected, otherwise blocks the loop).")
    args = parser.parse_args()

    follower_config = SOFollowerRobotConfig(
        port=args.follower_port,
        id=args.id,
        use_degrees=True,
    )
    leader_config = SOLeaderTeleopConfig(
        port=args.leader_port,
        id=args.id,
        use_degrees=True,
    )

    # Initialize the robot and teleoperator
    follower = SOFollower(follower_config)
    leader = SOLeader(leader_config)

    # NOTE: Using the URDF from the SO-ARM100 repo mounted in the container
    follower_kinematics_solver = RobotKinematics(
        urdf_path="/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    # NOTE: Using the URDF from the SO-ARM100 repo mounted in the container
    leader_kinematics_solver = RobotKinematics(
        urdf_path="/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    # Build pipeline to convert teleop joints to EE action
    leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # build pipeline to convert EE action to robot joints
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.4, -0.4, 0.0], "max": [0.4, 0.4, 0.5]},  # Safe workspace bounds in meters
                max_ee_step_m=0.05,  # Max 5cm step per iteration (was 10cm)
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

    # Connect to the robot and teleoperator
    follower.connect()
    leader.connect()

    # Init rerun viewer (only when explicitly requested via --viz)
    use_viz = args.viz and _rerun_available
    if args.viz and not _rerun_available:
        print("Warning: --viz requested but rerun-sdk is not installed. Visualization disabled.")
    if use_viz:
        init_rerun(session_name="so100_so100_EE_teleop")

    print("Starting teleop loop... Press Ctrl+C to stop.")
    print("Timing instrumentation enabled - printing every 30 iterations")
    loop_count = 0

    try:
        while True:
            start_loop_t = time.perf_counter()

            # Get robot observation
            t0 = time.perf_counter()
            robot_obs = follower.get_observation()
            t_obs = time.perf_counter() - t0

            # Get teleop observation
            t1 = time.perf_counter()
            leader_joints_obs = leader.get_action()
            print("")
            print("leader_joints_obs:", leader_joints_obs)
            t_get_action = time.perf_counter() - t1

            # teleop joints -> teleop EE action
            t2 = time.perf_counter()
            leader_ee_act = leader_to_ee(leader_joints_obs)
            print("leader_ee_act:", leader_ee_act)
            t_teleop_proc = time.perf_counter() - t2

            # teleop EE -> robot joints
            t3 = time.perf_counter()
            print("robot_obs:", robot_obs)
            follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))
            t_robot_proc = time.perf_counter() - t3

            # Send action to robot
            t4 = time.perf_counter()
            _ = follower.send_action(follower_joints_act)
            print("Sending action to robot:", follower_joints_act)
            t_send = time.perf_counter() - t4

            # Visualize
            t5 = time.perf_counter()
            if use_viz:
                log_rerun_data(observation=leader_ee_act, action=follower_joints_act)
            t_visualize = time.perf_counter() - t5

            # Calculate sleep time
            dt_s = time.perf_counter() - start_loop_t
            sleep_time = max(1.0 / FPS - dt_s, 0.0)
            precise_sleep(sleep_time)

            # Print timing every 30 iterations (~1 second at 30 FPS)
            loop_count += 1
            if loop_count % 30 == 0:
                actual_fps = 1.0 / (dt_s + sleep_time)
                print(f"[Loop {loop_count}] Total: {dt_s*1000:.1f}ms | "
                      f"GetObs: {t_obs*1000:.1f}ms | "
                      f"GetAct: {t_get_action*1000:.1f}ms | "
                      f"TeleopProc: {t_teleop_proc*1000:.1f}ms | "
                      f"RobotProc: {t_robot_proc*1000:.1f}ms | "
                      f"Send: {t_send*1000:.1f}ms | "
                      f"Visualize: {t_visualize*1000:.1f}ms | "
                      f"Sleep: {sleep_time*1000:.1f}ms | "
                      f"FPS: {actual_fps:.1f}")
    except KeyboardInterrupt:
        print("\nStopping teleop...")
    finally:
        follower.disconnect()
        leader.disconnect()
        print("Arms disconnected.")


if __name__ == "__main__":
    main()
