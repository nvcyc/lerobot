#!/usr/bin/env python
"""Debug recording setup with direct and record-loop control paths.

This script uses the same robot, leader, kinematics, and processor setup as the
recording scripts. It can run either the known-working direct teleop-style loop
or `record_loop()` without dataset writes so we can isolate behavior.
"""

import argparse
import time

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
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30
FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
URDF_PATH = "/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
EE_BOUNDS = {"min": [-0.4, -0.4, 0.0], "max": [0.4, 0.4, 0.5]}
MAX_EE_STEP_M = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run record-style setup with the direct teleop loop for debugging."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Seconds to run the debug loop. Use 0 or a negative value to run forever.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print actions but do not send commands to the follower. Direct mode only.",
    )
    parser.add_argument(
        "--mode",
        choices=("direct", "record-loop"),
        default="direct",
        help="Run the direct teleop loop or call lerobot_record.record_loop() without a dataset.",
    )
    parser.add_argument(
        "--observation-processor",
        choices=("fk", "separate-fk", "identity"),
        default="fk",
        help=(
            "Observation processor passed to record_loop mode. "
            "record_cy.py uses fk, which shares kinematics with IK."
        ),
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable Rerun visualization.",
    )
    parser.add_argument(
        "--connect-order",
        choices=("teleop", "record"),
        default="teleop",
        help="Use teleop order (follower then leader) or record order (leader then follower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run and args.mode != "direct":
        raise ValueError("--dry-run is only supported with --mode direct")

    follower_config = SOFollowerRobotConfig(
        port=FOLLOWER_PORT,
        id="None",
        use_degrees=True,
    )
    leader_config = SOLeaderTeleopConfig(
        port=LEADER_PORT,
        id="None",
        use_degrees=True,
    )

    follower = SOFollower(follower_config)
    leader = SOLeader(leader_config)

    follower_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )
    follower_observation_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )
    leader_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    leader_joints_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver,
                motor_names=list(leader.bus.motors.keys()),
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )
    follower_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    follower_joints_to_ee_separate = RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=follower_observation_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    follower_observation_passthrough = RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    ee_to_follower_joints = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        [
            EEBoundsAndSafety(
                end_effector_bounds=EE_BOUNDS,
                max_ee_step_m=MAX_EE_STEP_M,
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

    try:
        if args.connect_order == "record":
            leader.connect()
            follower.connect()
        else:
            follower.connect()
            leader.connect()

        if not args.no_rerun:
            try:
                init_rerun(session_name="record_teleop_debug")
            except Exception as exc:
                print(f"Could not initialize Rerun: {exc}")

        print("Starting record teleop debug loop...")
        print(f"Mode: {args.mode}")
        print(f"Duration: {'forever' if args.duration <= 0 else f'{args.duration:.1f}s'}")
        print(f"Dry run: {args.dry_run}")
        print(f"Connect order: {args.connect_order}")
        print("Press Ctrl+C to stop.")

        if args.mode == "record-loop":
            observation_processors = {
                "fk": follower_joints_to_ee,
                "separate-fk": follower_joints_to_ee_separate,
                "identity": follower_observation_passthrough,
            }
            observation_processor = observation_processors[args.observation_processor]
            events = {
                "exit_early": False,
                "stop_recording": False,
                "rerecord_episode": False,
            }
            record_loop(
                robot=follower,
                events=events,
                fps=FPS,
                teleop=leader,
                dataset=None,
                control_time_s=args.duration,
                single_task="Debug record loop without dataset",
                display_data=False,
                teleop_action_processor=leader_joints_to_ee,
                robot_action_processor=ee_to_follower_joints,
                robot_observation_processor=observation_processor,
            )
            return

        loop_count = 0
        start_run_t = time.perf_counter()
        while args.duration <= 0 or time.perf_counter() - start_run_t < args.duration:
            start_loop_t = time.perf_counter()

            t0 = time.perf_counter()
            robot_obs = follower.get_observation()
            t_obs = time.perf_counter() - t0

            t1 = time.perf_counter()
            leader_joints_obs = leader.get_action()
            t_get_action = time.perf_counter() - t1

            t2 = time.perf_counter()
            leader_ee_act = leader_joints_to_ee(leader_joints_obs)
            t_teleop_proc = time.perf_counter() - t2

            t3 = time.perf_counter()
            follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))
            t_robot_proc = time.perf_counter() - t3

            t4 = time.perf_counter()
            sent_action = None
            if not args.dry_run:
                sent_action = follower.send_action(follower_joints_act)
            t_send = time.perf_counter() - t4

            print("")
            print("leader_joints_obs:", leader_joints_obs)
            print("leader_ee_act:", leader_ee_act)
            print("robot_obs:", robot_obs)
            print("follower_joints_act:", follower_joints_act)
            if sent_action is not None:
                print("sent_action:", sent_action)

            if not args.no_rerun:
                try:
                    log_rerun_data(observation=leader_ee_act, action=follower_joints_act)
                except Exception as exc:
                    print(f"Could not log Rerun data: {exc}")

            dt_s = time.perf_counter() - start_loop_t
            sleep_time = max(1.0 / FPS - dt_s, 0.0)
            precise_sleep(sleep_time)

            loop_count += 1
            if loop_count % 30 == 0:
                actual_fps = 1.0 / (dt_s + sleep_time)
                print(
                    f"[Loop {loop_count}] Total: {dt_s * 1000:.1f}ms | "
                    f"GetObs: {t_obs * 1000:.1f}ms | "
                    f"GetAct: {t_get_action * 1000:.1f}ms | "
                    f"TeleopProc: {t_teleop_proc * 1000:.1f}ms | "
                    f"RobotProc: {t_robot_proc * 1000:.1f}ms | "
                    f"Send: {t_send * 1000:.1f}ms | "
                    f"Sleep: {sleep_time * 1000:.1f}ms | "
                    f"FPS: {actual_fps:.1f}"
                )

    except KeyboardInterrupt:
        print("\nStopping debug loop.")
    finally:
        if leader.is_connected:
            leader.disconnect()
        if follower.is_connected:
            follower.disconnect()


if __name__ == "__main__":
    main()
