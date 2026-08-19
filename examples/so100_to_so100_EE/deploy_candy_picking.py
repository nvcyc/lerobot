#!/usr/bin/env python3
"""Deploy the local candy-picking EE-delta ACT policy on an SO-ARM101."""

import argparse
import queue
import threading
import time

import numpy as np
import torch

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEEObservation,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.robot_utils import precise_sleep

from record_candy_picking import RelativeDeltaToAbsoluteEE


URDF_PATH = "/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
STATE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos")
ACTION_KEYS = (
    "ee.delta_x", "ee.delta_y", "ee.delta_z", "ee.delta_wx", "ee.delta_wy", "ee.delta_wz",
    "ee.delta_gripper_pos",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--id", default="orangebean")
    parser.add_argument("--follower-port", default="/dev/ttyACM0")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 runs until stopped.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Load the policy only; do not open hardware.")
    return parser.parse_args()


def input_thread(events: queue.Queue):
    while True:
        try:
            input()
        except EOFError:
            events.put("stop")
            return
        events.put("toggle")


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Check the JetPack 7 CDI GPU device.")

    policy = ACTPolicy.from_pretrained(args.policy_path).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    if args.dry_run:
        print(f"Loaded {args.policy_path} on {device}; no hardware opened.")
        return

    cameras = {
        "left": RealSenseCameraConfig(serial_number_or_name="244422300478", fps=30, width=640, height=480),
        "right": RealSenseCameraConfig(serial_number_or_name="035322250292", fps=30, width=640, height=480),
    }
    robot = SOFollower(SOFollowerRobotConfig(
        port=args.follower_port, id=args.id, use_degrees=True, cameras=cameras,
        max_relative_target=10.0,
    ))
    names = list(robot.bus.motors.keys())
    fk = RobotKinematics(URDF_PATH, "gripper_frame_link", names)
    action_kinematics = RobotKinematics(URDF_PATH, "gripper_frame_link", names)
    ik = RobotKinematics(URDF_PATH, "gripper_frame_link", names)
    observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEEObservation(fk, names)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            RelativeDeltaToAbsoluteEE(action_kinematics, names),
            EEBoundsAndSafety({"min": [-0.4, -0.4, 0.0], "max": [0.4, 0.4, 0.5]}, max_ee_step_m=0.05),
            InverseKinematicsEEToJoints(ik, names, initial_guess_current_joints=False),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    events = queue.Queue()
    threading.Thread(target=input_thread, args=(events,), daemon=True).start()
    robot.connect()
    running = False
    started = time.monotonic()
    print("Ready. Press Enter to start; Enter again halts. Ctrl-C or EOF stops and disconnects.")
    try:
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            try:
                while True:
                    event = events.get_nowait()
                    if event == "stop": return
                    running = not running
                    print("Running." if running else "Halted.")
            except queue.Empty:
                pass
            if not running:
                time.sleep(0.05)
                continue
            tick = time.monotonic()
            raw = robot.get_observation()
            ee = observation_processor(raw)
            frame = {
                "observation.state": np.asarray([ee[k] for k in STATE_KEYS], dtype=np.float32),
                "observation.images.left": raw["left"],
                "observation.images.right": raw["right"],
            }
            action = predict_action(frame, policy, device, preprocessor, postprocessor, False)
            values = {key: float(action.squeeze(0).cpu()[i]) for i, key in enumerate(ACTION_KEYS)}
            robot.send_action(action_processor((values, raw)))
            precise_sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        robot.disconnect()
        print("Arms and cameras disconnected.")


if __name__ == "__main__":
    main()
