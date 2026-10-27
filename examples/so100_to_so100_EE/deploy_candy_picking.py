#!/usr/bin/env python3
"""Deploy the local candy-picking EE-delta ACT policy on an SO-ARM101."""

import argparse
import math
import queue
import threading
import time

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference
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
from camera_registry import make_camera_config
from async_chunks import AsyncChunkInference, run_async_control


URDF_PATH = "/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
STATE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos")
ACTION_KEYS = (
    "ee.delta_x", "ee.delta_y", "ee.delta_z", "ee.delta_wx", "ee.delta_wy", "ee.delta_wz",
    "ee.delta_gripper_pos",
)


def policy_camera_names(policy: ACTPolicy) -> tuple[str, ...]:
    """Return registry camera names required by the policy checkpoint."""
    prefix = "observation.images."
    names = tuple(key.removeprefix(prefix) for key in policy.config.input_features if key.startswith(prefix))
    if not names:
        raise ValueError("The policy has no visual input features.")
    return names


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--id", default="orangebean")
    parser.add_argument("--follower-port", default="/dev/ttyACM0")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 runs until stopped.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Load the policy only; do not open hardware.")
    parser.add_argument("--async-inference", action="store_true",
                        help="Experimental asynchronous ACT chunks; no flow/diffusion RTC conditioning.")
    parser.add_argument("--replan-steps", type=int, default=10,
                        help="Minimum executed steps between async requests (default: 10).")
    parser.add_argument("--max-chunk-age", type=float, default=3.0,
                        help="Maximum seconds since an async chunk's observation (default: 3).")
    parser.add_argument("--inference-timeout", type=float, default=3.0,
                        help="Maximum seconds holding without a fresh async action (default: 3).")
    parser.add_argument("--inference-threads", type=int, default=4,
                        help="CPU torch threads in async mode, leaving capacity for control (default: 4).")
    parser.add_argument("--benchmark-inference", type=int, default=0, metavar="CHUNKS",
                        help="With --dry-run --async-inference, time CHUNKS using synthetic inputs.")
    args = parser.parse_args()
    for name in ("fps", "max_chunk_age", "inference_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be nonnegative and finite")
    if args.replan_steps < 1 or args.inference_threads < 1 or args.benchmark_inference < 0:
        parser.error("Step/thread counts must be positive; benchmark count must be nonnegative")
    if args.benchmark_inference and not (args.dry_run and args.async_inference):
        parser.error("--benchmark-inference requires --dry-run --async-inference")
    return args


def make_chunk_predictor(policy, device, preprocessor, postprocessor):
    """Same observation/normalization path as sync inference, returning all deltas."""
    if policy.config.temporal_ensemble_coeff is not None or policy.config.n_obs_steps != 1:
        raise ValueError("Async ACT requires one observation and temporal ensembling disabled")

    def predict(frame):
        with torch.inference_mode():
            batch = prepare_observation_for_inference(dict(frame), device)
            actions = policy.predict_action_chunk(preprocessor(batch))
            actions = postprocessor(actions)
            if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != len(ACTION_KEYS):
                raise ValueError(f"Unexpected ACT chunk shape: {tuple(actions.shape)}")
            return actions.squeeze(0).cpu().tolist()

    return predict


def benchmark_chunks(policy, predict, args):
    """Exercise the actual worker without cameras, serial ports or motor commands."""
    frame = {"observation.state": np.zeros(7, dtype=np.float32)}
    for key, feature in policy.config.input_features.items():
        if key.startswith("observation.images."):
            channels, height, width = feature.shape
            frame[key] = np.zeros((height, width, channels), dtype=np.uint8)
    worker = AsyncChunkInference(predict, max_age_s=args.inference_timeout)
    try:
        for index in range(args.benchmark_inference):
            worker.reset()
            started = time.monotonic()
            worker.submit(frame, 0, started)
            while worker.action(0) is None:
                if time.monotonic() - started > args.inference_timeout:
                    raise RuntimeError("Synthetic inference exceeded --inference-timeout")
                time.sleep(0.005)
            print(f"Synthetic chunk {index + 1}: {worker.latency_s * 1000:.1f} ms; "
                  f"{policy.config.chunk_size} finite EE delta actions; no hardware opened.")
    finally:
        worker.close()


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

    if args.async_inference:
        torch.set_num_threads(args.inference_threads)

    policy = ACTPolicy.from_pretrained(args.policy_path).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    predict_chunk = None
    if args.async_inference:
        predict_chunk = make_chunk_predictor(policy, device, preprocessor, postprocessor)
        if args.replan_steps >= policy.config.chunk_size:
            raise ValueError("--replan-steps must be smaller than the policy chunk size")
        print(f"Async ACT: replan after {args.replan_steps} executed steps when the worker is free; "
              f"maximum observation age {args.max_chunk_age:g}s.")
    if args.dry_run:
        if args.benchmark_inference:
            benchmark_chunks(policy, predict_chunk, args)
        print(f"Loaded {args.policy_path} on {device}; no hardware opened.")
        return

    camera_names = policy_camera_names(policy)
    cameras = make_camera_config(camera_names)
    print(f"Policy camera inputs: {', '.join(camera_names)}")
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
    worker = None
    try:
        if args.async_inference:
            worker = AsyncChunkInference(
                predict_chunk, replan_steps=args.replan_steps, max_age_s=args.max_chunk_age,
            )

            def observe():
                raw = robot.get_observation()
                ee = observation_processor(raw)
                frame = {"observation.state": np.asarray([ee[k] for k in STATE_KEYS], dtype=np.float32)}
                frame.update({f"observation.images.{name}": raw[name] for name in camera_names})
                return frame, raw

            def prepare_action(row, raw):
                return action_processor((dict(zip(ACTION_KEYS, row, strict=True)), raw))

            def hold_position():
                positions = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
                robot.send_action({f"{name}.pos": value for name, value in positions.items()})

            run_async_control(
                engine=worker, events=events, observe=observe, prepare_action=prepare_action,
                send_action=robot.send_action, hold_position=hold_position,
                reset_motion=action_processor.reset,
                fps=args.fps, duration=args.duration, starvation_timeout=args.inference_timeout,
            )
            return
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
            }
            frame.update({f"observation.images.{name}": raw[name] for name in camera_names})
            action = predict_action(frame, policy, device, preprocessor, postprocessor, False)
            values = {key: float(action.squeeze(0).cpu()[i]) for i, key in enumerate(ACTION_KEYS)}
            robot.send_action(action_processor((values, raw)))
            precise_sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        try:
            robot.disconnect()
            print("Arms and cameras disconnected.")
        finally:
            if worker is not None:
                worker.close()
                print(f"Async ACT: {worker.inferences} chunks, {worker.discarded} discarded; "
                      f"last inference {worker.latency_s * 1000:.1f} ms.")


if __name__ == "__main__":
    main()
