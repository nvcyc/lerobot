#!/usr/bin/env python
"""
Record candy-picking demonstrations with SO-ARM100 leader-follower teleoperation.

Hardware Setup:
  - Follower: /dev/ttyACM0 (robot doing the task)
  - Leader: /dev/ttyACM1 (human controls this one)
  - Cameras: left/right Intel RealSense D455 + SO-ARM101 wrist UVC camera

Usage:
  cd /workspace/lerobot/examples/so100_to_so100_EE
  python record_candy_picking.py

Controls during recording:
  - Before an episode, teleoperate freely and press Enter to start recording
  - Press 's' to STOP recording current episode and save
  - Press 'r' to RE-RECORD current episode (discard and redo)
  - Press 'q' to QUIT recording session
"""

import argparse
import select
import sys
import termios
import threading
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.processor.pipeline import IdentityProcessorStep, RobotActionProcessorStep
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEEAction,
    ForwardKinematicsJointsToEEObservation,
    InverseKinematicsEEToJoints,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
# Text-to-speech not available in headless Docker - using print() instead
# from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_visualization
from camera_registry import ALL_CAMERA_NAMES, make_camera_config

# ============================================================================
# CONFIGURATION - Customize these values
# ============================================================================

# Dataset configuration
NUM_EPISODES = 50  # Number of demonstrations to collect
FPS = 30  # Control frequency (Hz)
EPISODE_TIME_SEC = 120  # Max time per episode (seconds)
TASK_DESCRIPTION = "Pick colored candy and place in front of person"
HF_REPO_ID = "local/candy-picking-relative-v1"  # Local storage (no upload)

# Hardware ports
FOLLOWER_PORT = "/dev/ttyACM0"  # Robot arm doing the task
LEADER_PORT = "/dev/ttyACM1"    # Arm you control by hand

# Camera configuration lives in camera_registry.py. The default profile
# records both side views and the UVC wrist camera in every episode.

# URDF path
URDF_PATH = "/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

# Safety bounds (meters, relative to robot base)
EE_BOUNDS = {
    "min": [-0.4, -0.4, 0.0],  # [x, y, z] minimum
    "max": [0.4, 0.4, 0.5]     # [x, y, z] maximum
}
MAX_EE_STEP_M = 0.05  # Max end-effector movement per step (5cm)

# ============================================================================
# MAIN RECORDING SCRIPT
# ============================================================================

ABSOLUTE_EE_KEYS = ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]
DELTA_EE_KEYS = ["delta_x", "delta_y", "delta_z", "delta_wx", "delta_wy", "delta_wz", "delta_gripper_pos"]
RELATIVE_EE_ACTION_DATASET_FEATURE = {
    "action": {
        "dtype": "float32",
        "shape": (len(DELTA_EE_KEYS),),
        "names": [f"ee.{key}" for key in DELTA_EE_KEYS],
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record candy-picking demonstrations with SO-ARM100 leader-follower teleoperation."
    )
    parser.add_argument(
        "--dataset-location",
        type=Path,
        default=None,
        help=(
            "Parent directory where datasets should be stored. "
            "When set, this dataset is stored under <dataset-location>/<repo-id>. "
            "When omitted, LeRobot's default cache location is used."
        ),
    )
    parser.add_argument(
        "--id",
        default="None",
        help="Calibration ID saved by arm.sh calibrate (e.g. orangebean). Default: None",
    )
    parser.add_argument(
        "--follower-port",
        default=FOLLOWER_PORT,
        help=f"Serial port for the follower arm. Default: {FOLLOWER_PORT}",
    )
    parser.add_argument(
        "--leader-port",
        default=LEADER_PORT,
        help=f"Serial port for the leader arm. Default: {LEADER_PORT}",
    )
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"Dataset repo id to record into. Default: {HF_REPO_ID}",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=NUM_EPISODES,
        help=f"Number of episodes to record. Default: {NUM_EPISODES}",
    )
    parser.add_argument(
        "--display-mode",
        choices=["foxglove", "rerun", "none"],
        default="foxglove",
        help=(
            "Live visualization while recording. 'foxglove' serves a WebSocket "
            "you can open in a browser; 'rerun' pushes to a Rerun viewer "
            "already running at --display-ip; 'none' disables it. "
            "Default: foxglove"
        ),
    )
    parser.add_argument(
        "--display-ip",
        default="192.168.88.101",
        help="Address of an existing Rerun viewer, for --display-mode rerun.",
    )
    parser.add_argument(
        "--display-port",
        type=int,
        default=None,
        help="Port for live visualization. Default: 8765 (foxglove) / 9876 (rerun).",
    )
    args = parser.parse_args()
    if args.display_port is None:
        args.display_port = 8765 if args.display_mode == "foxglove" else 9876
    return args


def _current_ee_from_observation(
    observation: RobotObservation, kinematics: RobotKinematics, motor_names: list[str]
) -> dict[str, float]:
    joint_values = np.array([float(observation[f"{name}.pos"]) for name in motor_names], dtype=float)
    transform = kinematics.forward_kinematics(joint_values)
    pos = transform[:3, 3]
    rotvec = Rotation.from_matrix(transform[:3, :3]).as_rotvec()

    return {
        "ee.x": float(pos[0]),
        "ee.y": float(pos[1]),
        "ee.z": float(pos[2]),
        "ee.wx": float(rotvec[0]),
        "ee.wy": float(rotvec[1]),
        "ee.wz": float(rotvec[2]),
        "ee.gripper_pos": float(observation["gripper.pos"]),
    }


@dataclass
class AbsoluteEEToRelativeDelta(RobotActionProcessorStep):
    """Convert an absolute EE target into deltas from the follower's current EE pose."""

    kinematics: RobotKinematics
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required to compute relative end-effector deltas.")

        current = _current_ee_from_observation(observation, self.kinematics, self.motor_names)

        current_rot = Rotation.from_rotvec([current["ee.wx"], current["ee.wy"], current["ee.wz"]])
        target_rot = Rotation.from_rotvec([action["ee.wx"], action["ee.wy"], action["ee.wz"]])
        delta_rot = target_rot * current_rot.inv()
        delta_rotvec = delta_rot.as_rotvec()

        return {
            "ee.delta_x": float(action["ee.x"] - current["ee.x"]),
            "ee.delta_y": float(action["ee.y"] - current["ee.y"]),
            "ee.delta_z": float(action["ee.z"] - current["ee.z"]),
            "ee.delta_wx": float(delta_rotvec[0]),
            "ee.delta_wy": float(delta_rotvec[1]),
            "ee.delta_wz": float(delta_rotvec[2]),
            "ee.delta_gripper_pos": float(action["ee.gripper_pos"] - current["ee.gripper_pos"]),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for key in ABSOLUTE_EE_KEYS:
            features[PipelineFeatureType.ACTION].pop(f"ee.{key}", None)
        for key in DELTA_EE_KEYS:
            features[PipelineFeatureType.ACTION][f"ee.{key}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@dataclass
class RelativeDeltaToAbsoluteEE(RobotActionProcessorStep):
    """Convert recorded/predicted EE deltas back into an absolute EE target for IK."""

    kinematics: RobotKinematics
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required to reconstruct an absolute end-effector target.")

        current = _current_ee_from_observation(observation, self.kinematics, self.motor_names)

        current_rot = Rotation.from_rotvec([current["ee.wx"], current["ee.wy"], current["ee.wz"]])
        delta_rot = Rotation.from_rotvec(
            [action["ee.delta_wx"], action["ee.delta_wy"], action["ee.delta_wz"]]
        )
        target_rotvec = (delta_rot * current_rot).as_rotvec()

        return {
            "ee.x": float(current["ee.x"] + action["ee.delta_x"]),
            "ee.y": float(current["ee.y"] + action["ee.delta_y"]),
            "ee.z": float(current["ee.z"] + action["ee.delta_z"]),
            "ee.wx": float(target_rotvec[0]),
            "ee.wy": float(target_rotvec[1]),
            "ee.wz": float(target_rotvec[2]),
            "ee.gripper_pos": float(current["ee.gripper_pos"] + action["ee.delta_gripper_pos"]),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for key in DELTA_EE_KEYS:
            features[PipelineFeatureType.ACTION].pop(f"ee.{key}", None)
        for key in ABSOLUTE_EE_KEYS:
            features[PipelineFeatureType.ACTION][f"ee.{key}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@contextmanager
def terminal_recording_controls(events: dict, *, waiting_for_start: bool = False):
    """Read episode controls, or Enter to leave the unrecorded teleop standby phase."""
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    stop = threading.Event()

    def key_loop():
        while not stop.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue

            key = sys.stdin.read(1).lower()
            if waiting_for_start and key in ("\r", "\n"):
                print("\nStarting recording now...")
                events["start_recording"] = True
                events["exit_early"] = True
            elif key == "s" and not waiting_for_start:
                print("\nEnding episode and saving...")
                events["exit_early"] = True
            elif key == "r" and not waiting_for_start:
                print("\nRe-recording episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == "q":
                print("\nStopping recording...")
                events["stop_recording"] = True
                events["exit_early"] = True

    try:
        tty.setcbreak(fd)
        thread = threading.Thread(target=key_loop, daemon=True)
        thread.start()
        yield
    finally:
        stop.set()
        thread.join(timeout=0.2)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    args = parse_args()
    repo_id = args.repo_id
    num_episodes = args.num_episodes
    dataset_root = args.dataset_location / repo_id if args.dataset_location is not None else None

    print("\n" + "=" * 60)
    print("SO-ARM100 Candy-Picking Data Collection")
    print("=" * 60)
    print()
    print(f"Target dataset: {repo_id}")
    print(f"Dataset location: {dataset_root if dataset_root is not None else 'LeRobot default'}")
    print(f"Calibration ID: {args.id}")
    print(f"Episodes to record: {num_episodes}")
    print(f"Episode duration: {EPISODE_TIME_SEC}s")
    print(f"FPS: {FPS}")
    print()

    camera_config = make_camera_config()
    print(f"Camera streams: {', '.join(ALL_CAMERA_NAMES)}")

    # Create follower configuration (with camera)
    follower_config = SOFollowerRobotConfig(
        port=args.follower_port,
        id=args.id,  # Uses <id>.json calibration
        use_degrees=True,  # CRITICAL: FK/IK expects degrees!
        cameras=camera_config,  # Attach camera to follower
    )

    # Create leader configuration
    leader_config = SOLeaderTeleopConfig(
        port=args.leader_port,
        id=args.id,  # Uses <id>.json calibration
        use_degrees=True,  # CRITICAL: FK/IK expects degrees!
    )

    # Initialize robots
    print("Initializing robots...")
    follower = SOFollower(follower_config)
    leader = SOLeader(leader_config)

    # Initialize kinematics solvers
    print(f"Loading URDF: {URDF_PATH}")
    follower_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )
    # Keep dataset FK separate from control IK because RobotKinematics mutates
    # its underlying placo robot state on each FK/IK call.
    follower_observation_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )
    follower_relative_action_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )
    follower_delta_control_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    leader_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    # Build processing pipelines
    print("Setting up processing pipelines...")

    # Pipeline: Follower joints -> EE observation (for dataset only)
    follower_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEEObservation(
                kinematics=follower_observation_kinematics_solver,
                motor_names=list(follower.bus.motors.keys())
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Pipeline: Identity processor for observations (IK needs raw joints for control)
    follower_observation_passthrough = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Pipeline: Leader joints -> absolute EE target -> relative EE delta action.
    # The dataset stores deltas from the follower's current EE pose while the
    # follower still executes the same absolute target after reconstruction.
    leader_joints_to_relative_ee = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ForwardKinematicsJointsToEEAction(
                kinematics=leader_kinematics_solver,
                motor_names=list(leader.bus.motors.keys())
            ),
            AbsoluteEEToRelativeDelta(
                kinematics=follower_relative_action_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Pipeline: relative EE delta action -> absolute EE target -> Follower joints
    relative_ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            RelativeDeltaToAbsoluteEE(
                kinematics=follower_delta_control_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
            ),
            EEBoundsAndSafety(
                end_effector_bounds=EE_BOUNDS,
                max_ee_step_m=MAX_EE_STEP_M,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=False,  # CRITICAL: Avoids IK local minima!
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Create dataset
    print(f"\nCreating dataset: {repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        fps=FPS,
        features=combine_feature_dicts(
            RELATIVE_EE_ACTION_DATASET_FEATURE,
            aggregate_pipeline_dataset_features(
                pipeline=follower_joints_to_ee,
                initial_features=create_initial_features(observation=follower.observation_features),
                use_videos=True,
            ),
        ),
        robot_type=follower.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Connect to hardware
    print("\nConnecting to hardware...")
    print("  (This may take a few seconds)")
    leader.connect()
    print("  ✓ Leader connected")
    follower.connect()
    print("  ✓ Follower connected")
    print("  ✓ Camera connected")

    # Terminal controls are handled locally because Docker may not expose an X11
    # display for LeRobot's pynput-based keyboard listener.
    listener = None
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "start_recording": False,
    }

    # Live visualization. The cameras are held by this process, so a separate
    # viewer cannot open them -- the frames have to be published from here.
    # "foxglove" serves a WebSocket anyone on the LAN can open in a browser;
    # "rerun" pushes to a Rerun viewer already running at --display-ip.
    display_mode = args.display_mode
    if display_mode != "none":
        try:
            if display_mode == "foxglove":
                init_visualization(
                    "foxglove",
                    session_name="candy_picking_recording",
                    ip="0.0.0.0",
                    port=args.display_port,
                )
                print(f"\n✓ Live camera streams at ws://<this-host>:{args.display_port}")
                print("  Open https://app.foxglove.dev and connect to that address.")
            else:
                init_visualization(
                    "rerun",
                    session_name="candy_picking_recording",
                    ip=args.display_ip,
                    port=args.display_port,
                )
                print(f"\n✓ Streaming to Rerun viewer at {args.display_ip}:{args.display_port}")
        except Exception as e:
            print(f"⚠️  Could not start live visualization: {e}")
            print("   Recording will continue without it.")
            display_mode = "none"

    print("\n" + "=" * 60)
    print("✓ READY TO RECORD")
    print("=" * 60)
    print()
    print("Terminal controls:")
    print("  - Before an episode: teleoperate freely, then press ENTER to record")
    print("  - During an episode: press 's' to finish and save")
    print("  - During an episode: press 'r' to discard and re-record")
    print("  - Press 'q' at any time to stop")
    print()

    try:
        if not leader.is_connected or not follower.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        episode_idx = 0
        while episode_idx < num_episodes and not events["stop_recording"]:
            print()
            print("=" * 60)
            print(f"🎬 Episode {episode_idx + 1} of {num_episodes}")
            print("=" * 60)
            print()
            print("Instructions:")
            print("  1. Teleoperate freely to position the arms or reset the workspace")
            print("  2. The follower mirrors the leader, but this standby motion is not recorded")
            print("  3. Press ENTER to begin recording immediately from the current pose")
            print("  4. Press 's' to save, or record for up to 2 minutes")
            print()

            # Keep the arms under teleop while waiting. Passing no dataset
            # means this phase cannot add video or actions to the episode.
            events["start_recording"] = False
            leader_joints_to_relative_ee.reset()
            relative_ee_to_follower_joints.reset()
            with terminal_recording_controls(events, waiting_for_start=True):
                record_loop(
                    robot=follower,
                    events=events,
                    fps=FPS,
                    teleop=leader,
                    control_time_s=float("inf"),
                    single_task=TASK_DESCRIPTION,
                    display_data=display_mode != "none",
                    display_mode=display_mode if display_mode != "none" else "rerun",
                    teleop_action_processor=leader_joints_to_relative_ee,
                    robot_action_processor=relative_ee_to_follower_joints,
                    robot_observation_processor=follower_joints_to_ee,
                )

            if events["stop_recording"]:
                break
            if not events["start_recording"]:
                continue

            print("Recording... (up to 2 minutes; press 's' to save)")

            leader_joints_to_relative_ee.reset()
            relative_ee_to_follower_joints.reset()

            # Main record loop
            with terminal_recording_controls(events):
                record_loop(
                    robot=follower,
                    events=events,
                    fps=FPS,
                    teleop=leader,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=display_mode != "none",
                    display_mode=display_mode if display_mode != "none" else "rerun",
                    teleop_action_processor=leader_joints_to_relative_ee,
                    robot_action_processor=relative_ee_to_follower_joints,
                    robot_observation_processor=follower_joints_to_ee,
                )

            # Handle episode completion
            if events["rerecord_episode"]:
                print("🔄 Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                leader_joints_to_relative_ee.reset()
                relative_ee_to_follower_joints.reset()
                continue

            # Save episode
            print()
            print("Saving episode...")
            dataset.save_episode()
            print(f"✓ Episode {episode_idx + 1} saved!")
            episode_idx += 1

        print()
        print("=" * 60)
        print("✓ RECORDING COMPLETE!")
        print("=" * 60)
        print()
        print(f"Total episodes recorded: {episode_idx}")
        print(f"Dataset saved to: {repo_id}")
        print()
        print("Next steps:")
        print("  1. Inspect dataset quality: python replay.py")
        print("  2. Upload to HuggingFace Hub (if not already)")
        print("  3. Start training on desktop GPU")
        print()

    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user")
    except Exception as e:
        print(f"\n\nERROR during recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if listener is not None:
            print("\nStopping keyboard listener...")
            try:
                listener.stop()
                print("✓ Keyboard listener stopped")
            except Exception:
                pass

        print("\nDisconnecting hardware...")
        try:
            follower.disconnect()
            print("✓ Follower disconnected")
        except Exception:
            pass
        try:
            leader.disconnect()
            print("✓ Leader disconnected")
        except Exception:
            pass

        print("\nFinalizing dataset...")
        try:
            dataset.finalize()
            print("✓ Dataset finalized")
        except Exception as e:
            print(f"⚠️  Could not finalize dataset: {e}")
        print()

if __name__ == "__main__":
    main()
