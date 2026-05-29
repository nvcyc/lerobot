#!/usr/bin/env python
"""
Record candy-picking demonstrations with SO-ARM100 leader-follower teleoperation.

Hardware Setup:
  - Follower: /dev/ttyACM0 (robot doing the task)
  - Leader: /dev/ttyACM1 (human controls this one)
  - Camera: Intel RealSense D455 (serial: 244422300478)

Usage:
  cd /workspace/lerobot/examples/so100_to_so100_EE
  python record_candy_picking.py

Controls during recording:
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

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.core import TransitionKey
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
from lerobot.utils.visualization_utils import init_rerun

# ============================================================================
# CONFIGURATION - Customize these values
# ============================================================================

# Dataset configuration
NUM_EPISODES = 50  # Number of demonstrations to collect
FPS = 30  # Control frequency (Hz)
EPISODE_TIME_SEC = 30  # Max time per episode (seconds)
RESET_TIME_SEC = 10  # Time to reset environment between episodes
TASK_DESCRIPTION = "Pick colored candy and place in front of person"
HF_REPO_ID = "local/candy-picking-relative-v1"  # Local storage (no upload)

# Hardware ports
FOLLOWER_PORT = "/dev/ttyACM0"  # Robot arm doing the task
LEADER_PORT = "/dev/ttyACM1"    # Arm you control by hand

# Camera configuration - Two D455 side views (RGB only)
CAMERA_LEFT_SERIAL = "244422300478"   # D455 Camera 1
CAMERA_RIGHT_SERIAL = "035322250292"  # D455 Camera 2
CAMERA_FPS = 30
CAMERA_WIDTH = 640   # 640x480 for speed, or 1280x720 for quality
CAMERA_HEIGHT = 480
USE_DEPTH = False    # RGB-only (no depth)

# Note: Both D455 cameras on USB 3.2 - full bandwidth available!
# If left/right cameras are swapped in your physical setup,
# just swap the serial numbers above!

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
    return parser.parse_args()


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
def terminal_recording_controls(events: dict):
    """Read single-key controls from the terminal without relying on pynput/X11."""
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
            if key == "s":
                print("\nEnding episode and saving...")
                events["exit_early"] = True
            elif key == "r":
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
    dataset_root = args.dataset_location / HF_REPO_ID if args.dataset_location is not None else None

    print("\n" + "=" * 60)
    print("SO-ARM100 Candy-Picking Data Collection")
    print("=" * 60)
    print()
    print(f"Target dataset: {HF_REPO_ID}")
    print(f"Dataset location: {dataset_root if dataset_root is not None else 'LeRobot default'}")
    print(f"Episodes to record: {NUM_EPISODES}")
    print(f"Episode duration: {EPISODE_TIME_SEC}s")
    print(f"FPS: {FPS}")
    print()

    # Create dual camera configuration (both cameras working!)
    camera_config = {
        "left": RealSenseCameraConfig(
            serial_number_or_name=CAMERA_LEFT_SERIAL,  # D435i
            fps=CAMERA_FPS,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            use_depth=USE_DEPTH,  # RGB-only
        ),
        "right": RealSenseCameraConfig(
            serial_number_or_name=CAMERA_RIGHT_SERIAL,  # D435
            fps=CAMERA_FPS,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            use_depth=USE_DEPTH,  # RGB-only
        )
    }

    # Create follower configuration (with camera)
    follower_config = SOFollowerRobotConfig(
        port=FOLLOWER_PORT,
        id="None",  # Uses None.json calibration
        use_degrees=True,  # CRITICAL: FK/IK expects degrees!
        cameras=camera_config,  # Attach camera to follower
    )

    # Create leader configuration
    leader_config = SOLeaderTeleopConfig(
        port=LEADER_PORT,
        id="None",  # Uses None.json calibration
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
    print(f"\nCreating dataset: {HF_REPO_ID}")
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
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
    }

    # Connect to Rerun viewer on remote desktop
    # Note: Change the IP address to your desktop's IP if different
    DESKTOP_IP = "192.168.88.101"  # ← CHANGE THIS to your desktop's IP address
    print(f"\nConnecting to Rerun viewer at {DESKTOP_IP}:9876...")
    try:
        init_rerun(session_name="candy_picking_recording", ip=DESKTOP_IP, port=9876)
        print("✓ Connected to Rerun viewer!")
    except Exception as e:
        print(f"⚠️  Could not connect to Rerun: {e}")
        print("   Recording will continue without visualization.")

    print("\n" + "=" * 60)
    print("✓ READY TO RECORD")
    print("=" * 60)
    print()
    print("Check your Rerun viewer on the desktop for live visualization!")
    print()
    print("Terminal controls:")
    print("  - Press 's' to finish and save the current episode")
    print("  - Press 'r' to discard and re-record the current episode")
    print("  - Press 'q' to stop recording after the current loop")
    print()

    try:
        if not leader.is_connected or not follower.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        episode_idx = 0
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            print()
            print("=" * 60)
            print(f"🎬 Recording episode {episode_idx + 1} of {NUM_EPISODES}")
            print("=" * 60)
            print()
            print("Instructions:")
            print("  1. Move leader arm to demonstrate picking a candy")
            print("  2. Follower will mimic your movements")
            print("  3. Camera will record RGB-D frames")
            print("  4. Press 's' when done to save episode")
            print()
            input("Press ENTER to start recording episode...")
            print()
            print("Recording... (move the leader arm now)")
            print()

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
                    display_data=True,  # Enable real-time visualization
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

            # Reset environment between episodes
            if not events["stop_recording"] and episode_idx < NUM_EPISODES:
                print()
                print("🔄 Reset the environment")
                print(f"You have {RESET_TIME_SEC} seconds to reset the workspace")
                print("  - Return arms to starting position")
                print("  - Rearrange candies for next episode")
                print()

                leader_joints_to_relative_ee.reset()
                relative_ee_to_follower_joints.reset()

                record_loop(
                    robot=follower,
                    events=events,
                    fps=FPS,
                    teleop=leader,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                    teleop_action_processor=leader_joints_to_relative_ee,
                    robot_action_processor=relative_ee_to_follower_joints,
                    robot_observation_processor=follower_joints_to_ee,
                )

        print()
        print("=" * 60)
        print("✓ RECORDING COMPLETE!")
        print("=" * 60)
        print()
        print(f"Total episodes recorded: {episode_idx}")
        print(f"Dataset saved to: {HF_REPO_ID}")
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
