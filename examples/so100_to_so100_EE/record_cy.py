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


from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
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
# from lerobot.utils.utils import log_say  # Not available in headless Docker
from lerobot.utils.visualization_utils import init_rerun  # Remote Rerun visualization

NUM_EPISODES = 50  # Increase for full dataset collection
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "Pick colored candies and place them in front of person"
HF_REPO_ID = "local/candy-picking-v1"  # Local dataset (no upload)


def main():
    # Create the robot and teleoperator configurations
    # Dual D455 cameras for left/right side views (RGB only)
    camera_config = {
        "left": RealSenseCameraConfig(
            serial_number_or_name="244422300478",  # D455 Camera 1
            fps=FPS,
            width=640,
            height=480,
            use_depth=False,  # RGB-only
        ),
        "right": RealSenseCameraConfig(
            serial_number_or_name="035322250292",  # D455 Camera 2
            fps=FPS,
            width=640,
            height=480,
            use_depth=False,  # RGB-only
        )
    }
    follower_config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id="None",  # Uses None.json calibration
        # cameras=camera_config,  # Temporarily disabled to test timing
        use_degrees=True  # FK expects degrees
    )
    leader_config = SOLeaderTeleopConfig(
        port="/dev/ttyACM1",
        id="None",  # Uses None.json calibration
        use_degrees=True  # FK expects degrees
    )

    # Initialize the robot and teleoperator
    follower = SOFollower(follower_config)
    leader = SOLeader(leader_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    follower_kinematics_solver = RobotKinematics(
        urdf_path="/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )
    # Keep dataset FK separate from control IK because RobotKinematics mutates
    # its underlying placo robot state on each FK/IK call.
    follower_observation_kinematics_solver = RobotKinematics(
        urdf_path="/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    leader_kinematics_solver = RobotKinematics(
        urdf_path="/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    # Build pipeline to convert follower joints to EE observation (for dataset only)
    follower_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=follower_observation_kinematics_solver,
                motor_names=list(follower.bus.motors.keys())
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Build identity processor for observations (IK needs raw joints for control)
    follower_observation_passthrough = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Build pipeline to convert leader joints to EE action
    leader_joints_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # Build pipeline to convert EE action to follower joints
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.4, -0.4, 0.0], "max": [0.4, 0.4, 0.5]},  # Realistic workspace bounds
                max_ee_step_m=0.05,  # Max 5cm step per iteration for stability
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=False,  # CRITICAL! Avoids local minima with wrapping joints
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=combine_feature_dicts(
            # Run the feature contract of the pipelines
            # This tells you how the features would look like after the pipeline steps
            aggregate_pipeline_dataset_features(
                pipeline=leader_joints_to_ee,
                initial_features=create_initial_features(action=leader.action_features),
                use_videos=True,
            ),
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

    # Connect the robot and teleoperator
    leader.connect()
    follower.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()

    # Connect to Rerun viewer on remote desktop
    # Change DESKTOP_IP to your desktop's IP address
    DESKTOP_IP = "192.168.88.101"  # ← UPDATE THIS!
    RERUN_PORT = 9876

    print(f"\n🔗 Connecting to Rerun viewer at {DESKTOP_IP}:{RERUN_PORT}...")
    try:
        init_rerun(session_name="candy_picking_recording", ip=DESKTOP_IP, port=RERUN_PORT)
        print("✓ Connected to Rerun! Check your desktop viewer.")
    except Exception as e:
        print(f"⚠️  Could not connect to Rerun: {e}")
        print("   Recording will continue without visualization.")

    try:
        if not leader.is_connected or not follower.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("Starting record loop...")
        episode_idx = 0
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            print(f"\n🎬 Recording episode {episode_idx + 1} of {NUM_EPISODES}")

            # Main record loop
            record_loop(
                robot=follower,
                events=events,
                fps=FPS,
                teleop=leader,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=leader_joints_to_ee,
                robot_action_processor=ee_to_follower_joints,
                robot_observation_processor=follower_joints_to_ee,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]
            ):
                print("\n🔄 Reset the environment")
                record_loop(
                    robot=follower,
                    events=events,
                    fps=FPS,
                    teleop=leader,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                    teleop_action_processor=leader_joints_to_ee,
                    robot_action_processor=ee_to_follower_joints,
                    robot_observation_processor=follower_joints_to_ee,
                )

            if events["rerecord_episode"]:
                print("\n🔄 Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            episode_idx += 1

    finally:
        # Clean up
        print("\n✓ Stop recording")
        leader.disconnect()
        follower.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
