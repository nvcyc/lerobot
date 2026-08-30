"""Physical camera registry for the SO-ARM101 recording and deployment setup.

Names in this module become LeRobot dataset keys (for example,
``observation.images.wrist``). Keep camera names stable once a dataset has
been recorded: trained policies use those same names during deployment.
"""

from __future__ import annotations

from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras import CameraConfig

CAMERA_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# UVC /dev/videoN values change when devices are plugged or rebooted. The
# by-id symlink is bound to this physical wrist camera and currently resolves
# to /dev/video12, so it is safe to store in the shared registry.
WRIST_CAMERA_PATH = Path(
    "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
)

ALL_CAMERA_NAMES = ("left", "right", "wrist")


def make_camera_config(names: tuple[str, ...] = ALL_CAMERA_NAMES) -> dict[str, CameraConfig]:
    """Create fresh LeRobot camera configs for the requested named streams."""
    definitions: dict[str, CameraConfig] = {
        "left": RealSenseCameraConfig(
            serial_number_or_name="244422300478",
            fps=CAMERA_FPS,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            use_depth=False,
        ),
        "right": RealSenseCameraConfig(
            serial_number_or_name="035322250292",
            fps=CAMERA_FPS,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            use_depth=False,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=WRIST_CAMERA_PATH,
            fps=CAMERA_FPS,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
        ),
    }
    unknown = set(names) - definitions.keys()
    if unknown:
        raise ValueError(f"Unknown camera name(s): {', '.join(sorted(unknown))}")
    return {name: definitions[name] for name in names}
