import os
from datetime import datetime

import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


DEFAULT_CONFIG = os.path.join(
    get_package_share_directory("cattle_cameras"),
    "config",
    "cameras.yaml",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path):
    """Load and validate the camera config yaml."""

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    cameras = config.get("cameras") or []
    if not cameras:
        raise RuntimeError(f"No cameras defined in {path}")

    default_suffixes = config.get("topic_suffixes") or []
    names = set()
    for cam in cameras:
        if "name" not in cam or "serial" not in cam:
            raise RuntimeError(f"Camera entry needs 'name' and 'serial': {cam}")
        if cam["name"] in names:
            raise RuntimeError(f"Duplicate camera name '{cam['name']}' in {path}")
        names.add(cam["name"])
        cam.setdefault("model", "zedx")
        cam.setdefault("topic_suffixes", default_suffixes)

    config["cameras"] = cameras
    return config


# ---------------------------------------------------------------------------
# ZED
# ---------------------------------------------------------------------------


def zed_node(camera):
    """Launch a single ZED camera from its config entry."""

    zed_launch = os.path.join(
        get_package_share_directory("zed_wrapper"),
        "launch",
        "zed_camera.launch.py",
    )

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_launch),
        launch_arguments={
            "camera_model": camera["model"],
            "serial_number": str(camera["serial"]),
            "camera_name": camera["name"],
        }.items(),
    )


def camera_topics(camera):
    """Full topic names to record for a camera."""

    return [
        f"/{camera['name']}/zed_node/{suffix}" for suffix in camera["topic_suffixes"]
    ]


# ---------------------------------------------------------------------------
# Rosbag
# ---------------------------------------------------------------------------


def bag_recorder(bag_directory, cameras):
    """Create one rosbag recording process covering all cameras."""

    bag_directory = os.path.expanduser(bag_directory)
    os.makedirs(bag_directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = os.path.join(bag_directory, f"sensor_data_{timestamp}")

    topics = [t for cam in cameras for t in camera_topics(cam)]

    recorder = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", bag_name, *topics],
        output="screen",
    )

    return bag_name, recorder


# ---------------------------------------------------------------------------
# Launch description
# ---------------------------------------------------------------------------


def launch_setup(context):
    config = load_config(LaunchConfiguration("cameras_config").perform(context))
    cameras = config["cameras"]

    zeds = [zed_node(cam) for cam in cameras]

    bag_name, recorder = bag_recorder(
        config.get("bag_directory", "~/zed_bags"), cameras
    )

    # Give the ZED wrappers time to initialize before starting rosbag.
    delayed_recorder = TimerAction(
        period=float(config.get("record_delay", 15.0)),
        actions=[
            LogInfo(
                msg=(
                    f"Starting rosbag recording for "
                    f"{', '.join(c['name'] for c in cameras)}:\n"
                    f"  {bag_name}"
                )
            ),
            recorder,
        ],
    )

    return [*zeds, delayed_recorder]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "cameras_config",
                default_value=DEFAULT_CONFIG,
                description="Path to the cameras yaml file",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
