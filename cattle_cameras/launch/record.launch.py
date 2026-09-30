import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZED_SERIAL = "48750829"
CAMERA_NAME = "zed_0"

# Absolute directory for bags
BAG_DIRECTORY = os.path.expanduser("~/zed_bags")

# Topics published by the current ZED ROS 2 wrapper
ZED_TOPIC_SUFFIXES = (
    "rgb/color/rect/image/compressed",
    "rgb/color/rect/camera_info",

    "depth/depth_registered/compressedDepth",
    "depth/depth_registered/camera_info",


    "status/health",
    "status/heartbeat",
)


# ---------------------------------------------------------------------------
# ZED
# ---------------------------------------------------------------------------

def zed_node(serial, camera_name):
    """Launch a single ZED X camera by serial number."""

    zed_launch = os.path.join(
        get_package_share_directory("zed_wrapper"),
        "launch",
        "zed_camera.launch.py",
    )

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_launch),
        launch_arguments={
            "camera_model": "zedx",
            "serial_number": str(serial),
            "camera_name": camera_name,
        }.items(),
    )


# ---------------------------------------------------------------------------
# Rosbag
# ---------------------------------------------------------------------------

def bag_recorder(camera_name):
    """Create the rosbag recording process."""

    os.makedirs(BAG_DIRECTORY, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    bag_name = os.path.join(
        BAG_DIRECTORY,
        f"sensor_data_{timestamp}",
    )

    topics = [
        f"/{camera_name}/zed_node/{suffix}"
        for suffix in ZED_TOPIC_SUFFIXES
    ]

    recorder = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-o",
            bag_name,
            *topics,
        ],
        output="screen",
    )

    return bag_name, recorder


# ---------------------------------------------------------------------------
# Launch description
# ---------------------------------------------------------------------------

def generate_launch_description():

    zed = zed_node(
        serial=ZED_SERIAL,
        camera_name=CAMERA_NAME,
    )

    bag_name, recorder = bag_recorder(CAMERA_NAME)

    # Give the ZED wrapper time to initialize before starting rosbag.
    delayed_recorder = TimerAction(
        period=15.0,
        actions=[
            LogInfo(
                msg=(
                    f"Starting rosbag recording:\n"
                    f"  {bag_name}"
                )
            ),
            recorder,
        ],
    )

    return LaunchDescription(
        [
            zed,
            delayed_recorder,
        ]
    )

