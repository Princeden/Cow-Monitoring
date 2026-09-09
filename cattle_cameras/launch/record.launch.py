import os
import re
import subprocess
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def get_zed_serials():
    """Serial numbers of all connected ZED cameras."""
    try:
        import pyzed.sl as sl

        devices = sl.Camera.get_device_list()
        return [dev.serial_number for dev in devices]
    except Exception as e:
        print(f"Failed to detect ZED cameras: {e}")
        return []


def zed_node(serial, namespace):
    """ZED launch wrapper for a single camera."""
    zed_launch = os.path.join(
        get_package_share_directory("zed_wrapper"),
        "launch",
        "zed_camera.launch.py",
    )

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_launch),
        launch_arguments={
            "camera_model": "zed2i",
            "serial_number": str(serial),
            "camera_name": namespace,
        }.items(),
    )


def get_camera_nodes():
    """Detect cameras and return launch objects, names, and active camera type."""
    cameras = []

    # Detect ZED
    zed_serials = get_zed_serials()
    zed_names = [f"zed_{i}" for i in range(len(zed_serials))]
    for serial, name in zip(zed_serials, zed_names):
        cameras.append(zed_node(serial, name))

    return cameras, zed_names


ZED_TOPIC_SUFFIXES = (
    "left/image_rect_color/compressed",
    "right/image_rect_color/compressed",
    "depth/depth_registered/compressedDepth",
    "left/camera_info",
    "right/camera_info",
    "depth/camera_info",
)


def bag_recorder(camera_names):
    topics = []

    for name in camera_names:
        topics.extend([f"/{name}/{suffix}" for suffix in ZED_TOPIC_SUFFIXES])

    bag_name = f"data/sensor_data_{datetime.now():%Y%m%d_%H%M%S}"

    action = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-s",
            "mcap",
            "-o",
            bag_name,
            *topics,
        ],
        output="screen",
    )
    return bag_name, action


def generate_launch_description():
    nodes = []
    camera_nodes, camera_names = get_camera_nodes()
    nodes.extend(camera_nodes)
    bag_name, recorder_action = bag_recorder(camera_names)

    nodes += [
        LogInfo(
            msg=f"Recording {len(camera_names)} camera(s) to '{bag_name}': "
            f"{', '.join(camera_names)}"
        ),
    ]
    nodes.append(recorder_action)

    return LaunchDescription(nodes)
