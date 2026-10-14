import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
# Detection-triggered recording
# ---------------------------------------------------------------------------


def detection_params(config):
    """Detector parameters from the yaml, with defaults suited to a Jetson."""

    detection = config.get("detection") or {}
    return {
        "weights": detection.get("weights", "box_model.pt"),
        # ROS can't type an empty list parameter, so [""] stands for "any class".
        "class_names": detection.get("class_names") or [""],
        "conf": float(detection.get("conf", 0.5)),
        "region": [float(v) for v in detection.get("region", [0.0, 0.0, 1.0, 1.0])],
        "start_frames": int(detection.get("start_frames", 3)),
        "stop_after": float(detection.get("stop_after", 10.0)),
        "max_rate": float(detection.get("max_rate", 5.0)),
        "imgsz": int(detection.get("imgsz", 640)),
        "device": str(detection.get("device", "")),
        "half": bool(detection.get("half", True)),
        "publish_debug_image": bool(detection.get("publish_debug_image", False)),
    }


def camera_pipeline(camera, config):
    """Detector + recorder pair: record this camera only while a calf is in view."""

    name = camera["name"]
    trigger_topic = f"/{name}/calf_present"
    image_suffix = config.get("detection_image_suffix", "rgb/color/rect/image")
    image_topic = f"/{name}/zed_node/{image_suffix}"

    detector = Node(
        package="cattle_cameras",
        executable="detector_node",
        name=f"{name}_detector",
        output="screen",
        parameters=[
            {
                **detection_params(config),
                "image_topic": image_topic,
                "trigger_topic": trigger_topic,
                "debug_image_topic": f"/{name}/detector/dbg_image",
            }
        ],
    )

    recorder = Node(
        package="cattle_cameras",
        executable="recorder_manager",
        name=f"{name}_recorder",
        output="screen",
        parameters=[
            {
                "trigger_topic": trigger_topic,
                "topics": camera_topics(camera),
                "bag_output_dir": config.get("bag_directory", "~/zed_bags"),
                "bag_prefix": name,
                "pre_roll": float(config.get("pre_roll", 5.0)),
            }
        ],
    )

    return [detector, recorder]


# ---------------------------------------------------------------------------
# Launch description
# ---------------------------------------------------------------------------


def launch_setup(context):
    config = load_config(LaunchConfiguration("cameras_config").perform(context))
    cameras = config["cameras"]

    actions = []
    for cam in cameras:
        actions.append(zed_node(cam))
        actions.extend(camera_pipeline(cam, config))

    actions.append(
        LogInfo(
            msg=(
                f"Recording {', '.join(c['name'] for c in cameras)} when a calf is "
                f"detected, into {config.get('bag_directory', '~/zed_bags')}"
            )
        )
    )
    return actions


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
