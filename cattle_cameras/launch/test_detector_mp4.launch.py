"""Feed an mp4 into the real detector_node and view its debug output live.

Runs mp4_publisher -> detector_node (unmodified detection logic, with
publish_debug_image on) -> detection_viewer, so what you see is exactly what
detector_node computes -- no reimplemented logic.

Usage:
    ros2 launch cattle_cameras test_detector_mp4.launch.py \
        video_path:=/path/to/video.mp4 \
        weights:=yolov8n.pt class_name:=person
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    image_topic = "/camera/camera/color/image_raw"
    debug_topic = "/detector_node/dbg_image"

    return LaunchDescription(
        [
            DeclareLaunchArgument("video_path", description="Path to mp4 file"),
            DeclareLaunchArgument("weights", default_value="yolov8n.pt"),
            DeclareLaunchArgument("class_name", default_value="cow"),
            DeclareLaunchArgument("conf", default_value="0.5"),
            DeclareLaunchArgument("fps", default_value="15.0"),
            DeclareLaunchArgument(
                "region", default_value="[0.33, 0.0, 0.66, 1.0]"
            ),
            DeclareLaunchArgument("loop", default_value="true"),
            Node(
                package="cattle_cameras",
                executable="mp4_publisher",
                name="mp4_publisher",
                parameters=[
                    {
                        "video_path": LaunchConfiguration("video_path"),
                        "image_topic": image_topic,
                        "fps": LaunchConfiguration("fps"),
                        "loop": LaunchConfiguration("loop"),
                    }
                ],
            ),
            Node(
                package="cattle_cameras",
                executable="detector_node",
                name="detector_node",
                parameters=[
                    {
                        "image_topic": image_topic,
                        "weights": LaunchConfiguration("weights"),
                        "class_name": LaunchConfiguration("class_name"),
                        "conf": LaunchConfiguration("conf"),
                        "region": LaunchConfiguration("region"),
                        "publish_debug_image": True,
                        "debug_image_topic": debug_topic,
                    }
                ],
            ),
            Node(
                package="cattle_cameras",
                executable="detection_viewer",
                name="detection_viewer",
                parameters=[{"image_topic": debug_topic}],
            ),
        ]
    )
