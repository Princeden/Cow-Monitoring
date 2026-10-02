### Recording Setup for Cattle Monitoring Project

This is a ros package for the Cattle Monitoring Project. It is designed to automatically detect realsense and zed cameras, and record the
compressed topic streams in a ros bag. Also included is a script for turning the bag back into video.

Usage after building is `ros2 launch cattle_cameras record.launch.py`

#### Camera configuration

Cameras are defined in `config/cameras.yaml` (serial number, name, optional model and per-camera topic list). `record.launch.py` starts a ZED wrapper for each entry and records each camera's topics into a single bag. To use a different file:

`ros2 launch cattle_cameras record.launch.py cameras_config:=/path/to/cameras.yaml`

After editing the yaml, rebuild (or use `colcon build --symlink-install`) so the installed copy updates.
