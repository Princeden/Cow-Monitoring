### Recording Setup for Cattle Monitoring Project

This is a ros package for the Cattle Monitoring Project. It launches the zed cameras and records their
compressed topic streams in a ros bag, but only while a calf is in frame. Also included is a script for turning the bag back into video.

Usage after building is `ros2 launch cattle_cameras record.launch.py`

#### Camera configuration

Cameras are defined in `config/cameras.yaml` (serial number, name, optional model and per-camera topic list). `record.launch.py` starts, for each entry, a ZED wrapper, a YOLO `detector_node` and a `recorder_manager`. The detector publishes `/<camera>/calf_present`; the recorder keeps the last `pre_roll` seconds of messages in memory and, when it turns true, starts a new bag (`<bag_directory>/<camera>_<timestamp>`) beginning with that buffer, and closes it once no calf has been seen for `stop_after` seconds. Detection settings live under `detection:` in the yaml. To use a different file:

`ros2 launch cattle_cameras record.launch.py cameras_config:=/path/to/cameras.yaml`

After editing the yaml, rebuild (or use `colcon build --symlink-install`) so the installed copy updates.

#### Running on a Jetson

- Install `ultralytics` together with the JetPack builds of PyTorch/torchvision (the pip wheels are CPU-only on aarch64).
- Export the weights to TensorRT on the Jetson itself (engines are device specific), then set `detection.weights` to the `.engine` file:

  `yolo export model=weights/box_model.pt format=engine half=True imgsz=640`

- Set `publish_debug_image: true` and run `ros2 run cattle_cameras detection_viewer --ros-args -p image_topic:=/zed_0/detector/dbg_image` to check what triggers recording.
- Lower `max_rate` if the board is overloaded (each camera runs its own detector).
