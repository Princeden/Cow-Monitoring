import os
import signal
import subprocess
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class RecorderManagerNode(Node):
    def __init__(self):
        super().__init__("recorder_manager_node")

        self.declare_parameter("trigger_topic", "/recording_trigger")
        self.declare_parameter("topics", ["/camera/camera/color/image_raw"])
        self.declare_parameter("bag_output_dir", "./bags")
        self.declare_parameter("bag_prefix", "clip")

        self.trigger_topic = self.get_parameter("trigger_topic").value
        self.topics = self.get_parameter("topics").value
        self.bag_output_dir = self.get_parameter("bag_output_dir").value
        self.bag_prefix = self.get_parameter("bag_prefix").value

        os.makedirs(self.bag_output_dir, exist_ok=True)

        self.recording = False
        self.process = None

        self.sub = self.create_subscription(
            Bool, self.trigger_topic, self.trigger_callback, 10
        )

        self.get_logger().info(
            f"Recorder manager ready. trigger_topic={self.trigger_topic} "
            f"topics={self.topics} output_dir={self.bag_output_dir}"
        )

    def trigger_callback(self, msg: Bool):
        if msg.data and not self.recording:
            self.start_recording()
        elif not msg.data and self.recording:
            self.stop_recording()

    def start_recording(self):
        bag_path = os.path.join(
            self.bag_output_dir, f"{self.bag_prefix}_{int(time.time())}"
        )
        cmd = ["ros2", "bag", "record", "-o", bag_path] + list(self.topics)
        self.get_logger().info(f"Starting bag record: {' '.join(cmd)}")

        # New process group so we can send SIGINT to the whole group on stop,
        # which lets ros2 bag record shut down and finalize the bag cleanly.
        self.process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
        )
        self.recording = True

    def stop_recording(self):
        if self.process is None:
            self.recording = False
            return

        self.get_logger().info("Stopping bag record")
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.get_logger().warn("ros2 bag record did not exit in time, killing it")
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        finally:
            self.process = None
            self.recording = False

    def destroy_node(self):
        if self.recording:
            self.stop_recording()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
