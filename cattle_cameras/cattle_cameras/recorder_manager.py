"""Record topics to a bag while a trigger is true, including a pre-roll.

Every recorded topic is subscribed to all the time (as serialized bytes, so
nothing is deserialized). The last `pre_roll` seconds are kept in memory; when
the trigger turns true a bag is opened, the buffer is written into it, and
live messages follow. The clip therefore starts before the detection.
"""

import os
from collections import deque
from datetime import datetime

import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import Bool


class RecorderManagerNode(Node):
    def __init__(self):
        super().__init__("recorder_manager_node")

        self.declare_parameter("trigger_topic", "/recording_trigger")
        self.declare_parameter("topics", ["/camera/camera/color/image_raw"])
        self.declare_parameter("bag_output_dir", "./bags")
        self.declare_parameter("bag_prefix", "clip")
        # Seconds of messages before the trigger to include in each bag.
        self.declare_parameter("pre_roll", 5.0)
        self.declare_parameter("storage_id", "sqlite3")

        self.trigger_topic = self.get_parameter("trigger_topic").value
        self.topics = list(self.get_parameter("topics").value)
        self.bag_output_dir = os.path.expanduser(
            self.get_parameter("bag_output_dir").value
        )
        self.bag_prefix = self.get_parameter("bag_prefix").value
        self.pre_roll_ns = int(self.get_parameter("pre_roll").value * 1e9)
        self.storage_id = self.get_parameter("storage_id").value

        os.makedirs(self.bag_output_dir, exist_ok=True)

        self.writer = None
        self.topic_types = {}  # topic -> type string, once subscribed
        self.topic_subs = []
        self.buffer = deque()  # (topic, serialized bytes, receive time ns)

        # Best effort is compatible with both reliable and best-effort publishers.
        self.data_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Topics may not exist yet (camera still starting), so keep looking
        # until every one has been subscribed.
        self.discover_timer = self.create_timer(1.0, self.discover_topics)

        self.sub = self.create_subscription(
            Bool, self.trigger_topic, self.trigger_callback, 10
        )

        self.get_logger().info(
            f"Recorder manager ready. trigger_topic={self.trigger_topic} "
            f"topics={self.topics} output_dir={self.bag_output_dir} "
            f"pre_roll={self.pre_roll_ns / 1e9}s"
        )

    @property
    def recording(self):
        return self.writer is not None

    def discover_topics(self):
        available = dict(self.get_topic_names_and_types())
        for topic in self.topics:
            if topic in self.topic_types or topic not in available:
                continue
            msg_type = available[topic][0]
            self.topic_subs.append(
                self.create_subscription(
                    get_message(msg_type),
                    topic,
                    lambda data, topic=topic: self.on_message(topic, data),
                    self.data_qos,
                    raw=True,
                )
            )
            self.topic_types[topic] = msg_type
            if self.recording:
                self.create_bag_topic(topic)
            self.get_logger().info(f"Subscribed to {topic} [{msg_type}]")

        if len(self.topic_types) == len(self.topics):
            self.discover_timer.cancel()

    def on_message(self, topic, data):
        stamp = self.get_clock().now().nanoseconds
        if self.recording:
            self.writer.write(topic, data, stamp)
            return

        self.buffer.append((topic, data, stamp))
        while self.buffer and stamp - self.buffer[0][2] > self.pre_roll_ns:
            self.buffer.popleft()

    def trigger_callback(self, msg: Bool):
        if msg.data and not self.recording:
            self.start_recording()
        elif not msg.data and self.recording:
            self.stop_recording()

    def create_bag_topic(self, topic):
        self.writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic, type=self.topic_types[topic], serialization_format="cdr"
            )
        )

    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_path = os.path.join(self.bag_output_dir, f"{self.bag_prefix}_{timestamp}")
        self.get_logger().info(
            f"Starting recording: {bag_path} "
            f"({len(self.buffer)} pre-roll messages)"
        )

        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(
            rosbag2_py.StorageOptions(uri=bag_path, storage_id=self.storage_id),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            ),
        )
        for topic in self.topic_types:
            self.create_bag_topic(topic)

        while self.buffer:
            self.writer.write(*self.buffer.popleft())

    def stop_recording(self):
        self.get_logger().info("Stopping recording")
        # Humble's SequentialWriter has no close(); dropping the only reference
        # runs its destructor, which flushes the bag and writes metadata.yaml.
        self.writer = None

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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
