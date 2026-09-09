"""Publish frames from an mp4 file as sensor_msgs/Image, looping at a fixed rate.

Lets you feed recorded video into detector_node exactly as if it were a live
camera, so the real node's logic runs unmodified against test footage.
"""

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2


class Mp4PublisherNode(Node):
    def __init__(self):
        super().__init__("mp4_publisher")

        self.declare_parameter("video_path", "")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("loop", True)

        video_path = self.get_parameter("video_path").value
        self.image_topic = self.get_parameter("image_topic").value
        fps = self.get_parameter("fps").value
        self.loop = self.get_parameter("loop").value

        if not video_path:
            raise ValueError("video_path parameter is required")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.image_topic, 10)
        self.timer = self.create_timer(1.0 / fps, self.on_timer)

        self.get_logger().info(
            f"Publishing '{video_path}' to {self.image_topic} at {fps} fps "
            f"(loop={self.loop})"
        )

    def on_timer(self):
        ok, frame = self.cap.read()

        if not ok:
            if not self.loop:
                self.get_logger().info("Video finished, shutting down")
                self.timer.cancel()
                rclpy.shutdown()
                return

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Mp4PublisherNode()
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
