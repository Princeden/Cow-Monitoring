import os
import time

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from ultralytics import YOLO


def resolve_weights(weights):
    """Allow bare filenames to refer to the package's installed weights dir."""
    if os.path.isabs(weights) or os.path.exists(weights):
        return weights
    packaged = os.path.join(
        get_package_share_directory("cattle_cameras"), "weights", weights
    )
    return packaged if os.path.exists(packaged) else weights


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")

        # --- Parameters ---
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("trigger_topic", "/recording_trigger")
        self.declare_parameter("weights", "yolov8n.pt")
        # Classes that count as "present". [""] (or empty) = any detected class.
        self.declare_parameter("class_names", ["cow"])
        self.declare_parameter("conf", 0.5)
        # region as fractions of frame width/height: [x1, y1, x2, y2]
        self.declare_parameter("region", [0.0, 0.0, 1.0, 1.0])
        # Consecutive positive frames needed before triggering.
        self.declare_parameter("start_frames", 3)
        # Seconds with no detection before the trigger is released.
        self.declare_parameter("stop_after", 10.0)
        # Max inference rate in Hz; frames arriving faster are dropped. <=0 = no limit.
        self.declare_parameter("max_rate", 5.0)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "")  # "" lets ultralytics pick (cuda:0 on Jetson)
        self.declare_parameter("half", True)
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("debug_image_topic", "~/dbg_image")

        self.image_topic = self.get_parameter("image_topic").value
        self.trigger_topic = self.get_parameter("trigger_topic").value
        weights = resolve_weights(self.get_parameter("weights").value)
        class_names = [c for c in self.get_parameter("class_names").value or [] if c]
        self.conf = self.get_parameter("conf").value
        self.region = self.get_parameter("region").value
        self.start_frames = self.get_parameter("start_frames").value
        self.stop_after = self.get_parameter("stop_after").value
        max_rate = self.get_parameter("max_rate").value
        self.min_period = 1.0 / max_rate if max_rate > 0 else 0.0
        self.imgsz = self.get_parameter("imgsz").value
        self.device = self.get_parameter("device").value or None
        self.half = self.get_parameter("half").value
        self.publish_debug_image = self.get_parameter("publish_debug_image").value
        debug_image_topic = self.get_parameter("debug_image_topic").value

        self.get_logger().info(f"Loading YOLO weights: {weights}")
        self.model = YOLO(weights)

        if class_names:
            self.target_class_ids = {
                i for i, name in self.model.names.items() if name in class_names
            }
            if not self.target_class_ids:
                raise ValueError(
                    f"None of {class_names} found in model classes: {self.model.names}"
                )
        else:
            self.target_class_ids = None  # any class

        self.bridge = CvBridge()
        self.last_inference = 0.0
        self.positive_streak = 0
        self.last_seen = None
        self.triggered = False

        # Depth 1: if inference is slower than the camera, only ever process
        # the newest frame instead of working through a backlog of stale ones.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos
        )
        self.pub = self.create_publisher(Bool, self.trigger_topic, 10)

        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(Image, debug_image_topic, 10)

        targets = class_names if class_names else "any"
        self.get_logger().info(
            f"Detector ready. image_topic={self.image_topic} "
            f"classes={targets} region={self.region} "
            f"start_frames={self.start_frames} stop_after={self.stop_after}s "
            f"max_rate={max_rate}Hz"
        )

    def region_in_pixels(self, width, height):
        x1, y1, x2, y2 = self.region
        return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

    @staticmethod
    def box_center_in_region(box_xyxy, region_px):
        bx1, by1, bx2, by2 = box_xyxy
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        rx1, ry1, rx2, ry2 = region_px
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def is_target(self, cls_id):
        return self.target_class_ids is None or cls_id in self.target_class_ids

    def draw_debug_box(self, frame, box, xyxy, in_region):
        color = (0, 0, 255) if in_region else (0, 255, 0)
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{self.model.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
        cv2.putText(
            frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    @staticmethod
    def draw_debug_region(frame, region_px, triggered):
        rx1, ry1, rx2, ry2 = region_px
        color = (0, 0, 255) if triggered else (255, 255, 255)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
        cv2.putText(
            frame, f"trigger={triggered}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

    def update_trigger(self, present, now):
        if present:
            self.positive_streak += 1
            self.last_seen = now
            if self.positive_streak >= self.start_frames:
                self.triggered = True
        else:
            self.positive_streak = 0
            if self.triggered and now - self.last_seen >= self.stop_after:
                self.triggered = False

    def image_callback(self, msg: Image):
        now = time.monotonic()
        if now - self.last_inference < self.min_period:
            return
        self.last_inference = now

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width = frame.shape[:2]
        region_px = self.region_in_pixels(width, height)

        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        present = False
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            in_region = self.is_target(int(box.cls[0])) and self.box_center_in_region(
                xyxy, region_px
            )
            present = present or in_region

            if self.debug_pub is not None:
                self.draw_debug_box(frame, box, xyxy, in_region)

        self.update_trigger(present, now)

        if self.debug_pub is not None:
            self.draw_debug_region(frame, region_px, self.triggered)
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

        # Publish every processed frame so a late-starting recorder still
        # picks up the current state.
        self.pub.publish(Bool(data=self.triggered))


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
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
