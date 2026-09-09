import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from collections import deque

from ultralytics import YOLO


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")

        # --- Parameters ---
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("trigger_topic", "/recording_trigger")
        self.declare_parameter("weights", "yolov8n.pt")
        self.declare_parameter("class_name", "cow")
        self.declare_parameter("conf", 0.5)
        # region as fractions of frame width/height: [x1, y1, x2, y2]
        self.declare_parameter("region", [0.33, 0.0, 0.66, 1.0])
        self.declare_parameter("debounce_frames", 8)
        self.declare_parameter("publish_only_on_change", False)
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("debug_image_topic", "~/dbg_image")

        self.image_topic = self.get_parameter("image_topic").value
        self.trigger_topic = self.get_parameter("trigger_topic").value
        weights = self.get_parameter("weights").value
        self.class_name = self.get_parameter("class_name").value
        self.conf = self.get_parameter("conf").value
        self.region = self.get_parameter("region").value
        debounce_frames = self.get_parameter("debounce_frames").value
        self.publish_only_on_change = self.get_parameter("publish_only_on_change").value
        self.publish_debug_image = self.get_parameter("publish_debug_image").value
        debug_image_topic = self.get_parameter("debug_image_topic").value

        self.get_logger().info(f"Loading YOLO weights: {weights}")
        self.model = YOLO(weights)

        class_ids = [
            i for i, name in self.model.names.items() if name == self.class_name
        ]
        if not class_ids:
            raise ValueError(
                f"Class '{self.class_name}' not found in model classes: {self.model.names}"
            )
        self.target_class_id = class_ids[0]

        self.bridge = CvBridge()
        self.presence_history = deque(maxlen=debounce_frames)
        self.last_published_state = None

        self.sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(Bool, self.trigger_topic, 10)

        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(Image, debug_image_topic, 10)

        self.get_logger().info(
            f"Detector ready. image_topic={self.image_topic} "
            f"class={self.class_name} region={self.region} "
            f"debounce_frames={debounce_frames}"
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

    def draw_debug_box(self, frame, box, xyxy, in_region):
        color = (0, 0, 255) if in_region else (0, 255, 0)
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{self.model.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
        cv2.putText(
            frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    @staticmethod
    def draw_debug_region(frame, region_px, object_in_region):
        rx1, ry1, rx2, ry2 = region_px
        color = (0, 0, 255) if object_in_region else (255, 255, 255)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
        cv2.putText(
            frame, f"trigger={object_in_region}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width = frame.shape[:2]
        region_px = self.region_in_pixels(width, height)

        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]

        object_in_region = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            in_region = cls_id == self.target_class_id and self.box_center_in_region(
                xyxy, region_px
            )
            object_in_region = object_in_region or in_region

            if self.debug_pub is not None:
                self.draw_debug_box(frame, box, xyxy, in_region)

        if self.debug_pub is not None:
            self.draw_debug_region(frame, region_px, object_in_region)
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

        self.presence_history.append(object_in_region)

        if len(self.presence_history) < self.presence_history.maxlen:
            return  # not enough history yet to make a stable decision

        should_trigger = None
        if all(self.presence_history):
            should_trigger = True
        elif not any(self.presence_history):
            should_trigger = False

        if should_trigger is None:
            return  # mixed window, hold current state

        if self.publish_only_on_change and should_trigger == self.last_published_state:
            return

        self.last_published_state = should_trigger
        self.pub.publish(Bool(data=should_trigger))


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
