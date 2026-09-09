import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def decode_compressed_depth(msg):
    """Decode sensor_msgs/msg/CompressedImage with format '16UC1; compressedDepth'."""
    raw = np.frombuffer(msg.data[12:], dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)


def make_reader(bag_path):
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    return reader


def find_image_topics(reader):
    topics = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics}

    image_topics = [
        t for t, ty in topic_types.items() if ty == "sensor_msgs/msg/CompressedImage"
    ]

    return topic_types, image_topics


def decode_frame(topic, topic_types, data):
    msg_type = get_message(topic_types[topic])
    msg = deserialize_message(data, msg_type)

    if "depth" in topic.lower():
        return decode_compressed_depth(msg)

    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def pass1_find_time_bounds(bag_path, image_topics):
    """Find first/last timestamps for each image topic."""
    reader = make_reader(bag_path)

    first_ts = {}
    last_ts = {}

    while reader.has_next():
        topic, _, timestamp = reader.read_next()

        if topic not in image_topics:
            continue

        first_ts.setdefault(topic, timestamp)
        last_ts[topic] = timestamp

    return first_ts, last_ts


def get_depth_png_dir(topic, output_dir):
    name = topic.strip("/").replace("/", "_")

    d = output_dir / "depth_png" / name
    d.mkdir(parents=True, exist_ok=True)

    return d


def get_color_png_dir(topic, output_dir):
    name = topic.strip("/").replace("/", "_")

    d = output_dir / "color_png" / name
    d.mkdir(parents=True, exist_ok=True)

    return d


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("bag", help="Path to rosbag")

    parser.add_argument("--output", default="output_videos", help="Output directory")

    parser.add_argument(
        "--fps", type=float, default=15.0, help="Output synchronization FPS"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Find topics
    # ------------------------------------------------------------------

    reader = make_reader(args.bag)

    topic_types, image_topics = find_image_topics(reader)

    print("Found topics:")

    for t in image_topics:
        print(" ", t)

    if not image_topics:
        print("No CompressedImage topics found.")
        return

    # ------------------------------------------------------------------
    # Find synchronization window
    # ------------------------------------------------------------------

    first_ts, last_ts = pass1_find_time_bounds(args.bag, image_topics)

    missing = [t for t in image_topics if t not in first_ts]

    if missing:
        print("Skipping empty topics:", missing)

        image_topics = [t for t in image_topics if t not in missing]

    sync_start = max(first_ts[t] for t in image_topics)

    sync_end = max(last_ts[t] for t in image_topics)

    period_ns = int(1e9 / args.fps)

    print(f"Sync window: {sync_start} -> {sync_end}")

    print(f"Output FPS: {args.fps}")

    # ------------------------------------------------------------------
    # Video writers
    # ------------------------------------------------------------------

    writers = {}

    def get_writer(topic, frame):
        if topic in writers:
            return writers[topic]

        name = topic.strip("/").replace("/", "_")

        h, w = frame.shape[:2]

        if "depth" in topic.lower():
            path = depth_dir / f"{name}.mp4"

            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h), False
            )

        else:
            path = color_dir / f"{name}.mp4"

            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
            )

        writers[topic] = writer

        return writer

    def render(topic, frame):
        if "depth" in topic.lower():
            return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return frame

    # ------------------------------------------------------------------
    # Decode pass
    # ------------------------------------------------------------------

    reader = make_reader(args.bag)

    latest_frame = {}

    next_tick = sync_start

    frame_count = 0

    def emit_tick():
        nonlocal frame_count

        has_color = any(
            "depth" not in t.lower() and t in latest_frame for t in image_topics
        )

        has_depth = any(
            "depth" in t.lower() and t in latest_frame for t in image_topics
        )

        # Do not save unless both exist
        if not (has_color and has_depth):
            return

        filename = f"{frame_count:06d}.png"

        for t in image_topics:
            frame = latest_frame.get(t)

            if frame is None:
                continue

            # Save video
            vis = render(t, frame)

            get_writer(t, vis).write(vis)

            # Save PNG
            if "depth" in t.lower():
                png_dir = get_depth_png_dir(t, output_dir)

                cv2.imwrite(str(png_dir / filename), frame)

            else:
                png_dir = get_color_png_dir(t, output_dir)

                cv2.imwrite(str(png_dir / filename), frame)

        frame_count += 1

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic not in image_topics:
            continue

        frame = decode_frame(topic, topic_types, data)

        if frame is not None:
            latest_frame[topic] = frame

        while next_tick <= timestamp and next_tick <= sync_end:
            if next_tick >= sync_start:
                emit_tick()

            next_tick += period_ns

    # Flush remaining frames
    while next_tick <= sync_end:
        emit_tick()

        next_tick += period_ns

    for writer in writers.values():
        writer.release()

    print(f"Saved synchronized frames: {frame_count}")


if __name__ == "__main__":
    main()

