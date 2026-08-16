#!/usr/bin/env python3
"""Data-collection node for the SO-100 arm.

Records observations (camera image + joint states) and actions (joint
position commands) to a ``rosbag2`` SQLite database while a human
teleoperates the arm (e.g. with ``teleop_keyboard``).

The bag is written in the Octo / Bridge-compatible format:

    /image            sensor_msgs/Image        (observation)
    /joint_states     sensor_msgs/JointState   (observation)
    /action           sensor_msgs/JointState   (action — commanded positions)

A metadata YAML file is written alongside the bag to document the
episode and the instruction string.

Usage
-----
    # Start the simulation and teleop, then in a third terminal:
    ros2 run vla_policy collect_data --ros-args \
        -p instruction:="pick up the red cube" \
        -p bag_dir:=/tmp/my_episode
"""

import os
import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.clock import Clock

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper_joint",
]


def _write_metadata_yaml(path, instruction, num_steps, fps):
    """Write a human-readable episode metadata file."""
    with open(path, "w") as f:
        f.write(f"instruction: \"{instruction}\"\n")
        f.write(f"num_steps: {num_steps}\n")
        f.write(f"fps: {fps}\n")
        f.write("observation_topics:\n")
        f.write("  - /image\n")
        f.write("  - /joint_states\n")
        f.write("action_topic: /action\n")
        f.write("joint_names:\n")
        for name in ARM_JOINTS:
            f.write(f"  - {name}\n")


class DataCollector(Node):
    def __init__(self):
        super().__init__("collect_data")
        self.set_parameters([Parameter("use_sim_time", value=True)])

        self.instruction = self.declare_parameter(
            "instruction", "pick up the object"
        ).value
        self.bag_dir = self.declare_parameter(
            "bag_dir", "/tmp/so100_episode"
        ).value
        self.fps = self.declare_parameter("fps", 10).value
        self.save_image = self.declare_parameter("save_image", True).value

        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_joints = None
        self._running = False
        self._step_count = 0

        self._bag_writer = None

        self.create_subscription(Image, "/image", self._image_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

    # ------------------------------------------------------------------ #
    # Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _image_cb(self, msg):
        with self._lock:
            self._latest_image = msg

    def _joint_cb(self, msg):
        with self._lock:
            self._latest_joints = msg

    # ------------------------------------------------------------------ #
    # ROS 2 bag writing                                                   #
    # ------------------------------------------------------------------ #

    def _init_bag(self):
        try:
            import rosbag2_py
            from rclpy.serialization import serialize_message

            self._rosbag2_py = rosbag2_py
            self._serialize_message = serialize_message

            storage_id = "sqlite3"
            conn = rosbag2_py.StorageOptions(
                uri=self.bag_dir, storage_id=storage_id
            )
            # Use CDR serialization for maximum compatibility
            converter = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            )
            self._bag_writer = rosbag2_py.SequentialWriter()
            self._bag_writer.open(conn, converter)

            # Image topic
            self._bag_writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name="/image",
                    type="sensor_msgs/msg/Image",
                    serialization_format="cdr",
                )
            )
            # Joint states topic (observations)
            self._bag_writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name="/joint_states",
                    type="sensor_msgs/msg/JointState",
                    serialization_format="cdr",
                )
            )
            # Action topic (commanded joint positions)
            self._bag_writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name="/action",
                    type="sensor_msgs/msg/JointState",
                    serialization_format="cdr",
                )
            )

            self.get_logger().info(f"Bag writer initialised at {self.bag_dir}")
            return True

        except ImportError:
            self.get_logger().fatal(
                "rosbag2_py is not installed. "
                "Install with: sudo apt install ros-jazzy-rosbag2-storage-sqlite3"
            )
            return False

    def _record_step(self, stamp):
        """Write one timestep of image + joint_states + action to the bag."""
        with self._lock:
            image = self._latest_image
            joints = self._latest_joints

        if image is None or joints is None:
            return

        ns = stamp.sec * 10**9 + stamp.nanosec

        # Image observation
        if self.save_image:
            self._bag_writer.write(
                "/image",
                self._serialize_message(image),
                ns,
            )

        # Joint state observation
        self._bag_writer.write(
            "/joint_states",
            self._serialize_message(joints),
            ns,
        )

        # Action = commanded joint positions (same as current state in
        # teleop mode; for autonomous mode this would be the policy output).
        action = JointState()
        action.header.stamp = stamp
        action.name = list(ARM_JOINTS)
        for name in ARM_JOINTS:
            if name in joints.name:
                idx = joints.name.index(name)
                action.position.append(joints.position[idx])
            else:
                action.position.append(0.0)
        action.velocity = []
        action.effort = []

        self._bag_writer.write(
            "/action",
            self._serialize_message(action),
            ns,
        )

        self._step_count += 1

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #

    def start(self):
        if not self._init_bag():
            return

        self._running = True
        self.get_logger().info(
            f"Recording episode (instruction: '{self.instruction}') "
            f"at {self.fps} FPS to {self.bag_dir}"
        )
        self.get_logger().info("Press Ctrl-C to stop recording.")

        period = 1.0 / self.fps
        try:
            while rclpy.ok() and self._running:
                rclpy.spin_once(self, timeout_sec=0.01)
                stamp = self.get_clock().now().to_msg()
                self._record_step(stamp)
                time.sleep(period)
        except KeyboardInterrupt:
            pass
        finally:
            self._finish()

    def _finish(self):
        self._running = False
        if self._bag_writer is not None:
            self._bag_writer.close()
            self._bag_writer = None

        # Write metadata
        meta_path = os.path.join(self.bag_dir, "episode.yaml")
        _write_metadata_yaml(
            meta_path, self.instruction, self._step_count, self.fps
        )
        self.get_logger().info(
            f"Recorded {self._step_count} steps -> {self.bag_dir}"
        )


def main():
    rclpy.init()
    node = DataCollector()
    try:
        node.start()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
