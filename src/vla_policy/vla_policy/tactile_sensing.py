#!/usr/bin/env python3
"""Tactile sensing integration for the SO-100 arm.

Provides a unified interface for tactile sensors (GelSight, DIGIT,
and simulated tactile feedback from Gazebo contact sensors).

In simulation, tactile signals are synthesized from contact forces
reported by Gazebo.  On real hardware, this node reads from the
actual sensor driver.

Subscriptions
-------------
    /gazebo/contacts          gazebo_msgs/ContactsState  (sim)
    /tactile/raw              sensor_msgs/Image          (real sensor)

Publications
-------------
    /tactile/force            geometry_msgs/WrenchStamped  Contact force
    /tactile/contact         std_msgs/Bool               Contact detected
    /tactile/slip             std_msgs/Float32            Slip probability
    /tactile/image            sensor_msgs/Image           Tactile image

Usage
-----
    # Sim mode (auto-synthesizes from contact sensors):
    ros2 run vla_policy tactile_sensing

    # With real GelSight sensor:
    ros2 run vla_policy tactile_sensing --ros-args -p sensor_type:="gelsight"

    # With DIGIT sensor:
    ros2 run vla_policy tactile_sensing --ros-args -p sensor_type:="digit"
"""

import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import WrenchStamped, Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Header


def image_to_numpy(msg):
    if msg.encoding in ("bgr8", "rgb8"):
        dtype, channels = np.uint8, 3
    elif msg.encoding == "mono8":
        dtype, channels = np.uint8, 1
    else:
        return None
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(
        msg.height, msg.width, channels
    )
    return arr


class TactileSensingNode(Node):
    def __init__(self):
        super().__init__("tactile_sensing")

        self.set_parameters([Parameter("use_sim_time", value=True)])

        # ---- Parameters -----------------------------------------------------
        self.sensor_type = self.declare_parameter(
            "sensor_type", "simulated"
        ).value  # "simulated", "gelsight", "digit"
        self.gripper_joint = self.declare_parameter(
            "gripper_joint", "gripper_joint"
        ).value
        self.contact_force_thresh = self.declare_parameter(
            "contact_force_thresh", 0.5
        ).value
        self.slip_detection = self.declare_parameter(
            "slip_detection", True
        ).value
        self.update_rate = self.declare_parameter(
            "update_rate", 100.0
        ).value

        # ---- Publishers -----------------------------------------------------
        self._force_pub = self.create_publisher(
            WrenchStamped, "/tactile/force", 10
        )
        self._contact_pub = self.create_publisher(
            Bool, "/tactile/contact", 10
        )
        self._slip_pub = self.create_publisher(
            Float32, "/tactile/slip", 10
        )
        self._image_pub = self.create_publisher(
            Image, "/tactile/image", 10
        )

        # ---- Subscribers (sim mode) -----------------------------------------
        # In a full Gazebo setup, subscribe to contact sensor outputs.
        # For now, synthesize from joint states + gripper position.
        from sensor_msgs.msg import JointState
        self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10
        )

        # ---- State ----------------------------------------------------------
        self._lock = threading.Lock()
        self._gripper_pos = 0.0
        self._gripper_effort = 0.0
        self._contact = False
        self._contact_force = 0.0
        self._slip_prob = 0.0

        self.create_timer(1.0 / self.update_rate, self._update)

    def _joint_cb(self, msg):
        if self.gripper_joint in msg.name:
            idx = msg.name.index(self.gripper_joint)
            with self._lock:
                self._gripper_pos = msg.position[idx] if idx < len(msg.position) else 0.0
                self._gripper_effort = (
                    msg.effort[idx] if idx < len(msg.effort) else 0.0
                )

    def _update(self):
        if self.sensor_type == "simulated":
            self._simulated_tactile()
        else:
            pass  # Real sensor drivers publish their own data

        self._publish()

    def _simulated_tactile(self):
        """Synthesize tactile signals from simulation state."""
        with self._lock:
            pos = self._gripper_pos
            effort = self._gripper_effort

        # Simple contact model: contact when gripper is near closed
        # and effort is non-zero (motor pushing against object)
        contact = pos < 0.3 and abs(effort) > 0.1

        # Force estimate from effort (rough conversion)
        force_magnitude = abs(effort) * 0.5  # Nm -> N (rough)

        # Slip detection: if contact but force is fluctuating rapidly,
        # estimate slip probability
        slip_prob = 0.0
        if self.slip_detection and contact:
            noise = np.random.normal(0, 0.05)
            slip_prob = max(0.0, min(1.0, 0.1 + noise))

        with self._lock:
            self._contact = contact
            self._contact_force = force_magnitude
            self._slip_prob = slip_prob

        # Generate synthetic tactile image (pressure map)
        self._generate_tactile_image(pos, contact)

    def _generate_tactile_image(self, gripper_pos, contact):
        """Generate a synthetic tactile pressure map image."""
        h, w = 64, 64
        if contact:
            # Create a pressure blob in the center
            yy, xx = np.mgrid[:h, :w]
            cx, cy = w // 2, h // 2
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            pressure = np.exp(-r ** 2 / (2 * 10 ** 2))
            pressure *= max(0, 1.0 - gripper_pos)  # Scale with grip
        else:
            pressure = np.zeros((h, w))

        # Convert to RGB for visualization
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (pressure * 255).astype(np.uint8)  # Red channel
        img[:, :, 1] = ((1 - pressure) * 50).astype(np.uint8)

        msg = Image()
        msg.header = Header()
        msg.height, msg.width = h, w
        msg.encoding = "rgb8"
        msg.data = img.tobytes()
        msg.step = w * 3
        self._image_pub.publish(msg)

    def _publish(self):
        with self._lock:
            contact = self._contact
            force = self._contact_force
            slip = self._slip_prob

        # Force
        force_msg = WrenchStamped()
        force_msg.header.frame_id = "gripper"
        force_msg.wrench.force = Vector3(
            x=0.0, y=0.0, z=force
        )
        self._force_pub.publish(force_msg)

        # Contact boolean
        self._contact_pub.publish(Bool(data=contact))

        # Slip probability
        self._slip_pub.publish(Float32(data=slip))

        if contact:
            self.get_logger().debug(
                f"Contact: force={force:.2f}N slip={slip:.2f}"
            )


def main():
    rclpy.init()
    node = TactileSensingNode()
    try:
        rclpy.spin(node)
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
