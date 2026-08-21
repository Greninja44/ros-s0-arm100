#!/usr/bin/env python3
"""SAM 2 segmentation node for the SO-100 arm.

Uses Meta's Segment Anything Model 2 (SAM 2) to perform zero-shot
object segmentation from the overhead camera.  Given a text prompt
(e.g. "red cube"), the node detects and segments the target object,
computes its 3D centroid (from depth), and publishes grasp-ready
PoseStamped messages.

Subscriptions
-------------
    /image              sensor_msgs/Image     RGB camera
    /depth/image        sensor_msgs/Image     Depth camera (32FC1)

Publications
-------------
    /sam2/mask          sensor_msgs/Image     Binary segmentation mask
    /sam2/centroid      geometry_msgs/PoseStamped  3D centroid of target
    /sam2/detections    vision_msgs/Detection2DArray  2D detections

Usage
-----
    ros2 run vla_policy sam2_segmentation --ros-args \
        -p prompt:="red cube"

    # mock mode (no SAM 2 / GPU install required): does real color-
    # threshold detection + depth back-projection instead of running the
    # actual SAM 2 model, so it still finds and localizes whichever
    # colored object the prompt names (red/blue/green/yellow) rather
    # than returning a hardcoded position. Good enough to validate and
    # use the whole image -> mask -> 3D pose -> grasp pipeline without
    # a checkpoint; swap to the real model by leaving mock:=false once
    # SAM 2 + a checkpoint are installed (see _load_model()).
    ros2 run vla_policy sam2_segmentation --ros-args -p mock:=true
"""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header


def image_to_numpy(msg):
    if msg.encoding in ("bgr8", "rgb8"):
        dtype, channels = np.uint8, 3
    elif msg.encoding == "mono8":
        dtype, channels = np.uint8, 1
    elif msg.encoding in ("32FC1", "16UC1"):
        dtype = np.float32 if msg.encoding == "32FC1" else np.uint16
        channels = 1
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(
        msg.height, msg.width, channels
    )
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


# Simple additive/subtractive color names a prompt like "red cube" or
# "the blue block" can name -- classical substitute for a real
# open-vocabulary detector when SAM 2 / Grounding DINO aren't installed
# (see SAM2SegmentationNode._mock_detection). Each entry is a predicate
# on an (H, W, 3) uint8 RGB image returning an (H, W) bool mask.
_COLOR_DETECTORS = {
    "red": lambda img: (
        (img[:, :, 0].astype(int) - img[:, :, 1].astype(int) > 40)
        & (img[:, :, 0].astype(int) - img[:, :, 2].astype(int) > 40)
    ),
    "blue": lambda img: (
        (img[:, :, 2].astype(int) - img[:, :, 0].astype(int) > 40)
        & (img[:, :, 2].astype(int) - img[:, :, 1].astype(int) > 40)
    ),
    "green": lambda img: (
        (img[:, :, 1].astype(int) - img[:, :, 0].astype(int) > 40)
        & (img[:, :, 1].astype(int) - img[:, :, 2].astype(int) > 40)
    ),
    "yellow": lambda img: (
        (img[:, :, 0].astype(int) - img[:, :, 2].astype(int) > 40)
        & (img[:, :, 1].astype(int) - img[:, :, 2].astype(int) > 40)
        & (np.abs(img[:, :, 0].astype(int) - img[:, :, 1].astype(int)) < 40)
    ),
}


def color_from_prompt(prompt):
    """Pick the first known color name mentioned in a text prompt."""
    prompt_lower = prompt.lower()
    for name in _COLOR_DETECTORS:
        if name in prompt_lower:
            return name
    return None


def camera_optical_to_world(x_opt, y_opt, z_opt, camera_position, camera_pitch):
    """Transform a point from the camera's optical frame to world frame.

    ``(x_opt, y_opt, z_opt)`` follows the standard camera optical-frame
    convention (X right, Y down, Z forward/depth), which is what
    pinhole back-projection from a pixel + depth naturally produces.
    The camera has no URDF/TF frame of its own (it's a standalone
    sensor in the world file, not attached to the robot), so this
    applies the sensor's known static world pose directly instead of
    looking up a transform. Only pitch (rotation about world Y) is
    modeled, matching the straight-down overhead camera every world
    here uses; extend this if a world ever gives the camera roll/yaw.
    """
    # Optical frame -> the sensor link's own (X-forward/Y-left/Z-up)
    # frame, via the standard REP 103 camera static transform.
    link_x = z_opt
    link_y = -x_opt
    link_z = -y_opt

    # Link frame -> world frame via the camera's pitch about world Y.
    c, s = np.cos(camera_pitch), np.sin(camera_pitch)
    world_dx = c * link_x + s * link_z
    world_dy = link_y
    world_dz = -s * link_x + c * link_z

    return np.array([
        camera_position[0] + world_dx,
        camera_position[1] + world_dy,
        camera_position[2] + world_dz,
    ])


class SAM2SegmentationNode(Node):
    def __init__(self):
        super().__init__("sam2_segmentation")

        self.set_parameters([Parameter("use_sim_time", value=True)])

        # ---- Parameters -----------------------------------------------------
        self.prompt = self.declare_parameter(
            "prompt", "red cube"
        ).value
        self.image_topic = self.declare_parameter(
            "image_topic", "/image"
        ).value
        self.depth_topic = self.declare_parameter(
            "depth_topic", "/depth/image"
        ).value
        self.camera_info_topic = self.declare_parameter(
            "camera_info_topic", "/image/camera_info"
        ).value
        self.confidence_thresh = self.declare_parameter(
            "confidence_thresh", 0.5
        ).value
        self.mock = self.declare_parameter("mock", False).value

        # Static camera extrinsics (see worlds/pick_place_depth.world's
        # overhead_camera): position in world frame, plus its downward
        # pitch. The camera isn't part of the robot URDF, so there's no
        # TF frame for it -- this is the only way to turn a depth pixel
        # into a world-frame grasp target. Override these if the camera
        # moves or a different world is used.
        self.camera_position = [
            self.declare_parameter(f"camera_{ax}", default).value
            for ax, default in zip(("x", "y", "z"), (0.1, 0.1, 1.2))
        ]
        self.camera_pitch = self.declare_parameter(
            "camera_pitch", np.pi / 2
        ).value

        # ---- Publishers -----------------------------------------------------
        self._mask_pub = self.create_publisher(Image, "/sam2/mask", 10)
        self._centroid_pub = self.create_publisher(
            PoseStamped, "/sam2/centroid", 10
        )

        # ---- Subscribers ----------------------------------------------------
        self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, 10)
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_cb, 10
        )

        # ---- State ----------------------------------------------------------
        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_depth = None
        self._camera_info = None
        self._sam2_model = None
        self._sam2_predictor = None

        self.create_timer(0.1, self._process_frame)

    # ------------------------------------------------------------------ #
    # Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _image_cb(self, msg):
        try:
            image = image_to_numpy(msg)
        except ValueError:
            return
        with self._lock:
            self._latest_image = image
            self._image_header = msg.header

    def _depth_cb(self, msg):
        try:
            depth = image_to_numpy(msg).astype(np.float32)
            if msg.encoding == "16UC1":
                depth = depth / 1000.0
        except ValueError:
            return
        with self._lock:
            self._latest_depth = depth

    def _camera_info_cb(self, msg):
        with self._lock:
            self._camera_info = msg

    # ------------------------------------------------------------------ #
    # SAM 2 model                                                         #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            self.get_logger().fatal(
                "SAM 2 is not installed. "
                "Install with: pip install sam-2\n"
                "Or run with mock:=true to test the pipeline."
            )
            return False

        self.get_logger().info("Loading SAM 2 model ...")
        sam2_model = build_sam2(
            "sam2_hiera_l.yaml",
            "sam2_hiera_large.pt",
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        )
        self._sam2_predictor = SAM2ImagePredictor(sam2_model)
        self.get_logger().info("SAM 2 model ready.")
        return True

    def _load_grounding_dino(self):
        """Load Grounding DINO for text-prompted detection."""
        try:
            from groundingdino.util.inference import load_model, predict
            self._gdino_model = load_model(
                "GroundingDINO_SwinT_OGC.py",
                "groundingdino_swint_ogc.pth",
            )
            self._gdino_predict = predict
            return True
        except ImportError:
            self.get_logger().warn(
                "Grounding DINO not installed. "
                "Falling back to color-based detection."
            )
            return False

    # ------------------------------------------------------------------ #
    # Processing                                                          #
    # ------------------------------------------------------------------ #

    def _process_frame(self):
        with self._lock:
            image = self._latest_image
            depth = self._latest_depth
            cam_info = self._camera_info

        if image is None:
            return

        if self.mock:
            self._mock_detection(image, depth, cam_info)
            return

        if self._sam2_predictor is None:
            if not self._load_model():
                return

        import torch

        self._sam2_predictor.set_image(image)

        # Try Grounding DINO for text-prompted box detection
        boxes = self._detect_with_text(image)

        if boxes is not None and len(boxes) > 0:
            # Use the first detected box
            box = boxes[0]
            masks, scores, _ = self._sam2_predictor.predict(
                box=box,
                multimask_output=True,
            )
        else:
            # Fall back to point-prompted segmentation (center of image)
            h, w = image.shape[:2]
            point_coords = np.array([[w // 2, h // 2]])
            point_labels = np.array([1])
            masks, scores, _ = self._sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        # Select best mask
        best_idx = scores.argmax()
        mask = masks[best_idx]
        score = float(scores[best_idx])

        self.get_logger().info(
            f"SAM 2 mask score: {score:.3f} "
            f"(threshold: {self.confidence_thresh})"
        )

        if score < self.confidence_thresh:
            return

        # Publish mask
        mask_msg = Image()
        mask_msg.header = self._image_header
        mask_msg.height, mask_msg.width = mask.shape
        mask_msg.encoding = "mono8"
        mask_msg.is_bigendian = False
        mask_msg.step = mask.shape[1]
        mask_msg.data = (mask.astype(np.uint8) * 255).tobytes()
        self._mask_pub.publish(mask_msg)

        # Compute 3D centroid
        centroid = self._compute_centroid(mask, depth, cam_info)
        if centroid is not None:
            pose_msg = PoseStamped()
            pose_msg.header = self._image_header
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position = Point(
                x=float(centroid[0]),
                y=float(centroid[1]),
                z=float(centroid[2]),
            )
            pose_msg.pose.orientation = Quaternion(
                x=0.0, y=0.0, z=0.0, w=1.0
            )
            self._centroid_pub.publish(pose_msg)
            self.get_logger().info(
                f"Target centroid: ({centroid[0]:.3f}, "
                f"{centroid[1]:.3f}, {centroid[2]:.3f})"
            )

    def _detect_with_text(self, image):
        """Use Grounding DINO for text-prompted detection."""
        if not hasattr(self, "_gdino_model") or self._gdino_model is None:
            if not self._load_grounding_dino():
                return None

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)

        import torch
        boxes, logits, phrases = self._gdino_predict(
            model=self._gdino_model,
            image=pil_img,
            caption=self.prompt,
            box_threshold=0.3,
            text_threshold=0.25,
        )
        if len(boxes) == 0:
            return None

        # Convert normalized boxes to pixel coords
        h, w = image.shape[:2]
        pixel_boxes = boxes.clone()
        pixel_boxes[:, 0] *= w
        pixel_boxes[:, 1] *= h
        pixel_boxes[:, 2] *= w
        pixel_boxes[:, 3] *= h

        return pixel_boxes.numpy()

    def _compute_centroid(self, mask, depth, cam_info):
        """Compute the world-frame 3D centroid from mask + depth + intrinsics."""
        if depth is None or cam_info is None:
            return None

        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None

        cx, cy = int(xs.mean()), int(ys.mean())

        if cx >= depth.shape[1] or cy >= depth.shape[0]:
            return None

        z = depth[cy, cx]
        if z <= 0 or np.isnan(z) or z > 5.0:
            return None

        fx = cam_info.k[0]
        fy = cam_info.k[4]
        px = cam_info.k[2]
        py = cam_info.k[5]

        x_opt = (cx - px) * z / fx
        y_opt = (cy - py) * z / fy

        return camera_optical_to_world(
            x_opt, y_opt, z, self.camera_position, self.camera_pitch
        )

    def _mock_detection(self, image, depth, cam_info):
        """Classical color-threshold stand-in for the real SAM 2 + Grounding
        DINO pipeline: finds the object matching the color named in
        ``prompt`` (red/blue/green/yellow) by simple RGB thresholding,
        then back-projects its pixel centroid through the depth image
        and camera intrinsics/pose the same way the real path does.
        Runs the whole image -> mask -> 3D pose pipeline for real, just
        with a much simpler (GPU-free) detector standing in for SAM 2.
        """
        color = color_from_prompt(self.prompt)
        if color is None:
            self.get_logger().warn(
                f"No known color in prompt {self.prompt!r} "
                f"(known: {sorted(_COLOR_DETECTORS)}); "
                "no detection published."
            )
            return

        mask = _COLOR_DETECTORS[color](image)
        if not mask.any():
            return

        mask_msg = Image()
        mask_msg.header = self._image_header
        mask_msg.height, mask_msg.width = mask.shape
        mask_msg.encoding = "mono8"
        mask_msg.step = mask.shape[1]
        mask_msg.data = (mask.astype(np.uint8) * 255).tobytes()
        self._mask_pub.publish(mask_msg)

        centroid = self._compute_centroid(mask, depth, cam_info)
        if centroid is None:
            return

        pose_msg = PoseStamped()
        pose_msg.header = self._image_header
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position = Point(
            x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2])
        )
        pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self._centroid_pub.publish(pose_msg)
        self.get_logger().info(
            f"Detected {color!r} object at world "
            f"({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})"
        )


def main():
    rclpy.init()
    node = SAM2SegmentationNode()
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
