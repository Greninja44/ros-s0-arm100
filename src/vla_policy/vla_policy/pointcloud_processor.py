#!/usr/bin/env python3
"""3D point cloud processing node for the SO-100 arm.

Converts depth images to colored point clouds, performs clustering,
and computes grasp poses for detected objects.  Uses the depth camera
and RGB camera to build a complete 3D scene representation.

Subscriptions
-------------
    /image              sensor_msgs/Image       RGB camera
    /depth/image        sensor_msgs/Image       Depth image (32FC1)
    /image/camera_info  sensor_msgs/CameraInfo  Camera intrinsics

Publications
-------------
    /pointcloud                 sensor_msgs/PointCloud2    Colored point cloud
    /pointcloud/clusters        visualization_msgs/MarkerArray  Cluster markers
    /pointcloud/grasp_poses     geometry_msgs/PoseArray    Grasp candidates
    /pointcloud/object_poses    geometry_msgs/PoseArray    Object centroids

Usage
-----
    ros2 run vla_policy pointcloud_processor
"""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from std_msgs.msg import Header

try:
    import struct
    HAS_STRUCT = True
except ImportError:
    HAS_STRUCT = False


def image_to_numpy(msg):
    if msg.encoding in ("bgr8", "rgb8"):
        dtype, channels = np.uint8, 3
    elif msg.encoding == "mono8":
        dtype, channels = np.uint8, 1
    elif msg.encoding == "32FC1":
        dtype, channels = np.float32, 1
    elif msg.encoding == "16UC1":
        dtype, channels = np.uint16, 1
    else:
        return None
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(
        msg.height, msg.width, channels
    )
    return arr


class PointcloudProcessorNode(Node):
    def __init__(self):
        super().__init__("pointcloud_processor")

        self.set_parameters([Parameter("use_sim_time", value=True)])

        # ---- Parameters -----------------------------------------------------
        self.image_topic = self.declare_parameter(
            "image_topic", "/image"
        ).value
        self.depth_topic = self.declare_parameter(
            "depth_topic", "/depth/image"
        ).value
        self.min_cluster_size = self.declare_parameter(
            "min_cluster_size", 50
        ).value
        self.voxel_size = self.declare_parameter(
            "voxel_size", 0.005
        ).value
        self.grasp_reach = self.declare_parameter(
            "grasp_reach", 0.08
        ).value
        self.max_grasp_candidates = self.declare_parameter(
            "max_grasp_candidates", 10
        ).value

        # ---- Publishers -----------------------------------------------------
        self._cloud_pub = self.create_publisher(
            PointCloud2, "/pointcloud", 10
        )
        self._grasp_pub = self.create_publisher(
            PoseArray, "/pointcloud/grasp_poses", 10
        )
        self._object_pub = self.create_publisher(
            PoseArray, "/pointcloud/object_poses", 10
        )

        # ---- Subscribers ----------------------------------------------------
        self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, 10)
        self.create_subscription(
            CameraInfo, "/image/camera_info", self._camera_info_cb, 10
        )

        # ---- State ----------------------------------------------------------
        self._lock = threading.Lock()
        self._rgb = None
        self._depth = None
        self._cam_info = None

        self.create_timer(0.1, self._process)

    # ------------------------------------------------------------------ #
    # Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _image_cb(self, msg):
        img = image_to_numpy(msg)
        if img is not None:
            with self._lock:
                self._rgb = img
                self._header = msg.header

    def _depth_cb(self, msg):
        depth = image_to_numpy(msg)
        if depth is not None:
            depth = depth.astype(np.float32)
            if msg.encoding == "16UC1":
                depth = depth / 1000.0
            with self._lock:
                self._depth = depth

    def _camera_info_cb(self, msg):
        with self._lock:
            self._cam_info = msg

    # ------------------------------------------------------------------ #
    # Point cloud generation                                              #
    # ------------------------------------------------------------------ #

    def _depth_to_pointcloud(self, rgb, depth, cam_info):
        """Convert depth + RGB + intrinsics to organized point cloud."""
        h, w = depth.shape
        fx = cam_info.k[0]
        fy = cam_info.k[4]
        cx = cam_info.k[2]
        cy = cam_info.k[5]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.copy()
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Mask invalid points
        valid = (z > 0.05) & (z < 5.0) & ~np.isnan(z)
        x[~valid] = 0.0
        y[~valid] = 0.0
        z[~valid] = 0.0

        # Build XYZRGB array
        points = np.stack([x, y, z], axis=-1)
        if rgb.shape[:2] == (h, w):
            colors = rgb.reshape(-1, 3) if rgb.ndim == 3 else np.stack(
                [rgb.reshape(-1)] * 3, axis=-1
            )
        else:
            colors = np.zeros((h * w, 3), dtype=np.uint8)

        points_flat = points.reshape(-1, 3)
        valid_flat = valid.reshape(-1)
        colors_flat = colors.reshape(-1, 3).astype(np.uint32)
        rgb_packed = (
            (colors_flat[:, 0].astype(np.uint32) << 16)
            | (colors_flat[:, 1].astype(np.uint32) << 8)
            | colors_flat[:, 2].astype(np.uint32)
        )

        return points_flat, valid_flat, rgb_packed

    def _points_to_pointcloud2(self, points, colors, header):
        """Convert numpy arrays to PointCloud2 message."""
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True

        data = bytearray()
        for i in range(len(points)):
            data.extend(struct.pack("fffI",
                float(points[i, 0]),
                float(points[i, 1]),
                float(points[i, 2]),
                int(colors[i]),
            ))
        msg.data = bytes(data)

        return msg

    # ------------------------------------------------------------------ #
    # Clustering                                                          #
    # ------------------------------------------------------------------ #

    def _cluster_objects(self, points, valid):
        """Simple Euclidean clustering for object detection."""
        valid_points = points[valid]
        if len(valid_points) < self.min_cluster_size:
            return []

        # Simple grid-based clustering
        voxel_size = self.voxel_size
        voxel_indices = np.floor(valid_points / voxel_size).astype(int)

        # Use hash-based grouping
        clusters = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(i)

        # Merge nearby voxels
        object_clusters = []
        visited = set()

        sorted_keys = sorted(clusters.keys())
        for key in sorted_keys:
            if key in visited:
                continue
            cluster_indices = list(clusters[key])
            visited.add(key)

            # Merge with neighbors
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        neighbor = (key[0]+dx, key[1]+dy, key[2]+dz)
                        if neighbor in clusters and neighbor not in visited:
                            cluster_indices.extend(clusters[neighbor])
                            visited.add(neighbor)

            if len(cluster_indices) >= self.min_cluster_size:
                cluster_points = valid_points[cluster_indices]
                centroid = cluster_points.mean(axis=0)
                size = cluster_points.max(axis=0) - cluster_points.min(axis=0)

                # Filter out table/floor (z < 0.02) and very large clusters
                if centroid[2] > 0.02 and size.max() < 0.3:
                    object_clusters.append({
                        "centroid": centroid,
                        "size": size,
                        "points": cluster_points,
                        "n_points": len(cluster_indices),
                    })

        return object_clusters

    # ------------------------------------------------------------------ #
    # Grasp generation                                                    #
    # ------------------------------------------------------------------ #

    def _generate_grasp_poses(self, clusters):
        """Generate candidate grasp poses for each cluster."""
        grasps = []
        for cluster in clusters:
            centroid = cluster["centroid"]
            size = cluster["size"]
            reach = max(size[0], size[1], self.grasp_reach) / 2.0 + 0.02

            for angle in np.linspace(0, np.pi, 4):
                q = Quaternion()
                # Simple approach: rotate around z-axis
                q.z = np.sin(angle / 2)
                q.w = np.cos(angle / 2)

                pose = Pose()
                pose.position = Point(
                    x=float(centroid[0]),
                    y=float(centroid[1]),
                    z=float(centroid[2] + 0.05),
                )
                pose.orientation = q
                grasps.append(pose)

                if len(grasps) >= self.max_grasp_candidates:
                    break
            if len(grasps) >= self.max_grasp_candidates:
                break

        return grasps

    # ------------------------------------------------------------------ #
    # Processing                                                          #
    # ------------------------------------------------------------------ #

    def _process(self):
        with self._lock:
            rgb = self._rgb
            depth = self._depth
            cam_info = self._cam_info
            header = getattr(self, "_header", Header())

        if rgb is None or depth is None or cam_info is None:
            return

        # Generate point cloud
        points, valid, colors = self._depth_to_pointcloud(rgb, depth, cam_info)

        # Publish point cloud
        cloud_msg = self._points_to_pointcloud2(points, colors, header)
        self._cloud_pub.publish(cloud_msg)

        # Cluster objects
        clusters = self._cluster_objects(points, valid)

        if clusters:
            self.get_logger().info(
                f"Detected {len(clusters)} object cluster(s)"
            )

        # Publish object centroids
        obj_array = PoseArray()
        obj_array.header = header
        for cluster in clusters:
            pose = Pose()
            pose.position = Point(
                x=float(cluster["centroid"][0]),
                y=float(cluster["centroid"][1]),
                z=float(cluster["centroid"][2]),
            )
            pose.orientation = Quaternion(w=1.0)
            obj_array.poses.append(pose)
        self._object_pub.publish(obj_array)

        # Generate and publish grasp poses
        grasps = self._generate_grasp_poses(clusters)
        if grasps:
            grasp_array = PoseArray()
            grasp_array.header = header
            grasp_array.poses = grasps
            self._grasp_pub.publish(grasp_array)


def main():
    rclpy.init()
    node = PointcloudProcessorNode()
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
