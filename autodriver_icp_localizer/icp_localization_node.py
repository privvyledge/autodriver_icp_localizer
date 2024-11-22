#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import numpy as np
import torch
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
import tf_transformations
import time
from typing import Optional, Tuple

class ICPLocalizationNode(Node):
    def __init__(self):
        super(ICPLocalizationNode, self).__init__('icp_localization_node')

        # Parameters
        self.declare_parameter('use_odom_hint', True)
        self.declare_parameter('max_icp_distance', 0.3)
        self.declare_parameter('min_points', 100)
        self.declare_parameter('reference_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('transform_timeout', 1.0)
        self.declare_parameter('guess_frame_tf', '')  # Empty string disables TF-based guess

        # Initialize device
        try:
            device_str = self.get_parameter('device').value
            # Check if CUDA is available when requested
            if device_str.startswith('cuda'):
                if not torch.cuda.is_available():
                    self.get_logger().warn('CUDA requested but not available. Falling back to CPU')
                    device_str = 'cpu'

            self.torch_device = torch.device(device_str)
            self.o3d_device = o3c.Device('CUDA:0' if device_str.startswith('cuda') else 'CPU:0')
            self.get_logger().info(f'Using PyTorch device: {self.torch_device}')
            self.get_logger().info(f'Using Open3D device: {self.o3d_device}')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize device {device_str}. Error: {str(e)}')
            self.get_logger().warn('Falling back to CPU')
            self.torch_device = torch.device('cpu')
            self.o3d_device = o3c.Device('CPU:0')

        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize subscribers
        self.scan_sub = self.create_subscription(
                LaserScan,
                'scan',
                self.scan_callback,
                10)

        self.odom_sub = self.create_subscription(
                Odometry,
                'odom',
                self.odom_callback,
                10)

        # Initialize publishers
        self.pose_pub = self.create_publisher(
                Odometry,
                'localization',
                10)

        # Initialize transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # State variables
        self.reference_cloud = None
        self.last_pose = np.eye(4)
        self.last_time = None
        self.last_velocity = np.zeros(3)
        self.last_angular_velocity = np.zeros(3)
        self.odom_hint = np.eye(4)

        # Cache for laser to base transform
        self.laser_to_base = None

    def get_transform(self, target_frame: str, source_frame: str, time: rclpy.time.Time) -> Optional[TransformStamped]:
        """Get transform between frames"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time,
                rclpy.duration.Duration(seconds=self.get_parameter('transform_timeout').value))
            return transform
        except Exception as e:
            self.get_logger().error(f'Failed to lookup transform: {str(e)}')
            return None

    def transform_to_matrix(self, transform: TransformStamped) -> np.ndarray:
        """Convert geometry_msgs Transform to 4x4 matrix"""
        trans = transform.transform.translation
        rot = transform.transform.rotation

        # Convert quaternion to rotation matrix
        quat = [rot.x, rot.y, rot.z, rot.w]
        rot_matrix = tf_transformations.quaternion_matrix(quat)

        # Set translation
        rot_matrix[0:3, 3] = [trans.x, trans.y, trans.z]

        return rot_matrix

    def get_guess_from_tf(self, timestamp: rclpy.time.Time) -> Optional[np.ndarray]:
        """Get initial guess from TF tree if enabled"""
        guess_frame = self.get_parameter('guess_frame_tf').value
        if not guess_frame:
            return None

        transform = self.get_transform(
            self.get_parameter('reference_frame').value,
            guess_frame,
            timestamp
        )

        if transform is None:
            return None

        return self.transform_to_matrix(transform)

    def laser_scan_to_point_cloud(self, scan_msg: LaserScan) -> Tuple[Optional[o3d.t.geometry.PointCloud], Optional[torch.Tensor]]:
        """Convert LaserScan to Open3D point cloud and PyTorch tensor on GPU"""
        # Get transform from laser to base if not cached
        if self.laser_to_base is None:
            transform = self.get_transform(
                self.get_parameter('base_frame').value,
                'laser_link',
                scan_msg.header.stamp)

            if transform is None:
                return None, None

            self.laser_to_base = torch.from_numpy(
                self.transform_to_matrix(transform)
            ).float().to(self.torch_device)

        # Convert scan to tensor operations
        angles = torch.arange(
            scan_msg.angle_min,
            scan_msg.angle_max + scan_msg.angle_increment,
            scan_msg.angle_increment,
            device=self.torch_device
        )

        ranges = torch.tensor(scan_msg.ranges, device=self.torch_device)
        valid = torch.logical_and(
            ranges > scan_msg.range_min,
            ranges < scan_msg.range_max
        )

        if torch.sum(valid) < self.get_parameter('min_points').value:
            self.get_logger().warn('Not enough valid points in scan')
            return None, None

        # Extract valid measurements
        valid_ranges = ranges[valid]
        valid_angles = angles[valid]

        # Convert to cartesian coordinates
        x = valid_ranges * torch.cos(valid_angles)
        y = valid_ranges * torch.sin(valid_angles)
        z = torch.zeros_like(x)

        # Stack points
        points = torch.stack([x, y, z], dim=1)

        # Convert to homogeneous coordinates
        points_h = torch.ones((points.shape[0], 4), device=self.torch_device)
        points_h[:, 0:3] = points

        # Transform to base frame
        points_base = (self.laser_to_base @ points_h.T).T[:, 0:3]

        # Create Open3D point cloud.
        # o3d_tensor = o3c.Tensor(
        #     points_base.cpu().numpy(),
        #     dtype=o3c.Dtype.Float32,
        #     device=self.o3d_device
        # )

        o3d_tensor = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(points_base))

        # todo: initialize this once above then update here
        pcd = o3d.t.geometry.PointCloud(device=self.o3d_device)
        pcd.point.positions = o3d_tensor

        return pcd, points_base

    def odom_callback(self, msg: Odometry) -> None:
        """Store odometry hint for ICP initialization"""
        if self.get_parameter('use_odom_hint').value:
            # Extract position
            pos = msg.pose.pose.position
            pos = np.array([pos.x, pos.y, pos.z])

            # Extract orientation
            quat = msg.pose.pose.orientation
            quat = [quat.x, quat.y, quat.z, quat.w]
            rot_mat = tf_transformations.quaternion_matrix(quat)[:3, :3]

            # Construct transformation matrix
            self.odom_hint = np.eye(4)
            self.odom_hint[:3, :3] = rot_mat
            self.odom_hint[:3, 3] = pos

    def scan_callback(self, msg: LaserScan) -> None:
        """Main callback for laser scan processing"""
        # Convert scan to point cloud
        current_cloud, points_tensor = self.laser_scan_to_point_cloud(msg)

        if current_cloud is None or points_tensor is None:
            return

        # Initialize reference cloud if needed
        if self.reference_cloud is None:
            self.reference_cloud = current_cloud
            self.last_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            return

        # Get initial guess (prioritize TF over odometry)
        init_guess = self.get_guess_from_tf(msg.header.stamp)
        if init_guess is None and self.get_parameter('use_odom_hint').value:
            init_guess = self.odom_hint
        if init_guess is None:
            init_guess = self.last_pose

        # Convert guess to tensor
        init_transform = o3c.Tensor(
            init_guess,
            dtype=o3c.Dtype.Float32,
            device=self.o3d_device
        )

        # Perform ICP. todo: initialize these once above and add point to plane option
        reg = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
        criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=50
        )

        try:
            result = o3d.t.pipelines.registration.icp(
                current_cloud, self.reference_cloud,
                self.get_parameter('max_icp_distance').value,
                init_transform,
                reg,
                criteria
            )
        except Exception as e:
            self.get_logger().error(f'ICP failed: {str(e)}')
            return

        # Extract transformation
        current_pose = result.transformation.cpu().numpy()

        # Calculate velocity by differentiation
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = current_time - self.last_time

        # run velocity estimation with CPU
        if dt > 0:
            # # Convert transforms to PyTorch for velocity calculations
            # curr_pose_torch = torch.from_numpy(current_pose).float().to(self.torch_device)
            # last_pose_torch = torch.from_numpy(self.last_pose).float().to(self.torch_device)

            # Linear velocity
            pos_diff = current_pose[:3, 3] - self.last_pose[:3, 3]
            current_velocity = pos_diff / dt

            # Angular velocity
            rot_diff = np.matmul(
                    current_pose[:3, :3],
                    self.last_pose[:3, :3].T
            )
            angle = np.arccos((np.trace(rot_diff) - 1) / 2)
            if angle > 1e-10:
                axis = np.array([
                    rot_diff[2, 1] - rot_diff[1, 2],
                    rot_diff[0, 2] - rot_diff[2, 0],
                    rot_diff[1, 0] - rot_diff[0, 1]
                ])
                axis = axis / (2 * np.sin(angle))
                current_angular_velocity = axis * angle / dt
            else:
                current_angular_velocity = np.zeros(3)

            # Apply simple low-pass filter
            alpha = 0.7
            self.last_velocity = alpha * current_velocity + (1 - alpha) * self.last_velocity
            self.last_angular_velocity = alpha * current_angular_velocity + (1 - alpha) * self.last_angular_velocity

        # Publish odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp
        odom_msg.header.frame_id = self.get_parameter('reference_frame').value
        odom_msg.child_frame_id = self.get_parameter('base_frame').value

        # Set pose
        odom_msg.pose.pose.position.x = current_pose[0, 3]
        odom_msg.pose.pose.position.y = current_pose[1, 3]
        odom_msg.pose.pose.position.z = current_pose[2, 3]

        quat = tf_transformations.quaternion_from_matrix(current_pose)
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Set velocity
        odom_msg.twist.twist.linear.x = self.last_velocity[0]
        odom_msg.twist.twist.linear.y = self.last_velocity[1]
        odom_msg.twist.twist.linear.z = self.last_velocity[2]

        odom_msg.twist.twist.angular.x = self.last_angular_velocity[0]
        odom_msg.twist.twist.angular.y = self.last_angular_velocity[1]
        odom_msg.twist.twist.angular.z = self.last_angular_velocity[2]

        self.pose_pub.publish(odom_msg)

        # Broadcast transform
        tf = TransformStamped()
        tf.header = odom_msg.header
        tf.child_frame_id = odom_msg.child_frame_id
        tf.transform.translation = odom_msg.pose.pose.position
        tf.transform.rotation = odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf)

        # Update state
        self.last_pose = current_pose
        self.last_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = ICPLocalizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()