#!/usr/bin/env python3
"""Starts robot NS's elevation map before mighty may plan on it.

The lidar never sees the ground under and right around the robot, which the
elevation mapper's planning grid treats as blocked. On the real rovers someone
drives them a bit first; this stands in for that:
  1. waits until the mapper's grid is centered on the robot (it's tracking it),
  2. publishes a rectangle of flat floor (frame map, z=0) on the mapper's input
     cloud for SEED_S seconds: the robot's footprint, with room to turn in
     place, and the strip ahead out to FRONT_M, past where the lidar first sees
     the ground. Assumes that area is empty floor.
  3. waits SETTLE_S more seconds for lidar,
  4. then relays the planning grid to mighty, which doesn't plan before its
     first one (config/elevation_sim.yaml renames the mapper's output).

Usage: elevation_start.py NS [FRONT_M] [SEED_S] [SETTLE_S]
"""
import sys

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener

ns = sys.argv[1]
front = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
back, half_width = 0.5, 0.5  # covers the 0.6 m x 0.6 m footprint turning in place
seed_s = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
settle_s = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
spacing = 0.05

rclpy.init()
node = rclpy.create_node('elevation_start', namespace=ns)
log = node.get_logger()
tf_buffer = Buffer()
TransformListener(tf_buffer, node)
seed_pub = node.create_publisher(PointCloud2, f'/{ns}/mid360_PointCloud2', qos_profile_sensor_data)
planning_pub = node.create_publisher(OccupancyGrid, f'/{ns}/planning_occ_2d_topic', 10)
latest = {}
node.create_subscription(OccupancyGrid, f'/{ns}/occ_2d_topic',
                         lambda m: latest.__setitem__('occ', m), qos_profile_sensor_data)
gate_open = False


def relay(msg):
    latest['planning'] = msg
    if gate_open:
        planning_pub.publish(msg)


node.create_subscription(OccupancyGrid, f'/{ns}/planning_occ_2d_topic_ungated', relay, 10)


def robot_pose():
    if not tf_buffer.can_transform('map', f'{ns}/base_link', rclpy.time.Time()):
        return None
    t = tf_buffer.lookup_transform('map', f'{ns}/base_link', rclpy.time.Time()).transform
    q = t.rotation
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    return t.translation.x, t.translation.y, yaw


def tracking():
    pose, grid = robot_pose(), latest.get('occ')
    if pose is None or grid is None:
        return False
    xy = np.array(pose[:2])
    i = grid.info
    center = np.array([i.origin.position.x + i.width * i.resolution / 2,
                       i.origin.position.y + i.height * i.resolution / 2])
    return np.linalg.norm(center - xy) < 1.0


def spin_for(seconds, each=None):
    end = node.get_clock().now().nanoseconds + seconds * 1e9
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        if each:
            each()
        rclpy.spin_once(node, timeout_sec=0.2)


while rclpy.ok() and not tracking():
    rclpy.spin_once(node, timeout_sec=0.5)

x, y, yaw = robot_pose()
u, v = np.meshgrid(np.arange(-back, front + spacing, spacing),
                   np.arange(-half_width, half_width + spacing, spacing))
u, v = u.ravel(), v.ravel()
seed = np.column_stack([x + u * np.cos(yaw) - v * np.sin(yaw), y + u * np.sin(yaw) + v * np.cos(yaw),
                        np.zeros(u.size)]).astype(np.float32)
log.info(f'seeding floor from {back} m behind to {front} m ahead of ({x:.2f}, {y:.2f}), '
         f'{2 * half_width} m wide, for {seed_s} s, then {settle_s} s of lidar')
spin_for(seed_s, lambda: seed_pub.publish(point_cloud2.create_cloud_xyz32(
    Header(frame_id='map', stamp=node.get_clock().now().to_msg()), seed)))
spin_for(settle_s)

gate_open = True
if 'planning' in latest:
    planning_pub.publish(latest['planning'])
log.info('map ready: relaying the planning grid to mighty')
rclpy.spin(node)
