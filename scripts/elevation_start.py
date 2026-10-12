#!/usr/bin/env python3
"""Starts robot NS's elevation map before mighty may plan on it.

The lidar never sees the ground under and right around the robot, which the
elevation mapper's planning grid treats as blocked. On the real rovers someone
drives them a bit first; this stands in for that:
  1. waits until the mapper's grid is centered on the robot (it's tracking it),
  2. publishes what is really within RADIUS_M of the robot (frame map) on the
     mapper's input cloud for SEED_S seconds: floor points, and a column of
     points up to each obstacle's top, from the boxes and cylinders of the
     Gazebo world $ROVER_SIM_WORLD (worlds/<name>.world; other shapes, e.g.
     meshes, are left to the lidar),
  3. waits SETTLE_S more seconds for lidar,
  4. then relays the planning grid to mighty, which doesn't plan before its
     first one (config/elevation_sim.yaml renames the mapper's output).

Usage: elevation_start.py NS [RADIUS_M] [SEED_S] [SETTLE_S]
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from scipy.spatial.transform import Rotation
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener

ns = sys.argv[1]
radius = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
seed_s = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
settle_s = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
spacing = 0.05
# Obstacles: shapes starting below this height (not overhead); columns up to at most it,
# which is also about where the mapper stops taking points above the lidar.
max_height = 1.0


def pose_matrix(element):
    pose = element.find('pose')
    v = [float(n) for n in pose.text.split()] if pose is not None and pose.text else [0.0] * 6
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', v[3:]).as_matrix()
    t[:3, 3] = v[:3]
    return t


def world_shapes(path):
    """(kind, world transform, dims) of every box and cylinder collision; models with other shapes."""
    shapes, skipped = [], set()

    def walk(model, parent):
        t_model = parent @ pose_matrix(model)
        for sub_model in model.findall('model'):
            walk(sub_model, t_model)
        for link in model.findall('link'):
            t_link = t_model @ pose_matrix(link)
            for collision in link.findall('collision'):
                t = t_link @ pose_matrix(collision)
                geometry = collision.find('geometry')
                if geometry.find('box') is not None:
                    shapes.append(('box', t, [float(n) for n in geometry.find('box/size').text.split()]))
                elif geometry.find('cylinder') is not None:
                    shapes.append(('cylinder', t, [float(geometry.find('cylinder/radius').text),
                                                   float(geometry.find('cylinder/length').text)]))
                elif geometry.find('plane') is None:
                    skipped.add(model.get('name'))

    for model in ET.parse(path).getroot().find('world').findall('model'):
        walk(model, np.eye(4))
    return shapes, skipped


def obstacle_tops(shapes, xy):
    """Top of the tallest obstacle standing on each xy point (0: floor)."""
    top = np.zeros(len(xy))
    for kind, t, dims in shapes:
        half = np.array(dims) / 2 if kind == 'box' else np.array([dims[0], dims[0], dims[1] / 2])
        corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T * half
        z = (corners @ t[:3, :3].T + t[:3, 3])[:, 2]
        if z.max() < 0.05 or z.min() > max_height:
            continue
        local = (np.column_stack([xy, np.full(len(xy), t[2, 3])]) - t[:3, 3]) @ t[:3, :3]
        if kind == 'box':
            inside = (np.abs(local[:, 0]) <= half[0]) & (np.abs(local[:, 1]) <= half[1])
        else:
            inside = local[:, 0]**2 + local[:, 1]**2 <= dims[0]**2
        top[inside] = np.maximum(top[inside], min(z.max(), max_height))
    return top


world = os.environ.get('ROVER_SIM_WORLD', 'empty')
world_file = os.path.join(get_package_share_directory('mighty'), 'worlds', f'{world}.world')

rclpy.init()
node = rclpy.create_node('elevation_start', namespace=ns)
log = node.get_logger()
if not os.path.exists(world_file):
    log.error(f'no world file {world_file} (ROVER_SIM_WORLD={world}); not starting {ns}\'s map')
    sys.exit(1)
shapes, skipped = world_shapes(world_file)
if skipped:
    log.warn(f'{world}: not seeding models with shapes other than boxes and cylinders: {sorted(skipped)}')
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


def robot_xy():
    if not tf_buffer.can_transform('map', f'{ns}/base_link', rclpy.time.Time()):
        return None
    t = tf_buffer.lookup_transform('map', f'{ns}/base_link', rclpy.time.Time()).transform.translation
    return np.array([t.x, t.y])


def tracking():
    xy, grid = robot_xy(), latest.get('occ')
    if xy is None or grid is None:
        return False
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

x, y = robot_xy()
r = np.arange(-radius, radius + spacing, spacing)
u, v = np.meshgrid(r, r)
inside = u**2 + v**2 <= radius**2
xy = np.column_stack([u[inside] + x, v[inside] + y])
top = obstacle_tops(shapes, xy)
points = [np.column_stack([xy[top == 0], np.zeros((top == 0).sum())])]
for z in np.arange(0.0, max_height + 1e-6, 0.1):
    column = top >= z
    column &= top > 0
    points.append(np.column_stack([xy[column], np.full(column.sum(), z)]))
seed = np.vstack(points).astype(np.float32)
log.info(f'seeding {world} within {radius} m of ({x:.2f}, {y:.2f}): '
         f'{(top == 0).mean():.0%} floor, {(top > 0).mean():.0%} obstacle; {seed_s} s, then {settle_s} s of lidar')
spin_for(seed_s, lambda: seed_pub.publish(point_cloud2.create_cloud_xyz32(
    Header(frame_id='map', stamp=node.get_clock().now().to_msg()), seed)))
spin_for(settle_s)

gate_open = True
if 'planning' in latest:
    planning_pub.publish(latest['planning'])
log.info('map ready: relaying the planning grid to mighty')
rclpy.spin(node)
