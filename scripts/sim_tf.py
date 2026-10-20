#!/usr/bin/env python3
"""world -> <ns>/map for every agent of a run, on /tf.

  sim_tf.py --spawn /rover_sim/run/spawn.yaml [--rate 10]

One node, sim_tf, publishes one TFMessage per tick with a transform per agent: the
spawn pose (x, y, 0, yaw about z) from <run>/spawn.yaml (written by sim.py), stamped
with wall time as the rest of the stack. It goes on periodic /tf rather than /tf_static
because rmw_zenoh does not deliver latched samples to late joiners (dlio_ws
scripts/map_odom_tf.py). Sim only: the robot has no world frame.

The pure parts (spawn.yaml parsing, the transforms) import without rclpy;
tests/test_sim_tf.py covers them, and checks them against sim.py's spawn.yaml and
frame_align (sim.py frame_align_transform: mighty's alignment between two of these maps).
"""

import argparse
import math
import re
import sys
from collections import namedtuple

AGENT_NAME_RE = re.compile(r'^[A-Z]{2}[0-9]{2}$')

SpawnPose = namedtuple('SpawnPose', 'name x y yaw')


class SpawnError(ValueError):
    pass


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def spawn_poses(data, source='spawn.yaml'):
    """[SpawnPose] from parsed spawn.yaml: {agents: [{name, x, y, yaw (rad), yaw_deg, ...}]}.
    yaw is authoritative; a yaw_deg that disagrees with it means the writer is broken."""
    if not isinstance(data, dict) or not isinstance(data.get('agents'), list) or not data['agents']:
        raise SpawnError('%s: expected {agents: [...]} with at least one agent' % source)
    out, seen = [], set()
    for i, a in enumerate(data['agents']):
        where = '%s: agents[%d]' % (source, i)
        if not isinstance(a, dict):
            raise SpawnError('%s: expected a mapping' % where)
        name = a.get('name')
        if not isinstance(name, str) or not AGENT_NAME_RE.match(name):
            raise SpawnError('%s: name %r is not two capital letters and two digits' % (where, name))
        if name in seen:
            raise SpawnError('%s: duplicate agent %s' % (where, name))
        seen.add(name)
        for k in ('x', 'y', 'yaw'):
            if not _num(a.get(k)):
                raise SpawnError('%s (%s): %s %r is not a finite number' % (where, name, k, a.get(k)))
        yaw_deg = a.get('yaw_deg')
        if yaw_deg is not None:
            if not _num(yaw_deg):
                raise SpawnError('%s (%s): yaw_deg %r is not a finite number' % (where, name, yaw_deg))
            diff = math.remainder(math.radians(yaw_deg) - a['yaw'], 2 * math.pi)
            if abs(diff) > 1e-6:
                raise SpawnError('%s (%s): yaw %r rad and yaw_deg %r disagree' % (where, name, a['yaw'], yaw_deg))
        out.append(SpawnPose(name, float(a['x']), float(a['y']), float(a['yaw'])))
    return out


def load_spawn(path):
    import yaml
    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise SpawnError('%s: %s' % (path, e))
    return spawn_poses(data, path)


def yaw_quat(yaw):
    """(x, y, z, w) of a rotation by yaw about z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def world_to_map(p):
    """(parent, child, (tx, ty, tz), (qx, qy, qz, qw)) of world -> <ns>/map."""
    return 'world', p.name + '/map', (p.x, p.y, 0.0), yaw_quat(p.yaw)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--spawn', default='/rover_sim/run/spawn.yaml')
    ap.add_argument('--rate', type=float, default=10.0, help='Hz')
    a = ap.parse_args(argv)
    if not a.rate > 0:
        ap.error('--rate must be > 0')
    return a


def main(argv=None):
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from tf2_ros import TransformBroadcaster

    argv = sys.argv if argv is None else argv
    args = parse_args(rclpy.utilities.remove_ros_args(argv)[1:])
    try:
        poses = load_spawn(args.spawn)
    except (OSError, SpawnError) as e:
        sys.exit('sim_tf: %s' % e)

    class SimTf(Node):
        def __init__(self):
            super().__init__('sim_tf')
            self.tfb = TransformBroadcaster(self)
            self.msgs = []
            for p in poses:
                parent, child, t, q = world_to_map(p)
                m = TransformStamped()
                m.header.frame_id, m.child_frame_id = parent, child
                tr, rot = m.transform.translation, m.transform.rotation
                tr.x, tr.y, tr.z = t
                rot.x, rot.y, rot.z, rot.w = q
                self.msgs.append(m)
                self.get_logger().info('%s -> %s: x %.3f y %.3f yaw %.4f rad (%.2f deg)'
                                       % (parent, child, p.x, p.y, p.yaw, math.degrees(p.yaw)))
            self.create_timer(1.0 / args.rate, self.tick)

        def tick(self):
            stamp = self.get_clock().now().to_msg()
            for m in self.msgs:
                m.header.stamp = stamp
            self.tfb.sendTransform(self.msgs)

    rclpy.init(args=argv)
    node = SimTf()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()            # SIGINT may already have shut the context down


if __name__ == '__main__':
    main()
