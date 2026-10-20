#!/usr/bin/env python3
"""mapping: fallback only (no GPU, "not parity"): one agent's global_mapper 2D grids on the
names map_fusion_node owns on the robot.

  grid_relay.py <ns> [--rate 5] [--origin-offset-cells 0.5]

On the robot map_fusion_node fuses the elevation mapper's elev/* with global_mapper's voxel/*
and owns <ns>/{occ_2d_topic, planning_occ_2d_topic, esdf_2d_topic} plus the live window
live/occ_2d_topic that persistent_map_node stitches. A fallback session runs neither the
elevation mapper nor map_fusion. global_mapper keeps the robot's pane command (every output
under voxel/, a fallback param file), and this node, grid_relay_<ns> in /<ns>, publishes
  voxel/occ_2d_topic   -> occ_2d_topic and live/occ_2d_topic   (mighty_node's A*, goal_selector;
                                                                persistent_map_node's window)
  voxel/esdf_2d_topic  -> esdf_2d_topic                         (mighty_node's ESDF cost)
so persistent_map runs the robot's own parameters. occ_2d_topic is the live window, not
map_fusion's composite with the persistent map. Nothing in mighty 3cf7320 subscribes
planning_occ_2d_topic, so it is not made.

What it does, matching the robot's map_fusion outputs as configured in
mapping_ws@c1233f7:config/mapping/map_fusion_*.yaml:
  rate   a --rate Hz timer (default 5) sends the newest grids it has not sent;
  stamp  the newest DLIO deskewed cloud's, so a dead lidar or DLIO shows as a frozen data stamp
         to mighty's map-age hold (map_max_age_sec) however often global_mapper publishes;
  start  nothing before the first cloud;
  origin moved by --origin-offset-cells (default +0.5 cell) in x and y, the robot's grid
         convention; data is passed through unchanged;
  QoS    out reliable, keep last 10, volatile (mighty subscribes best effort, which matches);
         in best effort, keep last 1.

It logs an error while another node also publishes one of its outputs (MIGHTY would get two
maps interleaved).

The pure parts (RelayState, the helpers) import without rclpy; tests/test_grid_relay.py
covers them.
"""

import argparse
import re
import sys
import time

AGENT_NAME_RE = re.compile(r'^[A-Z]{2}[0-9]{2}$')

OCC_IN, ESDF_IN = 'voxel/occ_2d_topic', 'voxel/esdf_2d_topic'
CLOUD_IN = 'dlio/odom_node/pointcloud/deskewed'
# output topic -> the input it carries (relative to /<ns>)
OUTPUTS = (('occ_2d_topic', OCC_IN), ('live/occ_2d_topic', OCC_IN), ('esdf_2d_topic', ESDF_IN))

RATE_HZ = 5.0
ORIGIN_OFFSET_CELLS = 0.5
IN_QOS = dict(reliability='best_effort', depth=1)    # as global_mapper publishes
OUT_QOS = dict(reliability='reliable', depth=10)     # as map_fusion publishes
CONTRACT_PERIOD_S = 5.0
CONTRACT_GRACE_S = 15.0           # zenoh graph discovery before the first check
STALE_S = 5.0


def node_name(ns):
    return 'grid_relay_' + ns


def shift_origin(info, cells):
    """Move a nav_msgs/MapMetaData's origin by `cells` cells in x and y (in place)."""
    d = cells * float(info.resolution)
    info.origin.position.x += d
    info.origin.position.y += d
    return info


def other_publishers(endpoints, ns, name):
    """Fully qualified names of the endpoints that are not this node, sorted."""
    me = ('/' + ns, name)
    out = set()
    for e in endpoints:
        ens = '/' + e.node_namespace.strip('/')
        if (ens, e.node_name) != me:
            out.add('%s/%s' % (ens.rstrip('/'), e.node_name))
    return sorted(out)


class RelayState:
    """What to publish on each tick, from the newest messages (duck-typed nav_msgs/
    OccupancyGrid, builtin_interfaces/Time). Each input message goes out at most once, with
    the newest cloud stamp and its origin moved; nothing goes out before the first cloud."""

    def __init__(self, origin_offset_cells=ORIGIN_OFFSET_CELLS):
        self.offset = origin_offset_cells
        self.pending = {}          # input topic -> newest message not yet sent
        self.cloud_stamp = None

    def on_grid(self, topic, msg):
        self.pending[topic] = msg

    def on_cloud(self, stamp):
        self.cloud_stamp = stamp

    def tick(self):
        """[(output topic, message)] to publish now (occ_2d_topic before live/ before esdf)."""
        if self.cloud_stamp is None or not self.pending:
            return []
        out = []
        for topic in (OCC_IN, ESDF_IN):
            msg = self.pending.pop(topic, None)
            if msg is None:
                continue
            msg.header.stamp = self.cloud_stamp
            shift_origin(msg.info, self.offset)
            out += [(dst, msg) for dst, src in OUTPUTS if src == topic]
        return out


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('ns', help='agent namespace, e.g. JR81')
    ap.add_argument('--rate', type=float, default=RATE_HZ, help='publish rate, Hz (default %(default)s)')
    ap.add_argument('--origin-offset-cells', type=float, default=ORIGIN_OFFSET_CELLS,
                    help='cells to move each grid origin by in x and y (default %(default)s)')
    a = ap.parse_args(argv)
    if not AGENT_NAME_RE.match(a.ns):
        ap.error('namespace %r is not two capital letters and two digits' % a.ns)
    if not 0 < a.rate <= 50:
        ap.error('--rate must be in (0, 50] Hz')
    if not abs(a.origin_offset_cells) <= 1.0:
        ap.error('--origin-offset-cells must be within [-1, 1]')
    return a


def main(argv=None):
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2

    argv = sys.argv if argv is None else argv
    args = parse_args(rclpy.utilities.remove_ros_args(argv)[1:])

    def qos(q):
        rel = ReliabilityPolicy.RELIABLE if q['reliability'] == 'reliable' else ReliabilityPolicy.BEST_EFFORT
        return QoSProfile(reliability=rel, history=HistoryPolicy.KEEP_LAST, depth=q['depth'],
                          durability=DurabilityPolicy.VOLATILE)

    class GridRelay(Node):
        def __init__(self):
            super().__init__(node_name(args.ns), namespace='/' + args.ns)
            self.state = RelayState(args.origin_offset_cells)
            self.last = {}
            self.pubs = {dst: self.create_publisher(OccupancyGrid, dst, qos(OUT_QOS)) for dst, _ in OUTPUTS}
            for src in (OCC_IN, ESDF_IN):
                self.create_subscription(OccupancyGrid, src, lambda m, s=src: self.on_grid(s, m), qos(IN_QOS))
            self.create_subscription(PointCloud2, CLOUD_IN, self.on_cloud, qos(IN_QOS))
            self.create_timer(1.0 / args.rate, self.tick)
            self.started = time.monotonic()
            self.create_timer(CONTRACT_PERIOD_S, self.check)
            ns = '/' + args.ns
            self.get_logger().info(
                '%s/{%s} -> %s at %.1f Hz, stamped with %s/%s, origin %+.2f cell in x and y; nothing '
                'before the first cloud' % (ns, ','.join((OCC_IN, ESDF_IN)), ', '.join('%s/%s' % (ns, d) for d, _ in OUTPUTS),
                                            args.rate, ns, CLOUD_IN, args.origin_offset_cells))

        def on_grid(self, src, msg):
            if src not in self.last:
                self.get_logger().info('first %s: %dx%d at %.3f m in %s'
                                       % (src, msg.info.width, msg.info.height, msg.info.resolution,
                                          msg.header.frame_id))
            self.last[src] = time.monotonic()
            self.state.on_grid(src, msg)

        def on_cloud(self, msg):
            if self.state.cloud_stamp is None:
                self.get_logger().info('first DLIO cloud: publishing from now on')
            self.state.on_cloud(msg.header.stamp)

        def tick(self):
            for dst, msg in self.state.tick():
                self.pubs[dst].publish(msg)

        def check(self):
            now = time.monotonic()
            for src in (OCC_IN, ESDF_IN):
                t = self.last.get(src)
                if t is not None and now - t > STALE_S:
                    self.get_logger().warning('nothing on %s for %.0f s (global_mapper down?)' % (src, now - t),
                                              throttle_duration_sec=30.0)
            if now - self.started < CONTRACT_GRACE_S:
                return
            for dst, _ in OUTPUTS:
                topic = '/%s/%s' % (args.ns, dst)
                others = other_publishers(self.get_publishers_info_by_topic(topic), args.ns, self.get_name())
                if others:
                    self.get_logger().error('%s is also published by %s: MIGHTY gets two maps interleaved '
                                            '(this relay is for mapping: fallback, without map_fusion_node)'
                                            % (topic, ', '.join(others)), throttle_duration_sec=30.0)

    rclpy.init(args=argv)
    node = GridRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
