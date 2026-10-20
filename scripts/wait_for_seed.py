#!/usr/bin/env python3
"""dlio: the DLIO pane's gate. Waits until one seed pose has actually arrived on /<ns>/world,
prints its z and exits 0; DLIO starts after it.

  wait_for_seed.py <ns> [--timeout 300]

DLIO registers scans as soon as its 3 s IMU calibration ends, seeded or not, and a seed that
arrives later leaves its odom anchored at z 0 instead of the lidar height. The seed pane's
`ros2 topic pub` running is not proof of delivery: rmw_zenoh 0.1.9 once delayed it by ~85 s.
A message received here, in DLIO's container, is. The seed pane keeps publishing at 2 Hz, so
DLIO's own subscription has its 3 s calibration to receive one too (`sim.py arm` checks the
odom height on an agent's first arm).

Exit 0 on a seed; 1 on --timeout (generous: a paused bring-up takes minutes); 128 + the
signal on SIGINT/SIGTERM. Every exit leaves the graph gently (as `sim.py status` does): the
subscription first, then the node. A message with another frame_id or a non-finite z is not
taken (DLIO would anchor odom there); the wait goes on and the timeout names it.

The pure parts import without rclpy; tests/test_wait_for_seed.py covers them.
"""

import argparse
import math
import re
import signal
import sys
import time

AGENT_NAME_RE = re.compile(r'^[A-Z]{2}[0-9]{2}$')
SEED_FRAME = 'world'               # the seed pane's header.frame_id (sim.py seed_pose_cmd)
TIMEOUT_S = 300.0
PROGRESS_S = 5.0
LEAVE_PAUSE_S = 0.5
EXIT_TIMEOUT = 1


def node_name(ns):
    return 'rover_sim_wait_for_seed_' + ns


def topic(ns):
    return '/%s/world' % ns


def parse_args(argv):
    p = argparse.ArgumentParser(description='Wait for the seed pose on /<ns>/world.')
    p.add_argument('ns', help='agent namespace, e.g. JR81')
    p.add_argument('--timeout', type=float, default=TIMEOUT_S, help='seconds (default %(default)g)')
    args = p.parse_args(argv)
    if not AGENT_NAME_RE.match(args.ns):
        p.error('ns %r is not an agent name like JR81' % args.ns)
    if not args.timeout > 0:
        p.error('--timeout must be > 0')
    return args


def seed_problem(frame_id, z):
    """Why a seed message is not taken, or None."""
    if frame_id != SEED_FRAME:
        return 'frame_id %r, not %r' % (frame_id, SEED_FRAME)
    if not math.isfinite(z):
        return 'z %r is not finite' % z
    return None


def waiting_line(ns, elapsed, rejected=None):
    return '[rover_sim] waiting for the %s seed pose on %s ... (%.0f s%s)' % (
        ns, topic(ns), elapsed, '; last one not taken: ' + rejected if rejected else '')


def received_line(ns, z, elapsed):
    return '[rover_sim] %s seed pose received on %s after %.1f s: z %.3f' % (ns, topic(ns), elapsed, z)


def timeout_line(ns, timeout, rejected=None):
    return ('[rover_sim] no %s seed pose on %s after %.0f s%s: is its seed pose pane publishing? '
            'Not starting DLIO (a late seed anchors odom at z 0)' % (
                ns, topic(ns), timeout, ' (only rejected ones: %s)' % rejected if rejected else ''))


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from geometry_msgs.msg import PoseStamped

    # rclpy's own handlers would shut the context down under us, before the gentle leave.
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: stop.append(signum))
    node = Node(node_name(args.ns), enable_rosout=False, start_parameter_services=False)
    got, rejected = [], []

    def on_seed(m):
        why = seed_problem(m.header.frame_id, m.pose.position.z)
        if why:
            rejected.append(why)
        elif not got:
            got.append(m.pose.position.z)

    # `ros2 topic pub`'s default: reliable, volatile, keep last 10.
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE)
    sub = node.create_subscription(PoseStamped, topic(args.ns), on_seed, qos)
    t0 = time.monotonic()
    print(waiting_line(args.ns, 0.0), flush=True)
    last = t0
    rc = EXIT_TIMEOUT
    try:
        while not stop:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if got:
                print(received_line(args.ns, got[0], now - t0), flush=True)
                rc = 0
                break
            if now - t0 >= args.timeout:
                print(timeout_line(args.ns, args.timeout, rejected[-1] if rejected else None),
                      file=sys.stderr, flush=True)
                break
            if now - last >= PROGRESS_S:
                last = now
                print(waiting_line(args.ns, now - t0, rejected[-1] if rejected else None), flush=True)
        if stop:
            rc = 128 + stop[0]
            print('[rover_sim] %s: stopped while waiting for the %s seed pose; not starting DLIO'
                  % (signal.Signals(stop[0]).name, args.ns), file=sys.stderr, flush=True)
    finally:
        node.destroy_subscription(sub)
        rclpy.spin_once(node, timeout_sec=0)
        time.sleep(LEAVE_PAUSE_S)
        node.destroy_node()
        rclpy.try_shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())
