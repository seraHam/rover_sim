"""Unit tests for scripts/drive_pattern.py's planner and controller (pure Python, no rclpy).

The controller is closed around a unicycle with what the sim's command path adds: command
latency, the diff drive's clamp and v slew (classes.yaml), and the JR's 0.992 in-place yaw gain.

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('drive_pattern', str(REPO / 'scripts' / 'drive_pattern.py'))
dp = importlib.util.module_from_spec(_spec)
sys.modules['drive_pattern'] = dp
_spec.loader.exec_module(dp)

CLASSES = REPO / 'src' / 'rover_sim' / 'config' / 'classes.yaml'


def drive(segs, start, speed=0.6, accel=0.5, omega=0.5, alpha=1.0, v_max=1.2, a_max=1.5, w_max=6.0,
          w_gain=0.992, latency=0.08, rate=20.0, t_max=400.0, twist=True):
    """Run a Driver against the plant; returns (driver, t, end pose, trace, commands)."""
    d = dp.Driver(segs, speed, accel, omega, alpha)
    x, y, yaw = start
    v = w = 0.0
    t, dt, nxt = 0.0, 0.001, 0.0
    queue, cmd, trace, cmds = [], (0.0, 0.0), [], []
    while t < t_max and not d.done:
        if t >= nxt - 1e-9:
            c = d.step(t, x, y, yaw, v if twist else None, w if twist else None)
            cmds.append((t, c[0], c[1]))
            queue.append((t + latency, c))
            nxt += 1.0 / rate
        while queue and queue[0][0] <= t:
            cmd = queue.pop(0)[1]
        vc = max(-v_max, min(v_max, cmd[0]))
        v += max(-a_max * dt, min(a_max * dt, vc - v))
        w = max(-w_max, min(w_max, cmd[1])) * w_gain
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = dp.wrap(yaw + w * dt)
        t += dt
        trace.append((x, y, yaw))
    return d, t, (x, y, yaw), trace, cmds


def dist_to_segment(p, a, b):
    ax, ay = a
    bx, by = b
    lx, ly = bx - ax, by - ay
    f = max(0.0, min(1.0, ((p[0] - ax) * lx + (p[1] - ay) * ly) / (lx * lx + ly * ly)))
    return math.hypot(p[0] - ax - f * lx, p[1] - ay - f * ly)


class PlanTest(unittest.TestCase):
    def test_square_geometry(self):
        p = dp.Pose2(2.0, -1.0, math.radians(30))
        segs = dp.plan_square(p, 12.5)
        lines = [s for s in segs if isinstance(s, dp.Line)]
        turns = [s for s in segs if isinstance(s, dp.Turn)]
        self.assertEqual(len(lines), 4)
        self.assertEqual(len(turns), 4)
        self.assertIsInstance(segs[0], dp.Line)                     # forward first
        self.assertAlmostEqual(math.atan2(lines[0].y1 - lines[0].y0, lines[0].x1 - lines[0].x0), p.yaw)
        self.assertEqual((lines[0].x0, lines[0].y0), (p.x, p.y))
        self.assertEqual((lines[-1].x1, lines[-1].y1), (p.x, p.y))  # closed
        for a, b in zip(lines, lines[1:]):
            self.assertAlmostEqual(a.x1, b.x0)
            self.assertAlmostEqual(a.y1, b.y0)
        # left: the second corner is to the left of the first leg
        self.assertGreater(math.sin(math.atan2(lines[1].y1 - p.y, lines[1].x1 - p.x) - p.yaw), 0)
        self.assertAlmostEqual(dp.wrap(turns[0].yaw - p.yaw), math.pi / 2)
        self.assertAlmostEqual(dp.wrap(turns[-1].yaw - p.yaw), 0.0)
        length, turn = dp.plan_extent(segs, p.yaw)
        self.assertAlmostEqual(length, 50.0)
        self.assertAlmostEqual(turn, 2 * math.pi)

    def test_square_right(self):
        segs = dp.plan_square(dp.Pose2(0, 0, 0), 5.0, left=False)
        self.assertAlmostEqual(segs[0].x1, 5.0)
        self.assertAlmostEqual(segs[2].y1, -5.0)
        self.assertAlmostEqual(segs[1].yaw, -math.pi / 2)

    def test_loop(self):
        p = dp.Pose2(1.0, 2.0, math.pi / 2)
        arc, turn = dp.plan_loop(p, 12.5)
        self.assertAlmostEqual(arc.r, 6.25)
        self.assertAlmostEqual(arc.cx, 1.0 - 6.25)                 # left of +y is -x
        self.assertAlmostEqual(arc.cy, 2.0)
        self.assertAlmostEqual(arc.sweep, 2 * math.pi)
        self.assertAlmostEqual(dp.plan_extent([arc, turn], p.yaw)[0], math.pi * 12.5)
        self.assertAlmostEqual(dp.plan_extent([arc, turn], p.yaw)[1], 0.0, places=9)
        self.assertLess(dp.plan_loop(p, 4.0, left=False)[0].sweep, 0)

    def test_waypoints(self):
        self.assertEqual(dp.parse_waypoints('1,2; -3.5,4;'), [(1.0, 2.0), (-3.5, 4.0)])
        for bad in ('', '1', '1,2,3', 'a,b', 'nan,1'):
            with self.assertRaises(dp.PlanError):
                dp.parse_waypoints(bad)
        segs = dp.plan_waypoints(dp.Pose2(0, 0, 0), [(0, 0), (3, 4), (3, 4), (3, 0)])
        self.assertEqual([type(s).__name__ for s in segs], ['Turn', 'Line', 'Turn', 'Line'])
        self.assertAlmostEqual(segs[0].yaw, math.atan2(4, 3))
        self.assertAlmostEqual(dp.plan_extent(segs, 0.0)[0], 9.0)

    def test_negative_waypoints_parse(self):
        argv = dp.join_negative_values(['--ns', 'JR81', '--pattern', 'waypoints', '--waypoints', '-2,0;-3.5,-1'])
        a = dp.build_parser().parse_args(argv)
        self.assertEqual(dp.parse_waypoints(a.waypoints), [(-2.0, 0.0), (-3.5, -1.0)])
        self.assertEqual(dp.join_negative_values(['--waypoints']), ['--waypoints'])

    def test_approach(self):
        anchor = dp.Pose2(2.0, 0.0, 1.0)
        self.assertEqual(dp.plan_approach(dp.Pose2(2.01, 0.0, 1.01), anchor), [])
        self.assertEqual(dp.plan_approach(dp.Pose2(2.0, 0.0, 1.2), anchor), [dp.Turn(1.0)])
        segs = dp.plan_approach(dp.Pose2(0.0, 0.0, 0.0), anchor)
        self.assertEqual([type(s).__name__ for s in segs], ['Turn', 'Line', 'Turn'])
        self.assertEqual(segs[-1].yaw, 1.0)

    def test_bad_size(self):
        for f in (dp.plan_square, dp.plan_loop):
            with self.assertRaises(dp.PlanError):
                f(dp.Pose2(0, 0, 0), 0.0)
        with self.assertRaises(dp.PlanError):
            dp.plan('spiral', dp.Pose2(0, 0, 0))
        # No default size: none fits every world and start.
        self.assertIsNone(dp.build_parser().parse_args(['--ns', 'JR81']).size)
        for pattern in ('square', 'loop'):
            with self.assertRaisesRegex(dp.PlanError, 'needs --size'):
                dp.plan(pattern, dp.Pose2(0, 0, 0))
        self.assertEqual(len(dp.plan('waypoints', dp.Pose2(0, 0, 0), waypoints=[(1.0, 0.0)])), 2)

    def test_walls(self):
        walls = [-8.0, -8.0, 8.0, 8.0]
        # smoke's JR81 at (-2, 0) facing +x: a 12.5 m square ends its first leg at x 10.5.
        big = dp.plan_square(dp.Pose2(-2.0, 0.0, 0.0), 12.5)
        self.assertEqual(dp.plan_outside(big, walls), (10.5, 0.0))
        self.assertIsNone(dp.plan_outside(dp.plan_square(dp.Pose2(-2.0, 0.0, 0.0), 3.0), walls))
        self.assertEqual(dp.plan_outside(dp.plan_square(dp.Pose2(-2.0, 0.0, 0.0), 9.6), walls), (7.6, 0.0))
        loop = dp.plan_loop(dp.Pose2(0.0, 0.0, 0.0), 15.2)                 # reaches y 15.2 > 7.5
        self.assertIsNotNone(dp.plan_outside(loop, walls))
        # A leg from a start near a wall back into the room is fine.
        self.assertIsNone(dp.plan_outside(dp.plan_waypoints(dp.Pose2(7.8, 0.0, 0.0), [(0.0, 0.0)]), walls))
        self.assertIsNone(dp.plan_outside([dp.Turn(1.0)], walls))
        # The odom_eval session's goals stay inside.
        for a in yaml.safe_load((REPO / 'config' / 'sessions' / 'odom_eval.yaml').read_text())['agents']:
            segs = dp.plan_waypoints(dp.Pose2(a['pose'][0], a['pose'][1], 0.0), [tuple(g) for g in a['goals']] * 4)
            self.assertIsNone(dp.plan_outside(segs, walls), a['name'])


class LimitsTest(unittest.TestCase):
    @unittest.skipUnless(CLASSES.is_file(), 'no src/rover_sim in this checkout mount')
    def test_repo_classes(self):
        path, classes = dp.load_classes(str(CLASSES))
        jr, rr = dp.class_limits(classes, 'jr'), dp.class_limits(classes, 'rr')
        self.assertEqual(jr, dp.Limits(1.2, 1.5, 6.0))
        self.assertEqual(rr, dp.Limits(1.0, 2.5, 1.5))
        # the defaults are inside both classes' limits
        a = dp.build_parser().parse_args(['--ns', 'JR81'])
        for lim in (jr, rr):
            self.assertLessEqual(a.speed, lim.v_max)
            self.assertLessEqual(a.accel, lim.accel)
            self.assertLessEqual(a.omega, lim.omega_max)

    @unittest.skipUnless(CLASSES.is_file(), 'no src/rover_sim in this checkout mount')
    def test_default_lookup_finds_the_checkout(self):
        path, classes = dp.load_classes()
        self.assertEqual(os.path.realpath(path), os.path.realpath(str(CLASSES)))
        self.assertIn('jr', classes)

    def test_errors(self):
        with self.assertRaises(dp.PlanError):
            dp.class_limits({'jr': {'v_max': 1.0}}, 'jr')
        with self.assertRaises(dp.PlanError):
            dp.class_limits({'jr': {'v_max': 1.0, 'linear_acceleration': 0, 'omega_max': 1}}, 'jr')
        with self.assertRaises(dp.PlanError):
            dp.class_limits({}, 'xx')
        with self.assertRaises(dp.PlanError):
            dp.load_classes('/nonexistent/classes.yaml')

    def test_clamp(self):
        msgs = []
        self.assertEqual(dp.clamp_to('--speed', 2.0, 1.2, msgs.append), 1.2)
        self.assertEqual(dp.clamp_to('--speed', 0.6, 1.2, msgs.append), 0.6)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(dp.clamp_to('--omega', 1.4, 1.0, msgs.append, "safety_node's ang_vel_max"), 1.0)
        self.assertIn("over safety_node's ang_vel_max 1", msgs[-1])

    # A teleop file with the robot file's layout (flat node keys, /** defaults, per-robot
    # overrides, as sim.py writes teleop_sim.yaml); the values are made up.
    TELEOP = {'/**/joy_teleop/teleop_twist_joy_node': {'ros__parameters': {'scale_linear': {'x': 0.5}}},
              '/**/safety_node': {'ros__parameters': {'lin_vel_max': 0.9, 'ang_vel_max': 1.1, 'alpha_tau_lin': 0.2}},
              '/JR81/safety_node': {'ros__parameters': {'ang_vel_max': 3.0}},
              '/JR8/safety_node': {'ros__parameters': {'ang_vel_max': 9.0}},
              '/JR81/joy_teleop/safety_node': {'ros__parameters': {'lin_vel_max': 9.0}}}

    def test_safety_limits(self):
        self.assertEqual(dp.safety_limits(self.TELEOP, 'JR81'), (0.9, 3.0))
        self.assertEqual(dp.safety_limits(self.TELEOP, 'RR83'), (0.9, 1.1))          # the /** block only
        self.assertEqual(dp.safety_limits({}, 'RR83'), (1.0, 1.0))                     # safety_node's defaults
        later = dict(self.TELEOP, **{'/**/safety_node': {'ros__parameters': {'ang_vel_max': 0.5}}})
        self.assertEqual(dp.safety_limits(later, 'JR81'), (1.0, 3.0))                 # file order: the ns block wins
        with self.assertRaises(dp.PlanError):
            dp.safety_limits({'/**/safety_node': {'ros__parameters': {'lin_vel_max': 'fast'}}}, 'JR81')
        for key, fqn, hit in (('/**', '/JR81/safety_node', True), ('/*/safety_node', '/JR81/safety_node', True),
                              ('/*/safety_node', '/a/JR81/safety_node', False),
                              ('/**/safety_node', '/safety_node', True),
                              ('JR81/safety_node', '/JR81/safety_node', True), ('/JR8*', '/JR81/safety_node', False)):
            self.assertIs(dp.params_key_matches(key, fqn), hit, (key, fqn))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'teleop_sim.yaml')
            self.assertIsNone(dp.load_safety_limits(path, 'JR81'))
            with open(path, 'w') as f:
                yaml.safe_dump(self.TELEOP, f)
            self.assertEqual(dp.load_safety_limits(path, 'JR81'), (0.9, 3.0))


class SpawnTest(unittest.TestCase):
    DATA = {'agents': [{'name': 'JR81', 'class': 'jr', 'id': 81, 'x': -2.0, 'y': 0.0, 'yaw': 0.0,
                        'yaw_deg': 0.0, 'lidar_height': 0.36, 'reference_robot': 'JR02'},
                       {'name': 'RR83', 'class': 'rr', 'id': 83, 'x': 2.0, 'y': 0.0, 'yaw': math.pi,
                        'yaw_deg': 180.0, 'lidar_height': 0.40, 'reference_robot': 'RR08'}]}

    def test_entry(self):
        pose, cls, h = dp.spawn_entry(self.DATA, 'RR83')
        self.assertEqual((pose, cls, h), (dp.Pose2(2.0, 0.0, math.pi), 'rr', 0.40))
        with self.assertRaises(dp.PlanError):
            dp.spawn_entry(self.DATA, 'JR82')
        with self.assertRaises(dp.PlanError):
            dp.spawn_entry({'agents': [{'name': 'JR81', 'x': 'a'}]}, 'JR81')
        with self.assertRaises(dp.PlanError):
            dp.spawn_entry([], 'JR81')

    def test_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'spawn.yaml')
            with open(p, 'w') as f:
                yaml.safe_dump(self.DATA, f)
            self.assertEqual(dp.load_spawn_entry(p, 'JR81')[0], dp.Pose2(-2.0, 0.0, 0.0))
            with open(p, 'w') as f:
                f.write('agents: [\n')
            with self.assertRaises(dp.PlanError):
                dp.load_spawn_entry(p, 'JR81')

    def test_walls_entry(self):
        self.assertEqual(dp.spawn_walls(dict(self.DATA, world='arena', walls=[-8, -8, 8, 8])),
                         ('arena', [-8.0, -8.0, 8.0, 8.0]))
        self.assertEqual(dp.spawn_walls(dict(self.DATA, world='empty')), ('empty', None))
        self.assertEqual(dp.spawn_walls(self.DATA), (None, None))                     # an older spawn.yaml
        for bad in ([1, 2, 3], [0, 0, -1, 1], ['a', 0, 1, 1]):
            with self.assertRaises(dp.PlanError):
                dp.spawn_walls(dict(self.DATA, walls=bad))

    def test_ns_class(self):
        self.assertEqual((dp.ns_class('JR81'), dp.ns_class('RR83'), dp.ns_class('XX01')), ('jr', 'rr', None))


class DriverTest(unittest.TestCase):
    def check_commands(self, cmds, speed, accel, omega, alpha, rate=20.0):
        for (t0, v0, w0), (t1, v1, w1) in zip(cmds, cmds[1:]):
            self.assertLessEqual(abs(v1), speed + 1e-9)
            self.assertLessEqual(abs(w1), omega + 1e-9)
            # ramps, except the drop to zero when a segment is reached (from creep speed)
            if v1 != 0.0:
                self.assertLessEqual(abs(v1 - v0), accel / rate + 1e-9)
            if w1 != 0.0:
                self.assertLessEqual(abs(w1 - w0), alpha / rate + 1e-9)
        self.assertEqual(cmds[-1][1:], (0.0, 0.0))

    def test_square_closes(self):
        start = (2.0, -1.0, math.radians(30))
        segs = dp.plan_square(dp.Pose2(*start), 12.5)
        d, t, end, trace, cmds = drive(segs, start)
        self.assertTrue(d.done)
        self.assertLess(math.hypot(end[0] - start[0], end[1] - start[1]), 0.015)
        self.assertLess(abs(dp.wrap(end[2] - start[2])), math.radians(1.5))
        lines = [s for s in segs if isinstance(s, dp.Line)]
        worst = max(min(dist_to_segment(p, (s.x0, s.y0), (s.x1, s.y1)) for s in lines) for p in trace[::10])
        self.assertLess(worst, 0.03)                                  # stays on the square
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(trace[::10], trace[10::10]))
        self.assertAlmostEqual(length, 50.0, delta=0.15)
        self.assertLess(t, 50.0 / 0.6 + 2 * math.pi / 0.5 + 8 * 3.0)  # no dawdling
        self.check_commands(cmds, 0.6, 0.5, 0.5, 1.0)

    def test_rr_square_right_without_measured_twist(self):
        """RR plant (v 1.0, slew 2.5, w 1.5, gain 1.0) and a settle with no twist feedback."""
        start = (0.0, 0.0, math.pi)
        d, t, end, trace, cmds = drive(dp.plan_square(dp.Pose2(*start), 4.0, left=False), start, speed=0.8,
                                       v_max=1.0, a_max=2.5, w_max=1.5, w_gain=1.0, twist=False)
        self.assertTrue(d.done)
        self.assertLess(math.hypot(end[0], end[1]), 0.015)
        self.assertLess(abs(dp.wrap(end[2] - math.pi)), math.radians(1.5))
        self.assertGreater(min(p[1] for p in trace), -0.05)          # right turns from facing -x: +y side
        self.check_commands(cmds, 0.8, 0.5, 0.5, 1.0)

    def test_loop_closes_without_stopping(self):
        start = (0.0, 0.0, 0.0)
        segs = dp.plan_loop(dp.Pose2(*start), 12.5)
        d, t, end, trace, cmds = drive(segs, start)
        self.assertTrue(d.done)
        self.assertLess(math.hypot(end[0], end[1]), 0.015)
        self.assertLess(abs(end[2]), math.radians(1.5))
        radial = max(abs(math.hypot(p[0] - segs[0].cx, p[1] - segs[0].cy) - 6.25) for p in trace[::10])
        self.assertLess(radial, 0.05)
        cruise = [c for c in cmds if 5.0 < c[0] < 60.0]
        self.assertTrue(all(c[1] > 0.55 for c in cruise))            # one continuous loop
        self.check_commands(cmds, 0.6, 0.5, 0.5, 1.0)

    def test_waypoints_reached(self):
        pts = [(3.0, 4.0), (-2.0, 1.0), (0.5, -0.5)]
        d, t, end, trace, cmds = drive(dp.plan_waypoints(dp.Pose2(0, 0, 0), pts), (0.0, 0.0, 0.0))
        self.assertTrue(d.done)
        self.assertLess(math.hypot(end[0] - 0.5, end[1] + 0.5), 0.03)
        for wp in pts:
            self.assertLess(min(math.hypot(p[0] - wp[0], p[1] - wp[1]) for p in trace), 0.03)

    def test_tight_loop_slows_to_hold_the_circle(self):
        """A loop of radius 0.5 m: at 0.6 m/s its feed-forward yaw rate (1.2 rad/s) alone would
        exceed the 0.5 rad/s clamp, so the arc is driven at 0.8 * 0.5 * 0.5 = 0.2 m/s."""
        start = (0.0, 0.0, 0.0)
        segs = dp.plan_loop(dp.Pose2(*start), 1.0)
        d, t, end, trace, cmds = drive(segs, start, t_max=120.0)
        self.assertTrue(d.done)
        self.assertAlmostEqual(d.arc_speed(0.5), 0.2)
        self.assertLessEqual(max(c[1] for c in cmds), 0.2 + 1e-9)
        radial = max(abs(math.hypot(p[0] - segs[0].cx, p[1] - segs[0].cy) - 0.5) for p in trace[::10])
        self.assertLess(radial, 0.05)
        self.assertLess(math.hypot(end[0], end[1]), 0.03)
        self.assertLess(t, 1.5 * d.duration(0.0) + 10.0)
        self.assertAlmostEqual(dp.Driver(segs, 0.6, 0.5, 0.5, 1.0).duration(0.0), math.pi / 0.2)

    def test_duration(self):
        segs = dp.plan_square(dp.Pose2(0, 0, 0), 3.0)
        self.assertAlmostEqual(dp.Driver(segs, 0.6, 0.5, 0.5, 1.0).duration(0.0), 12.0 / 0.6 + 2 * math.pi / 0.5)
        big = dp.plan_loop(dp.Pose2(0, 0, 0), 12.5)                    # 0.8 * 0.5 * 6.25 > 0.6: full speed
        self.assertAlmostEqual(dp.Driver(big, 0.6, 0.5, 0.5, 1.0).duration(0.0), math.pi * 12.5 / 0.6)

    def test_bad_gains(self):
        with self.assertRaises(dp.PlanError):
            dp.Driver([], 0.6, 0.0, 0.5, 1.0)

    def test_empty_plan_is_done(self):
        d = dp.Driver([], 0.6, 0.5, 0.5, 1.0)
        self.assertTrue(d.done)
        self.assertEqual(d.step(0.0, 0, 0, 0), (0.0, 0.0))


class FakeNode:
    """What DriveRunner needs of an rclpy node."""

    def __init__(self):
        self.logs, self.pubs = [], []

    def get_logger(self):
        log = types.SimpleNamespace()
        log.info = lambda m: self.logs.append(('info', m))
        log.warn = lambda m: self.logs.append(('warn', m))
        log.error = lambda m: self.logs.append(('error', m))
        return log

    def create_subscription(self, *a):
        pass

    def create_publisher(self, msg_type, topic, depth):
        pub = types.SimpleNamespace(topic=topic, sent=[])
        pub.publish = pub.sent.append
        self.pubs.append(pub)
        return pub

    def create_timer(self, period, cb):
        return types.SimpleNamespace(cancel=lambda: None)


FAKE_MODULES = {'geometry_msgs': types.ModuleType('geometry_msgs'),
                'geometry_msgs.msg': types.SimpleNamespace(Twist=lambda: types.SimpleNamespace(
                    linear=types.SimpleNamespace(), angular=types.SimpleNamespace())),
                'nav_msgs': types.ModuleType('nav_msgs'), 'nav_msgs.msg': types.SimpleNamespace(Odometry=object),
                'rclpy': types.ModuleType('rclpy'),
                'rclpy.qos': types.SimpleNamespace(HistoryPolicy=types.SimpleNamespace(KEEP_LAST=1),
                                                   ReliabilityPolicy=types.SimpleNamespace(BEST_EFFORT=2),
                                                   QoSProfile=lambda **k: None)}


@unittest.skipUnless(CLASSES.is_file(), 'no src/rover_sim in this checkout mount')
class RunnerGateTest(unittest.TestCase):
    """DriveRunner refuses to command anything that is not a sim agent, and a plan that leaves
    the world's walls, before it publishes anything."""

    SPAWN = dict(SpawnTest.DATA, world='arena', walls=[-8.0, -8.0, 8.0, 8.0])

    def runner(self, argv, env=dp.SIM_SESSION_URI, spawn=None, teleop=None):
        with tempfile.TemporaryDirectory() as d:
            sp, tp = os.path.join(d, 'spawn.yaml'), os.path.join(d, 'teleop_sim.yaml')
            with open(sp, 'w') as f:
                yaml.safe_dump(spawn or self.SPAWN, f)
            if teleop:
                with open(tp, 'w') as f:
                    yaml.safe_dump(teleop, f)
            args = dp.build_parser().parse_args(['--classes', str(CLASSES), '--teleop', tp] + argv)
            node = FakeNode()
            environ = {k: v for k, v in os.environ.items() if k != 'ZENOH_SESSION_CONFIG_URI'}
            if env is not None:
                environ['ZENOH_SESSION_CONFIG_URI'] = env
            with mock.patch.dict(sys.modules, FAKE_MODULES), mock.patch.dict(os.environ, environ, clear=True):
                return dp.DriveRunner(node, args, args.ns, sp), node

    def test_sim_only(self):
        for env in (None, '/etc/zenoh/session.json5'):
            with self.assertRaisesRegex(dp.PlanError, 'not in a rover_sim container'):
                self.runner(['--ns', 'JR81', '--pattern', 'square', '--size', '3'], env=env)
        # Even with --from-here, the agent must be one the sim spawned.
        with self.assertRaisesRegex(dp.PlanError, 'no agent JR02'):
            self.runner(['--ns', 'JR02', '--from-here', '--pattern', 'square', '--size', '3'])
        r, node = self.runner(['--ns', 'JR81', '--from-here', '--pattern', 'square', '--size', '3'])
        self.assertIsNone(r.anchor)
        r.stop_now()                                                  # never zeros to an agent it has not driven
        self.assertEqual(node.pubs[0].sent, [])

    def test_plan_checks(self):
        with self.assertRaisesRegex(dp.PlanError, 'needs --size'):
            self.runner(['--ns', 'JR81'])
        with self.assertRaisesRegex(dp.PlanError, r'drives to \(10.50, 0.00\), within 0.5 m of the walls of arena'):
            self.runner(['--ns', 'JR81', '--size', '12.5'])
        with self.assertRaisesRegex(dp.PlanError, 'walls of arena'):
            self.runner(['--ns', 'RR83', '--pattern', 'waypoints', '--waypoints', '0,0;0,7.8'])
        r, node = self.runner(['--ns', 'JR81', '--size', '3'])
        self.assertEqual((r.anchor, node.pubs[0].topic), (dp.Pose2(-2.0, 0.0, 0.0), '/JR81/cmd_vel_auto'))
        # empty: no walls, so a big square is up to the user.
        self.runner(['--ns', 'JR81', '--size', '12.5'], spawn=dict(SpawnTest.DATA, world='empty'))

    def test_safety_clamp(self):
        r, node = self.runner(['--ns', 'RR83', '--size', '3', '--omega', '1.4', '--speed', '0.95'],
                              teleop=LimitsTest.TELEOP)
        self.assertEqual((r.omega, r.speed), (1.1, 0.9))             # class: 1.5 and 1.0; /** safety_node: 1.1, 0.9
        warns = [m for k, m in node.logs if k == 'warn']
        self.assertTrue(any("over /RR83/safety_node's ang_vel_max 1.1" in m for m in warns), warns)
        r, node = self.runner(['--ns', 'RR83', '--size', '3', '--omega', '1.4'])
        self.assertEqual(r.omega, 1.4)                               # no teleop file: the class limit only
        self.assertTrue(any('clamped to the class limits only' in m for k, m in node.logs))

    def test_main_refuses_before_rclpy_opens_a_session(self):
        inits = []
        rclpy = types.ModuleType('rclpy')
        rclpy.init = lambda **k: inits.append(k)
        rclpy.utilities = types.SimpleNamespace(remove_ros_args=lambda argv: list(argv))
        fake = dict(FAKE_MODULES, rclpy=rclpy, **{'rclpy.node': types.SimpleNamespace(Node=object),
                                                    'rclpy.signals': types.SimpleNamespace(SignalHandlerOptions=None)})
        with tempfile.TemporaryDirectory() as d:
            sp = os.path.join(d, 'spawn.yaml')
            with open(sp, 'w') as f:
                yaml.safe_dump(self.SPAWN, f)
            for env, argv, says in ((None, ['--ns', 'JR81', '--size', '3'], 'not in a rover_sim container'),
                                    (dp.SIM_SESSION_URI, ['--ns', 'JR02', '--size', '3'], 'no agent JR02'),
                                    (dp.SIM_SESSION_URI, ['--ns', 'JR81'], 'needs --size')):
                environ = {k: v for k, v in os.environ.items() if k != 'ZENOH_SESSION_CONFIG_URI'}
                if env:
                    environ['ZENOH_SESSION_CONFIG_URI'] = env
                err = io.StringIO()
                with mock.patch.dict(sys.modules, fake), mock.patch.dict(os.environ, environ, clear=True), \
                        contextlib.redirect_stderr(err):
                    self.assertEqual(dp.main(['drive_pattern.py', '--spawn', sp] + argv), 2)
                self.assertIn(says, err.getvalue())
        self.assertEqual(inits, [])

    def test_no_auto_mode(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            dp.build_parser().parse_args(['--ns', 'JR81', '--auto-mode'])


if __name__ == '__main__':
    unittest.main()
