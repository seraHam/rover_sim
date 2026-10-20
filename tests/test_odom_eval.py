"""Unit tests for scripts/odom_eval.py's math and grading (pure Python, no rclpy).

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('odom_eval', str(REPO / 'scripts' / 'odom_eval.py'))
oe = importlib.util.module_from_spec(_spec)
sys.modules['odom_eval'] = oe
_spec.loader.exec_module(oe)

T0 = 1790000000 * oe.NS_PER_S        # a wall stamp, as the bridge's
H = 0.36


def rand_pose(rng):
    q = oe.qnormalize(tuple(rng.uniform(-1, 1) for _ in range(4)))
    return (tuple(rng.uniform(-5, 5) for _ in range(3)), q)


def rot_matrix(q):
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def chassis_path(n=3600, rate=100.0):
    """Ground truth chassis poses: 10 m along the heading, a 90 deg turn in place, 5 m more,
    then still, from a spawn at (2, -1) facing +y. [(t s, x, y, yaw)]"""
    out, v, w = [], 0.5, 0.5
    x, y, yaw = 2.0, -1.0, math.pi / 2
    t1, t2 = 10.0 / v, 10.0 / v + (math.pi / 2) / w
    for k in range(n):
        t = k / rate
        if t <= t1:
            px, py, pyaw = x, y + v * t, yaw
        elif t <= t2:
            px, py, pyaw = x, y + 10.0, yaw - w * (t - t1)
        else:
            px, py, pyaw = x + v * min(t - t2, 10.0), y + 10.0, 0.0
        out.append((t, px, py, pyaw))
    return out


def raw_from(path, spawn, dlio_of, map_odom=None, h=H):
    """A recording: truth from the chassis path, DLIO = what dlio_of(t, true base_link in
    world) says base_link is in the world, expressed in <ns>/odom as DLIO publishes it."""
    world_odom = oe.compose(spawn, map_odom or oe.IDENTITY)
    truth, dlio = [], []
    for t, x, y, yaw in path:
        ns = T0 + int(round(t * oe.NS_PER_S))
        truth.append([ns, x, y, 0.0] + list(oe.yaw_quat(yaw)))
        base = ((x, y, h), oe.yaw_quat(yaw))
        est = dlio_of(t, base)
        if est is None:
            continue
        p = oe.compose(oe.inverse(world_odom), est)
        dlio.append([ns] + list(p[0]) + list(p[1]) + [ns + 4000000])
    return {'ns': 'JR81', 'lidar_height': h, 'world_map': oe.pose_list(spawn),
            'map_odom': oe.pose_list(map_odom) if map_odom else None, 'truth': truth, 'dlio': dlio}


SPAWN = ((2.0, -1.0, 0.0), oe.yaw_quat(math.pi / 2))


class Se3Test(unittest.TestCase):
    def test_rotation_matches_matrix(self):
        rng = random.Random(1)
        for _ in range(50):
            q = rand_pose(rng)[1]
            v = tuple(rng.uniform(-3, 3) for _ in range(3))
            m = rot_matrix(q)
            want = tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))
            for a, b in zip(oe.qrot(q, v), want):
                self.assertAlmostEqual(a, b, places=12)

    def test_compose_inverse(self):
        rng = random.Random(2)
        for _ in range(50):
            a, b = rand_pose(rng), rand_pose(rng)
            ident = oe.compose(a, oe.inverse(a))
            self.assertTrue(oe.is_identity(ident, 1e-9), ident)
            back = oe.compose(oe.inverse(a), oe.compose(a, b))
            for u, v in zip(back[0], b[0]):
                self.assertAlmostEqual(u, v, places=9)
            self.assertLess(oe.quat_angle(oe.qmul(oe.qconj(back[1]), b[1])), 1e-7)

    def test_compose_planar(self):
        a = ((1.0, 2.0, 0.0), oe.yaw_quat(math.pi / 2))
        b = ((3.0, 0.0, 0.5), oe.yaw_quat(0.25))
        c = oe.compose(a, b)
        self.assertAlmostEqual(c[0][0], 1.0)
        self.assertAlmostEqual(c[0][1], 5.0)
        self.assertAlmostEqual(c[0][2], 0.5)
        self.assertAlmostEqual(oe.quat_yaw(c[1]), math.pi / 2 + 0.25)

    def test_yaw_and_angle(self):
        for yaw in (-3.0, -1.0, 0.0, 0.7, 3.1):
            self.assertAlmostEqual(oe.quat_yaw(oe.yaw_quat(yaw)), yaw)
            self.assertAlmostEqual(oe.quat_angle(oe.yaw_quat(yaw)), abs(yaw))
        self.assertAlmostEqual(oe.quat_angle(tuple(-c for c in oe.yaw_quat(0.3))), 0.3)   # double cover

    def test_slerp(self):
        q0, q1 = oe.yaw_quat(0.2), oe.yaw_quat(1.4)
        self.assertAlmostEqual(oe.quat_yaw(oe.slerp(q0, q1, 0.25)), 0.5)
        self.assertAlmostEqual(oe.quat_yaw(oe.slerp(q0, tuple(-c for c in q1), 0.5)), 0.8)
        self.assertAlmostEqual(oe.quat_yaw(oe.slerp(q0, oe.yaw_quat(0.2001), 0.5)), 0.20005)
        # across +-pi: the short way round
        self.assertAlmostEqual(abs(oe.quat_yaw(oe.slerp(oe.yaw_quat(3.0), oe.yaw_quat(-3.0), 0.5))), math.pi)


class TrackTest(unittest.TestCase):
    def setUp(self):
        self.tr = oe.Track([(t, ((t, 2 * t, 0.0), oe.yaw_quat(t))) for t in (2.0, 0.0, 1.0, 3.0, 3.5)])

    def test_interpolation(self):
        p = self.tr.pose_at(0.25, 1.0)
        self.assertAlmostEqual(p[0][0], 0.25)
        self.assertAlmostEqual(p[0][1], 0.5)
        self.assertAlmostEqual(oe.quat_yaw(p[1]), 0.25)
        self.assertEqual(self.tr.position_at(3.5, 1.0), (3.5, 7.0, 0.0))    # the last sample itself

    def test_outside_and_gaps(self):
        self.assertIsNone(self.tr.pose_at(-0.01, 1.0))
        self.assertIsNone(self.tr.pose_at(3.6, 1.0))
        self.assertIsNone(self.tr.position_at(2.5, 0.9))                   # 2 -> 3 is a 1 s gap
        self.assertIsNotNone(self.tr.position_at(3.2, 0.9))


class HelpersTest(unittest.TestCase):
    def test_path_length_ignores_jitter(self):
        pts = [(0.001 * (i % 2), 0.0, 0.0) for i in range(1000)]          # 1 mm jitter at rest
        self.assertEqual(oe.path_length(pts), 0.0)
        line = [(0.01 * i, 0.0, 0.0) for i in range(501)]
        self.assertAlmostEqual(oe.path_length(line), 5.0, places=6)

    def test_se2_align(self):
        rng = random.Random(3)
        ref = [(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(100)]
        th, dx, dy = 0.3, 0.5, -0.2
        c, s = math.cos(th), math.sin(th)
        # est = R(-th) (ref - d): the inverse motion maps est back onto ref
        est = [(c * (x - dx) + s * (y - dy), -s * (x - dx) + c * (y - dy)) for x, y in ref]
        ax, ay, ath, err = oe.se2_align(est, ref)
        self.assertAlmostEqual(ath, th)
        self.assertAlmostEqual(ax, dx)
        self.assertAlmostEqual(ay, dy)
        self.assertLess(err, 1e-9)

    def test_best_offset_stays_in_range(self):
        tr = oe.Track([(k / 100.0, ((0.01 * k, 0.0, 0.0), oe.IDENTITY[1])) for k in range(3000)])
        dl = [(k / 100.0, tr.position_at(k / 100.0 - 0.5, 0.1)) for k in range(100, 2900)]
        d, _ = oe.best_offset(dl, tr, 0.1, search=0.3)
        self.assertGreaterEqual(d, -0.3)

    def test_best_offset_finds_a_delay(self):
        tr = oe.Track([(k / 100.0, ((math.sin(k / 100.0), math.cos(k / 100.0), 0.0), oe.IDENTITY[1]))
                       for k in range(3000)])
        dl = [(k / 100.0, tr.position_at(k / 100.0 - 0.047, 0.1)) for k in range(100, 2900)]
        d, e = oe.best_offset(dl, tr, 0.1)
        self.assertAlmostEqual(d, -0.047, places=3)
        self.assertLess(e, 1e-3)


class EvaluateTest(unittest.TestCase):
    def test_perfect(self):
        r = oe.evaluate(raw_from(chassis_path(), SPAWN, lambda t, b: b))
        self.assertEqual(r['verdict'], 'PASS')
        self.assertLess(r['ate_m']['rmse'], 1e-9)
        self.assertLess(r['yaw_deg']['max'], 1e-6)
        self.assertAlmostEqual(r['path_length_m'], 15.0, delta=0.01)
        self.assertEqual(r['samples']['dlio_matched'], 3600)
        self.assertAlmostEqual(r['rates_hz']['dlio_stamps'], 100.0, places=6)
        self.assertTrue(r['line'].startswith('odom_eval JR81: PASS ATE 0.000 m < 0.10 m'), r['line'])

    def test_constant_offset(self):
        def off(t, b):
            return ((b[0][0] + 0.03, b[0][1] - 0.04, b[0][2] + 0.02), b[1])
        r = oe.evaluate(raw_from(chassis_path(), SPAWN, off))
        self.assertAlmostEqual(r['ate_m']['xy_rmse'], 0.05, places=9)
        self.assertAlmostEqual(r['ate_m']['xy_max'], 0.05, places=9)
        self.assertAlmostEqual(r['ate_m']['z_rmse'], 0.02, places=9)
        self.assertAlmostEqual(r['ate_m']['z_mean'], 0.02, places=9)
        self.assertAlmostEqual(r['ate_m']['rmse'], math.sqrt(0.05 ** 2 + 0.02 ** 2), places=9)
        self.assertAlmostEqual(r['end']['xy_m'], 0.05, places=9)
        self.assertAlmostEqual(r['end']['drift_pct'], 100 * 0.05 / 15.0, places=3)
        self.assertLess(r['se2_aligned']['xy_rmse_m'], 1e-9)
        self.assertAlmostEqual(r['se2_aligned']['dx_m'], -0.03, places=9)
        self.assertEqual(r['verdict'], 'PASS')

    def test_drift_with_distance(self):
        """A 1 % scale error from the start: the error grows with the distance from it."""
        def drift(t, b):
            sx, sy = SPAWN[0][0], SPAWN[0][1]
            return ((sx + 1.01 * (b[0][0] - sx), sy + 1.01 * (b[0][1] - sy), b[0][2]), b[1])
        r = oe.evaluate(raw_from(chassis_path(), SPAWN, drift))
        end = 0.01 * math.hypot(5.0, 10.0)
        self.assertAlmostEqual(r['end']['xy_m'], end, places=6)
        self.assertAlmostEqual(r['end']['drift_pct'], 100 * end / 15.0, places=2)
        self.assertAlmostEqual(r['ate_m']['xy_max'], end, places=6)
        self.assertEqual(r['verdict'], 'PASS')

    def test_fail_and_target(self):
        def off(t, b):
            return ((b[0][0] + 0.12, b[0][1], b[0][2]), b[1])
        raw = raw_from(chassis_path(), SPAWN, off)
        self.assertEqual(oe.evaluate(raw)['verdict'], 'FAIL')
        self.assertIn('>= 0.10 m target', oe.evaluate(raw)['line'])
        self.assertEqual(oe.evaluate(raw, target=0.15)['verdict'], 'PASS')

    def test_yaw_error_only(self):
        def yawed(t, b):
            return (b[0], oe.qmul(b[1], oe.yaw_quat(math.radians(2.0))))
        r = oe.evaluate(raw_from(chassis_path(), SPAWN, yawed))
        self.assertAlmostEqual(r['yaw_deg']['rmse'], 2.0, places=6)
        self.assertAlmostEqual(r['yaw_deg']['mean'], 2.0, places=6)
        self.assertAlmostEqual(r['rotation_deg']['max'], 2.0, places=6)
        self.assertLess(r['ate_m']['rmse'], 1e-9)

    def test_tilt_shows_in_rotation_not_yaw(self):
        def tilted(t, b):
            return (b[0], oe.qmul(b[1], (math.sin(math.radians(1.5) / 2), 0.0, 0.0, math.cos(math.radians(1.5) / 2))))
        r = oe.evaluate(raw_from(chassis_path(), SPAWN, tilted))
        self.assertLess(r['yaw_deg']['max'], 1e-6)
        self.assertAlmostEqual(r['rotation_deg']['rmse'], 1.5, places=6)

    def test_delayed_stamps(self):
        """DLIO stamped t carrying where the agent was 60 ms earlier: graded at offset 0,
        found by the diagnostic."""
        path = chassis_path()
        lookup = {round(t, 2): (x, y, yaw) for t, x, y, yaw in path}

        def late(t, b):
            if t < 0.06:
                return None
            x, y, yaw = lookup[round(t - 0.06, 2)]
            return ((x, y, H), oe.yaw_quat(yaw))
        r = oe.evaluate(raw_from(path, SPAWN, late))
        self.assertAlmostEqual(r['best_offset']['offset_s'], -0.06, places=3)
        self.assertLess(r['best_offset']['ate_rmse_m'], 1e-6)
        self.assertAlmostEqual(r['ate_m']['xy_max'], 0.5 * 0.06, places=6)      # at 0.5 m/s
        self.assertIn('best offset -60 ms', r['line'])

    def test_spawn_and_map_odom_are_composed(self):
        mo = ((0.4, -0.3, 0.0), oe.yaw_quat(0.2))
        raw = raw_from(chassis_path(), SPAWN, lambda t, b: b, map_odom=mo)
        self.assertLess(oe.evaluate(raw)['ate_m']['rmse'], 1e-9)
        raw['map_odom'] = None                                  # graded as identity: off by mo
        r = oe.evaluate(raw)
        self.assertGreater(r['ate_m']['xy_rmse'], 0.4)
        spawned = dict(raw_from(chassis_path(), SPAWN, lambda t, b: b), world_map=oe.pose_list(oe.IDENTITY))
        self.assertGreater(oe.evaluate(spawned)['ate_m']['xy_rmse'], 1.0)

    def test_lever_arm(self):
        raw = raw_from(chassis_path(), SPAWN, lambda t, b: b, h=0.40)
        self.assertLess(oe.evaluate(raw)['ate_m']['z_max'], 1e-9)
        raw['lidar_height'] = 0.36
        self.assertAlmostEqual(oe.evaluate(raw)['ate_m']['z_mean'], 0.04, places=9)

    def test_startup_samples_are_dropped(self):
        raw = raw_from(chassis_path(), SPAWN, lambda t, b: b)
        raw['dlio'][:0] = [[0, 0.0, 0.0, H, 0.0, 0.0, 0.0, 1.0, T0]] * 5           # before any IMU
        raw['dlio'][5:5] = [[T0 + 1000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, T0]] * 3  # before the seed
        raw['dlio'].append(list(raw['dlio'][-1]))                                    # a repeated stamp
        raw['dlio'].append([T0 + 2000, 1.0, 0.0, H, 0.0, 0.0, 0.0, 0.0, T0])          # no rotation at all
        raw['truth'].append([T0 + 3000, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        r = oe.evaluate(raw)
        s = r['samples']
        self.assertEqual((s['dlio_zero_stamp'], s['dlio_pre_seed'], s['dlio_duplicate_stamps'],
                          s['dlio_bad_quaternion'], s['truth_dropped']), (5, 3, 1, 1, 1))
        self.assertEqual(s['dlio_matched'], 3601)
        self.assertLess(r['ate_m']['rmse'], 1e-9)

    def test_unmatched_outside_truth(self):
        raw = raw_from(chassis_path(), SPAWN, lambda t, b: b)
        raw['truth'] = raw['truth'][100:-100]
        r = oe.evaluate(raw)
        self.assertEqual(r['samples']['dlio_unmatched'], 200)
        raw['truth'] = []
        self.assertEqual(oe.evaluate(raw)['verdict'], 'NO DATA')

    def test_seed_pose_before_dlio_runs_is_not_graded(self):
        """The seed pose DLIO holds through its IMU calibration equals the truth by construction:
        grading starts at dlio_live_ns, the first sample DLIO computed."""
        def off(t, b):                       # 5 cm off once DLIO runs (t >= 3 s)
            return b if t < 3.0 else ((b[0][0] + 0.05, b[0][1], b[0][2]), b[1])
        path = [(k / 100.0, 2.0, -1.0, math.pi / 2) for k in range(600)]
        raw = raw_from(path, SPAWN, off)
        self.assertAlmostEqual(oe.evaluate(raw)['ate_m']['rmse'], 0.05 * math.sqrt(0.5), places=9)   # old recording
        raw['dlio_live_ns'] = T0 + 3 * oe.NS_PER_S
        r = oe.evaluate(raw)
        self.assertAlmostEqual(r['ate_m']['rmse'], 0.05, places=9)
        self.assertEqual((r['samples']['dlio_pre_live'], r['samples']['dlio_matched']), (300, 300))
        raw['dlio_live_ns'] = None                                  # DLIO never ran while recording
        r = oe.evaluate(raw)
        self.assertEqual(r['verdict'], 'NO DATA')
        self.assertIn('(600 before DLIO ran)', r['line'])

    def test_seed_pose_at_rest_is_graded(self):
        """DLIO's seed pose (0, 0, h) in odom while calibrating is the spawn pose: no error."""
        path = [(k / 100.0, 2.0, -1.0, math.pi / 2) for k in range(300)]
        r = oe.evaluate(raw_from(path, SPAWN, lambda t, b: b))
        self.assertLess(r['ate_m']['rmse'], 1e-12)
        self.assertEqual(r['path_length_m'], 0.0)
        self.assertIsNone(r['end']['drift_pct'])
        self.assertIn('no path', r['line'])
        self.assertIn('barely moved', r['best_offset']['note'])
        self.assertIsNone(r['best_offset']['offset_s'])
        self.assertIn('best offset n/a', r['line'])


class LiveGateTest(unittest.TestCase):
    ZERO = (0.0,) * 6
    SEED = [0.0, 0.0, H, 0.0, 0.0, 0.0, 1.0]

    def test_startup_sequence(self):
        g = oe.LiveGate()
        g.feed([0] + self.SEED, self.ZERO, 1.0)                              # no IMU yet
        g.feed([T0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], self.ZERO, 2.0)     # no seed yet
        self.assertIsNone(g.seed)
        for k in range(300):                                                  # calibrating, still
            g.feed([T0 + k] + self.SEED, self.ZERO, 3.0)
        self.assertIsNone(g.since)
        g.feed([T0 + 400, 1e-5, 0.0, H, 0.0, 0.0, 0.0, 1.0], self.ZERO, 6.0)   # running
        self.assertEqual((g.since, g.since_ns), (6.0, T0 + 400))
        g.feed([T0 + 500, 2e-5, 0.0, H, 0.0, 0.0, 0.0, 1.0], self.ZERO, 7.0)
        self.assertEqual((g.since, g.since_ns), (6.0, T0 + 400))

    def test_twist_alone_counts(self):
        g = oe.LiveGate()
        g.feed([T0] + self.SEED, self.ZERO, 1.0)
        g.feed([T0 + 1] + self.SEED, (0.0, 0.0, 0.0, 0.0, 0.0, 1e-4), 1.5)
        self.assertEqual((g.since, g.since_ns), (1.5, T0 + 1))
        g = oe.LiveGate()                                    # recording started while DLIO already ran
        g.feed([T0 + 7] + self.SEED, (0.01, 0.0, 0.0, 0.0, 0.0, 0.0), 1.0)
        self.assertEqual(g.since_ns, T0 + 7)


class CliTest(unittest.TestCase):
    def test_offline_round_trip(self):
        def off(t, b):
            return ((b[0][0], b[0][1] + 0.08, b[0][2]), b[1])
        raw = raw_from(chassis_path(), SPAWN, off)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            src, out = os.path.join(d, 'raw.json'), os.path.join(d, 'report.json')
            with open(src, 'w') as f:
                json.dump(raw, f)
            self.assertEqual(oe.main(['--offline', src, '--out', out]), 0)
            with open(out) as f:
                rep = json.load(f)
            self.assertAlmostEqual(rep['ate_m']['xy_rmse'], 0.08, places=9)
            self.assertEqual(oe.main(['--offline', src, '--out', out, '--target', '0.05']), 1)
            raw['drive'] = {'code': 3}
            with open(src, 'w') as f:
                json.dump(raw, f)
            self.assertEqual(oe.main(['--offline', src, '--out', out]), 3)
            with open(out) as f:
                self.assertTrue(json.load(f)['line'].endswith('DRIVE DID NOT COMPLETE (timeout)'))
            raw['truth'] = []
            with open(src, 'w') as f:
                json.dump(raw, f)
            self.assertEqual(oe.main(['--offline', src, '--out', out]), 2)

    def test_drive_refused_outside_the_sim(self):
        """--drive is refused before rclpy is even imported when this is not a sim container."""
        data = {'agents': [{'name': 'JR81', 'class': 'jr', 'id': 81, 'x': -2.0, 'y': 0.5, 'yaw': 0.5,
                            'lidar_height': 0.36, 'reference_robot': 'JR02'}]}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'spawn.yaml')
            with open(p, 'w') as f:
                yaml.safe_dump(data, f)
            environ = {k: v for k, v in os.environ.items() if k != 'ZENOH_SESSION_CONFIG_URI'}
            err = io.StringIO()
            with unittest.mock.patch.dict(os.environ, environ, clear=True), \
                    unittest.mock.patch.dict(sys.modules, {'rclpy': None}), contextlib.redirect_stderr(err):
                self.assertEqual(oe.main(['--ns', 'JR81', '--spawn', p, '--drive', '--size', '3']), 2)
            self.assertIn('not in a rover_sim container', err.getvalue())
            with unittest.mock.patch.dict(os.environ, {'ZENOH_SESSION_CONFIG_URI': oe.drive_pattern.SIM_SESSION_URI}), \
                    unittest.mock.patch.dict(sys.modules, {'rclpy': None}), contextlib.redirect_stderr(err):
                self.assertEqual(oe.main(['--ns', 'JR81', '--spawn', p, '--drive', '--pattern', 'loop']), 2)
            self.assertIn('needs --size', err.getvalue())

    def test_split_argv(self):
        self.assertEqual(oe.split_argv(['--ns', 'JR81', '--drive', '--pattern', 'loop']),
                         (['--ns', 'JR81'], ['--pattern', 'loop']))
        self.assertEqual(oe.split_argv(['--ns', 'JR81']), (['--ns', 'JR81'], None))

    def test_spawn_pose(self):
        data = {'agents': [{'name': 'JR81', 'class': 'jr', 'id': 81, 'x': -2.0, 'y': 0.5, 'yaw': 0.5,
                            'yaw_deg': math.degrees(0.5), 'lidar_height': 0.36, 'reference_robot': 'JR02'},
                           {'name': 'RR83', 'class': 'rr', 'id': 83, 'x': 2.0, 'y': 0.0, 'yaw': math.pi,
                            'yaw_deg': 180.0, 'lidar_height': 0.40, 'reference_robot': 'RR08'}]}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'spawn.yaml')
            with open(p, 'w') as f:
                yaml.safe_dump(data, f)
            pose, h = oe.spawn_pose(p, 'RR83')
            self.assertEqual(pose[0], (2.0, 0.0, 0.0))
            self.assertAlmostEqual(oe.quat_yaw(pose[1]), math.pi)
            self.assertEqual(h, 0.40)
            with self.assertRaises(oe.drive_pattern.PlanError):
                oe.spawn_pose(p, 'JR99')
            with self.assertRaises(oe.drive_pattern.PlanError):
                oe.spawn_pose(os.path.join(d, 'missing.yaml'), 'JR81')

    def test_check_frames(self):
        f = oe.check_frames({'dlio': ['JR81/odom', 'JR81/base_link'], 'truth': ['world', 'chassis']}, 'JR81')
        self.assertTrue(f['dlio']['ok'])
        self.assertFalse(f['truth']['ok'])
        self.assertIsNone(oe.check_frames({}, 'JR81')['dlio']['seen'])


if __name__ == '__main__':
    unittest.main()
