#!/usr/bin/env python3
"""Unit tests for scripts/truth_odom.py (odometry: truth, NOT PARITY): DLIO's cfg, the frames, the
ground-truth buffer, the gates and names (pure Python, the host), and the cloud (numpy: the images).

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, str(REPO / 'scripts' / (name + '.py')))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


to = load('truth_odom')
oe = sys.modules['odom_eval']
dp = sys.modules['drive_pattern']
sim = load('sim')
try:
    import numpy as np
except ImportError:
    np = None

# The two classes' lidar pitch, as tools/reference/jr02/tf/tf_tree.yaml records the robots' mounts
# (JR02 29.32 deg, RR08 20.5716 deg), and their crop box half-extents.
JR_PITCH, RR_PITCH = 29.32, 20.5716


def pitch_R(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [round(v, 7) for v in (c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)]


def cfg_texts(pitch, crop, calibration=True, extra_dlio='', extra_params=''):
    """Fixture DLIO cfg in its two files' shape (/** blocks, slash keys), built from public values."""
    r = pitch_R(pitch)
    dlio = ('/**:\n  ros__parameters:\n    imu/calibration: %s\n'
            '    extrinsics/baselink2imu/t: [0., 0., 0.]\n    extrinsics/baselink2imu/R: %s\n'
            '    extrinsics/baselink2lidar/t: [0., 0., 0.]\n'
            '    # extrinsics/baselink2lidar/R: [1., 0., 0., 0., 1., 0., 0., 0., 1.]\n'
            '    extrinsics/baselink2lidar/R: %s\n%s' % ('true' if calibration else 'false', r, r, extra_dlio))
    params = ('/**:\n  ros__parameters:\n    frames/odom: odom\n    frames/baselink: base_link\n'
              '    frames/lidar: lidar\n    frames/imu: imu\n    odom/imu/calibration/time: 3.0\n'
              '    odom/preprocessing/cropBoxFilter/size: %s   # half-extent [m]\n%s' % (crop, extra_params))
    return [('dlio.yaml', dlio), ('params.yaml', params)]


def spawn_data(odometry='truth'):
    return {'world': 'arena', 'agents': [
        {'name': 'JR81', 'class': 'jr', 'id': 81, 'x': 2.0, 'y': -1.0, 'yaw': math.pi / 2, 'yaw_deg': 90.0,
         'lidar_height': 0.36, 'reference_robot': 'JR02', 'odometry': odometry},
        {'name': 'RR83', 'class': 'rr', 'id': 83, 'x': -2.0, 'y': 0.0, 'yaw': 0.0, 'yaw_deg': 0.0,
         'lidar_height': 0.40, 'reference_robot': 'RR08'}]}


def close(test, a, b, places=6):
    for x, y in zip(a, b):
        test.assertAlmostEqual(x, y, places=places)


def pitch_of(q):
    return to.rpy_deg(q)[1]


class Cfg(unittest.TestCase):
    def test_classes(self):
        jr, rr = to.load_dlio_cfg(cfg_texts(JR_PITCH, 0.5)), to.load_dlio_cfg(cfg_texts(RR_PITCH, 0.3))
        self.assertAlmostEqual(pitch_of(jr.base_lidar[1]), JR_PITCH, places=4)
        self.assertAlmostEqual(pitch_of(rr.base_lidar[1]), RR_PITCH, places=4)
        self.assertAlmostEqual(pitch_of(rr.base_imu[1]), RR_PITCH, places=4)
        self.assertEqual((jr.crop, rr.crop, jr.hold_s), (0.5, 0.3, 3.0))
        self.assertEqual(rr.frames, {'odom': 'odom', 'baselink': 'base_link', 'imu': 'imu'})
        self.assertEqual(rr.base_lidar[0], (0.0, 0.0, 0.0))
        self.assertEqual(rr.sources, 'dlio.yaml + params.yaml')

    def test_later_file_wins_and_comments_are_comments(self):
        # params.yaml after dlio.yaml (dlio.launch.py's order): its value is the one DLIO runs.
        texts = cfg_texts(JR_PITCH, 0.5, extra_dlio='    odom/preprocessing/cropBoxFilter/size: 1.0\n')
        self.assertEqual(to.load_dlio_cfg(texts).crop, 0.5)
        # The commented-out identity R above the real one is not read.
        self.assertAlmostEqual(pitch_of(to.load_dlio_cfg(texts).base_lidar[1]), JR_PITCH, places=4)

    def test_nested_keys_and_node_blocks(self):
        nested = ('dlio_odom_node:\n  ros__parameters:\n    odom:\n      preprocessing:\n        cropBoxFilter:\n'
                  '          size: 0.25\n')
        cfg = to.load_dlio_cfg(cfg_texts(RR_PITCH, 0.3) + [('extra.yaml', nested)])
        self.assertEqual(cfg.crop, 0.25)
        other = 'some_other_node:\n  ros__parameters:\n    odom/preprocessing/cropBoxFilter/size: 9.0\n'
        self.assertEqual(to.load_dlio_cfg(cfg_texts(RR_PITCH, 0.3) + [('other.yaml', other)]).crop, 0.3)

    def test_calibration_off_holds_nothing(self):
        self.assertEqual(to.load_dlio_cfg(cfg_texts(JR_PITCH, 0.5, calibration=False)).hold_s, 0.0)

    def test_refusals(self):
        (dn, dt), (pn, pt) = cfg_texts(JR_PITCH, 0.5)
        cases = [
            ([(dn, dt.replace('extrinsics/baselink2lidar/R:', '#')), (pn, pt)], 'baselink2lidar'),
            ([(dn, dt), (pn, pt.replace('frames/imu: imu', ''))], 'frames/imu'),
            ([(dn, dt), (pn, pt.replace('cropBoxFilter/size: 0.5', 'cropBoxFilter/size: -1'))], 'cropBoxFilter'),
            ([(dn, dt.replace('imu/calibration: true', 'imu/calibration: 1')), (pn, pt)], 'imu/calibration'),
            ([(dn, dt), (pn, pt.replace('odom/imu/calibration/time: 3.0', ''))], 'calibration/time'),
            ([(dn, dt.replace('[0., 0., 0.]', '[0., 0.]', 1)), (pn, pt)], 'baselink2imu'),
            ([(dn, '- a list'), (pn, pt)], 'not a ROS 2 parameter file'),
            ([(dn, '/**:\n  nothing: 1\n'), (pn, pt)], 'ros__parameters'),
            ([(dn, '{'), (pn, pt)], 'dlio.yaml'),
        ]
        for texts, why in cases:
            with self.assertRaises(to.SetupError) as cm:
                to.load_dlio_cfg(texts)
            self.assertIn(why, str(cm.exception))

    def test_not_a_rotation(self):
        (dn, dt), (pn, pt) = cfg_texts(JR_PITCH, 0.5)
        r = pitch_R(JR_PITCH)
        for bad, why in ((r[:8] + [2.0], 'not a rotation'), ([-v for v in r], 'not a proper rotation')):
            with self.assertRaises(to.SetupError) as cm:
                to.load_dlio_cfg([(dn, dt.replace('baselink2lidar/R: %s' % r, 'baselink2lidar/R: %s' % bad)),
                                  (pn, pt)])
            self.assertIn(why, str(cm.exception))

    def test_read_dir(self):
        with tempfile.TemporaryDirectory() as d:
            for name, text in cfg_texts(RR_PITCH, 0.3):
                Path(d, name).write_text(text)
            self.assertEqual(to.read_dlio_cfg(d).crop, 0.3)
            os.remove(os.path.join(d, 'params.yaml'))
            with self.assertRaises(to.SetupError):
                to.read_dlio_cfg(d)

    def test_rot_round_trip(self):
        rng = random.Random(3)
        for _ in range(50):
            q = oe.qnormalize(tuple(rng.uniform(-1, 1) for _ in range(4)))
            back = to.rot_to_quat(to.quat_to_rot(q))
            if back[3] * q[3] < 0:
                back = tuple(-c for c in back)
            close(self, back, q)


class Spawn(unittest.TestCase):
    def test_truth_agent(self):
        world_map, h = to.check_spawn(spawn_data(), 'JR81')
        self.assertEqual(h, 0.36)
        close(self, world_map[0], (2.0, -1.0, 0.0))
        close(self, world_map[1], oe.yaw_quat(math.pi / 2))

    def test_refuses_dlio_and_unknown(self):
        for data, ns, why in ((spawn_data('dlio'), 'JR81', 'odometry: dlio'),
                              (spawn_data(), 'RR83', 'odometry: dlio'),         # no key: dlio
                              (spawn_data(), 'JR99', 'no agent JR99'),
                              (spawn_data('gps'), 'JR81', "odometry 'gps'")):
            with self.assertRaises(to.SetupError) as cm:
                to.check_spawn(data, ns)
            self.assertIn(why, str(cm.exception))

    def test_spawn_odometry(self):
        self.assertEqual(dp.spawn_odometry(spawn_data(), 'JR81'), 'truth')
        self.assertEqual(dp.spawn_odometry(spawn_data(), 'RR83'), 'dlio')
        with self.assertRaises(dp.PlanError):
            dp.spawn_odometry({'agents': 'x'}, 'JR81')


class Frames(unittest.TestCase):
    def frames(self, pitch=JR_PITCH, h=0.36, spawn=((2.0, -1.0, 0.0), oe.yaw_quat(math.pi / 2))):
        return to.Frames(spawn, h, to.load_dlio_cfg(cfg_texts(pitch, 0.5)))

    def test_at_spawn_base_link_is_the_odom_origin_at_lidar_height(self):
        f = self.frames()
        p, q = f.base_in_odom(((2.0, -1.0, 0.0), oe.yaw_quat(math.pi / 2)))
        close(self, p, (0.0, 0.0, 0.36))
        close(self, q, (0.0, 0.0, 0.0, 1.0))

    def test_moves_in_the_spawn_frame(self):
        # Spawned facing world +y: 1 m along world +y is 1 m along odom +x; a 90 deg left turn more.
        f = self.frames()
        p, q = f.base_in_odom(((2.0, 0.0, 0.0), oe.yaw_quat(math.pi)))
        close(self, p, (1.0, 0.0, 0.36))
        self.assertAlmostEqual(oe.quat_yaw(q), math.pi / 2)

    def test_pitched_chassis_moves_base_link_with_the_lever(self):
        f = self.frames(spawn=oe.IDENTITY)
        pitch = math.radians(10)
        q = (0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2))
        p, _ = f.base_in_odom(((0.0, 0.0, 0.0), q))
        close(self, p, (0.36 * math.sin(pitch), 0.0, 0.36 * math.cos(pitch)))

    def test_lidar_in_odom_is_the_cfg_extrinsic(self):
        f = self.frames(pitch=RR_PITCH, h=0.40, spawn=oe.IDENTITY)
        lidar = f.lidar_in_odom(f.base_in_odom(oe.IDENTITY))
        close(self, lidar[0], (0.0, 0.0, 0.40))
        self.assertAlmostEqual(pitch_of(lidar[1]), RR_PITCH, places=4)

    def test_twist(self):
        f = self.frames(spawn=oe.IDENTITY)
        # Driving forward at 0.5 m/s, turned 90 deg left in odom: linear in odom axes is +y.
        base = f.base_in_odom(((0.0, 0.0, 0.0), oe.yaw_quat(math.pi / 2)))
        v, w = f.twist(base, (0.5, 0.0, 0.0), (0.0, 0.0, 0.3))
        close(self, v, (0.0, 0.5, 0.0))
        self.assertEqual(w, (0.0, 0.0, 0.3))                      # angular stays in base_link axes
        # A pitch rate moves base_link, 0.36 m up, forward: w x lever.
        v, _ = f.twist(f.base_in_odom(oe.IDENTITY), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        close(self, v, (0.36, 0.0, 0.0))


class Buffer(unittest.TestCase):
    def pose(self, x, yaw=0.0):
        return ((x, 0.0, 0.0), oe.yaw_quat(yaw))

    def test_interpolates(self):
        b = to.GtBuffer()
        self.assertTrue(b.add(10.0, self.pose(0.0)))
        self.assertTrue(b.add(10.1, self.pose(1.0, math.pi / 2)))
        p, q = b.at(10.025)
        close(self, p, (0.25, 0.0, 0.0))
        self.assertAlmostEqual(oe.quat_yaw(q), math.pi / 8)
        self.assertEqual(b.at(10.1), self.pose(1.0, math.pi / 2))
        self.assertIsNone(b.at(9.9))
        self.assertIsNone(b.at(10.2))

    def test_order_window_and_restart(self):
        b = to.GtBuffer(keep_s=2.0, restart_s=1.0)
        for k in range(31):
            b.add(10.0 + 0.1 * k, self.pose(k))
        self.assertEqual((b.t[0], b.t[-1]), (11.0, 13.0))         # 2 s kept
        self.assertFalse(b.add(13.0, self.pose(0)))               # not newer: dropped
        self.assertFalse(b.add(12.5, self.pose(0)))               # out of order: dropped
        self.assertTrue(b.add(5.0, self.pose(7)))                 # far older: the bridge restarted
        self.assertEqual((b.t, b.resets), ([5.0], 1))

    def test_scan_pose(self):
        b = to.GtBuffer()
        self.assertEqual(to.scan_pose(b, 10.0, 0.0), ('wait', None))
        self.assertEqual(to.scan_pose(b, 10.0, 1.0), ('drop', None))      # no truth at all
        b.add(10.0, self.pose(0.0))
        b.add(10.1, self.pose(1.0))
        what, p = to.scan_pose(b, 10.05, 0.0)
        self.assertEqual(what, 'pose')
        close(self, p[0], (0.5, 0.0, 0.0))
        self.assertEqual(to.scan_pose(b, 10.12, 0.0), ('wait', None))     # newer than the truth
        self.assertEqual(to.scan_pose(b, 10.12, to.SCAN_WAIT_S), ('held', self.pose(1.0)))
        self.assertEqual(to.scan_pose(b, 9.0, 0.0), ('drop', None))       # older than the buffer

    def test_cloud_gate(self):
        g = to.CloudGate(3.0)
        self.assertFalse(g.open(10.0))                            # paused: no truth yet
        g.saw_truth(10.0)
        g.saw_truth(11.0)                                         # the first one counts
        self.assertFalse(g.open(12.9))
        self.assertTrue(g.open(13.0))
        z = to.CloudGate(0.0)
        z.saw_truth(10.0)
        self.assertTrue(z.open(10.0))


class Names(unittest.TestCase):
    def test_session_payload(self):
        text, sid = to.session_payload('JR81', 1792510852383482000)
        self.assertEqual(sid, '1792510852383482000-truth-JR81')
        payload = json.loads(text)
        self.assertEqual((payload['id'], payload['ns']), (sid, 'JR81'))
        self.assertAlmostEqual(payload['start'], 1792510852.383482, places=5)
        self.assertEqual(sim.dlio_session_id(text), sid)           # what arm and status read
        self.assertEqual(int(sid.split('-')[0]), 1792510852383482000)   # dlio_session.py's start

    def test_node_name_is_the_parity_sim_only(self):
        allowed = yaml.safe_load((REPO / 'tools' / 'reference' / 'jr02' / 'allowed_deviations.yaml').read_text())
        self.assertIn('/{ns}/' + to.node_name('{ns}'), allowed['odometry']['truth']['sim_only'])

    def test_constants_agree(self):
        self.assertEqual(to.ODOMETRY_MODES, sim.ODOMETRY_MODES)
        self.assertEqual(dp.ODOMETRY_MODES, sim.ODOMETRY_MODES)
        self.assertEqual(to.DLIO_CFG_DIR, sim.DLIO_CFG_IN)
        self.assertEqual(to.SPAWN_DEFAULT, sim.run_path(sim.SPAWN_FILE))

    def test_summary(self):
        line = to.summary('RR83', to.load_dlio_cfg(cfg_texts(RR_PITCH, 0.3)), 0.40)
        for part in ('baselink2lidar rpy (0.0000, 20.5716, 0.0000) deg', 'crop 0.30 m', 'IMU calibration 3.0 s',
                     'frames RR83/odom RR83/base_link RR83/imu', 'lidar height 0.40 m'):
            self.assertIn(part, line)


class Main(unittest.TestCase):
    def files(self, d, odometry='truth'):
        Path(d, 'spawn.yaml').write_text(yaml.safe_dump(spawn_data(odometry)))
        for name, text in cfg_texts(JR_PITCH, 0.5):
            Path(d, name).write_text(text)
        return ['--spawn', os.path.join(d, 'spawn.yaml'), '--dlio-cfg', d]

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = to.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_check(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.run_main(['JR81', '--check'] + self.files(d))
            self.assertEqual(rc, 0)
            self.assertIn('baselink2lidar rpy (0.0000, 29.3200, 0.0000) deg', out)

    def test_refusals_exit_2_before_ros(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _, err = self.run_main(['JR81', '--check'] + self.files(d, 'dlio'))
            self.assertEqual(rc, 2)
            self.assertIn('refusing to run beside it', err)
            rc, _, err = self.run_main(['JR81', '--spawn', os.path.join(d, 'none.yaml'), '--check'])
            self.assertEqual(rc, 2)
            # Not --check, off the sim's session: refused before rclpy is imported.
            env = os.environ.pop('ZENOH_SESSION_CONFIG_URI', None)
            try:
                rc, _, err = self.run_main(['JR81'] + self.files(d))
            finally:
                if env is not None:
                    os.environ['ZENOH_SESSION_CONFIG_URI'] = env
            self.assertEqual(rc, 2)
            self.assertIn('not in a rover_sim container', err)


@unittest.skipIf(np is None, 'numpy: the gazebo and DLIO images have it')
class Cloud(unittest.TestCase):
    def raw(self, pts):
        """livox_sim_bridge.py's 26-byte points (x y z intensity, tag line, timestamp)."""
        return b''.join(struct.pack('<4f2Bd', x, y, z, i, 0, 0, 7.5) for x, y, z, i in pts)

    def test_crop(self):
        xyz = np.array([[0.5, 0.5, 0.5], [0.51, 0.0, 0.0], [0.0, 0.0, -0.49], [np.nan, 1.0, 1.0], [2.0, 0.0, 0.0]])
        self.assertEqual(to.crop_mask(np, xyz, 0.5).tolist(), [False, True, False, False, True])

    def test_layout_and_transform(self):
        cfg = to.load_dlio_cfg(cfg_texts(RR_PITCH, 0.3))
        f = to.Frames(((1.0, 2.0, 0.0), oe.yaw_quat(0.7)), 0.40, cfg)
        chassis = ((3.0, 1.0, 0.02), oe.qnormalize((0.01, -0.02, 0.4, 0.9)))
        odom_lidar = f.lidar_in_odom(f.base_in_odom(chassis))
        pts = [(2.0, 0.5, -0.3, 10.0), (0.1, 0.1, 0.1, 20.0), (5.0, -1.0, 1.0, 30.0)]   # the 2nd is cropped
        data, n = to.cloud_out(np, self.raw(pts), len(pts), odom_lidar, cfg.crop)
        self.assertEqual((n, len(data)), (2, 2 * to.OUT_STEP))
        out = np.frombuffer(data, dtype=to.out_dtype(np))
        for row, (x, y, z, i) in zip(out, (pts[0], pts[2])):
            want = oe.compose(odom_lidar, ((x, y, z), (0.0, 0.0, 0.0, 1.0)))[0]
            close(self, (row['x'], row['y'], row['z']), want, places=4)
            self.assertEqual((row['intensity'], row['timestamp']), (i, 7.5))
        # The fields DLIO's pcl::toROSMsg declares, all inside its 32-byte point.
        self.assertEqual([f[1] for f in to.OUT_FIELDS], [0, 4, 8, 16, 24, 24, 24])
        self.assertEqual(to.out_dtype(np).itemsize, to.OUT_STEP)
        self.assertEqual(to.livox_dtype(np).itemsize, 26)


if __name__ == '__main__':
    unittest.main()
