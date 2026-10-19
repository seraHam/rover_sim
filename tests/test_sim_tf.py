"""Unit tests for scripts/sim_tf.py's pure logic (no rclpy).

    python3 -m unittest discover -s tests -v
"""
import importlib.util
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('sim_tf', str(REPO / 'scripts' / 'sim_tf.py'))
sim_tf = importlib.util.module_from_spec(_spec)
sys.modules['sim_tf'] = sim_tf
_spec.loader.exec_module(sim_tf)


def spawn(*agents):
    return {'agents': [dict(a) for a in agents]}


JR = {'name': 'JR81', 'class': 'jr', 'id': 81, 'x': -2.0, 'y': 0.0, 'yaw': 0.0, 'yaw_deg': 0.0,
      'lidar_height': 0.36, 'reference_robot': 'JR02'}
RR = {'name': 'RR83', 'class': 'rr', 'id': 83, 'x': 2.0, 'y': 1.0, 'yaw': math.pi, 'yaw_deg': 180.0,
      'lidar_height': 0.40, 'reference_robot': 'RR08'}


def compose(a, b):
    """Planar (x, y, yaw) composition a * b."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def quat_yaw(q):
    return 2 * math.atan2(q[2], q[3])


class SpawnTest(unittest.TestCase):
    def test_parse(self):
        poses = sim_tf.spawn_poses(spawn(JR, RR))
        self.assertEqual(poses, [sim_tf.SpawnPose('JR81', -2.0, 0.0, 0.0),
                                 sim_tf.SpawnPose('RR83', 2.0, 1.0, math.pi)])

    def test_load_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'spawn.yaml')
            with open(p, 'w') as f:
                yaml.safe_dump(spawn(JR, RR), f)
            self.assertEqual([q.name for q in sim_tf.load_spawn(p)], ['JR81', 'RR83'])
            with open(p, 'w') as f:
                f.write('agents: [\n')
            with self.assertRaises(sim_tf.SpawnError):
                sim_tf.load_spawn(p)

    def test_yaw_deg_optional_but_checked(self):
        a = dict(JR)
        del a['yaw_deg']
        sim_tf.spawn_poses(spawn(a))
        b = dict(RR, yaw=-math.pi)          # same angle, other branch
        sim_tf.spawn_poses(spawn(b))
        with self.assertRaisesRegex(sim_tf.SpawnError, 'disagree'):
            sim_tf.spawn_poses(spawn(dict(JR, yaw_deg=90.0)))     # degrees written as yaw?

    def test_errors(self):
        bad = [
            ({}, 'agents'),
            (spawn(), 'agents'),
            (spawn(dict(JR, name='jr81')), 'name'),
            (spawn(JR, dict(RR, name='JR81')), 'duplicate'),
            (spawn(dict(JR, x=float('nan'))), 'x'),
            (spawn(dict(JR, y='1')), 'y'),
            (spawn(dict(JR, yaw=True)), 'yaw'),
            ({'agents': ['JR81']}, 'mapping'),
        ]
        for data, msg in bad:
            with self.assertRaisesRegex(sim_tf.SpawnError, msg):
                sim_tf.spawn_poses(data)


class TransformTest(unittest.TestCase):
    def test_world_to_map(self):
        p = sim_tf.spawn_poses(spawn(RR))[0]
        parent, child, t, q = sim_tf.world_to_map(p)
        self.assertEqual((parent, child), ('world', 'RR83/map'))
        self.assertEqual(t, (2.0, 1.0, 0.0))
        self.assertAlmostEqual(sum(v * v for v in q), 1.0)
        self.assertAlmostEqual(q[0], 0.0)
        self.assertAlmostEqual(q[1], 0.0)
        self.assertAlmostEqual(abs(quat_yaw(q)), math.pi)

    def test_yaw_quat(self):
        for yaw in (0.0, 0.3, -1.2, math.pi / 2, 3.0):
            self.assertAlmostEqual(quat_yaw(sim_tf.yaw_quat(yaw)), yaw)

    def test_args(self):
        a = sim_tf.parse_args([])
        self.assertEqual((a.spawn, a.rate), ('/rover_sim/run/spawn.yaml', 10.0))
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sim_tf.parse_args(['--rate', '0'])


def load_sim():
    if 'sim' in sys.modules:                 # test_sim.py's copy, when run together
        return sys.modules['sim']
    spec = importlib.util.spec_from_file_location('sim', str(REPO / 'scripts' / 'sim.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['sim'] = mod
    spec.loader.exec_module(mod)
    return mod


class SimPyContract(unittest.TestCase):
    """sim.py writes spawn.yaml and the frame_align pairs; sim_tf reads the one and its
    world -> <ns>/map frames must agree with the other."""

    @classmethod
    def setUpClass(cls):
        cls.sim = sim = load_sim()
        if not all(hasattr(sim, f) for f in ('spawn_table', 'generated_yaml', 'read_pins',
                                             'frame_align_pairs', 'frame_align_transform')):
            raise unittest.SkipTest('sim.py has no spawn_table/frame_align yet')

    def sessions(self):
        return [self.sim.load_session(str(p)) for p in sorted(self.sim.SESSIONS_DIR.glob('*.yaml'))]

    def test_spawn_yaml_parses(self):
        pins = self.sim.read_pins()
        for s in self.sessions():
            text = self.sim.generated_yaml(s, 'spawn', self.sim.spawn_table(s, pins))
            poses = sim_tf.spawn_poses(yaml.safe_load(text), s.name)
            self.assertEqual([(p.name, p.x, p.y) for p in poses], [(a.name, a.x, a.y) for a in s.agents])
            for p, a in zip(poses, s.agents):
                self.assertAlmostEqual(math.remainder(p.yaw - math.radians(a.yaw_deg), 2 * math.pi), 0.0)

    def test_frame_align_matches_world_frames(self):
        for s in self.sessions():
            poses = {p.name: p for p in sim_tf.spawn_poses(self.sim.spawn_table(s, self.sim.read_pins()))}
            for e, o in self.sim.frame_align_pairs(s):
                tx, ty, tz, qx, qy, qz, qw = self.sim.frame_align_transform(e, o)
                pe, po = poses[e.name], poses[o.name]
                _, _, te, qe = sim_tf.world_to_map(pe)
                x, y, yaw = compose((te[0], te[1], quat_yaw(qe)), (tx, ty, quat_yaw((qx, qy, qz, qw))))
                self.assertAlmostEqual(x, po.x, msg='%s %s' % (e.name, o.name))
                self.assertAlmostEqual(y, po.y)
                self.assertAlmostEqual(math.remainder(yaw - po.yaw, 2 * math.pi), 0.0)


if __name__ == '__main__':
    unittest.main()
