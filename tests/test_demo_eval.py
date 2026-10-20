"""Unit tests for scripts/demo_eval.py: geometry, footprints, the world's boxes, serialized
messages, the pane-log scan and the grading (pure Python; the probe against a fake rclpy).

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
_spec = importlib.util.spec_from_file_location('demo_eval', str(REPO / 'scripts' / 'demo_eval.py'))
de = importlib.util.module_from_spec(_spec)
sys.modules['demo_eval'] = de
_spec.loader.exec_module(de)
import goal_relay  # noqa: E402
import livox_sim_bridge  # noqa: E402

T0 = 1790000000.0
JR_FP = (-0.266, 0.176, -0.1905, 0.1905)          # the spawned JR URDF's collision footprint
RR_FP = (-0.251, 0.257, -0.202, 0.202)


def sim_module():
    if 'sim' in sys.modules:
        return sys.modules['sim']
    spec = importlib.util.spec_from_file_location('sim', str(REPO / 'scripts' / 'sim.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['sim'] = mod
    spec.loader.exec_module(mod)
    return mod


class SharedConstantsTest(unittest.TestCase):
    def test_same_as_sim(self):
        sim = sim_module()
        self.assertEqual(de.RUN_DIR_DEFAULT, sim.RUN_DIR_DEFAULT)
        self.assertEqual(de.RUN_DIR_IN, sim.RUN_DIR_IN)
        self.assertEqual(de.LOG_ROOT, sim.LOG_ROOT)
        self.assertEqual(list(de.SIM_CONTAINERS), list(sim.CONTAINERS))
        self.assertEqual(de.SIM_PROJECT, sim.PROJECT)
        self.assertEqual(de.WORLDS_DIR, sim.WORLDS_DIR)
        self.assertIn('mighty_ws', de.PROBE_SETUP)
        self.assertIn('/home/swarm/code/mighty_ws/install/setup.bash', sim.CONTAINERS[de.PROBE_CONTAINER].setup)

    def test_refuses_a_broken_isolation_before_the_probe(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'run'
            logs = Path(d) / 'logs'
            run.mkdir()
            logs.mkdir()
            (run / 'session.json').write_text(json.dumps({'name': 'smoke', 'demo': 'swap', 'world': 'arena',
                                                          'agents': [{'name': 'JR81', 'goals': [[1, 0]]}]}))
            (run / 'spawn.yaml').write_text(yaml.safe_dump({'agents': [{'name': 'JR81', 'x': 0.0, 'y': 0.0,
                                                                         'yaw': 0.0}]}))
            err = io.StringIO()
            with mock.patch.object(de, 'isolation_problems', lambda: ['network rover-sim is not internal']), \
                    mock.patch.object(de, 'footprints', lambda r, a: {}), \
                    mock.patch.object(de, 'sim_container', lambda c: c), \
                    mock.patch.object(de.subprocess, 'run', lambda *a, **k: calls.append(a) or None), \
                    contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = de.main(['--run-dir', str(run), '--log-dir', str(logs), '--duration', '1'])
        self.assertEqual((code, calls), (2, []))                      # no docker exec of the probe
        self.assertIn('not isolated', err.getvalue())

    def test_refuses_non_sim_before_docker(self):
        for bad in ('acl-c2-zenoh', 'acl-c2-master', 'some-exporter', 'sim', ''):
            with mock.patch.object(de.subprocess, 'run', side_effect=AssertionError('docker called')):
                with self.assertRaisesRegex(de.EvalError, 'refusing container'):
                    de.sim_container(bad)

    def test_probe_command(self):
        inner, budget = de.probe_command(300, ['JR81', 'RR83'], True)
        self.assertTrue(inner.startswith(de.PROBE_SETUP + ' && exec timeout -k 5 %d python3 ' % budget))
        self.assertIn('demo_eval.py record --duration 300 --spawn /rover_sim/run/spawn.yaml --agents JR81,RR83 '
                      '--maps', inner)
        self.assertGreater(budget, 300 + 2 * 8 * goal_relay.LEAVE_GAP_S)


class GeometryTest(unittest.TestCase):
    def sq(self, x, y, h=0.5, yaw=0.0):
        return de.rect_poly(x, y, yaw, (-h, h, -h, h))

    def test_distances(self):
        self.assertAlmostEqual(de.poly_distance(self.sq(0, 0), self.sq(2, 0)), 1.0)
        self.assertAlmostEqual(de.poly_distance(self.sq(0, 0), self.sq(2, 2)), math.sqrt(2))
        self.assertEqual(de.poly_distance(self.sq(0, 0), self.sq(1, 0)), 0.0)            # touching
        self.assertEqual(de.poly_distance(self.sq(0, 0), self.sq(0.3, 0.2)), 0.0)        # overlapping
        self.assertEqual(de.poly_distance(self.sq(0, 0, 2.0), self.sq(0.1, 0.1)), 0.0)   # one inside the other
        # a diamond's corner towards a square's face
        d = de.poly_distance(self.sq(0, 0), self.sq(0.5 + 0.1 + math.sqrt(0.5), 0, yaw=math.pi / 4))
        self.assertAlmostEqual(d, 0.1, places=9)

    def test_rect_and_point(self):
        p = de.rect_poly(1.0, 2.0, math.pi / 2, (-0.2, 0.4, -0.1, 0.1))
        self.assertEqual([tuple(round(v, 9) for v in c) for c in p],
                         [(1.1, 1.8), (1.1, 2.4), (0.9, 2.4), (0.9, 1.8)])
        self.assertTrue(de.inside((1.0, 2.0), p))
        b = de.Box('b', 0.0, 0.0, 0.5, math.pi / 2, 2.0, 1.0, 1.0)
        self.assertAlmostEqual(de.point_box(b, 0.0, 2.0), 1.0)                           # the long side is y now
        self.assertAlmostEqual(de.point_box(b, 1.5, 0.0), 1.0)
        self.assertEqual(de.point_box(b, 0.1, 0.1), 0.0)
        self.assertAlmostEqual(de.circumradius(JR_FP), math.hypot(0.266, 0.1905))


URDF = '''<robot name="t">
  <link name="chassis">
    <collision><origin xyz="-0.045 0 0.1545" rpy="0 0 0"/><geometry><box size="0.43 0.277 0.172"/></geometry></collision>
    <collision><origin xyz="-0.045 0 0.2405"/><geometry><box size="0.442 0.381 0.006"/></geometry></collision>
  </link>
  <link name="left_wheel"><collision><origin rpy="1.5707963267948966 0 0"/>
    <geometry><cylinder length="0.01" radius="0.0975"/></geometry></collision></link>
  <link name="caster"><collision><geometry><sphere radius="0.039"/></geometry></collision></link>
  <link name="base_link"/>
  <link name="lidar"/>
  <joint name="w" type="continuous"><parent link="chassis"/><child link="left_wheel"/>
    <origin xyz="0 0.1635 0.0975" rpy="0 0 0"/></joint>
  <joint name="c" type="fixed"><parent link="chassis"/><child link="caster"/><origin xyz="-0.3 0 0.039"/></joint>
  <joint name="b" type="fixed"><parent link="chassis"/><child link="base_link"/><origin xyz="0 0 0.36"/></joint>
  <joint name="l" type="fixed"><parent link="base_link"/><child link="lidar"/><origin rpy="0 0.51 0"/></joint>
</robot>'''


class FootprintTest(unittest.TestCase):
    def test_jr_like(self):
        xmin, xmax, ymin, ymax = de.urdf_footprint(URDF)
        self.assertEqual((xmax, ymin, ymax), (0.176, -0.1905, 0.1905))
        self.assertAlmostEqual(xmin, -0.339, places=3)                       # the moved caster sticks out
        # the wheel is a cylinder about y: it reaches its radius along x, half its length along y
        wheel_only = URDF.replace('<origin xyz="-0.3 0 0.039"/>', '<origin xyz="0 0 0.039"/>')
        self.assertEqual(de.urdf_footprint(wheel_only)[0], -0.266)

    def test_spawned_urdfs(self):
        run = Path(de.RUN_DIR_DEFAULT) / 'urdf'
        files = {'jr': sorted(run.glob('JR*.urdf')), 'rr': sorted(run.glob('RR*.urdf'))} if run.is_dir() else {}
        if not files.get('jr') or not files.get('rr'):
            self.skipTest('no spawned URDFs in %s (scripts/sim.py up writes them)' % run)
        self.assertEqual(de.urdf_footprint(files['jr'][0].read_text()), JR_FP)
        self.assertEqual(de.urdf_footprint(files['rr'][0].read_text()), RR_FP)

    def test_errors(self):
        with self.assertRaisesRegex(de.EvalError, 'root links'):
            de.urdf_footprint('<robot><link name="a"/><link name="b"/></robot>')
        with self.assertRaisesRegex(de.EvalError, 'mesh'):
            de.urdf_footprint('<robot><link name="a"><collision><geometry><mesh filename="x.stl"/></geometry>'
                              '</collision></link></robot>')


SDF = '''<sdf version="1.7"><world name="default">
  <model name="ground"><link name="l"><collision name="g"><geometry><plane><normal>0 0 1</normal></plane></geometry>
  </collision></link></model>
  <model name="m"><pose>1 0 0 0 0 1.5707963267948966</pose>
    <link name="l"><pose>1 0 0 0 0 0</pose>
      <collision name="c"><pose>0 1 0.5 0 0 0.1</pose><geometry><box><size>2 1 1</size></box></geometry></collision>
      <collision name="high"><pose>0 0 3 0 0 0</pose><geometry><box><size>1 1 1</size></box></geometry></collision>
    </link>
    <model name="inner"><pose>0 2 0 0 0 0</pose><link name="k">
      <collision name="d"><geometry><box><size>1 1 1</size></box></geometry></collision></link></model>
  </model>
</world></sdf>'''


class WorldTest(unittest.TestCase):
    def test_composition(self):
        boxes = {b.name: b for b in de.world_boxes(SDF)}
        self.assertEqual(sorted(boxes), ['m/inner/k/d', 'm/l/c'])           # plane skipped, high box out
        c = boxes['m/l/c']
        # model at (1, 0) facing +y: link 1 m ahead -> (1, 1); collision 1 m to the link's left -> (0, 1)
        self.assertEqual((round(c.cx, 6), round(c.cy, 6), c.cz, round(c.yaw, 6)), (0.0, 1.0, 0.5, 1.670796))
        d = boxes['m/inner/k/d']
        self.assertEqual((round(d.cx, 6), round(d.cy, 6)), (-1.0, 0.0))

    def test_refusals(self):
        for bad, says in ((SDF.replace('0 0 1.5707963267948966', '0.1 0 1.5'), 'roll/pitch'),
                          (SDF.replace('<size>2 1 1</size></box>', '<size>2 1 1</size></box><cylinder/>')
                           .replace('<box><size>2 1 1</size></box>', '<cylinder><radius>1</radius></cylinder>'),
                           'only box and plane'),
                          (SDF.replace('<model name="ground">', '<include><uri>model://x</uri></include>'
                                       '<model name="ground">'), 'include'),
                          (SDF.replace('<pose>1 0 0 0 0 0</pose>', '<pose relative_to="m">1 0 0 0 0 0</pose>'),
                           'relative to')):
            with self.assertRaisesRegex(de.EvalError, says):
                de.world_boxes(bad)

    def test_shipped_worlds(self):
        """Derived from the world files themselves: the arena's walls are worlds.yaml's, and the
        office starts keep the clearances config/sessions/office_explore.yaml states."""
        arena = de.world_boxes((de.WORLDS_DIR / 'arena.world').read_text())
        self.assertEqual(len(arena), 11)
        walls = yaml.safe_load((REPO / 'src' / 'rover_sim' / 'config' / 'worlds.yaml').read_text())['walls']['arena']
        faces = {'x': [], 'y': []}
        for b in arena:
            if b.name.split('/')[-1].startswith('wall_'):
                if b.sx < b.sy:
                    faces['x'].append(b.cx - math.copysign(b.sx / 2, b.cx))
                else:
                    faces['y'].append(b.cy - math.copysign(b.sy / 2, b.cy))
        self.assertEqual((min(faces['x']), min(faces['y']), max(faces['x']), max(faces['y'])), tuple(walls))
        office = de.world_boxes((de.WORLDS_DIR / 'ACL_office_simple.world').read_text())
        self.assertEqual(len(office), 115)    # 86 walls + 19 pilasters + 10 shelves (the ceiling is overhead)
        session = yaml.safe_load((REPO / 'config' / 'sessions' / 'office_explore.yaml').read_text())
        clear = {a['name']: min(de.point_box(b, a['pose'][0], a['pose'][1]) for b in office)
                 for a in session['agents']}
        self.assertEqual({n: round(v, 2) for n, v in clear.items()},
                         {'JR81': 1.20, 'JR82': 1.20, 'RR83': 1.19, 'RR84': 1.10})


def cdr_string(out, text):
    raw = text.encode() + b'\0'
    out += b'\0' * ((-(len(out) - 4)) % 4)
    out += struct.pack('<I', len(raw)) + raw


def marker_cdr(frame, ns, stamp=(5, 6)):
    out = bytearray(b'\x00\x01\x00\x00' + struct.pack('<iI', *stamp))
    cdr_string(out, frame)
    cdr_string(out, ns)
    out += b'\0' * ((-(len(out) - 4)) % 4) + struct.pack('<i', 83) + b'\0' * 64
    return bytes(out)


def dyntraj_cdr(frame, bbox, ident):
    out = bytearray(b'\x00\x01\x00\x00' + struct.pack('<iI', 7, 8))
    cdr_string(out, frame)
    out += b'\0' * ((-(len(out) - 4)) % 4) + struct.pack('<I', len(bbox)) + struct.pack('<%df' % len(bbox), *bbox)
    out += b'\0' * ((-(len(out) - 4)) % 4) + struct.pack('<i', ident) + b'\0' * 64
    return bytes(out)


class CdrTest(unittest.TestCase):
    def test_marker_and_dyntraj(self):
        for frame in ('JR81/map', 'JR8', 'J', ''):                          # every padding case
            self.assertEqual(de.marker_ns(marker_cdr(frame, 'other_agent_traj_83')), 'other_agent_traj_83')
            for bbox in ([], [0.6], [0.6, 0.6, 0.5]):
                self.assertEqual(de.dyntraj_id(dyntraj_cdr(frame, bbox, 84)), 84)
        self.assertAlmostEqual(de.stamp_of(marker_cdr('x', 'y', (10, 500000000))), 10.5)
        with self.assertRaises(ValueError):
            de.marker_ns(b'\x01\x00' + b'\0' * 30)

    def test_against_rclpy(self):
        try:
            from rclpy.serialization import serialize_message
            from visualization_msgs.msg import Marker
        except ImportError:
            self.skipTest('no rclpy / visualization_msgs here (the container run checks it)')
        for frame in ('JR81/map', 'abc', ''):
            m = Marker()
            m.header.frame_id, m.ns, m.id = frame, 'other_agent_traj_82', 3
            m.header.stamp.sec, m.header.stamp.nanosec = 12, 250000000
            buf = serialize_message(m)
            self.assertEqual(de.marker_ns(buf), 'other_agent_traj_82')
            self.assertAlmostEqual(de.stamp_of(buf), 12.25)


def ros_line(level, t, logger, msg, color=True):
    text = '[%s] [%.9f] [%s]: %s' % (level, t, logger, msg)
    return ('\x1b[33m%s\x1b[0m' % text if color else text) + '\r\n'


class LogsTest(unittest.TestCase):
    def test_a_restarted_pane_log(self):
        # `sim.py start goals` renames goal-relay.log (rotate_pane_logs) and a new one takes its path:
        # the old file counts from where it was, the new one from its start, not the other way round.
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'goals').mkdir()
            log = Path(d) / 'goals' / 'goal-relay.log'
            log.write_text(ros_line('ERROR', T0, 'sim.goal_relay', 'old history') * 3)
            start = de.log_sizes(d)
            with log.open('a') as f:
                f.write(ros_line('ERROR', T0 + 1, 'sim.goal_relay', 'before the restart'))
            log.rename(Path(d) / 'goals' / 'goal-relay.20261020_150000.log')
            log.write_text(ros_line('ERROR', T0 + 2, 'sim.goal_relay', 'after the restart'))
            out = de.scan_logs(d, start, de.log_sizes(d))
            self.assertEqual(sorted(e[2] for e in out['errors']), ['after the restart', 'before the restart'])
            # A recording from before the inode (bytes only) still grades by path.
            old = dict((k, v[0]) for k, v in start.items())
            self.assertEqual(de.scan_logs(d, old, dict((k, v[0]) for k, v in de.log_sizes(d).items()))['lines'],
                             4 + 1)      # the renamed file is new to it; the new one shrank: from its start

    def test_scan_new_content_only(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'mighty').mkdir()
            mn, gs = Path(d) / 'mighty' / 'mighty-node-JR81.log', Path(d) / 'mighty' / 'goal-selector-JR81.log'
            mn.write_text(ros_line('ERROR', T0, 'JR81.mighty_node', 'old error') + 'Dropping traj old\n')
            gs.write_text('')
            start = de.log_sizes(d)
            with mn.open('a') as f:
                f.write('[mighty-1] ' + ros_line('ERROR', T0 + 1, 'JR81.mighty_node', 'Map hold: holding (maps 2.0 s old)'))
                f.write('[mighty-1] ' + ros_line('WARN', T0 + 2, 'JR81.mighty_node',
                                                 'Dropping traj from agent 83 — no frame alignment received yet'))
                f.write('[mighty-1] ' + ros_line('ERROR', T0 + 3, 'JR81.mighty_node', 'Goal z is out of bounds: 5.0'))
                f.write('\x1b[?2004h$ echo hi\n')
            with gs.open('a') as f:
                f.write('[goal_selector-1] ' + ros_line('INFO', T0 + 4, 'JR81.goal_selector',
                                                        'Exploration: -> frontier 7 at (1.00, 2.00) (centroid ...)'))
                f.write(ros_line('INFO', T0 + 5, 'JR81.goal_selector', 'Manual goal (1, 2, 0) committed', False))
            (Path(d) / 'goals').mkdir()
            rec = {'agent': 'JR81', 'kind': 'swap', 'result': 'reached', 't_end': T0 + 6}
            (Path(d) / 'goals' / 'goal-relay.log').write_text(
                '[INFO] [%.3f] [sim.goal_relay]: %s\n2026-10-19T10:00:00.123Z ERROR zenoh::net::routing: link lost\n'
                % (T0 + 6, goal_relay.leg_line(rec)))
            out = de.scan_logs(d, start, de.log_sizes(d))
            self.assertEqual(dict(out['dropping_traj']), {'JR81': 1})            # not the old line
            self.assertEqual(dict(out['frontiers']), {'JR81': 1})
            self.assertEqual(out['legs'], [rec])
            self.assertEqual(dict(out['expected_errors']), {'mighty_node: Map hold:': 1})
            self.assertEqual([(l, m) for _, l, m in out['errors']],
                             [('zenoh zenoh::net::routing:', 'link lost'),
                              ('JR81.mighty_node', 'Goal z is out of bounds: 5.0')]
                             if out['errors'][0][1].startswith('zenoh') else
                             [('JR81.mighty_node', 'Goal z is out of bounds: 5.0'),
                              ('zenoh zenoh::net::routing:', 'link lost')])
            self.assertEqual(de.error_kinds(out['errors'])[0][0], 1)
            allowed = de.scan_logs(d, start, de.log_sizes(d), allowed=[r'Goal z is out of bounds', 'link lost'])
            self.assertEqual(allowed['errors'], [])
            self.assertEqual(sum(allowed['expected_errors'].values()), 3)


def drive(rows, t0, t1, p0, p1, yaw, dt=0.05):
    n = int(round((t1 - t0) / dt))
    for k in range(n + 1):
        f = k / n
        rows.append((t0 + k * dt, p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]), yaw))


def swap_recording(gap=0.4, stop_at=None):
    """Two agents on the arena's diagonal, driving at each other and stopping `gap` m apart."""
    rec = {'mode': 'swap', 'agents': ['JR81', 'RR83'], 't0': T0, 't1': T0 + 60.0, 'truth': {}, 'est': {},
           'session': {'name': 'multi_agent_swap'},
           'spawn': {'agents': [{'name': 'JR81', 'x': 2.0, 'y': 0.0, 'yaw': math.pi},
                                {'name': 'RR83', 'x': -2.0, 'y': 0.0, 'yaw': 0.0}]},
           'footprints': {'JR81': list(JR_FP), 'RR83': list(RR_FP)},
           'rtf': {'JR81': [(T0 + k, 1.0) for k in range(60)], 'RR83': [(T0 + k, 0.999) for k in range(60)]},
           'trajs': {'81': 300, '83': 280}, 'traj_received': {'JR81': {'83': 270}, 'RR83': {'81': 299}},
           'term_goal': {}, 'coverage': []}
    # JR81's front is at +0.176 along its heading (-x here), RR83's at +0.257: stop with `gap` between them
    jr_stop = (stop_at if stop_at is not None else (gap / 2 + 0.176), 0.0)
    rr_stop = (-(gap / 2 + 0.257), 0.0)
    rec['truth']['JR81'], rec['truth']['RR83'] = [], []
    drive(rec['truth']['JR81'], T0, T0 + 30, (2.0, 0.0), jr_stop, math.pi)
    drive(rec['truth']['JR81'], T0 + 30.05, T0 + 60, jr_stop, jr_stop, math.pi)
    drive(rec['truth']['RR83'], T0, T0 + 30, (-2.0, 0.0), rr_stop, 0.0)
    drive(rec['truth']['RR83'], T0 + 30.05, T0 + 60, rr_stop, rr_stop, 0.0)
    for n, (sx, yaw) in (('JR81', (2.0, math.pi)), ('RR83', (-2.0, 0.0))):
        sp = types.SimpleNamespace(x=sx, y=0.0, yaw=yaw)
        rec['est'][n] = [(t, *goal_relay.to_map(sp, x, y)) for t, x, y, _ in rec['truth'][n][::2]]
    return rec


def legs_for(rec, n_reached=2, timeouts=0):
    legs = []
    for n in rec['agents']:
        for k in range(n_reached + timeouts):
            legs.append({'agent': n, 'kind': 'swap', 'result': 'reached' if k < n_reached else 'timeout',
                         't_end': T0 + 10 + k, 'duration_s': 9.5, 'leg': k + 1,
                         'target_world': list(rec['truth'][n][int((10 + k) / 0.05)][1:3])})
    return legs


def opts(**kw):
    o = dict(min_legs=2, goal_radius=0.5, contact=0.02, coverage_growth=2.0, rtf_min=0.95, odom_max=0.5)
    o.update(kw)
    return types.SimpleNamespace(**o)


def empty_logs(**kw):
    out = {'files': 3, 'lines': 100, 'dropping_traj': {}, 'errors': [], 'expected_errors': {}, 'frontiers': {},
           'legs': [], 'rtf_violations': []}
    out.update(kw)
    return out


class GradeTest(unittest.TestCase):
    BOX = [de.Box('arena/link/post', 0.0, 3.0, 0.5, 0.0, 0.5, 0.5, 1.0)]

    def crit(self, report):
        return {c['name']: c['pass'] for c in report['criteria']}

    def test_swap_pass(self):
        rec = swap_recording(gap=0.4)
        r = de.grade(rec, empty_logs(legs=legs_for(rec)), self.BOX, opts())
        self.assertEqual(self.crit(r), {'legs': True, 'separation': True, 'contact': True, 'trajs': True,
                                        'errors': True, 'rtf': True, 'odometry': True}, r)
        self.assertTrue(r['pass'])
        sep = r['criteria'][1]['detail']
        self.assertEqual((sep['worst']['pair'], sep['worst']['min_m']), ('JR81-RR83', 0.4))
        self.assertIn('0.400 m (JR81-RR83 at 30.0 s)', r['criteria'][1]['summary'])
        lines = de.report_lines(r)
        self.assertTrue(lines[0].startswith('PASS  legs'))
        self.assertTrue(lines[-1].startswith('PASS: swap demo of multi_agent_swap, 60 s, 7 criteria'))

    def test_swap_failures(self):
        rec = swap_recording(gap=-0.1)                                    # overlapping chassis
        legs = legs_for(rec, n_reached=1, timeouts=1)
        logs = empty_logs(legs=legs, dropping_traj={'JR81': 2}, errors=[('m/x.log', 'JR81.mighty', 'boom')],
                          rtf_violations=['2026-10-19 RR83 rtf 0.5'])
        rec['traj_received']['JR81'] = {}
        rec['rtf']['RR83'][10:17] = [(T0 + 10 + k, 0.9) for k in range(7)]
        rec['est']['RR83'] = [(t, x + 1.0, y) for t, x, y in rec['est']['RR83']]
        r = de.grade(rec, logs, self.BOX, opts())
        self.assertEqual(self.crit(r), {'legs': False, 'separation': False, 'contact': True, 'trajs': False,
                                        'errors': False, 'rtf': False, 'odometry': False})
        sep = r['criteria'][1]['detail']['pairs']['JR81-RR83']
        self.assertEqual((sep['min_m'], len(sep['contacts'])), (0.0, 1))
        self.assertAlmostEqual(sep['contacts'][0][0], T0 + 28.8, delta=0.3)   # contact from ~29 s on, held
        self.assertEqual(r['criteria'][3]['detail']['pairs']['JR81<-RR83'], {'sent': 280, 'accepted': 0, 'ok': False})
        self.assertEqual(r['criteria'][5]['detail']['agents']['RR83']['longest_below_s'], 6.0)   # 7 samples
        self.assertEqual(r['criteria'][6]['detail']['RR83']['max_m'], 1.0)
        self.assertEqual(r['criteria'][6]['detail']['RR83']['rmse_m'], 1.0)
        self.assertEqual(r['criteria'][6]['detail']['JR81']['rmse_m'], 0.0)
        self.assertAlmostEqual(r['criteria'][6]['detail']['JR81']['path_m'], 2.0 - 0.126, delta=0.06)   # 0.1 m steps
        self.assertFalse(r['pass'])

    def test_one_agent_has_no_separation_to_fail(self):
        rec = swap_recording(gap=0.4)
        for k in ('truth', 'est', 'rtf', 'footprints'):
            rec[k].pop('RR83')
        rec['agents'] = ['JR81']
        r = de.grade(rec, empty_logs(legs=[l for l in legs_for(rec) if l['agent'] == 'JR81']), self.BOX, opts())
        self.assertTrue(r['criteria'][1]['pass'], r['criteria'][1])
        self.assertIn('n/a (one agent)', r['criteria'][1]['summary'])

    def test_truth_agents_are_not_graded(self):
        # odometry: truth (NOT PARITY): the estimate is the truth; with DLIO's drift on it, a truth
        # agent would fail, so the drift below shows the criterion does not grade it.
        rec = swap_recording(gap=0.4)
        rec['est']['JR81'] = [(t, x + 1.0, y) for t, x, y in rec['est']['JR81']]
        for a in rec['spawn']['agents']:
            a['odometry'] = 'truth'
        r = de.grade(rec, empty_logs(legs=legs_for(rec)), self.BOX, opts())
        odo = r['criteria'][6]
        self.assertEqual((odo['name'], odo['pass'], odo.get('na')), ('odometry', True, True))
        self.assertTrue(odo['summary'].startswith('n/a: odometry: truth (NOT PARITY), not graded: '), odo['summary'])
        self.assertEqual(odo['detail']['truth_check']['JR81']['max_m'], 1.0)    # reported, not graded
        self.assertEqual(odo['detail']['truth_check']['RR83']['max_m'], 0.0)
        self.assertTrue(r['pass'])
        self.assertEqual(r['odometry'], {'JR81': 'truth', 'RR83': 'truth'})
        lines = de.report_lines(r)
        self.assertTrue(lines[6].startswith('N/A   odometry'), lines[6])
        self.assertEqual(lines[-1], 'PASS: swap demo of multi_agent_swap, 60 s, 7 criteria (1 n/a); '
                                    'NOT PARITY: odometry truth for JR81, RR83')

    def test_mixed_grades_only_dlio(self):
        rec = swap_recording(gap=0.4)
        rec['spawn']['agents'][0]['odometry'] = 'truth'                    # JR81 truth, RR83 DLIO
        rec['est']['JR81'] = [(t, x + 1.0, y) for t, x, y in rec['est']['JR81']]
        r = de.grade(rec, empty_logs(legs=legs_for(rec)), self.BOX, opts())
        odo = r['criteria'][6]
        self.assertTrue(odo['pass'])
        self.assertNotIn('na', odo)
        self.assertIn('RR83 max 0.0 m', odo['summary'])
        self.assertIn('not graded: JR81 residual max 1.0 m', odo['summary'])
        self.assertEqual(sorted(odo['detail']), ['RR83', 'truth_check'])
        rec['est']['RR83'] = [(t, x + 1.0, y) for t, x, y in rec['est']['RR83']]
        r = de.grade(rec, empty_logs(legs=legs_for(rec)), self.BOX, opts())
        self.assertFalse(r['criteria'][6]['pass'])                         # RR83's DLIO still grades
        self.assertTrue(de.report_lines(r)[-1].endswith('7 criteria; NOT PARITY: odometry truth for JR81'))

    def test_odometry_modes(self):
        spawn = {'agents': [{'name': 'JR81', 'odometry': 'truth'}, {'name': 'RR83'}]}
        self.assertEqual(de.odometry_modes(spawn, ['JR81', 'RR83', 'JR82']),
                         {'JR81': 'truth', 'RR83': 'dlio', 'JR82': 'dlio'})
        self.assertEqual(de.odometry_modes({}, ['JR81']), {'JR81': 'dlio'})

    def test_rtf_rule_is_the_bridges(self):
        # livox_sim_bridge.RtfMonitor: more than hold_s since the first sample below; NaN (the bridge
        # has under 1 s of on-time IMU) neither starts nor ends a run and is not in min or mean.
        nan = float('nan')
        six = [(T0 + k, 0.9 if 10 <= k < 16 else 1.0) for k in range(30)]        # 6 below: 5 s
        ok, out = de.rtf_summary({'JR81': six}, 0.95)
        self.assertTrue(ok)
        self.assertEqual(out['JR81']['longest_below_s'], 5.0)
        seven = [(t, 0.9 if 10 <= t - T0 < 17 else v) for t, v in six]            # 7 below: 6 s
        self.assertFalse(de.rtf_summary({'JR81': seven}, 0.95)[0])
        gappy = [(t, nan if t - T0 in (12, 20) else v) for t, v in seven]          # NaN inside the run
        ok, out = de.rtf_summary({'JR81': gappy}, 0.95)
        self.assertFalse(ok)
        self.assertEqual((out['JR81']['unknown'], out['JR81']['n'], out['JR81']['min']), (2, 28, 0.9))
        self.assertAlmostEqual(out['JR81']['mean'], (22 * 1.0 + 6 * 0.9) / 28, places=4)

    def test_reached_but_far(self):
        rec = swap_recording()
        legs = legs_for(rec)
        legs[0]['target_world'] = [legs[0]['target_world'][0] + 2.0, 0.0]
        ok, per = de.leg_summary(legs, rec['agents'], T0, T0 + 60, 2, rec['truth'])
        self.assertFalse(ok)
        self.assertEqual(per['JR81']['reached_but_far'][0]['leg'], 1)
        ok, per = de.leg_summary(legs, rec['agents'], T0 + 10.5, T0 + 60, 1, rec['truth'])   # leg 1 out of the window
        self.assertTrue(ok)
        self.assertEqual(per['JR81']['reached'], 1)

    def test_box_contact(self):
        rec = swap_recording()
        rows = []
        drive(rows, T0, T0 + 10, (0.0, 0.0), (0.0, 3.0 - 0.25 - 0.176 - 0.01), math.pi / 2)
        bc = de.box_contact({'JR81': rows}, {'JR81': JR_FP}, self.BOX, 0.02)
        self.assertEqual((bc['JR81']['box'], bc['JR81']['min_m']), ('arena/link/post', 0.01))
        self.assertEqual(len(bc['JR81']['contacts'][0]['spans']), 1)
        far = de.box_contact({'JR81': rec['truth']['JR81']}, {'JR81': JR_FP}, self.BOX, 0.02)
        self.assertEqual(far['JR81']['contacts'], [])

    def test_trajs_need_something_to_judge(self):
        agents = ['JR81', 'RR83']
        self.assertFalse(de.traj_summary({}, {}, agents)[0])
        self.assertTrue(de.traj_summary({}, {}, ['JR81'])[0])
        ok, pairs = de.traj_summary({'81': 100, '83': 5}, {'RR83': {'81': 60}}, agents)
        self.assertTrue(ok)
        self.assertEqual(pairs['JR81<-RR83'], {'sent': 5, 'accepted': 0, 'ok': True})   # too few to judge

    def test_explore(self):
        rec = swap_recording()
        rec['mode'] = 'explore'
        rec['coverage'] = [(T0, 20.0, {'JR81': 12.0}), (T0 + 30, 40.0, {}), (T0 + 60, 55.5, {'JR81': 30.0})]
        rec['term_goal'] = {'JR81': [(T0, 1, 0, 0), (T0 + 1, 1, 0, 0), (T0 + 5, 2, 1, 1)], 'RR83': []}
        logs = empty_logs(frontiers={'RR83': 3})
        r = de.grade(rec, logs, self.BOX, opts())
        c = self.crit(r)
        self.assertTrue(c['frontiers'] and c['coverage'])
        self.assertEqual(r['criteria'][0]['detail']['term_goal_commitments'], {'JR81': 2, 'RR83': 0})
        self.assertEqual(r['criteria'][1]['detail']['growth_m2'], 35.5)
        rec['coverage'] = rec['coverage'][:1]
        r = de.grade(rec, empty_logs(), self.BOX, opts())
        self.assertFalse(self.crit(r)['coverage'] or self.crit(r)['frontiers'])


class HostTest(unittest.TestCase):
    def test_pick_mode(self):
        self.assertEqual(de.pick_mode('auto', {'demo': 'explore'}), 'explore')
        self.assertEqual(de.pick_mode('auto', {'demo': 'none', 'agents': [{'goals': [[1, 2]]}]}), 'swap')
        self.assertEqual(de.pick_mode('auto', {'agents': [{'goals': []}]}), 'explore')
        self.assertEqual(de.pick_mode('swap', {'demo': 'explore'}), 'swap')

    def test_grade_a_saved_recording(self):
        rec = swap_recording()
        with tempfile.TemporaryDirectory() as d:
            logs = Path(d) / 'logs'
            (logs / 'goals').mkdir(parents=True)
            text = ''.join('[INFO] [%.3f] [sim.goal_relay]: %s\n' % (g['t_end'], goal_relay.leg_line(g))
                           for g in legs_for(rec))
            (logs / 'goals' / 'goal-relay.log').write_text(text)
            rec.update(world='arena', host={'log_dir': str(logs), 'log_start': {}, 'log_end': de.log_sizes(logs),
                                            'run_dir': d, 'rtf_log': [0, 0]})
            path = Path(d) / 'swap_x_recording.json'
            path.write_text(json.dumps(rec))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = de.main(['grade', '--recording', str(path)])
            self.assertEqual(code, 0, out.getvalue())
            report = json.loads((Path(d) / 'swap_x_report.json').read_text())
            self.assertTrue(report['pass'])
            self.assertEqual((report['world']['file'], report['world']['boxes']), ('src/rover_sim/worlds/arena.world', 11))
            self.assertIn('PASS  legs', out.getvalue())
            # a recording named otherwise is not overwritten by its report
            other = Path(d) / 'rec.json'
            other.write_text(json.dumps(rec))
            with contextlib.redirect_stdout(io.StringIO()):
                de.main(['grade', '--recording', str(other)])
            self.assertTrue((Path(d) / 'rec.json.report.json').is_file())
            self.assertEqual(json.loads(other.read_text())['mode'], 'swap')

    def test_run_records_in_sim_mighty_then_grades(self):
        """The host run: log offsets before and after, the probe through docker exec, the report."""
        rec = swap_recording()
        probe = {k: rec[k] for k in ('truth', 'est', 'rtf', 'trajs', 'traj_received', 'term_goal', 'coverage')}
        probe.update(t0=rec['t0'], t1=rec['t1'], agents=rec['agents'], mode={}, goal_reached={}, notes=[],
                     version=1, stopped_early=False)
        with tempfile.TemporaryDirectory() as d:
            run, logs = Path(d) / 'run', Path(d) / 'logs'
            (run / 'urdf').mkdir(parents=True)
            (logs / 'goals').mkdir(parents=True)
            (run / 'session.json').write_text(json.dumps({'name': 'multi_agent_swap', 'demo': 'swap',
                                                          'world': 'arena', 'agents': []}))
            (run / 'spawn.yaml').write_text(yaml.safe_dump(dict(rec['spawn'], world='arena')))
            for n in rec['agents']:
                (run / 'urdf' / ('%s.urdf' % n)).write_text(URDF)
            relay = logs / 'goals' / 'goal-relay.log'
            relay.write_text('[INFO] [1.0] [sim.goal_relay]: LEG {"agent": "JR81", "kind": "swap", "result": '
                             '"timeout", "t_end": %f}\n' % (T0 + 5))          # before the window: not graded
            calls = []

            def fake_run(cmd, **kw):
                calls.append(cmd)
                if cmd[1] == 'inspect':
                    return NS(returncode=0, stdout='true rover-sim-stack\n')
                with relay.open('a') as f:                              # legs logged while it records
                    for g in legs_for(rec):
                        f.write('[INFO] [%.3f] [sim.goal_relay]: %s\n' % (g['t_end'], goal_relay.leg_line(g)))
                return NS(returncode=0, stdout='noise\n' + de.MARK + json.dumps(probe) + '\n')
            out = io.StringIO()
            with mock.patch.object(de.subprocess, 'run', fake_run), contextlib.redirect_stdout(out), \
                    mock.patch.object(de, 'isolation_problems', lambda: []):
                code = de.main(['--run-dir', str(run), '--log-dir', str(logs), '--duration', '60'])
            self.assertEqual(code, 0, out.getvalue())
            self.assertEqual(calls[-1][:4], ['docker', 'exec', 'sim-mighty', 'bash'])
            self.assertIn('demo_eval.py record --duration 60 --spawn /rover_sim/run/spawn.yaml --agents JR81,RR83',
                          calls[-1][5])
            self.assertNotIn('--maps', calls[-1][5])                       # a swap reads no grids
            reports = sorted((logs / 'demo_eval').glob('swap_*_report.json'))
            saved = sorted((logs / 'demo_eval').glob('swap_*_recording.json'))
            self.assertEqual((len(reports), len(saved)), (1, 1))
            report = json.loads(reports[0].read_text())
            legs = report['criteria'][0]
            self.assertEqual((legs['name'], legs['pass'], legs['detail']['JR81']['timeout']), ('legs', True, 0))
            self.assertEqual(json.loads(saved[0].read_text())['footprints']['JR81'][1], 0.176)

    def test_record_refuses_outside_the_sim(self):
        environ = {k: v for k, v in os.environ.items() if k != 'ZENOH_SESSION_CONFIG_URI'}
        err = io.StringIO()
        with mock.patch.dict(os.environ, environ, clear=True), mock.patch.dict(sys.modules, {'rclpy': None}), \
                contextlib.redirect_stderr(err):
            self.assertEqual(de.main(['record', '--duration', '1']), 2)
        self.assertIn('not in a rover_sim container', err.getvalue())


# ---------------------------------------------------------------------------- the probe, fake rclpy

class NS(types.SimpleNamespace):
    pass


class FakeNode:
    def __init__(self):
        self.subs = []

    def create_subscription(self, msg_type, topic, cb, qos, raw=False):
        s = NS(type=msg_type, topic=topic, cb=cb, qos=qos, raw=raw)
        self.subs.append(s)
        return s

    def topic(self, name):
        (s,) = [s for s in self.subs if s.topic == name]
        return s


QOS = NS(QoSProfile=lambda **kw: dict(kw), HistoryPolicy=NS(KEEP_LAST='keep_last'),
         ReliabilityPolicy=NS(RELIABLE='reliable', BEST_EFFORT='best_effort'))
FAKE = {'rclpy': types.ModuleType('rclpy'), 'rclpy.qos': QOS,
        'geometry_msgs': types.ModuleType('geometry_msgs'), 'geometry_msgs.msg': NS(PoseStamped='PoseStamped'),
        'nav_msgs': types.ModuleType('nav_msgs'), 'nav_msgs.msg': NS(OccupancyGrid='OccupancyGrid', Odometry='Odometry'),
        'std_msgs': types.ModuleType('std_msgs'), 'std_msgs.msg': NS(Empty='Empty', Float64='Float64', String='String'),
        'visualization_msgs': types.ModuleType('visualization_msgs'), 'visualization_msgs.msg': NS(Marker='Marker'),
        'dynus_interfaces': types.ModuleType('dynus_interfaces'), 'dynus_interfaces.msg': NS(DynTraj='DynTraj')}


def odom_cdr(t, x, y, yaw):
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    return livox_sim_bridge.odom_build(int(round(t * 1e9)), 'world', 'JR81/chassis', (x, y, 0.0) + q + (0.0,) * 78)


class RecorderTest(unittest.TestCase):
    def recorder(self, maps=True):
        spawns = {'JR81': NS(name='JR81', x=2.0, y=0.0, yaw=math.pi), 'RR83': NS(name='RR83', x=-2.0, y=0.0, yaw=0.0)}
        node = FakeNode()
        with mock.patch.dict(sys.modules, FAKE):
            return de.Recorder(node, spawns, ['JR81', 'RR83'], maps), node

    def test_endpoints(self):
        r, node = self.recorder()
        topics = {s.topic: s for s in node.subs}
        self.assertEqual(len(topics), 2 * 8 + 1)
        self.assertEqual(len(r.entities), len(node.subs))
        for raw in ('/JR81/ground_truth/odom_wall', '/RR83/dlio/odom_node/odom', '/JR81/traj_received', '/trajs'):
            self.assertTrue(topics[raw].raw, raw)
        self.assertEqual(topics['/JR81/term_goal'].qos['reliability'], 'reliable')
        self.assertEqual(topics['/RR83/persistent_map/grid'].qos['reliability'], 'best_effort')

    def test_samples(self):
        r, node = self.recorder(maps=False)
        cb = node.topic('/JR81/ground_truth/odom_wall').cb
        for k in range(100):                                               # 100 Hz in, about 20 Hz kept
            cb(odom_cdr(T0 + k * 0.01, 1.0, 2.0, 0.5))
        times = [row[0] for row in r.truth['JR81']]
        self.assertTrue(20 <= len(times) <= 25, len(times))
        self.assertGreaterEqual(min(b - a for a, b in zip(times, times[1:])), 0.8 * de.TRUTH_DT - 1e-6)
        self.assertEqual(r.truth['JR81'][0], (T0, 1.0, 2.0, 0.5))
        node.topic('/RR83/dlio/odom_node/odom').cb(odom_cdr(T0, 0.5, 0.25, 0.0))
        self.assertEqual(r.est['RR83'], [(T0, 0.5, 0.25)])
        for _ in range(3):
            node.topic('/RR83/traj_received').cb(marker_cdr('RR83/map', 'other_agent_traj_81'))
        node.topic('/RR83/traj_received').cb(marker_cdr('RR83/map', 'velocity'))
        node.topic('/trajs').cb(dyntraj_cdr('JR81/map', [0.6, 0.6, 0.5], 81))
        node.topic('/JR81/mode').cb(NS(data='DISARMED'))
        node.topic('/JR81/mode').cb(NS(data='DISARMED'))
        node.topic('/JR81/mode').cb(NS(data='AUTO'))
        res = r.result()
        self.assertEqual((res['traj_received']['RR83'], res['trajs']), ({'81': 3}, {'81': 1}))
        self.assertEqual([m for _, m in res['mode']['JR81']], ['DISARMED', 'AUTO'])
        json.dumps(res)                                                      # plain data

    def test_polled(self):
        polled = []
        poller = NS(add=lambda t, topic, qos, handler, raw, latest: polled.append((topic, qos['depth'], latest)),
                    entities=lambda: [('poll', 'subscription', p) for p in polled])
        spawns = {'JR81': NS(name='JR81', x=0.0, y=0.0, yaw=0.0)}
        node = FakeNode()
        with mock.patch.dict(sys.modules, FAKE):
            r = de.Recorder(node, spawns, ['JR81'], False, poller)
        self.assertEqual(sorted(polled), [('/JR81/dlio/odom_node/odom', 1, True),
                                          ('/JR81/ground_truth/odom_wall', 1, True),
                                          ('/JR81/traj_received', 400, False), ('/trajs', 400, False)])
        self.assertEqual(len(r.all_entities()), len(polled) + len(node.subs))
        self.assertNotIn('/trajs', {s.topic for s in node.subs})

    def test_coverage(self):
        if importlib.util.find_spec('numpy') is None:
            self.skipTest('no numpy here')
        r, node = self.recorder()

        def grid(n, data, ox=0.0, oy=0.0):
            info = NS(resolution=0.1, width=4, height=2, origin=NS(position=NS(x=ox, y=oy, z=0.0),
                                                                   orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)))
            node.topic('/%s/persistent_map/grid' % n).cb(NS(info=info, data=data))

        import array
        grid('JR81', array.array('b', [0, 100, -1, -1, 0, 0, 0, 0]))          # 6 known cells
        # RR83 faces +x from (-2, 0), JR81 faces -x from (2, 0): RR83's grid at (3.6, -0.2) in its map
        # covers world x 1.6..2.0, the cells of JR81's grid at its map's (0..0.4, 0..0.2) mirrored.
        grid('RR83', [0] * 8, ox=3.6, oy=-0.2)
        r.sample_coverage(T0)
        t, union, per = r.coverage[0]
        self.assertEqual(per, {'JR81': 0.06, 'RR83': 0.08})
        self.assertAlmostEqual(union, 0.08, places=6)                         # JR81's six lie inside RR83's
        r.sample_coverage(T0 + 1)
        self.assertEqual(len(r.coverage), 1)                                  # every COVERAGE_DT s


if __name__ == '__main__':
    unittest.main()
