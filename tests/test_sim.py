"""Unit tests for scripts/sim.py's pure logic (no docker, no tmux).

    python3 -m unittest discover -s tests -v
"""
import base64
import contextlib
import dataclasses
import importlib.util
import io
import ipaddress
import json
import math
import os
import random
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('sim', str(REPO / 'scripts' / 'sim.py'))
sim = importlib.util.module_from_spec(_spec)
sys.modules['sim'] = sim
_spec.loader.exec_module(sim)


def session(**over):
    data = {'name': 't', 'world': 'empty', 'mapping': 'fallback', 'rviz': False,
            'agents': [{'name': 'JR81', 'class': 'jr', 'pose': [0, 0, 0]},
                       {'name': 'RR83', 'class': 'rr', 'pose': [2.0, 1.0, 90]}]}
    data.update(over)
    return data


def errors_of(data):
    try:
        sim.validate_session(data, 'test')
    except sim.SessionError as e:
        return str(e)
    return ''


class SessionValidation(unittest.TestCase):
    def test_smoke_file(self):
        s = sim.load_session('smoke')
        self.assertEqual(s.name, 'smoke')
        self.assertEqual(s.world, 'arena')
        self.assertEqual(s.mapping, 'fallback')
        self.assertFalse(s.rviz)
        self.assertFalse(s.gui)
        self.assertEqual([(a.name, a.cls, a.id) for a in s.agents], [('JR81', 'jr', 81), ('RR83', 'rr', 83)])
        self.assertEqual(s.num_agents, 83)
        self.assertEqual(s.source, 'config/sessions/smoke.yaml')

    def test_every_shipped_session_validates(self):
        for p in sorted((REPO / 'config' / 'sessions').glob('*.yaml')):
            sim.load_session(str(p))

    def test_defaults(self):
        d = session()
        del d['mapping'], d['rviz']
        s = sim.validate_session(d)
        self.assertEqual((s.mapping, s.rviz, s.gui), ('gpu', False, False))
        self.assertTrue(sim.validate_session(session(gui=True)).gui)
        self.assertTrue(sim.validate_session(session(gui=True)).to_json()['gui'])

    def test_yaw_radians(self):
        def yaw(deg):
            return sim.Agent('JR81', 'jr', 0.0, 0.0, deg).yaw
        self.assertEqual(yaw(0.0), 0.0)
        self.assertEqual(yaw(-135.0), math.radians(-135.0))      # exact inside [-180, 180]
        self.assertEqual(yaw(180.0), math.pi)
        self.assertAlmostEqual(yaw(270.0), -math.pi / 2)
        self.assertAlmostEqual(yaw(-450.0), -math.pi / 2)
        for deg in range(-720, 721, 15):
            self.assertLessEqual(abs(yaw(float(deg))), math.pi)  # spawn_agent.launch.py refuses |yaw| > 2 pi

    def test_valid_with_goals(self):
        d = session()
        d['agents'][0]['goals'] = [[1, 2], [3.5, -1, 90]]
        s = sim.validate_session(d)
        self.assertEqual(s.agents[0].goals, ((1.0, 2.0), (3.5, -1.0, 90.0)))

    def test_bad_names(self):
        for name in ('J81', 'jr81', 'JR8', 'JR811', 'JRX1', 81):
            d = session()
            d['agents'][0]['name'] = name
            self.assertIn('two capital letters and two digits', errors_of(d), name)

    def test_unknown_prefix(self):
        d = session(agents=[{'name': 'NX01', 'class': 'jr', 'pose': [0, 0, 0]}])
        e = errors_of(d)
        self.assertIn('prefix NX is not a robot class', e)
        self.assertIn('class jr needs the name prefix JR', e)

    def test_class_mismatch(self):
        d = session(agents=[{'name': 'JR81', 'class': 'rr', 'pose': [0, 0, 0]}])
        self.assertIn('class rr needs the name prefix RR', errors_of(d))

    def test_bad_class(self):
        d = session(agents=[{'name': 'JR81', 'class': 'jk', 'pose': [0, 0, 0]}])
        self.assertIn("class: 'jk' is not one of jr, rr", errors_of(d))

    def test_suffix_unique_across_classes(self):
        d = session(agents=[{'name': 'JR81', 'class': 'jr', 'pose': [0, 0, 0]},
                            {'name': 'RR81', 'class': 'rr', 'pose': [1, 0, 0]}])
        self.assertIn('id 81 already used by JR81', errors_of(d))

    def test_duplicate_name(self):
        d = session(agents=[{'name': 'JR81', 'class': 'jr', 'pose': [0, 0, 0]},
                            {'name': 'JR81', 'class': 'jr', 'pose': [1, 0, 0]}])
        self.assertIn('duplicate of agents[0]', errors_of(d))

    def test_id_00(self):
        d = session(agents=[{'name': 'JR00', 'class': 'jr', 'pose': [0, 0, 0]}])
        self.assertIn('id 00', errors_of(d))

    def test_bad_pose(self):
        for pose in ([0, 0], [0, 0, 0, 0], [0, 'a', 0], [True, 0, 0], [0, 0, float('nan')], None):
            d = session()
            d['agents'][1]['pose'] = pose
            self.assertIn('is not [x, y, yaw_deg]', errors_of(d), pose)

    def test_top_level(self):
        e = errors_of(session(mapping='cpu', rviz='yes', gui=1, world='bad world', extra=1))
        for want in ("mapping: 'cpu'", "rviz: 'yes'", 'gui: 1 is not', "world: 'bad world'", 'unknown key(s): extra'):
            self.assertIn(want, e)
        self.assertIn('need a non-empty list', errors_of(session(agents=[])))
        self.assertIn('top level', errors_of([1, 2]))

    def test_all_errors_reported_at_once(self):
        d = session(mapping='x', agents=[{'name': 'JR81', 'class': 'rr', 'pose': [0]},
                                         {'name': 'RR81', 'class': 'rr', 'pose': [0, 0, 0], 'color': 1}])
        e = errors_of(d)
        for want in ('mapping', 'class rr needs', 'pose', 'id 81 already used', 'unknown key(s): color'):
            self.assertIn(want, e)

    def test_loadtest_agents(self):
        agents = sim.loadtest_agents(4)
        self.assertEqual([(a.name, a.cls) for a in agents],
                         [('JR81', 'jr'), ('RR82', 'rr'), ('JR83', 'jr'), ('RR84', 'rr')])
        with self.assertRaises(sim.SimError):
            sim.loadtest_agents(20)


class Panes(unittest.TestCase):
    ctx = sim.RunContext(run_dir=Path(sim.RUN_DIR_DEFAULT))

    def test_router_pane(self):
        panes = sim.zenoh_panes(sim.validate_session(session()), self.ctx)
        self.assertEqual(len(panes), 1)
        p = panes[0]
        self.assertEqual((p.container, p.slug, p.gate), ('sim-drive', 'router', False))
        line = sim.pane_command(p)
        argv = shlex.split(line)
        self.assertEqual(argv[:7], ['docker', 'exec', '-it', '-e', 'RUST_LOG=zenoh=info', 'sim-drive', 'bash'])
        self.assertEqual(argv[7], '-c')
        inner = argv[8]
        self.assertEqual(len(argv), 9)
        self.assertTrue(inner.startswith('source /opt/ros/humble/setup.bash && '
                                         'source /home/swarm/code/rover_ws/install/setup.bash && '))
        self.assertTrue(inner.endswith('ros2 run rmw_zenoh_cpp rmw_zenohd --ros-args -r __node:=zenoh_router'))
        self.assertIn('$ZENOH_ROUTER_CONFIG_URI', inner)
        self.assertNotIn('/dev/tcp', inner)        # the router never waits on itself
        # The single-quoted argument keeps $VARs for the container's bash.
        self.assertIn("bash -c 'source", line)

    def test_gated_panes_for_every_container(self):
        for name, c in sim.CONTAINERS.items():
            # A single quote inside the command must survive the host-side quoting.
            p = sim.Pane('t', 's', name, 'echo "$ROBOT it\'s"', env=(('A', 'b c'),))
            argv = shlex.split(sim.pane_command(p))
            self.assertEqual(argv[:5], ['docker', 'exec', '-it', '-e', 'A=b c'])
            self.assertEqual(argv[5], name)
            inner = argv[-1]
            self.assertIn('until (exec 3<>/dev/tcp/sim-drive/7447 && exec 3>&-)', inner)
            self.assertNotIn('echo >/dev/tcp', inner)   # the gate never writes into the router
            self.assertIn('test -r "$ZENOH_SESSION_CONFIG_URI"', inner)
            for probe in c.probes:
                self.assertIn('python3 %s' % probe, inner)
            self.assertTrue(inner.endswith('echo "$ROBOT it\'s"'))
            for f in c.setup:
                self.assertIn('source ' + f, inner)
            # Valid bash, as typed into the pane and as run in the container.
            subprocess.run(['bash', '-n', '-c', inner], check=True)

    def test_known_probe_paths(self):
        self.assertEqual(sim.CONTAINERS['sim-drive'].probes[0],
                         '/home/swarm/code/rover_ws/install/share/rover_bringup/scripts/wait_zenoh_ready.py')
        self.assertEqual(sim.CONTAINERS['sim-dlio-jr'].probes[0], '/home/swarm/code/dlio_ws/scripts/wait_zenoh_ready.py')
        self.assertEqual(sim.CONTAINERS['sim-mapping'].probes[0],
                         '/home/swarm/code/mapping_ws/install/share/rover_bringup/scripts/wait_zenoh_ready.py')

    def test_require_readable_runs(self):
        snippet = sim.require_readable('CFG')
        with tempfile.NamedTemporaryFile() as f:
            ok = subprocess.run(['bash', '-c', snippet + ' && echo ok'], env=dict(os.environ, CFG=f.name),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(ok.stdout.strip(), 'ok')
        env = dict(os.environ)
        env.pop('CFG', None)
        bad = subprocess.run(['bash', '-c', snippet + ' && echo ok'], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn('CFG=<unset> is not readable', bad.stderr)

    def test_overlays_optional(self):
        snippet = sim.setup_snippet(sim.Container('sim-x', (), overlays=('/nonexistent/setup.bash',)))
        self.assertEqual(subprocess.run(['bash', '-c', snippet + ' && echo ok'], stdout=subprocess.PIPE,
                                        universal_newlines=True).stdout.strip(), 'ok')

    def test_sim_only(self):
        for bad in ('acl-c2-master', 'acl-c2-zenoh', 'node-exporter', 'rover-sim'):
            with self.assertRaises(sim.SimError):
                sim.sim_only(bad)
            with self.assertRaises(sim.SimError):
                sim.pane_command(sim.Pane('t', 's', bad, 'true'))
        self.assertEqual(sim.sim_only('sim-drive'), 'sim-drive')

    def test_dlio_container_by_class(self):
        s = sim.validate_session(session())
        self.assertEqual([sim.dlio_container(a) for a in s.agents], ['sim-dlio-jr', 'sim-dlio-rr'])

    def test_tmux_windows(self):
        mk = lambda i, w: sim.Pane('t%d' % i, 's%d' % i, 'sim-mighty', 'true', window=w)
        panes = [mk(i, 'JR81') for i in range(4)] + [mk(i, 'main') for i in range(4, 21)] + [mk(99, 'JR81')]
        wins = sim.tmux_windows(panes, max_per_window=8)
        self.assertEqual([(n, len(ps)) for n, ps in wins],
                         [('JR81', 5), ('main', 8), ('main-2', 8), ('main-3', 1)])

    def test_history_limit_before_every_pane(self):
        # tmux applies history-limit only to panes created after it is set.
        calls = []

        class Done:
            stdout = '%1\n%2\n'
            returncode = 0

        old = sim.tmux
        sim.tmux = lambda *a, **k: (calls.append(a), Done())[1]
        try:
            with tempfile.TemporaryDirectory() as d:
                panes = [sim.Pane('a', 'a', 'sim-drive', 'true'), sim.Pane('b', 'b', 'sim-drive', 'true')]
                sim.start_stack(sim.STACKS['zenoh'], panes, Path(d))
        finally:
            sim.tmux = old
        verbs = [c[0] for c in calls]
        limit = next(i for i, c in enumerate(calls) if c[0] == 'set-option' and 'history-limit' in c)
        self.assertEqual(verbs.index('new-session'), limit - 1)       # creates only the placeholder
        self.assertEqual(calls[limit - 1][5], sim.TMUX_PLACEHOLDER)
        self.assertLess(limit, verbs.index('new-window'))
        self.assertLess(limit, verbs.index('split-window'))
        self.assertIn(('kill-window', '-t', '=rsim-zenoh:' + sim.TMUX_PLACEHOLDER), calls)
        self.assertEqual(calls[-1], ('select-window', '-t', '=rsim-zenoh:main'))
        # Panes are found by slug, as a shell prompt may retitle them.
        self.assertIn(('set-option', '-p', '-t', '%1', '@rsim_slug', 'a'), calls)
        self.assertIn(('set-option', '-p', '-t', '%2', '@rsim_slug', 'b'), calls)


class Stacks(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(list(sim.STACKS), ['zenoh', 'gazebo', 'drive', 'frame_align', 'dlio', 'mapping',
                                            'elevation', 'mighty', 'rviz'])
        self.assertEqual([s.name for s in sim.STACKS.values() if s.implemented],
                         ['zenoh', 'gazebo', 'drive', 'frame_align', 'dlio'])
        self.assertEqual(sim.STACKS['dlio'].containers, ('sim-dlio-jr', 'sim-dlio-rr'))
        for s in sim.STACKS.values():
            for c in s.containers:
                self.assertIn(c, sim.CONTAINERS)
        self.assertEqual(sim.STACKS['frame_align'].containers, ('sim-mighty',))
        # Only stacks that open windows get X access (xhost +SI:localuser:root): gazebo
        # only for gzclient.
        plain, gui = sim.validate_session(session()), sim.validate_session(session(gui=True))
        self.assertEqual([s.name for s in sim.STACKS.values() if s.needs_display(plain)], ['rviz'])
        self.assertEqual([s.name for s in sim.STACKS.values() if s.needs_display(gui)], ['gazebo', 'rviz'])

    def test_select_default(self):
        s = sim.validate_session(session())
        self.assertEqual([x.name for x in sim.select_stacks(s)], ['zenoh', 'gazebo', 'drive', 'frame_align', 'dlio'])
        self.assertEqual(sim.pending_stacks(s), ['mapping (Phase 4)', 'mighty (Phase 5)'])
        gpu = sim.validate_session(session(mapping='gpu', rviz=True))
        self.assertIn('elevation (Phase 4)', sim.pending_stacks(gpu))
        self.assertIn('rviz (Phase 5)', sim.pending_stacks(gpu))
        self.assertEqual(sim.session_profiles(gpu), ['gpu'])
        self.assertEqual(sim.session_profiles(s), [])
        # One agent: nothing to align.
        one = sim.validate_session(session(agents=[{'name': 'JR81', 'class': 'jr', 'pose': [0, 0, 0]}]))
        self.assertEqual([x.name for x in sim.select_stacks(one)], ['zenoh', 'gazebo', 'drive', 'dlio'])

    def test_select_requested(self):
        s = sim.validate_session(session())
        self.assertEqual([x.name for x in sim.select_stacks(s, ['zenoh'])], ['zenoh'])
        self.assertEqual([x.name for x in sim.select_stacks(s, ['drive', 'gazebo'])], ['gazebo', 'drive'])
        self.assertEqual([x.name for x in sim.select_stacks(s, ['dlio', 'zenoh'])], ['zenoh', 'dlio'])
        with self.assertRaisesRegex(sim.SimError, 'Phase 4'):
            sim.select_stacks(s, ['mapping'])
        with self.assertRaisesRegex(sim.SimError, 'unknown stack'):
            sim.select_stacks(s, ['nope'])
        one = sim.validate_session(session(agents=[{'name': 'JR81', 'class': 'jr', 'pose': [0, 0, 0]}]))
        with self.assertRaisesRegex(sim.SimError, r'frame_align stack does not apply .*1 agent'):
            sim.select_stacks(one, ['frame_align'])

    def test_generate_steps_once_each(self):
        st = sim.STACKS
        self.assertEqual(sim.generate_steps([st['zenoh'], st['gazebo'], st['drive'], st['frame_align'], st['dlio']]),
                         [sim.write_agent_files, sim.link_calib, sim.write_teleop_file, sim.write_dlio_launch])
        self.assertEqual(sim.generate_steps([st['drive']]), [sim.write_agent_files, sim.link_calib,
                                                             sim.write_teleop_file])
        self.assertEqual(sim.generate_steps([st['dlio']]), [sim.write_dlio_launch])
        self.assertEqual(sim.generate_steps([st['gazebo']]), [sim.write_agent_files])
        self.assertEqual(sim.generate_steps([st['zenoh']]), [])

    def test_placeholders_raise(self):
        s = sim.validate_session(session(mapping='gpu', rviz=True))
        ctx = sim.RunContext(run_dir=Path('/tmp/x'))
        for st in sim.STACKS.values():
            if not st.implemented:
                with self.assertRaisesRegex(sim.SimError, 'Phase %d' % st.phase):
                    st.panes(s, ctx)


def inner_of(p):
    """The in-container command of a pane, as docker exec receives it."""
    return shlex.split(sim.pane_command(p))[-1]


def launch_args(cmd, verb):
    """k:=v arguments of the first `verb ...` step of an && chain."""
    part = next(c for c in cmd.split(' && ') if c.startswith(verb))
    return OrderedDict(t.split(':=', 1) for t in shlex.split(part) if ':=' in t)


def pose(x, y, yaw):
    """3x3 homogeneous 2D transform."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]]


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


class FakeProc:
    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


SWAP = sim.load_session('multi_agent_swap')


class ShippedSessions(unittest.TestCase):
    def test_multi_agent_swap(self):
        s = SWAP
        self.assertEqual([(a.name, a.cls) for a in s.agents],
                         [('JR81', 'jr'), ('JR82', 'jr'), ('RR83', 'rr'), ('RR84', 'rr')])
        self.assertEqual((s.world, s.mapping, s.rviz, s.gui), ('arena', 'gpu', False, False))
        for a in s.agents:
            self.assertAlmostEqual(math.hypot(a.x, a.y), 4.0, delta=0.01)          # an 8 m circle
            facing = math.degrees(math.atan2(-a.y, -a.x))
            self.assertAlmostEqual(math.remainder(a.yaw_deg - facing, 360.0), 0.0, delta=0.5)
            opposite = [o for o in s.agents if math.hypot(o.x + a.x, o.y + a.y) < 1e-6]
            self.assertEqual(len(opposite), 1, a.name)
            self.assertEqual(a.goals, ((opposite[0].x, opposite[0].y),), a.name)
            self.assertNotEqual(opposite[0].cls, a.cls)                         # crossings are cross-class

    def test_office_explore(self):
        s = sim.load_session('office_explore')
        self.assertEqual([(a.name, a.cls) for a in s.agents], [(a.name, a.cls) for a in SWAP.agents])
        self.assertEqual((s.world, s.mapping, s.rviz, s.gui), ('ACL_office_simple', 'gpu', False, False))
        self.assertTrue(all(not a.goals for a in s.agents))
        world = REPO / 'src' / 'rover_sim' / 'worlds' / 'ACL_office_simple.world'
        if not world.is_file():
            self.skipTest('no %s' % world)
        boxes = world_boxes(world)
        self.assertGreater(len(boxes), 50)
        for a in s.agents:
            # Free: over 1 m from every wall, and inside the building (a wall in each of
            # four directions).
            self.assertGreater(min(box_distance(a.x, a.y, b) for b in boxes), 1.0, a.name)
            for ang in (0.0, math.pi / 2, math.pi, -math.pi / 2):
                self.assertTrue(any(min(box_distance(a.x + r * 0.1 * math.cos(ang), a.y + r * 0.1 * math.sin(ang), b)
                                        for b in boxes) == 0.0 for r in range(1, 300)), (a.name, ang))
        for a in s.agents:
            for o in s.agents:
                if o is not a:
                    self.assertGreater(math.hypot(a.x - o.x, a.y - o.y), 5.0)


def world_boxes(path):
    """2D footprints (cx, cy, yaw, sx, sy) of the box collisions of a Gazebo world."""
    def pose_of(el):
        p = el.find('pose')
        v = [float(t) for t in p.text.split()] if p is not None and p.text else [0.0] * 6
        return v[0], v[1], v[5]

    def compose(a, b):
        c, s = math.cos(a[2]), math.sin(a[2])
        return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]

    out = []
    for m in ET.parse(str(path)).getroot().find('world').findall('model'):
        for link in m.findall('link'):
            lp = compose(pose_of(m), pose_of(link))
            for col in link.findall('collision'):
                size = col.find('geometry/box/size')
                if size is not None:
                    sx, sy, _ = (float(t) for t in size.text.split())
                    out.append(compose(lp, pose_of(col)) + (sx, sy))
    return out


def box_distance(x, y, b):
    cx, cy, yaw, sx, sy = b
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * (x - cx) + s * (y - cy), -s * (x - cx) + c * (y - cy)
    return math.hypot(max(abs(lx) - sx / 2, 0.0), max(abs(ly) - sy / 2, 0.0))


class FrameAlign(unittest.TestCase):
    def test_transform_composes(self):
        # T_world_ego * T_ego_other == T_world_other, for any two spawn poses.
        rnd = random.Random(7)
        for _ in range(200):
            e = sim.Agent('JR81', 'jr', rnd.uniform(-20, 20), rnd.uniform(-20, 20), rnd.uniform(-360, 360))
            o = sim.Agent('RR83', 'rr', rnd.uniform(-20, 20), rnd.uniform(-20, 20), rnd.uniform(-360, 360))
            tx, ty, tz, qx, qy, qz, qw = sim.frame_align_transform(e, o)
            self.assertEqual((tz, qx, qy), (0.0, 0.0, 0.0))
            self.assertGreaterEqual(qw, 0.0)
            self.assertAlmostEqual(qz * qz + qw * qw, 1.0)
            got = mul(pose(e.x, e.y, e.yaw), pose(tx, ty, 2 * math.atan2(qz, qw)))
            want = pose(o.x, o.y, o.yaw)
            for i in range(2):
                for j in range(3):
                    self.assertAlmostEqual(got[i][j], want[i][j], places=9)

    def test_known_pair(self):
        # Ego facing +y at the origin; other at (1, 2) facing +x: 2 m ahead, 1 m to the right.
        e, o = sim.Agent('JR81', 'jr', 0.0, 0.0, 90.0), sim.Agent('RR83', 'rr', 1.0, 2.0, 0.0)
        t = sim.frame_align_transform(e, o)
        for got, want in zip(t, (2.0, -1.0, 0.0, 0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))):
            self.assertAlmostEqual(got, want)

    def test_names_cross_class(self):
        jr, rr = sim.Agent('JR81', 'jr', 0, 0, 0), sim.Agent('RR83', 'rr', 1, 0, 0)
        self.assertEqual(sim.frame_align_name(jr, rr), 'JR83')     # ego's prefix, other's digits
        self.assertEqual(sim.frame_align_name(rr, jr), 'RR81')
        s = sim.validate_session(session())
        table = sim.frame_align_table(s)['pairs']
        self.assertEqual([(p['ego'], p['other'], p['topic'], p['node']) for p in table],
                         [('JR81', 'RR83', '/frame_align/JR81/JR83', 'frame_align_JR81_RR83'),
                          ('RR83', 'JR81', '/frame_align/RR83/RR81', 'frame_align_RR83_JR81')])
        self.assertEqual(list(table[0])[4:], list(sim.FRAME_ALIGN_KEYS))

    def test_pairs(self):
        pairs = sim.frame_align_pairs(SWAP)
        self.assertEqual(len(pairs), 12)
        self.assertEqual(len({(e.name, o.name) for e, o in pairs}), 12)
        self.assertFalse([1 for e, o in pairs if e.name == o.name])

    def test_panes(self):
        ctx = sim.RunContext(run_dir=Path('/tmp/x'))
        panes = sim.frame_align_panes(SWAP, ctx)
        self.assertEqual(len(panes), 12)
        self.assertEqual(len({p.slug for p in panes}), 12)
        self.assertEqual([w for w, _ in sim.tmux_windows(panes)], ['align-JR81', 'align-JR82', 'align-RR83',
                                                                   'align-RR84'])
        by_slug = {p.slug: p for p in panes}
        p = by_slug['frame-align-JR82-RR84']
        self.assertEqual(p.container, 'sim-mighty')
        argv = shlex.split(p.cmd)
        self.assertEqual(argv[:6], ['ros2', 'run', 'mighty', 'frame_align_publisher.py', '--ros-args', '-r'])
        self.assertEqual(argv[6], '__node:=frame_align_JR82_RR84')
        params = OrderedDict(argv[i + 1].split(':=', 1) for i in range(7, len(argv), 2) if argv[i] == '-p')
        self.assertEqual(list(params), ['ego_name', 'other_name'] + list(sim.FRAME_ALIGN_KEYS))
        self.assertEqual((params['ego_name'], params['other_name']), ('JR82', 'JR84'))
        # Doubles: a bare integer would be refused by the publisher's double parameters.
        for k in sim.FRAME_ALIGN_KEYS:
            self.assertIn('.', params[k], k)
        jr82, rr84 = SWAP.agents[1], SWAP.agents[3]
        for k, v in zip(sim.FRAME_ALIGN_KEYS, sim.frame_align_transform(jr82, rr84)):
            self.assertAlmostEqual(float(params[k]), v, places=8)
        subprocess.run(['bash', '-n', '-c', inner_of(p)], check=True)

    def test_fmt_float(self):
        for v, want in ((1.0, '1.0'), (-2.5, '-2.5'), (0.0, '0.0'), (-0.0, '0.0'), (-1e-12, '0.0'),
                        (5.656854249492, '5.656854249'), (1e-5, '0.00001'), (123456.0, '123456.0')):
            self.assertEqual(sim.fmt_float(v), want)


class CpuSplit(unittest.TestCase):
    def test_split(self):
        self.assertEqual(sim.split_cpus([2, 3, 4, 5]), ([2, 3, 4], [5]))     # sim-gazebo's cpu_budget cores
        self.assertEqual(sim.split_cpus(range(8)), ([0, 1, 2, 3, 4, 5], [6, 7]))
        self.assertEqual(sim.split_cpus([5, 2, 4, 3, 3]), ([2, 3, 4], [5]))
        self.assertEqual(sim.split_cpus([2, 3, 4, 5], spare=2), ([2, 3], [4, 5]))
        self.assertEqual(sim.split_cpus([2, 3, 4, 5], spare=9), ([2], [3, 4, 5]))   # gzserver keeps one
        self.assertEqual(sim.split_cpus([2, 3]), ([2], [3]))
        self.assertEqual(sim.split_cpus([7]), ([7], [7]))                   # nothing to split: shared

    def test_pin_self(self):
        ctx = sim.RunContext(run_dir=Path('/tmp/x'))
        self.assertEqual(sim.pin_self('gz', ctx), 'taskset -cp "$(python3 /rover_sim/scripts/sim.py cpus gz)" $$')
        spare = sim.RunContext(run_dir=Path('/tmp/x'), gz_spare_cpus=2)
        self.assertIn('cpus rest --spare 2)', sim.pin_self('rest', spare))

    def test_spare_from_env(self):
        with mock.patch.dict(os.environ, {'ROVER_SIM_GZ_SPARE_CPUS': '2'}):
            self.assertEqual(sim.gz_spare_from_env(), 2)
        for bad in ('0', '-1', 'x'):
            with mock.patch.dict(os.environ, {'ROVER_SIM_GZ_SPARE_CPUS': bad}):
                with self.assertRaises(sim.SimError):
                    sim.gz_spare_from_env()
        env = dict(os.environ)
        env.pop('ROVER_SIM_GZ_SPARE_CPUS', None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(sim.gz_spare_from_env())


class Phase2Panes(unittest.TestCase):
    ctx = sim.RunContext(run_dir=Path(sim.RUN_DIR_DEFAULT))

    def test_gazebo_panes(self):
        panes = sim.gazebo_panes(SWAP, self.ctx)
        self.assertEqual([p.slug for p in panes], ['gzserver', 'spawn', 'sim-tf', 'bridge-JR81', 'bridge-JR82',
                                                   'bridge-RR83', 'bridge-RR84'])
        self.assertEqual({p.container for p in panes}, {'sim-gazebo'})
        self.assertEqual([(w, len(ps)) for w, ps in sim.tmux_windows(panes)], [('gazebo', 3), ('bridges', 4)])
        by = {p.slug: p for p in panes}
        self.assertEqual(by['gzserver'].cmd, sim.pin_self('gz', self.ctx) + ' && ros2 launch rover_sim '
                         'world.launch.py world:=arena gui:=false paused:=true')
        self.assertEqual(by['sim-tf'].cmd, sim.pin_self('rest', self.ctx)
                         + ' && python3 /rover_sim/scripts/sim_tf.py --spawn /rover_sim/run/spawn.yaml')
        self.assertEqual(by['bridge-RR83'].cmd, sim.pin_self('rest', self.ctx)
                         + ' && python3 /rover_sim/scripts/livox_sim_bridge.py --ns RR83')
        # Only gzserver runs on gzserver's cores.
        for p in panes:
            self.assertTrue(p.cmd.startswith(sim.pin_self('gz' if p.slug == 'gzserver' else 'rest', self.ctx)))
            self.assertTrue(p.gate)
            inner = inner_of(p)
            subprocess.run(['bash', '-n', '-c', inner], check=True)
            self.assertIn('source /usr/share/gazebo/setup.bash && source /opt/livox_ws/install/setup.bash && '
                          'source /home/swarm/code/rover_sim_ws/install/setup.bash', inner)
            # $$ is the container's bash: it must reach it unexpanded (single-quoted on the host).
            self.assertIn("'", sim.pane_command(p).split(' bash -c ', 1)[1][:1])
            self.assertIn(' $$ && ', inner)

    def test_gui(self):
        s = sim.validate_session(session(gui=True))
        panes = sim.gazebo_panes(s, self.ctx)
        # gzserver keeps its cores to itself; gzclient runs on the rest's, in its own pane.
        self.assertIn('world:=empty gui:=false paused:=true', panes[0].cmd)
        gzc = [p for p in panes if p.slug == 'gzclient']
        self.assertEqual(len(gzc), 1)
        self.assertEqual(gzc[0].cmd, sim.pin_self('rest', self.ctx)
                         + ' && ros2 launch rover_sim world.launch.py server:=false gui:=true')
        self.assertEqual(gzc[0].window, 'gazebo')
        self.assertNotIn('gzclient', [p.slug for p in sim.gazebo_panes(SWAP, self.ctx)])

    def test_spawn_sequence(self):
        cmd = sim.spawn_cmd(SWAP, self.ctx)
        steps = cmd.split(' && ')
        self.assertEqual(steps[0], sim.pin_self('rest', self.ctx))
        wait = next(i for i, c in enumerate(steps) if '/spawn_entity' in c and c.startswith('until '))
        spawns = [i for i, c in enumerate(steps) if c.startswith('ros2 launch rover_sim spawn_agent.launch.py')]
        unpause = [i for i, c in enumerate(steps) if 'unpause_physics' in c]
        self.assertEqual(len(spawns), 4)
        self.assertEqual(len(unpause), 1)
        self.assertLess(wait, spawns[0])
        self.assertEqual(unpause[0], spawns[-1] + 1)                   # once, after the last spawn
        # Every CLI has a SIGKILL backstop: a hung call fails the pane, and up says so.
        self.assertEqual(steps[unpause[0]], 'timeout -k 2 30 ros2 service call /unpause_physics std_srvs/srv/Empty')
        self.assertEqual(steps[-1], 'exec sleep infinity')
        self.assertIn('until timeout -k 2 15 ros2 service list --no-daemon', steps[wait])
        subprocess.run(['bash', '-n', '-c', cmd], check=True)

    def test_spawn_agent_args(self):
        rr84 = SWAP.agents[3]
        args = launch_args(sim.spawn_agent_cmd(rr84, 1500), 'ros2 launch rover_sim spawn_agent.launch.py')
        self.assertEqual(list(args), ['namespace', 'class', 'x', 'y', 'yaw', 'lidar_mount_file', 'lidar_height',
                                      'lidar_samples', 'urdf_file'])
        self.assertEqual((args['namespace'], args['class'], args['x'], args['y']), ('RR84', 'rr', '2.83', '-2.83'))
        self.assertAlmostEqual(float(args['yaw']), math.radians(135.0), places=8)     # radians
        self.assertEqual(args['lidar_mount_file'], '/rover_sim/run/sensor_calib/RR84_lidar_mount.yaml')
        self.assertEqual(float(args['lidar_height']), 0.40)
        self.assertEqual(args['lidar_samples'], '1500')
        self.assertEqual(args['urdf_file'], '/rover_sim/run/urdf/RR84.urdf')
        jr = launch_args(sim.spawn_agent_cmd(SWAP.agents[0], 500), 'ros2 launch')
        self.assertEqual(float(jr['lidar_height']), 0.36)

    def test_spawn_uses_the_world_budget(self):
        for world in ('empty', 'arena', 'ACL_office_simple'):
            s = dataclasses.replace(SWAP, world=world)
            want = 'lidar_samples:=%d ' % sim.world_lidar_samples(world)
            self.assertEqual(sim.spawn_cmd(s, self.ctx).count(want), len(s.agents), world)

    def test_world_lidar_samples(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'worlds.yaml'
            p.write_text('lidar_samples_4_agents:\n  empty: 4000\n  bad: 0\n')
            self.assertEqual(sim.world_lidar_samples('empty', p), 4000)
            self.assertEqual(sim.world_lidar_samples('other', p), sim.LIDAR_SAMPLES_DEFAULT)
            with self.assertRaises(sim.SimError):
                sim.world_lidar_samples('bad', p)
            p.write_text('lidar_samples_4_agents: [\n')
            with self.assertRaises(sim.SimError):
                sim.world_lidar_samples('empty', p)
        # The shipped file has a budget for every shipped world.
        for w in sim.WORLDS_DIR.glob('*.world'):
            doc = yaml.safe_load(sim.WORLD_BUDGETS.read_text())
            self.assertIn(w.stem, doc['lidar_samples_4_agents'])

    def test_world_walls(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'worlds.yaml'
            p.write_text('walls:\n  box: [-2, -1, 2.5, 1]\n  flipped: [2, 0, -2, 1]\n  short: [1, 2]\n')
            self.assertEqual(sim.world_walls('box', p), [-2.0, -1.0, 2.5, 1.0])
            self.assertIsNone(sim.world_walls('other', p))
            for bad in ('flipped', 'short'):
                with self.assertRaisesRegex(sim.SimError, 'not \\[xmin, ymin, xmax, ymax\\]'):
                    sim.world_walls(bad, p)
            p.write_text('lidar_samples_4_agents: {}\n')
            self.assertIsNone(sim.world_walls('box', p))
        # The arena's inner wall faces (arena.world: 0.2 m walls centred at +-8.1).
        self.assertEqual(sim.world_walls('arena'), [-8.0, -8.0, 8.0, 8.0])
        self.assertIsNone(sim.world_walls('ACL_office_simple'))

    def test_clear_ros_logs(self):
        calls = []
        with mock.patch.object(sim, 'dexec', lambda n, args, **k: calls.append((n, args, k))):
            sim.clear_ros_logs(['sim-drive', 'sim-gazebo'])
        self.assertEqual(sorted(c[0] for c in calls), ['sim-drive', 'sim-gazebo'])
        self.assertEqual(calls[0][1], ['sh', '-c', 'rm -rf -- "${ROS_LOG_DIR:-${ROS_HOME:-$HOME/.ros}/log}"'])
        self.assertFalse(calls[0][2]['check'])

    def test_spare_cpus_reach_every_pinned_pane(self):
        ctx = sim.RunContext(run_dir=Path('/tmp/x'), gz_spare_cpus=2)
        for p in sim.gazebo_panes(SWAP, ctx):
            self.assertIn('--spare 2)" $$', p.cmd)

    def test_drive_panes(self):
        panes = sim.drive_panes(SWAP, self.ctx)
        self.assertEqual([p.slug for p in panes], ['teleop-safety-' + a.name for a in SWAP.agents]
                         + ['description-tf-' + a.name for a in SWAP.agents])
        self.assertEqual([(w, len(ps)) for w, ps in sim.tmux_windows(panes)], [('teleop', 4), ('description', 4)])
        p = panes[6]
        self.assertEqual((p.container, p.window, p.gate), ('sim-drive', 'description', True))
        self.assertEqual(p.cmd, 'ros2 launch rover_description state_publisher.launch.py robot_name:=RR83 '
                                'ns_tf_frame_ids:=true')
        argv = shlex.split(sim.pane_command(p))
        self.assertEqual(argv[:9], ['docker', 'exec', '-it', '-e', 'VEHTYPE=RR', '-e', 'VEHNUM=83', '-e',
                                    'ROBOT_NAME=RR83'])
        subprocess.run(['bash', '-n', '-c', argv[-1]], check=True)


SAMPLE_MOUNT = b'# test mount\nxyz: [0.0, 0.0, 0.0]\nrpy: [0.0, 0.5, 0.0]   # pitch\n'
SAMPLE_MOUNT_RR = b'# test mount rr\nxyz: [0.0, 0.0, 0.0]\nrpy: [0.0, 0.35, 0.0]\n'
SAMPLE_NPY = b'\x93NUMPY\x01\x00' + bytes(range(40))


class RunFiles(unittest.TestCase):
    def test_reference_robots_from_pins(self):
        pins = sim.read_pins()
        self.assertEqual((sim.reference_robot('jr', pins), sim.reference_robot('rr', pins)), ('JR02', 'RR08'))
        self.assertEqual((pins['DLIO_JR_ROBOT'], pins['DLIO_RR_ROBOT']), ('JR02', 'RR08'))
        for bad in ({}, {'DLIO_JR_ROBOT': 'RR08'}, {'DLIO_JR_ROBOT': 'JR2'}):
            with self.assertRaises(sim.SimError):
                sim.reference_robot('jr', bad)

    def test_class_table(self):
        self.assertEqual([(k, c.prefix, c.lidar_height) for k, c in sim.ROBOT_CLASSES.items()],
                         [('jr', 'JR', 0.36), ('rr', 'RR', 0.40)])
        self.assertEqual(sim.CLASS_PREFIXES, OrderedDict([('jr', 'JR'), ('rr', 'RR')]))
        # Same heights as the Gazebo models' own defaults.
        classes = REPO / 'src' / 'rover_sim' / 'config' / 'classes.yaml'
        if classes.is_file():
            doc = yaml.safe_load(classes.read_text())
            for k, c in sim.ROBOT_CLASSES.items():
                self.assertEqual(doc[k]['lidar_height'], c.lidar_height, k)

    def test_spawn_table(self):
        s = sim.validate_session(session())
        pins = {'DLIO_JR_ROBOT': 'JR02', 'DLIO_RR_ROBOT': 'RR08'}
        t = sim.spawn_table(s, pins)
        self.assertEqual(list(t), ['world', 'agents'])                 # empty has no walls
        self.assertEqual(t['world'], 'empty')
        arena = sim.spawn_table(sim.validate_session(session(world='arena')), pins)
        self.assertEqual((list(arena), arena['walls']), (['world', 'walls', 'agents'], [-8.0, -8.0, 8.0, 8.0]))
        self.assertEqual([list(a) for a in t['agents']],
                         [['name', 'class', 'id', 'x', 'y', 'yaw', 'yaw_deg', 'lidar_height', 'reference_robot']] * 2)
        rr = t['agents'][1]
        self.assertEqual((rr['name'], rr['class'], rr['id'], rr['x'], rr['y'], rr['yaw_deg'], rr['lidar_height'],
                          rr['reference_robot']), ('RR83', 'rr', 83, 2.0, 1.0, 90.0, 0.40, 'RR08'))
        self.assertAlmostEqual(rr['yaw'], math.pi / 2)
        text = sim.generated_yaml(s, 'Agents', t)
        self.assertTrue(text.startswith('# Agents for session t'))
        self.assertEqual(yaml.safe_load(text), t)

    def test_mount_header(self):
        a = sim.Agent('JR81', 'jr', 0, 0, 0)
        pins = {'DRIVE_IMAGE': 'rover-sim/drive:9b0b4ba', 'DRIVE_SHA': '9b0b4ba5e56930b67fd2061a395e926147ca99b2'}
        h = sim.mount_header(a, 'JR02', pins)
        self.assertTrue(h.startswith(b'# rover_sim: JR81 '))
        self.assertTrue(h.endswith(b'\n'))
        self.assertEqual(h.count(b'\n'), 1)
        self.assertIn(b'JR02_lidar_mount.yaml verbatim from rover-sim/drive:9b0b4ba (rover 9b0b4ba)', h)
        full = h + SAMPLE_MOUNT
        self.assertEqual(full[len(h):], SAMPLE_MOUNT)
        self.assertEqual(yaml.safe_load(full.decode()), yaml.safe_load(SAMPLE_MOUNT.decode()))
        sim.check_mount(full, 'test')

    def test_check_mount(self):
        sim.check_mount(SAMPLE_MOUNT, 'ok')
        for bad in (b'xyz: [0, 0, 0]\n', b'xyz: [0, 0]\nrpy: [0, 0, 0]\n', b'xyz: [a, 0, 0]\nrpy: [0, 0, 0]\n',
                    b'- 1\n', b': :\n', b'\xff\xfe'):
            with self.assertRaises(sim.SimError, msg=bad):
                sim.check_mount(bad, 'bad')

    def test_write_agent_files(self):
        s = SWAP
        calls = []

        def fake_dexec(name, args, **kw):
            calls.append((name, args))
            path = args[-1]
            data = {'JR02_lidar_mount.yaml': SAMPLE_MOUNT, 'RR08_lidar_mount.yaml': SAMPLE_MOUNT_RR,
                    'JR02_calib.npy': SAMPLE_NPY, 'RR08_calib.npy': SAMPLE_NPY + b'rr'}.get(os.path.basename(path))
            return FakeProc(0, base64.b64encode(data).decode()) if data else FakeProc(1, '', 'no such file')

        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'sensor_calib').mkdir()
            (run / 'sensor_calib' / 'JR99_calib.npy').write_text('stale')
            (run / 'urdf').mkdir()
            (run / 'urdf' / 'JR99.urdf').write_text('stale')
            with mock.patch.object(sim, 'dexec', fake_dexec), mock.patch.object(sim, 'running', lambda n: True), \
                    contextlib.redirect_stdout(io.StringIO()):
                sim.write_agent_files(s, sim.RunContext(run_dir=run))
            # Read from the drive image's rover_description, once per reference robot and file.
            self.assertEqual({c[0] for c in calls}, {'sim-drive'})
            self.assertEqual(sorted(c[1][-1] for c in calls), sorted(
                sim.RD_SHARE + '/sensor_calib/' + f for f in ('JR02_lidar_mount.yaml', 'JR02_calib.npy',
                                                              'RR08_lidar_mount.yaml', 'RR08_calib.npy')))
            self.assertEqual(sorted(p.name for p in (run / 'sensor_calib').iterdir()), sorted(
                '%s_%s' % (a.name, f) for a in s.agents for f in ('lidar_mount.yaml', 'calib.npy')))
            self.assertEqual(list((run / 'urdf').iterdir()), [])
            m = (run / 'sensor_calib' / 'RR84_lidar_mount.yaml').read_bytes()
            first, rest = m.split(b'\n', 1)
            self.assertTrue(first.startswith(b'# rover_sim: RR84 (class rr) uses RR08'))
            self.assertEqual(rest, SAMPLE_MOUNT_RR)
            self.assertEqual((run / 'sensor_calib' / 'JR82_calib.npy').read_bytes(), SAMPLE_NPY)
            self.assertEqual((run / 'sensor_calib' / 'RR83_calib.npy').read_bytes(), SAMPLE_NPY + b'rr')
            spawn = yaml.safe_load((run / 'spawn.yaml').read_text())
            self.assertEqual(spawn, sim.spawn_table(s, sim.read_pins()))
            fa = yaml.safe_load((run / 'frame_align.yaml').read_text())
            self.assertEqual(len(fa['pairs']), 12)
            # drive_pattern.py's limits, readable from every container's /rover_sim/run.
            self.assertEqual((run / 'classes.yaml').read_bytes(), sim.CLASSES_YAML.read_bytes())
            # Refuses to start without sim-drive, the calibration's source.
            with mock.patch.object(sim, 'running', lambda n: False):
                with self.assertRaisesRegex(sim.SimError, 'sim-drive'):
                    sim.write_agent_files(s, sim.RunContext(run_dir=run))

    def test_run_dir_writes_never_follow_links(self):
        # The containers write the run dir as root: a link planted there must not make
        # sim.py write or delete anything outside it.
        with tempfile.TemporaryDirectory() as d:
            run, outside = Path(d) / 'run', Path(d) / 'outside'
            run.mkdir()
            outside.mkdir()
            (outside / 'precious').write_text('keep')
            (run / '.spawn.yaml.tmp').symlink_to(outside / 'precious')       # the temp name
            sim.write_file(run / 'spawn.yaml', b'a: 1\n')
            self.assertEqual((outside / 'precious').read_text(), 'keep')
            self.assertEqual((run / 'spawn.yaml').read_bytes(), b'a: 1\n')
            (run / 'spawn.yaml').unlink()
            (run / 'spawn.yaml').symlink_to(outside / 'precious')            # the name itself
            sim.write_file(run / 'spawn.yaml', b'a: 2\n')
            self.assertFalse((run / 'spawn.yaml').is_symlink())
            self.assertEqual((outside / 'precious').read_text(), 'keep')
            self.assertEqual(sorted(p.name for p in run.iterdir()), ['spawn.yaml'])
            (run / 'urdf').symlink_to(outside, target_is_directory=True)       # a whole subdir
            with self.assertRaises(sim.SimError):
                sim.write_file(run / 'urdf' / 'JR81.urdf', b'x')
            sim.empty_own_dir(run / 'urdf')
            self.assertTrue((run / 'urdf').is_dir() and not (run / 'urdf').is_symlink())
            self.assertEqual(sorted(p.name for p in outside.iterdir()), ['precious'])
            (run / 'urdf' / 'old.urdf').write_text('x')
            (run / 'urdf' / 'link').symlink_to(outside / 'precious')
            (run / 'urdf' / 'sub').mkdir()
            sim.empty_own_dir(run / 'urdf')
            self.assertEqual(sorted(p.name for p in (run / 'urdf').iterdir()), ['sub'])
            self.assertEqual((outside / 'precious').read_text(), 'keep')
            (run / 'file').write_text('')
            with self.assertRaises(sim.SimError):
                sim.empty_own_dir(run / 'file')
            sim.write_session_file(sim.validate_session(session()), run)
            self.assertEqual(json.loads((run / 'session.json').read_text())['name'], 't')

    def test_read_file_script(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'a.npy'
            f.write_bytes(SAMPLE_NPY)
            (Path(d) / 'link.npy').symlink_to(f)

            def run(p):
                return subprocess.run(['bash', '-c', sim._READ_FILE_SH, 'read', str(p)], stdout=subprocess.PIPE,
                                      universal_newlines=True)
            ok = run(f)
            self.assertEqual((ok.returncode, base64.b64decode(ok.stdout)), (0, SAMPLE_NPY))
            self.assertNotEqual(run(Path(d) / 'link.npy').returncode, 0)     # never one of the run's own links
            self.assertNotEqual(run(Path(d) / 'missing').returncode, 0)

    def test_link_calib_script(self):
        # The script link_calib runs in sim-drive, here against a temp "share".
        with tempfile.TemporaryDirectory() as d:
            share, src = Path(d) / 'share', Path(d) / 'run'
            share.mkdir()
            src.mkdir()
            for n in ('JR81', 'RR83'):
                (src / (n + '_lidar_mount.yaml')).write_bytes(SAMPLE_MOUNT)
                (src / (n + '_calib.npy')).write_bytes(SAMPLE_NPY)
            (share / 'JR02_lidar_mount.yaml').write_bytes(SAMPLE_MOUNT)       # a real robot's own file
            (share / 'RR83_calib.npy').symlink_to(Path(d) / 'stale')          # an earlier run's link

            def run(*names):
                return subprocess.run(['bash', '-c', sim._LINK_CALIB_SH, 'link', str(share), str(src)] + list(names),
                                      stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(run('JR81', 'RR83').returncode, 0)
            for n in ('JR81', 'RR83'):
                for f in (n + '_lidar_mount.yaml', n + '_calib.npy'):
                    self.assertTrue((share / f).is_symlink())
                    self.assertEqual(os.readlink(str(share / f)), str(src / f))
            self.assertEqual(run('JR81').returncode, 0)                       # idempotent
            (src / 'JR02_lidar_mount.yaml').write_bytes(SAMPLE_MOUNT)
            (src / 'JR02_calib.npy').write_bytes(SAMPLE_NPY)
            p = run('JR02')
            self.assertEqual(p.returncode, 3)
            self.assertIn("image's own file", p.stderr)
            self.assertFalse((share / 'JR02_lidar_mount.yaml').is_symlink())
            self.assertEqual(run('RR99').returncode, 1)                        # nothing generated for it

    def test_link_calib(self):
        seen = []

        def fake(rc):
            def dexec(name, args, **kw):
                seen.append((name, args))
                return FakeProc(rc, '', 'boom')
            return dexec
        s = sim.validate_session(session())
        with mock.patch.object(sim, 'dexec', fake(0)):
            sim.link_calib(s, sim.RunContext(run_dir=Path('/tmp/x')))
        name, args = seen[0]
        self.assertEqual(name, 'sim-drive')
        self.assertEqual(args[:3], ['bash', '-c', sim._LINK_CALIB_SH])
        self.assertEqual(args[3:], ['link', sim.RD_SHARE + '/sensor_calib', '/rover_sim/run/sensor_calib',
                                    'JR81', 'RR83'])
        with mock.patch.object(sim, 'dexec', fake(3)):
            with self.assertRaisesRegex(sim.SimError, 'rename that agent'):
                sim.link_calib(s, sim.RunContext(run_dir=Path('/tmp/x')))

    def test_world_problem(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'empty.world').write_text('<sdf/>')
            self.assertIsNone(sim.world_problem('empty', Path(d)))
            self.assertIn('have: empty', sim.world_problem('office', Path(d)))
        self.assertIn('have: none', sim.world_problem('x', Path('/nonexistent')))
        if sim.WORLDS_DIR.is_dir():
            for p in sorted(SESSIONS_DIR.glob('*.yaml')):
                self.assertIsNone(sim.world_problem(sim.load_session(str(p)).world), p.name)


SESSIONS_DIR = REPO / 'config' / 'sessions'


class SimTimeStatus(unittest.TestCase):
    def test_parse(self):
        out = ('@@ /JR81/sim/rtf 0.998\n@@ /JR81/sim/time_offset 1760812345.25\n@@ /RR83/sim/rtf\n'
               '@@ /JR81/dlio/odom_node/odom 100.00 0.012 301\n@@ /JR81/mode AUTO\nzenoh noise\n@@\n')
        vals = sim.parse_status(out)
        self.assertEqual(vals, {'/JR81/sim/rtf': ['0.998'], '/JR81/sim/time_offset': ['1760812345.25'],
                                '/RR83/sim/rtf': [], '/JR81/dlio/odom_node/odom': ['100.00', '0.012', '301'],
                                '/JR81/mode': ['AUTO']})
        self.assertEqual(sim._num(vals, '/JR81/sim/rtf'), 0.998)
        self.assertEqual(sim._num(vals, '/JR81/dlio/odom_node/odom', 1), 0.012)
        for topic in ('/RR83/sim/rtf', '/JR81/mode', '/nope'):
            self.assertIsNone(sim._num(vals, topic))

    def test_specs(self):
        self.assertEqual(sim.status_specs(['JR81'], ['gazebo']), ['f64:/JR81/sim/rtf', 'f64:/JR81/sim/time_offset'])
        self.assertEqual(sim.status_specs(['JR81', 'RR83'], {'dlio', 'drive', 'mighty'}), [
            'odom:/JR81/dlio/odom_node/odom', 'count:/JR81/dlio/odom_node/pointcloud/deskewed', 'str:/JR81/mode',
            'odom:/RR83/dlio/odom_node/odom', 'count:/RR83/dlio/odom_node/pointcloud/deskewed', 'str:/RR83/mode'])
        self.assertEqual(sim.status_specs(['JR81'], ['zenoh']), [])
        for spec in sim.status_specs(['JR81'], ['gazebo', 'dlio', 'drive']):
            kind, topic = spec.split(':', 1)
            self.assertIn(kind, ('f64', 'str', 'odom', 'count'))
            self.assertTrue(topic.startswith('/JR81/'))

    def test_lines(self):
        vals = {'/JR81/sim/rtf': ['0.998'], '/JR81/sim/time_offset': ['12.3456'], '/JR82/sim/rtf': ['0.9']}
        lines = sim.sim_time_lines(['JR81', 'JR82', 'RR83'], vals)
        self.assertEqual(lines[0], 'JR81  rtf 0.998  time_offset 12.346 s')
        self.assertEqual(lines[1], 'JR82  rtf 0.900  LOW  time_offset -')
        self.assertIn('RR83  no sim/rtf or sim/time_offset', lines[2])

    def test_agent_lines(self):
        vals = sim.parse_status(
            '@@ /JR81/dlio/odom_node/odom 100.00 0.012 301\n@@ /JR81/dlio/odom_node/pointcloud/deskewed 10.00 31\n'
            '@@ /JR81/mode AUTO\n@@ /JR82/dlio/odom_node/odom 100.00 inf 300\n@@ /JR82/dlio/odom_node/pointcloud/deskewed\n'
            '@@ /JR82/mode DISARMED\n@@ /RR83/dlio/odom_node/odom 99.0 2.500 290\n@@ /RR83/mode\n')
        lines = sim.agent_lines(['JR81', 'JR82', 'RR83', 'RR84'], vals, {'dlio', 'drive'})
        self.assertEqual(lines[0], 'JR81  DLIO odom 100 Hz, stamp age 0.012 s,  scans 10.0 Hz,  safety AUTO')
        self.assertEqual(lines[1], 'JR82  DLIO odom 100 Hz, unstamped: no IMU has reached DLIO,  no scans (IMU '
                                   'calibration, seed pose or lidar),  safety DISARMED')
        self.assertIn('stamp age 2.500 s  STALE', lines[2])
        self.assertIn('safety: no /RR83/mode', lines[2])
        self.assertIn('RR84  DLIO odom: none in 8 s', lines[3])
        self.assertEqual(sim.agent_lines(['JR81'], vals, {'drive'}), ['JR81  safety AUTO'])

    def test_query_script(self):
        subprocess.run(['bash', '-n', '-c', sim._STATUS_SH], check=True)
        # One process for every topic, killed if it outlives its timeout (a stuck zenoh session
        # holds the router up until its socket closes).
        self.assertEqual(sim._STATUS_SH, 'exec timeout -k 2 %d python3 -c "$SIM_STATUS_PY" %d 3 "$@"'
                         % (sim.ECHO_TIMEOUT_S + 4, sim.ECHO_TIMEOUT_S))
        compile(sim._STATUS_PY, '<sim_status>', 'exec')
        self.assertNotIn('ros2 topic echo', sim._STATUS_PY)

    def test_query_script_logic(self):
        world = {'pubs': {'/JR81/sim/rtf': {'hz': 50, 'value': 0.999},
                          '/JR81/dlio/odom_node/odom': {'hz': 100, 'stamp': 'now'},
                          '/RR83/dlio/odom_node/odom': {'hz': 100, 'stamp': 0},
                          '/JR81/dlio/odom_node/pointcloud/deskewed': {'hz': 10},
                          '/JR81/mode': {'hz': 20, 'value': 'AUTO'}}}
        specs = ['f64:/JR81/sim/rtf', 'f64:/JR81/sim/time_offset', 'odom:/JR81/dlio/odom_node/odom',
                 'odom:/RR83/dlio/odom_node/odom', 'count:/JR81/dlio/odom_node/pointcloud/deskewed', 'str:/JR81/mode']
        out = run_with_fake_ros(sim._STATUS_PY, ['1.5', '0.5'] + specs, world)
        vals = sim.parse_status(out)
        self.assertEqual(sorted(vals), sorted(s.split(':', 1)[1] for s in specs))
        self.assertEqual(vals['/JR81/sim/rtf'], ['0.999'])
        self.assertEqual(vals['/JR81/sim/time_offset'], [])                     # never published
        self.assertEqual(vals['/JR81/mode'], ['AUTO'])
        hz, age, n = vals['/JR81/dlio/odom_node/odom']
        self.assertGreater(float(hz), 50)
        self.assertLess(float(age), 0.5)
        self.assertEqual(vals['/RR83/dlio/odom_node/odom'][1], 'inf')            # stamp 0: no IMU yet
        self.assertGreater(float(vals['/JR81/dlio/odom_node/pointcloud/deskewed'][0]), 5)

    def test_values_query(self):
        seen = []

        def dexec(name, args, **kw):
            seen.append((name, args, kw))
            return FakeProc(0, '@@ /JR81/sim/rtf 1.0\n@@ /JR81/sim/time_offset\n')
        with mock.patch.object(sim, 'dexec', dexec):
            vals = sim.status_values(['f64:/JR81/sim/rtf', 'f64:/JR81/sim/time_offset'])
        self.assertEqual(vals, {'/JR81/sim/rtf': ['1.0'], '/JR81/sim/time_offset': []})
        name, args, kw = seen[0]
        self.assertEqual(name, 'sim-drive')                 # not sim-gazebo, where the bridges run
        self.assertEqual(args[-3:], ['status', 'f64:/JR81/sim/rtf', 'f64:/JR81/sim/time_offset'])
        self.assertEqual(dict(kw['env']), {'SIM_STATUS_PY': sim._STATUS_PY})
        self.assertNotIn('cpus', args[2])
        self.assertTrue(args[2].endswith(sim._STATUS_SH))
        subprocess.run(['bash', '-n', '-c', args[2]], check=True)

        def hung(name, args, **kw):
            raise sim.SimError('timed out after 28s')
        with mock.patch.object(sim, 'dexec', hung), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(sim.status_values(['f64:/JR81/sim/rtf']), {})

    def test_status_reads_running_stacks(self):
        class Panes:
            returncode = 0
            stdout = 'rsim-dlio\tJR81\t0\tJR81 dlio odometry\tdocker\nrsim-drive\tteleop\t0\tx\tdocker\n'

        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'session.json').write_text(json.dumps(sim.validate_session(session()).to_json()))
            asked = []
            patches = [mock.patch.object(sim, 'running_sim_containers', lambda: ['sim-drive']),
                       mock.patch.object(sim, 'docker_inspect', lambda kind, name: None),
                       mock.patch.object(sim, 'resolve_run_dir', lambda: run),
                       mock.patch.object(sim, 'check_isolation', lambda: []),
                       mock.patch.object(sim, 'link_snapshot', lambda up: {'router': 1, 'p2p': 0, 'listen': 1,
                                                                           'syn_sent': 0}),
                       mock.patch.object(sim, 'router_up', lambda: True),
                       mock.patch.object(sim, 'tmux', lambda *a, **k: Panes()),
                       mock.patch.object(sim, 'status_values', lambda specs: asked.append(specs) or sim.parse_status(
                           '@@ /JR81/mode DISARMED\n'))]
            for p in patches:
                p.start()
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(sim.cmd_status(None), 0)
            finally:
                for p in patches:
                    p.stop()
        self.assertEqual(asked, [sim.status_specs(['JR81', 'RR83'], {'dlio', 'drive'})])
        self.assertIn('sim time: n/a (the gazebo stack is not running)', out.getvalue())
        self.assertIn('JR81  DLIO odom: none in 8 s', out.getvalue())
        self.assertIn('safety DISARMED', out.getvalue())

    def test_violations_and_rotation(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            self.assertEqual(sim.rtf_violations(run), [])
            (run / sim.RTF_LOG).write_text('\n2026-10-18T20:00:00 JR81 rtf 0.91 for 5.2 s\n\n')
            self.assertEqual(sim.rtf_violations(run), ['2026-10-18T20:00:00 JR81 rtf 0.91 for 5.2 s'])
            sim.rotate_rtf_log(run)                                            # what `up` does
            self.assertEqual(sim.rtf_violations(run), [])
            self.assertIn('JR81', (run / 'rtf_violations.prev.log').read_text())
            sim.rotate_rtf_log(run)                                            # nothing to rotate
            if os.getuid() != 0:
                (run / sim.RTF_LOG).write_text('x\n')
                os.chmod(str(run / sim.RTF_LOG), 0)
                self.assertEqual(len(sim.rtf_violations(run)), 1)              # unreadable: not valid
                os.chmod(str(run / sim.RTF_LOG), 0o600)

    def test_status_exit_code(self):
        class NoTmux:
            stdout, returncode = '', 1

        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'session.json').write_text(json.dumps(sim.validate_session(session()).to_json()))
            patches = [mock.patch.object(sim, 'running_sim_containers', lambda: []),
                       mock.patch.object(sim, 'docker_inspect', lambda kind, name: None),
                       mock.patch.object(sim, 'tmux', lambda *a, **k: NoTmux()),
                       mock.patch.dict(os.environ, {'ROVER_SIM_RUN': str(run)})]
            for p in patches:
                p.start()
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(sim.cmd_status(None), 0)
                self.assertIn('sim time: n/a', out.getvalue())
                (run / sim.RTF_LOG).write_text('JR81 rtf 0.80 below 0.95 for 6.0 s\n')
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(sim.cmd_status(None), 2)
                self.assertIn('!! RUN INVALID: the real-time factor was below 0.95 for over 5 s', out.getvalue())
                self.assertIn('!!   JR81 rtf 0.80', out.getvalue())
            finally:
                for p in patches:
                    p.stop()

    def test_status_with_broken_isolation_opens_no_ros_session(self):
        class Panes:
            returncode = 0
            stdout = 'rsim-gazebo\tgazebo\t0\tgzserver\tdocker\n'

        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'session.json').write_text(json.dumps(sim.validate_session(session()).to_json()))
            queried = []
            patches = [mock.patch.object(sim, 'running_sim_containers', lambda: ['sim-gazebo']),
                       mock.patch.object(sim, 'docker_inspect', lambda kind, name: None),
                       mock.patch.object(sim, 'resolve_run_dir', lambda: run),
                       mock.patch.object(sim, 'check_isolation', lambda: ['network rover-sim is not internal']),
                       mock.patch.object(sim, 'tmux', lambda *a, **k: Panes()),
                       mock.patch.object(sim, 'status_values', lambda *a, **k: queried.append(a) or {}),
                       mock.patch.object(sim, 'dexec', lambda *a, **k: queried.append(a))]
            for p in patches:
                p.start()
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(sim.cmd_status(None), 3)
                self.assertEqual(queried, [])
                self.assertIn('isolation: BROKEN', out.getvalue())
                self.assertIn('sim time: not queried (isolation BROKEN)', out.getvalue())
            finally:
                for p in patches:
                    p.stop()

    def test_banner(self):
        with mock.patch.object(sys.stdout, 'isatty', lambda: True):
            b = sim.red_banner(['bad'])
        self.assertTrue(b.startswith('\033[1;31m'))
        self.assertTrue(b.endswith('\033[0m'))
        self.assertIn('!! bad', b)


class Json5AndZenoh(unittest.TestCase):
    def cfg(self, name):
        return sim.json5_loads((REPO / 'docker' / 'zenoh' / name).read_text())

    def test_json5_subset(self):
        text = """// it's a comment with "quotes"
        { a: 1, 'b': 'it\\'s', "c": [1, 2, ], /* block */ d: { e: -1, f: +2, g: 0x10, h: .5, i: 5. },
          j: true, k: null, l: "tcp/[::]:7447", }"""
        self.assertEqual(sim.json5_loads(text), {'a': 1, 'b': "it's", 'c': [1, 2],
                                                 'd': {'e': -1, 'f': 2, 'g': 16, 'h': 0.5, 'i': 5.0},
                                                 'j': True, 'k': None, 'l': 'tcp/[::]:7447'})

    def test_session_config(self):
        s = self.cfg('session.json5')
        self.assertEqual(s['mode'], 'peer')
        self.assertEqual(s['connect']['endpoints'], ['tcp/sim-drive:7447'])
        self.assertEqual(s['listen']['endpoints'], [])
        self.assertFalse(s['scouting']['multicast']['enabled'])
        self.assertTrue(s['scouting']['gossip']['enabled'])
        self.assertFalse(s['scouting']['gossip']['multihop'])
        self.assertNotIn('peer', s['scouting']['gossip']['autoconnect']['peer'])
        self.assertEqual(s['transport']['link']['tx']['queue']['congestion_control']['block']['wait_before_close'],
                         5000000)
        self.assertNotIn('qos', s)                 # no fleet zid rules
        self.assertNotIn('id', s)
        self.assertNotIn('7448', json.dumps(s))

    def test_router_config(self):
        r = self.cfg('router.json5')
        self.assertEqual(r['mode'], 'router')
        self.assertEqual(r['listen']['endpoints'], ['tcp/[::]:7447'])
        self.assertEqual(r['connect']['endpoints'], [])
        self.assertFalse(r['scouting']['multicast']['enabled'])
        self.assertEqual(r['scouting']['gossip']['autoconnect']['router'], [])
        self.assertTrue(r['routing']['router']['peers_failover_brokering'])
        self.assertNotIn('id', r)

    def test_variants(self):
        base = self.cfg('session.json5')
        self.assertIsNone(sim.session_variant(base, 'shipped'))
        c = sim.session_variant(base, 'client')
        self.assertEqual(c['mode'], 'client')
        self.assertEqual(c['connect']['timeout_ms']['client'], -1)
        self.assertEqual(c['connect']['endpoints'], ['tcp/sim-drive:7447'])
        lm = sim.session_variant(base, 'local-mesh')
        self.assertEqual(lm['listen']['endpoints'], ['tcp/localhost:0'])
        self.assertEqual(lm['scouting']['gossip']['autoconnect']['peer'], ['router', 'peer'])
        m = sim.session_variant(base, 'mesh')
        self.assertEqual(m['listen']['endpoints'], ['tcp/0.0.0.0:0'])
        for v in (c, lm, m):
            self.assertFalse(v['scouting']['multicast']['enabled'])
            self.assertEqual(v['connect']['endpoints'], ['tcp/sim-drive:7447'])
        self.assertEqual(base['listen']['endpoints'], [])      # base not mutated
        with self.assertRaises(sim.SimError):
            sim.session_variant(base, 'nope')


class ComposeFile(unittest.TestCase):
    def setUp(self):
        self.doc = yaml.safe_load((REPO / 'docker' / 'compose.stack.yaml').read_text())

    def test_isolated_network(self):
        net = self.doc['networks']['rover-sim']
        self.assertEqual(net['name'], 'rover-sim')
        self.assertIs(net['internal'], True)
        self.assertEqual(net['driver_opts']['com.docker.network.bridge.inhibit_ipv4'], 'true')
        self.assertEqual(self.doc['name'], sim.PROJECT)

    def test_services(self):
        services = self.doc['services']
        self.assertEqual(sorted(services), sorted(sim.CONTAINERS))
        for name, svc in services.items():
            self.assertEqual(svc['container_name'], name)
            self.assertEqual(svc['networks'], ['rover-sim'], name)
            self.assertEqual(svc['command'], ['sleep', 'infinity'], name)
            self.assertIs(svc['init'], True, name)
            self.assertEqual(svc['env_file'], '../config/sim.env', name)
            self.assertEqual(svc['pull_policy'], 'never', name)
            for bad in ('network_mode', 'ports', 'privileged', 'cap_add', 'extra_hosts', 'ipc', 'pid'):
                self.assertNotIn(bad, svc, name)
            self.assertEqual(svc['sysctls'], {'net.ipv6.conf.all.disable_ipv6': 1,
                                              'net.ipv6.conf.default.disable_ipv6': 1}, name)
            self.assertEqual(svc['ulimits']['nofile'], {'soft': 65536, 'hard': 65536}, name)
            vols = svc['volumes']
            for want in ('../docker/zenoh:/rover_sim/zenoh:ro', '../scripts:/rover_sim/scripts:ro',
                         '../config:/rover_sim/config:ro'):
                self.assertIn(want, vols, name)
            # No default run dir: a shared /tmp path is unsafe (sim.py check_run_dir).
            run = [v for v in vols if isinstance(v, str) and v.endswith(':/rover_sim/run')]
            self.assertEqual(len(run), 1, name)
            self.assertTrue(run[0].startswith('${ROVER_SIM_RUN:?'), name)
            self.assertEqual(svc['cpuset'], '${%s:-}' % sim.cpuset_var(name), name)
            for v in vols:
                self.assertFalse(sim.docker_socket_mount(sim._volume_source(v)), v)
        self.assertIn('/tmp/.X11-unix:/tmp/.X11-unix:rw', services['sim-gazebo']['volumes'])
        self.assertEqual(services['sim-elevation']['profiles'], ['gpu'])
        dev = services['sim-elevation']['deploy']['resources']['reservations']['devices'][0]
        self.assertEqual((dev['driver'], dev['capabilities']), ('nvidia', ['gpu']))

    def test_images_from_pin_file(self):
        want = {'sim-gazebo': 'GAZEBO_IMAGE', 'sim-drive': 'DRIVE_IMAGE', 'sim-dlio-jr': 'DLIO_JR_IMAGE',
                'sim-dlio-rr': 'DLIO_RR_IMAGE', 'sim-mapping': 'MAPPING_IMAGE',
                'sim-elevation': 'ELEVATION_IMAGE', 'sim-mighty': 'MIGHTY_IMAGE'}
        for name, var in want.items():
            self.assertTrue(self.doc['services'][name]['image'].startswith('${%s:?' % var), name)
        stacks_env = REPO / 'docker' / 'stacks.env'
        if stacks_env.exists():
            keys = {l.split('=', 1)[0] for l in stacks_env.read_text().splitlines() if '=' in l and l[0] != '#'}
            self.assertLessEqual(set(want.values()), keys)

    def test_sim_env(self):
        env = dict(l.split('=', 1) for l in (REPO / 'config' / 'sim.env').read_text().splitlines()
                   if l and not l.startswith('#'))
        self.assertEqual(env, {'RMW_IMPLEMENTATION': 'rmw_zenoh_cpp', 'ROS_DOMAIN_ID': '22',
                               'ZENOH_ROUTER_CONFIG_URI': '/rover_sim/zenoh/router.json5',
                               'ZENOH_SESSION_CONFIG_URI': '/rover_sim/zenoh/session.json5'})
        for uri in (env['ZENOH_ROUTER_CONFIG_URI'], env['ZENOH_SESSION_CONFIG_URI']):
            self.assertTrue((REPO / 'docker' / 'zenoh' / Path(uri).name).is_file())

    def test_local_override_filter(self):
        old = sim.LOCAL_COMPOSE_FILE
        try:
            with tempfile.TemporaryDirectory() as d:
                sim.LOCAL_COMPOSE_FILE = Path(d) / 'compose.local.yaml'
                self.assertIsNone(sim.local_override())
                sim.LOCAL_COMPOSE_FILE.write_text('services:\n  rover-sim:\n    deploy: !reset {}\n')
                self.assertIsNone(sim.local_override())
                sim.LOCAL_COMPOSE_FILE.write_text('services:\n  sim-gazebo:\n    runtime: nvidia\n'
                                                  '    cpuset: !override "0-3"\n'
                                                  '    volumes: ["../src:/src:ro", "/tmp/.X11-unix:/tmp/.X11-unix"]\n')
                self.assertEqual(sim.local_override(), sim.LOCAL_COMPOSE_FILE)
                for bad, why in (('    pid: host\n', 'sets pid'),
                                 ('    network_mode: host\n', 'sets network_mode'),
                                 ('    networks: !reset [default]\n', 'sets networks'),
                                 ('    privileged: true\n', 'sets privileged'),
                                 ('    cap_add: [NET_ADMIN]\n', 'sets cap_add'),
                                 ('    ports: ["7447:7447"]\n', 'sets ports'),
                                 ('    devices: [/dev/ttyACM0]\n', 'sets devices'),
                                 ('    sysctls: {net.ipv6.conf.all.disable_ipv6: 0}\n', 'sets sysctls'),
                                 ('    volumes: [/var/run/docker.sock:/var/run/docker.sock]\n', 'docker socket'),
                                 ('    volumes: [{type: bind, source: /run, target: /hostrun}]\n', 'docker socket'),
                                 ('    volumes: ["/:/host:ro"]\n', 'docker socket')):
                    sim.LOCAL_COMPOSE_FILE.write_text('services:\n  sim-drive:\n' + bad)
                    with self.assertRaisesRegex(sim.SimError, why):
                        sim.local_override()
                sim.LOCAL_COMPOSE_FILE.write_text('networks:\n  rover-sim: {internal: false}\n')
                with self.assertRaisesRegex(sim.SimError, 'top-level networks'):
                    sim.local_override()
        finally:
            sim.LOCAL_COMPOSE_FILE = old

    def test_cpu_budget(self):
        b = sim.cpu_budget(20, sim.CONTAINERS)
        self.assertEqual(b['sim-gazebo'], '2-5')                    # gzserver's cores, nobody else's
        self.assertEqual({v for k, v in b.items() if k != 'sim-gazebo'}, {'6-19'})
        self.assertEqual(sorted(b), sorted(sim.CONTAINERS))
        self.assertEqual(sim.cpu_budget(8, sim.CONTAINERS), {})       # too small to split: unpinned
        self.assertEqual(sim.cpuset_var('sim-dlio-jr'), 'ROVER_SIM_CPUSET_DLIO_JR')


def enc4(ip, port):
    return '%s:%04X' % (ipaddress.IPv4Address(ip).packed[::-1].hex().upper(), port)


def enc6(ip, port):
    raw = ipaddress.IPv6Address(ip).packed
    return '%s:%04X' % (''.join(raw[i:i + 4][::-1].hex() for i in range(0, 16, 4)).upper(), port)


def table(rows, v6=False):
    head = '  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n'
    enc = enc6 if v6 else enc4
    return head + ''.join('%4d: %s %s %s 00000000:00000000 00:00000000 00000000     0        0 1 1\n'
                          % (i, enc(a, p), enc(b, q), st) for i, (a, p, b, q, st) in enumerate(rows))


class TcpTables(unittest.TestCase):
    def test_parse(self):
        socks = sim.parse_proc_net_tcp(table([('127.0.0.1', 8080, '0.0.0.0', 0, '0A'),
                                              ('10.20.0.5', 40000, '10.20.0.3', 7447, '01')]))
        self.assertEqual(socks, [('127.0.0.1', 8080, '0.0.0.0', 0, 'LISTEN'),
                                 ('10.20.0.5', 40000, '10.20.0.3', 7447, 'ESTABLISHED')])
        socks6 = sim.parse_proc_net_tcp(table([('::ffff:10.20.0.3', 7447, '::ffff:10.20.0.5', 40000, '01'),
                                               ('::', 7447, '::', 0, '0A')], v6=True))
        self.assertEqual(socks6, [('10.20.0.3', 7447, '10.20.0.5', 40000, 'ESTABLISHED'),
                                  ('::', 7447, '::', 0, 'LISTEN')])

    def test_parse_live_host_table(self):
        for f in ('/proc/net/tcp', '/proc/net/tcp6'):
            if os.path.exists(f):
                sim.parse_proc_net_tcp(Path(f).read_text())

    def test_analyze(self):
        r_ip = '10.20.0.3'
        tables = {
            'sim-drive': sim.parse_proc_net_tcp(table([('::', 7447, '::', 0, '0A')], v6=True) + table([])
                                                + table([('::ffff:10.20.0.3', 7447, '::ffff:10.20.0.5', 40000, '01'),
                                                         ('::ffff:10.20.0.3', 7447, '::ffff:10.20.0.3', 41000, '01'),
                                                         ('::ffff:10.20.0.3', 41000, '::ffff:10.20.0.3', 7447, '01')],
                                                        v6=True)),
            'sim-mighty': sim.parse_proc_net_tcp(table([('10.20.0.5', 40000, '10.20.0.3', 7447, '01'),
                                                        ('127.0.0.1', 5000, '127.0.0.1', 5001, '01'),
                                                        ('127.0.0.1', 5001, '127.0.0.1', 5000, '01'),
                                                        ('10.20.0.5', 40002, '10.20.0.6', 7447, '02')])),
            'sim-mapping': sim.parse_proc_net_tcp(table([('127.0.0.1', 5000, '127.0.0.1', 5001, '01'),
                                                         ('127.0.0.11', 37553, '0.0.0.0', 0, '0A')])),
        }
        a = sim.analyze_links(tables, r_ip)
        # mighty->router, drive-internal session->router, and two loopback links in
        # different netns that share addresses.
        self.assertEqual((a['total'], a['router'], a['p2p']), (4, 2, 2))
        # The docker DNS listener in sim-mapping is not counted.
        self.assertEqual((a['listen'], a['syn_sent']), (1, 1))
        self.assertEqual(a['per_container']['sim-mighty']['established'], 3)


class Processes(unittest.TestCase):
    def test_keep_container_command_tree(self):
        P = sim.Proc
        procs = [P(1, 0, 1, 100, '/sbin/docker-init -- /ros_entrypoint.sh sleep infinity'),
                 P(7, 1, 7, 101, 'sleep infinity'),
                 P(50, 0, 50, 500, 'bash -c source ... && ros2 run rmw_zenoh_cpp rmw_zenohd'),
                 P(60, 50, 50, 501, '/opt/ros/humble/lib/rmw_zenoh_cpp/rmw_zenohd'),
                 P(70, 1, 65, 600, 'python3 orphan_from_a_pane.py'),
                 P(80, 0, 0, 700, 'python3 /rover_sim/run/loadtest/x/lt_agent.py launch p sim-drive')]
        targets, groups = sim.stop_targets(procs)
        self.assertEqual([p.pid for p in targets], [50, 60, 70, 80])
        self.assertEqual(groups, [50, 65])               # never group 0 or the init's
        targets, groups = sim.stop_targets(procs, match='lt_agent.py')
        self.assertEqual(([p.pid for p in targets], groups), ([80], []))

    def test_without_init_fails_closed(self):
        P = sim.Proc
        procs = [P(1, 0, 1, 100, 'sleep infinity'), P(9, 0, 9, 200, 'bash'), P(10, 1, 9, 201, 'orphan')]
        self.assertEqual(sim.stop_targets(procs), ([], []))
        self.assertEqual(sim.stop_targets([]), ([], []))

    def test_host_pid_namespace_fails_closed(self):
        # What a container sharing the host's PID namespace would list: never a target.
        P = sim.Proc
        procs = [P(1, 0, 1, 1, '/sbin/init splash'), P(900, 1, 900, 50, '/usr/bin/dockerd -H fd://'),
                 P(4000, 1, 4000, 90, 'zenohd --rest-http-port 8001'),
                 P(4100, 1, 4100, 95, 'python3 -m acl_c2_master'),
                 P(5000, 1, 5000, 99, '/sbin/docker-init -- sleep infinity')]
        self.assertEqual(sim.stop_targets(procs), ([], []))
        self.assertEqual(sim.stop_targets(procs, match='zenohd'), ([], []))

    def test_stop_processes_needs_private_pid_namespace(self):
        old_inspect, old_procs = sim.docker_inspect, sim.container_procs
        seen = []
        sim.container_procs = lambda name: seen.append(name) or []
        try:
            for mode, ok in (('', True), ('host', False), ('container:acl-c2-master', False), (None, False)):
                sim.docker_inspect = lambda kind, name: {'HostConfig': {} if mode is None else {'PidMode': mode}}
                seen.clear()
                self.assertEqual(sim.stop_processes('sim-drive'), 0)
                self.assertEqual(seen, ['sim-drive'] if ok else [], mode)
        finally:
            sim.docker_inspect, sim.container_procs = old_inspect, old_procs

    def test_ps_probe_runs(self):
        out = subprocess.run([sys.executable, '-c', sim._PS_PY], stdout=subprocess.PIPE, check=True).stdout
        rows = [sim.Proc(*r) for r in json.loads(out)]
        self.assertTrue(rows)
        self.assertNotIn(os.getpid() + 10 ** 9, [r.pid for r in rows])
        compile(sim._KILL_PY, '_KILL_PY', 'exec')


class Isolation(unittest.TestCase):
    def test_probe_script(self):
        subprocess.run(['bash', '-n', '-c', sim._PROBE_SH], check=True)
        # Locally a closed port on loopback is refused: that is a "host is reachable" hit.
        out = subprocess.run(['bash', '-c', sim._PROBE_SH, 'probe', '127.0.0.1/1'], stdout=subprocess.PIPE,
                             universal_newlines=True).stdout.splitlines()
        hits = [sim.classify_probe(l) for l in out if l.startswith('PROBE')]
        self.assertEqual(len(hits), 1)
        self.assertIn('connection refused', hits[0])

    def test_probe_writes_nothing(self):
        # If isolation were broken the probe's target is the fleet C2 router: connect, close,
        # and send no byte.
        srv = socket.socket()
        srv.bind(('127.0.0.1', 0))
        srv.listen(1)
        got = []

        def serve():
            conn, _ = srv.accept()
            conn.settimeout(3)
            try:
                got.append(conn.recv(16))
            finally:
                conn.close()

        t = threading.Thread(target=serve)
        t.start()
        out = subprocess.run(['bash', '-c', sim._PROBE_SH, 'probe', '127.0.0.1/%d' % srv.getsockname()[1]],
                             stdout=subprocess.PIPE, universal_newlines=True).stdout
        t.join(5)
        srv.close()
        self.assertEqual(got, [b''])                           # orderly close, no data
        self.assertIn('is open', sim.classify_probe(out.strip()))
        subprocess.run(['bash', '-n', '-c', sim.tcp_wait()], check=True)

    def test_default_routes(self):
        head = 'Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n'
        onlink = 'eth0\t000014AC\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0\n'
        default = 'eth0\t00000000\t010014AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n'
        self.assertEqual(sim.default_routes(head + onlink), [])
        self.assertEqual(sim.default_routes(head + onlink + default), ['eth0 via 172.20.0.1'])
        if os.path.exists('/proc/net/route'):
            sim.default_routes(Path('/proc/net/route').read_text())

    def test_namespace_problems(self):
        self.assertEqual(sim.namespace_problems('sim-x', {'PidMode': '', 'IpcMode': 'private', 'UsernsMode': ''}), [])
        self.assertEqual(sim.namespace_problems('sim-x', {'IpcMode': 'shareable'}), [])
        bad = sim.namespace_problems('sim-x', {'PidMode': 'host', 'IpcMode': 'host', 'UsernsMode': 'host'})
        self.assertEqual(len(bad), 3)
        self.assertTrue(sim.namespace_problems('sim-x', {'IpcMode': 'container:acl-c2-master'}))

    def test_docker_socket_mount(self):
        for src in ('/var/run/docker.sock', '/run/docker.sock', '/run/docker', '/var/run/docker/',
                    '/run/containerd/containerd.sock', '/home/u/.docker/run/docker.sock', '/', '/run', '/var/run'):
            self.assertTrue(sim.docker_socket_mount(src), src)
        for src in ('/tmp/.X11-unix', '/home/user/rover_sim/scripts', '../config', '/tmp/rover_sim-1/run'):
            self.assertFalse(sim.docker_socket_mount(src), src)

    def test_ipv6_addresses(self):
        text = ('00000000000000000000000000000001 01 80 10 80       lo\n'
                'fe800000000000000042acfffe120002 1b 40 20 80     eth0\n')
        self.assertEqual(sim.ipv6_addresses(text), ['fe80::42:acff:fe12:2%eth0'])
        self.assertEqual(sim.ipv6_addresses(''), [])
        if os.path.exists('/proc/net/if_inet6'):
            sim.ipv6_addresses(Path('/proc/net/if_inet6').read_text())

    def test_classify(self):
        self.assertIsNone(sim.classify_probe('PROBE 172.30.0.1/7447 124 '))
        self.assertIsNone(sim.classify_probe('PROBE 10.0.0.2/7447 1 bash: connect: Network is unreachable'))
        self.assertIn('open', sim.classify_probe('PROBE 10.0.0.2/7447 0 '))
        self.assertIn('refused', sim.classify_probe('PROBE 10.0.0.2/7447 1 bash: connect: Connection refused'))
        self.assertIsNone(sim.classify_probe('garbage'))


class RunDir(unittest.TestCase):
    def test_default_is_per_user(self):
        self.assertEqual(sim.RUN_DIR_DEFAULT, '/tmp/rover_sim-%d/run' % os.getuid())

    def test_safe_and_unsafe(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d).resolve()
            os.chmod(str(base), 0o700)
            run = base / 'a' / 'run'
            sim.prepare_run_dir(run)                          # creates a/ and run/, mode 700
            for p in (base / 'a', run):
                self.assertEqual(stat.S_IMODE(os.lstat(str(p)).st_mode), 0o700)
            self.assertFalse(sim.check_run_dir(run))
            os.chmod(str(run), 0o777)
            with self.assertRaisesRegex(sim.SimError, 'writable by group or others'):
                sim.check_run_dir(run)
            os.chmod(str(run), 0o700)
            os.chmod(str(base / 'a'), 0o775)                   # group-writable parent
            with self.assertRaisesRegex(sim.SimError, 'writable by group or others'):
                sim.check_run_dir(run)
            os.chmod(str(base / 'a'), 0o700)
            link = base / 'link'
            link.symlink_to(base / 'a')
            with self.assertRaisesRegex(sim.SimError, 'symlink'):
                sim.check_run_dir(link / 'run')
            with self.assertRaisesRegex(sim.SimError, 'symlink'):
                sim.prepare_run_dir(link / 'new')
            self.assertFalse((base / 'a' / 'new').exists())
            (base / 'file').write_text('')
            with self.assertRaisesRegex(sim.SimError, 'not a directory'):
                sim.check_run_dir(base / 'file')

    def test_sticky_tmp_parent(self):
        # /tmp itself (root, sticky, world-writable) is fine as a parent; its entry must be ours.
        if not os.path.isdir('/tmp') or not os.stat('/tmp').st_mode & stat.S_ISVTX:
            self.skipTest('no sticky /tmp')
        d = Path(tempfile.mkdtemp(dir='/tmp'))
        try:
            self.assertFalse(sim.check_run_dir(d))
        finally:
            os.rmdir(str(d))

    def test_not_ours(self):
        if os.getuid() == 0:
            self.skipTest('root owns everything')
        with self.assertRaisesRegex(sim.SimError, 'owned by uid 0'):
            sim.check_run_dir(Path('/usr'))


class Provenance(unittest.TestCase):
    def test_image_provenance(self):
        labels = {
            'pin': {'rover_sim.source': 'pin', 'rover_sim.dirty': 'false'},
            'self-clean': {'rover_sim.source': 'rover_sim', 'rover_sim.dirty': 'false'},
            'self-dirty': {'rover_sim.source': 'rover_sim', 'rover_sim.dirty': 'true',
                           'org.opencontainers.image.revision': '19ada3213b55'},
            'local': {'rover_sim.source': 'local:/home/u/mighty', 'rover_sim.dirty': 'false'},
            'reused': {'org.opencontainers.image.version': '22.04'},
        }
        old = sim.docker_inspect
        sim.docker_inspect = lambda kind, name: {'Config': {'Labels': labels[name]}} if name in labels else None
        try:
            out = sim.image_provenance(OrderedDict((k, k) for k in list(labels) + ['missing']))
        finally:
            sim.docker_inspect = old
        self.assertEqual(len(out), 3)
        self.assertIn('uncommitted', out[0])
        self.assertIn('LOCAL build', out[1])
        self.assertIn('did not build', out[2])


class LoadTestPlan(unittest.TestCase):
    ALL = list(sim.CONTAINERS)

    def plan(self, n=4, available=None, profile='robot'):
        return sim.build_loadtest_plan(sim.loadtest_agents(n), self.ALL if available is None else available, profile)

    def test_counts_and_containers(self):
        plan = self.plan()
        procs = plan['procs']
        self.assertEqual(len(procs), 4 * sum(sim.LT_PROFILES['robot'].values()))
        self.assertEqual(len({p['id'] for p in procs}), len(procs))
        self.assertEqual(len({(p['ns'], p['node']) for p in procs}), len(procs))
        dlio = {p['agent']: p['container'] for p in procs if p['stack'] == 'dlio'}
        self.assertEqual(dlio, {'JR81': 'sim-dlio-jr', 'RR82': 'sim-dlio-rr', 'JR83': 'sim-dlio-jr',
                                'RR84': 'sim-dlio-rr'})
        for p in procs:
            self.assertIn(p['container'], sim.CONTAINERS)
            for t, _ in p['pubs']:
                self.assertIn(t, plan['topic_rates'])

    def test_expectations(self):
        procs = {p['id']: p for p in self.plan()['procs']}
        tf_pubs = sorted(i for i, p in procs.items() if any(t == sim.LT_TF for t, _ in p['pubs']))
        self.assertEqual(len(tf_pubs), 4 * len(sim.LT_TF_PUBLISHERS))
        g0 = procs['JR81.gazebo0']
        self.assertEqual(sorted(g0['subs'][sim.LT_TF]), [i for i in tf_pubs if i != 'JR81.gazebo0'])
        self.assertEqual(g0['subs']['/JR81/lt/drive0/t0'], ['JR81.drive0'])
        m0 = procs['RR82.mighty0']
        self.assertEqual(sorted(m0['subs'][sim.LT_TRAJS]), ['JR81.mighty0', 'JR83.mighty0', 'RR84.mighty0'])
        for up in ('dlio', 'mapping', 'elevation'):
            self.assertEqual(m0['subs']['/RR82/lt/%s0/t1' % up], ['RR82.%s0' % up])
        m1 = procs['RR82.mighty1']
        self.assertNotIn(sim.LT_TF, m1['subs'])
        self.assertNotIn(sim.LT_TRAJS, m1['subs'])
        self.assertIn('/RR82/lt/dlio0/t0', m1['subs'])
        # Subscriptions only ever name existing publishers.
        for p in procs.values():
            for t, ids in p['subs'].items():
                for i in ids:
                    self.assertIn(t, [x for x, _ in procs[i]['pubs']])

    def test_without_elevation(self):
        avail = [c for c in self.ALL if c != 'sim-elevation']
        procs = {p['id']: p for p in self.plan(available=avail)['procs']}
        self.assertFalse([i for i in procs if '.elevation' in i])
        self.assertFalse([t for t in procs['JR81.mighty0']['subs'] if 'elevation' in t])

    def test_single_agent_light(self):
        procs = self.plan(1, profile='light')['procs']
        self.assertEqual(len(procs), 6)
        m0 = next(p for p in procs if p['stack'] == 'mighty')
        self.assertEqual(m0['subs'][sim.LT_TRAJS], [])

    def test_summary(self):
        plan = self.plan(1, profile='light')
        results = {}
        for k, p in enumerate(plan['procs']):
            results[p['id']] = {'id': p['id'], 't_start': 100.0 + k * 0.1, 't_init': 100.5 + k * 0.1,
                                't_ready': 102.0 + k * 0.1, 't_end': 112.0 + k * 0.1,
                                'counts_ready': {t: 0 for t in p['subs']},
                                'counts_end': {t: 10 * len(ids) * plan['topic_rates'][t] for t, ids in p['subs'].items()}}
        s = sim.summarize_loadtest(plan, results, 10.0)
        self.assertTrue(s['pass'])
        self.assertAlmostEqual(s['discovery_s'], 2.5)
        self.assertAlmostEqual(s['delivery'], 1.0)
        self.assertEqual(s['ready'], 6)
        del results[plan['procs'][0]['id']]['t_ready']
        s = sim.summarize_loadtest(plan, results, 10.0)
        self.assertFalse(s['pass'])
        self.assertIsNone(s['discovery_s'])
        self.assertEqual(s['not_ready'], [plan['procs'][0]['id']])

    def test_agent_script_compiles(self):
        compile(sim.LT_AGENT_PY, 'lt_agent.py', 'exec')


class Cli(unittest.TestCase):
    def dry_run(self, name):
        return subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up', name, '--dry-run'],
                              stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout

    def test_dry_run(self):
        out = self.dry_run('smoke')
        for stack in ('zenoh', 'gazebo', 'drive', 'frame_align'):
            self.assertIn('stack %s -> tmux session rsim-%s' % (stack, stack), out)
        self.assertIn('rmw_zenohd', out)
        self.assertIn('world.launch.py world:=arena gui:=false paused:=true', out)
        self.assertIn('JR81  jr  id 81  pose (-2.00, 0.00, 0.0 deg)  calibrated from JR02, lidar at 0.36 m', out)
        self.assertIn('RR83  rr  id 83  pose (2.00, 0.00, 180.0 deg)  calibrated from RR08, lidar at 0.40 m', out)
        self.assertIn('before the panes: write_agent_files, link_calib, write_teleop_file, write_dlio_launch', out)
        self.assertIn('not started yet (later phases): mapping (Phase 4), mighty (Phase 5)', out)
        self.assertIn('stack dlio -> tmux session rsim-dlio', out)
        self.assertIn('not arming: auto_arm is off, so every safety_node stays DISARMED (scripts/sim.py arm)', out)
        self.assertIn('after the panes: arm JR81, JR82, RR83, RR84: /<ns>/auto_mode once its DLIO registers scans',
                      self.dry_run('odom_eval'))

    def test_dry_run_four_agents(self):
        out = self.dry_run('multi_agent_swap')
        self.assertEqual(out.count('frame_align_publisher.py'), 12)
        for ego in ('JR81', 'JR82', 'RR83', 'RR84'):
            self.assertIn('  window align-%s\n' % ego, out)
            self.assertIn('livox_sim_bridge.py --ns %s' % ego, out)
        self.assertIn('-p ego_name:=RR84 -p other_name:=RR81', out)
        self.assertIn('world:=ACL_office_simple', self.dry_run('office_explore'))

    def test_bad_session_exit_code(self):
        p = subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up', 'nope', '--dry-run'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn('available: multi_agent_swap, odom_eval, office_explore, smoke', p.stderr)

    def test_cpus_command(self):
        # What every sim-gazebo pane runs first (pin_self), here under a 4-core affinity.
        if shutil.which('taskset') is None or len(os.sched_getaffinity(0)) < 4:
            self.skipTest('needs taskset and 4 cores')
        cores = sorted(os.sched_getaffinity(0))[:4]
        lst = ','.join(map(str, cores))

        def run(*a):
            return subprocess.run(['taskset', '-c', lst, sys.executable, str(REPO / 'scripts' / 'sim.py'), 'cpus']
                                  + list(a), stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.strip()
        self.assertEqual(run('gz'), ','.join(map(str, cores[:3])))
        self.assertEqual(run('rest'), str(cores[3]))
        self.assertEqual(run('rest', '--spare', '2'), ','.join(map(str, cores[2:])))


# ---------------------------------------------------------------------------- Phase 3

# Just enough of rclpy and the message packages for the embedded query scripts to run on the
# host: publishers are a world description (topic -> rate and value), service calls change it.
FAKE_ROS = {
    'rclpy/__init__.py': r'''
import json, os, time
WORLD = json.loads(os.environ['FAKE_WORLD'])
def init(args=None):
    pass
def try_shutdown():
    pass
def spin_once(node, timeout_sec=None):
    time.sleep(0.002)
    now = time.time()
    for sub in node.subs:
        spec = WORLD['pubs'].get(sub.topic)
        if spec and now - sub.last >= 1.0 / spec['hz']:
            sub.last = now
            sub.deliver(spec, now)
''',
    'rclpy/node.py': r'''
import sys
import rclpy
class Msg:
    pass
def stamp(t):
    s = Msg()
    s.sec, s.nanosec = int(t), int(round((t - int(t)) * 1e9))
    return s
class Sub:
    def __init__(self, topic, cb, raw):
        self.topic, self.cb, self.raw, self.last = topic, cb, raw, 0.0
    def deliver(self, spec, now):
        if self.raw:
            return self.cb(b'\0' * 64)
        m, m.header, m.pose = Msg(), Msg(), Msg()
        m.data = spec.get('value')
        m.header.stamp = stamp(now if spec.get('stamp', 'now') == 'now' else spec['stamp'])
        m.pose.pose = Msg()
        m.pose.pose.position = Msg()
        m.pose.pose.position.z = spec.get('z', 0.0)
        self.cb(m)
class Future:
    def __init__(self, r):
        self.r = r
    def done(self):
        return True
    def result(self):
        return self.r
class Client:
    def __init__(self, name):
        self.name = name
    def service_is_ready(self):
        return self.name in rclpy.WORLD.get('services', {})
    def call_async(self, req):
        spec = rclpy.WORLD['services'][self.name]
        rclpy.WORLD.setdefault('calls', []).append(self.name)
        if spec.get('sets'):
            rclpy.WORLD['pubs'][spec['mode_topic']]['value'] = spec['sets']
        r = Msg()
        r.success, r.message = spec['success'], spec['message']
        return Future(r)
class Node:
    def __init__(self, name):
        self.subs = []
    def create_subscription(self, msg_type, topic, cb, qos, raw=False):
        self.subs.append(Sub(topic, cb, raw))
    def create_client(self, srv_type, name):
        return Client(name)
    def destroy_node(self):
        print('CALLS %s' % ' '.join(rclpy.WORLD.get('calls', [])), file=sys.stderr)
''',
    'rclpy/qos.py': 'class ReliabilityPolicy:\n    BEST_EFFORT = 2\nclass QoSProfile:\n    def __init__(self, **kw):\n'
                    '        pass\n',
    'nav_msgs/__init__.py': '', 'nav_msgs/msg.py': 'class Odometry:\n    pass\n',
    'sensor_msgs/__init__.py': '', 'sensor_msgs/msg.py': 'class PointCloud2:\n    pass\n',
    'std_msgs/__init__.py': '', 'std_msgs/msg.py': 'class Float64:\n    pass\nclass String:\n    pass\n',
    'std_srvs/__init__.py': '', 'std_srvs/srv.py': 'class Trigger:\n    class Request:\n        pass\n',
}


def run_with_fake_ros(script, args, world, calls=None):
    """Runs an embedded rclpy script against FAKE_ROS; returns its stdout (and the services it
    called into `calls`)."""
    with tempfile.TemporaryDirectory() as d:
        for rel, text in FAKE_ROS.items():
            (Path(d) / rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(d) / rel).write_text(text)
        p = subprocess.run([sys.executable, '-c', script] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=60,
                           env=dict(os.environ, PYTHONPATH=d, FAKE_WORLD=json.dumps(world)))
    if p.returncode != 0:
        raise AssertionError(p.stderr)
    if calls is not None:
        calls.extend(next(l for l in p.stderr.splitlines() if l.startswith('CALLS')).split()[1:])
    return p.stdout


def safety_world(*agents, **over):
    """A world with each agent's safety_node (mode + both Trigger services) and its DLIO:
    agents are (ns, mode, odom stamp 'now' | 0 | None, scans?), DLIO's odom at the seed height."""
    w = {'pubs': {}, 'services': {}}
    for ns, mode, stamp, scans in agents:
        w['pubs']['/%s/mode' % ns] = {'hz': 20, 'value': mode}
        if stamp is not None:
            w['pubs']['/%s/dlio/odom_node/odom' % ns] = {'hz': 100, 'stamp': stamp, 'z': sim.lidar_height_of(ns)}
        if scans:
            w['pubs']['/%s/dlio/odom_node/pointcloud/deskewed' % ns] = {'hz': 10}
        for srv, sets in (('auto_mode', 'AUTO'), ('disarm', 'DISARMED')):
            w['services']['/%s/%s' % (ns, srv)] = {'success': True, 'sets': sets, 'mode_topic': '/%s/mode' % ns,
                                                    'message': 'rover switched to %s mode by %s service' % (sets, srv)}
    w.update(over)
    return w


# Synthetic teleop file with the robot file's layout (flat, fully qualified node keys, /** defaults,
# per-robot overrides); the values are made up.
SAMPLE_TELEOP = """# test teleop file
/**/joy_teleop/joy_node:
  ros__parameters:
    deadzone: 0.1

/**/joy_teleop/teleop_twist_joy_node:
  ros__parameters:
    scale_angular:
      yaw: 0.5

/**/safety_node:
  ros__parameters:
    lin_vel_max : 1.0 # comment
    ang_vel_max : 1.0
    alpha_tau_lin: .2

/JR02/joy_teleop/teleop_twist_joy_node:
  ros__parameters:
    scale_angular:
      yaw: 1.5
/JR02/safety_node:
  ros__parameters:
    ang_vel_max: 3.5

/JR09/safety_node:
  ros__parameters:
    ang_vel_max: 9.0

/JR02:
  ros__parameters:
    not_a_node_block: 1
"""
PINS = {'DLIO_JR_ROBOT': 'JR02', 'DLIO_RR_ROBOT': 'RR08', 'DLIO_SHA': 'a' * 40}


def rcl_params(doc, fqn):
    """What a node of that name gets from a params file: every matching key's parameters,
    flattened, in file order (parity_check.merge_param_files does the full rcl rule)."""
    def flat(d, prefix=''):
        out = {}
        for k, v in d.items():
            out.update(flat(v, prefix + k + '.') if isinstance(v, dict) else {prefix + k: v})
        return out
    out = {}
    for key, block in doc.items():
        if key == fqn or (key.startswith('/**/') and fqn.endswith('/' + key[4:])):
            out.update(flat(block['ros__parameters']))
    return out


class TeleopFile(unittest.TestCase):
    def gen(self, s=None, base=SAMPLE_TELEOP):
        return sim.teleop_sim_yaml(base, s or SWAP, PINS, 'rover-sim/drive:x:teleop_logitech.yaml')

    def test_flat_blocks_cloned_per_agent(self):
        text = self.gen()
        doc = yaml.safe_load(text)
        base = yaml.safe_load(SAMPLE_TELEOP)
        for k, v in base.items():
            self.assertEqual(doc[k], v, k)                                  # the base file is kept whole
        new = [k for k in doc if k not in base]
        self.assertEqual(new, ['/JR81/joy_teleop/teleop_twist_joy_node', '/JR81/safety_node',
                               '/JR82/joy_teleop/teleop_twist_joy_node', '/JR82/safety_node'])
        for k in new:
            self.assertEqual(list(doc[k]), ['ros__parameters'])            # flat: one node per key
            self.assertEqual(doc[k], base[k.replace(k[1:5], 'JR02', 1)])
        # JR agents get JR02's override, RR agents (RR08 has no block) the /** defaults.
        self.assertEqual(rcl_params(doc, '/JR81/safety_node')['ang_vel_max'], 3.5)
        self.assertEqual(rcl_params(doc, '/JR81/joy_teleop/teleop_twist_joy_node')['scale_angular.yaw'], 1.5)
        self.assertEqual(rcl_params(doc, '/RR83/safety_node')['ang_vel_max'], 1.0)
        self.assertEqual(rcl_params(doc, '/RR83/safety_node')['alpha_tau_lin'], 0.2)
        self.assertIn('# RR83: RR08 has no blocks of its own; the /** defaults apply.', text)
        self.assertTrue(text.startswith('# rover_sim: teleop_sim.yaml for session multi_agent_swap'))
        self.assertIn('\n' + SAMPLE_TELEOP.rstrip('\n') + '\n', text)            # verbatim, comments too
        self.assertNotIn('$(', text)                                         # launch reads it with allow_substs

    def test_refuses(self):
        real = sim.validate_session(session(agents=[{'name': 'JR09', 'class': 'jr', 'pose': [0, 0, 0]}]))
        with self.assertRaisesRegex(sim.SimError, 'a real robot'):
            self.gen(real)
        for bad, why in (('JR02:\n  safety_node:\n    ros__parameters: {a: 1}\n', 'not a flat'),
                         ('/a/b:\n  ros__parameters: {x: 1}\n/a/b:\n  ros__parameters: {x: 2}\n', 'duplicate key'),
                         ('/a/b:\n  ros__parameters: {x: 1}\n  extra: 1\n', 'not a flat'),
                         ('- 1\n', 'node blocks'), ('a: [\n', 'teleop'), ('', 'node blocks')):
            with self.assertRaisesRegex(sim.SimError, why, msg=bad):
                self.gen(base=bad)

    def test_pinned_file(self):
        # The real file, from the pinned rover checkout build_images.sh keeps (not tracked here).
        f = next(iter(sorted((REPO / '.cache' / 'ws').glob('rover@*/src/rover/rover_control/config/'
                                                           'teleop_logitech.yaml'))), None)
        if f is None:
            self.skipTest('no pinned rover checkout under .cache/ws')
        doc = yaml.safe_load(self.gen(base=f.read_text()))
        ref = yaml.safe_load(f.read_text())
        self.assertEqual(rcl_params(doc, '/JR81/safety_node')['ang_vel_max'],
                         rcl_params(ref, '/JR02/safety_node')['ang_vel_max'])
        self.assertEqual(rcl_params(doc, '/RR84/safety_node'), rcl_params(ref, '/RR08/safety_node'))
        self.assertFalse([k for k in doc if k.startswith(('/RR83/', '/RR84/'))])

    def test_write_teleop_file(self):
        calls = []

        def fake_dexec(name, args, **kw):
            calls.append((name, args[-1]))
            return FakeProc(0, base64.b64encode(SAMPLE_TELEOP.encode()).decode())

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(sim, 'dexec', fake_dexec), mock.patch.object(sim, 'running', lambda n: True), \
                    mock.patch.object(sim, 'read_pins', lambda: dict(PINS, DRIVE_IMAGE='img', DRIVE_SHA='9b0b4ba5e5')), \
                    contextlib.redirect_stdout(io.StringIO()):
                sim.write_teleop_file(SWAP, sim.RunContext(run_dir=Path(d)))
            self.assertEqual(calls, [('sim-drive', sim.TELEOP_SRC)])
            text = (Path(d) / 'teleop_sim.yaml').read_text()
            self.assertIn('img:%s (rover 9b0b4ba)' % sim.TELEOP_SRC, text)
            self.assertIn('/JR82/safety_node', yaml.safe_load(text))
            with mock.patch.object(sim, 'running', lambda n: False):
                with self.assertRaisesRegex(sim.SimError, 'sim-drive'):
                    sim.write_teleop_file(SWAP, sim.RunContext(run_dir=Path(d)))


class Phase3Panes(unittest.TestCase):
    ctx = sim.RunContext(run_dir=Path(sim.RUN_DIR_DEFAULT))

    def test_teleop_pane(self):
        p = sim.drive_panes(SWAP, self.ctx)[0]
        self.assertEqual((p.slug, p.title, p.container, p.window), ('teleop-safety-JR81', 'JR81 teleop + safety',
                                                                     'sim-drive', 'teleop'))
        self.assertEqual(p.cmd, 'ros2 launch rover_control teleop.launch.yaml robot_name:=JR81 '
                                'teleop_file:=/rover_sim/run/teleop_sim.yaml')
        self.assertEqual(p.env, sim.robot_env(SWAP.agents[0]))
        argv = shlex.split(sim.pane_command(p))
        self.assertIn('python3 /home/swarm/code/rover_ws/install/share/rover_bringup/scripts/wait_zenoh_ready.py',
                      argv[-1])
        subprocess.run(['bash', '-n', '-c', argv[-1]], check=True)

    def test_dlio_panes(self):
        panes = sim.dlio_panes(SWAP, self.ctx)
        self.assertEqual([(w, [p.slug for p in ps]) for w, ps in sim.tmux_windows(panes)],
                         [(a.name, ['dlio-' + a.name, 'seed-' + a.name, 'map-odom-tf-' + a.name]) for a in SWAP.agents])
        by = {p.slug: p for p in panes}
        for a in SWAP.agents:
            for kind in ('dlio', 'seed', 'map-odom-tf'):
                p = by['%s-%s' % (kind, a.name)]
                self.assertEqual(p.container, 'sim-dlio-' + a.cls)
                self.assertEqual(p.env, sim.robot_env(a))     # ROBOT_NAME: wait_zenoh_ready.py's probe topic
                self.assertTrue(p.gate)
                inner = inner_of(p)
                # The robot's gate: the router, then dlio_ws's own wait_zenoh_ready.py (mounted).
                self.assertIn('if [ -f /home/swarm/code/dlio_ws/scripts/wait_zenoh_ready.py ]; then python3 '
                              '/home/swarm/code/dlio_ws/scripts/wait_zenoh_ready.py;', inner)
                self.assertIn('source /home/swarm/code/dlio_ws/install/setup.bash', inner)
                subprocess.run(['bash', '-n', '-c', inner], check=True)
        self.assertEqual(by['dlio-RR83'].cmd, sim.seed_wait(SWAP.agents[2]) + ' && ros2 launch '
                         '/rover_sim/run/dlio_sim.launch.py namespace:=RR83 initial_pose_topic:=world two_d_only:=false')
        # DLIO starts only once its own seed publisher runs: the pattern finds that process's
        # command line, never a pane's bash -c (which contains it) or another agent's seed.
        pat = re.search(r'pgrep -f "([^"]+)"', by['dlio-RR83'].cmd).group(1)
        seed_proc = '/usr/bin/python3 /opt/ros/humble/bin/ros2 ' + by['seed-RR83'].cmd[len('ros2 '):]
        self.assertTrue(re.search(pat, seed_proc))
        for p in panes:
            self.assertFalse(re.search(pat, 'bash -c ' + sim.pane_inner(p)), p.slug)
        self.assertFalse(re.search(pat, seed_proc.replace('/RR83/', '/RR84/')))
        self.assertIn('&& sleep %d' % sim.SEED_LEAD_S, by['dlio-RR83'].cmd)
        self.assertEqual(by['seed-JR82'].cmd, 'ros2 topic pub -r 2 /JR82/world geometry_msgs/msg/PoseStamped '
                                              '"{header: {frame_id: world}, pose: {position: {z: 0.36}, '
                                              'orientation: {w: 1}}}"')
        self.assertIn('{z: 0.4}', by['seed-RR84'].cmd)                    # the class's lidar height
        self.assertEqual(by['map-odom-tf-RR84'].cmd, 'python3 /home/swarm/code/dlio_ws/scripts/map_odom_tf.py RR84 '
                                                     '--ros-args -r __node:=map_odom_tf_broadcaster_RR84')
        # The seed's YAML reaches ros2 topic pub as one argument with the class's z.
        seed = shlex.split(by['seed-RR83'].cmd)
        self.assertEqual(seed[:7], ['ros2', 'topic', 'pub', '-r', '2', '/RR83/world', 'geometry_msgs/msg/PoseStamped'])
        self.assertEqual(len(seed), 8)
        self.assertEqual(yaml.safe_load(seed[-1]), {'header': {'frame_id': 'world'},
                                                    'pose': {'position': {'z': 0.4}, 'orientation': {'w': 1}}})
        # Fixed robot node names become unique per agent, as allowed_deviations.yaml renames them.
        allowed = yaml.safe_load((REPO / 'tools' / 'reference' / 'jr02' / 'allowed_deviations.yaml').read_text())
        self.assertEqual(allowed['rename'], {'/dlio_map_node': '/dlio_map_node_{ns}',
                                             '/map_odom_tf_broadcaster': '/map_odom_tf_broadcaster_{ns}'})

    def test_dlio_wrapper_launch(self):
        """Runs generate_launch_description() against a recording stand-in for launch."""
        made = []

        def cls(name):
            class Action:
                def __init__(self, *a, **kw):
                    self.kind, self.a, self.kw = name, a, kw
                    made.append(self)
            Action.__name__ = name
            return Action

        mods = {}
        for mod, names in (('launch', ['LaunchDescription']), ('launch.actions', ['DeclareLaunchArgument',
                                                                               'IncludeLaunchDescription']),
                           ('launch.launch_description_sources', ['PythonLaunchDescriptionSource']),
                           ('launch.substitutions', ['LaunchConfiguration']), ('launch_ros', []),
                           ('launch_ros.actions', ['SetRemap'])):
            m = type(sys)(mod)
            for n in names:
                setattr(m, n, cls(n))
            mods[mod] = m
        ns = {}
        with mock.patch.dict(sys.modules, mods):
            exec(compile(sim.DLIO_SIM_LAUNCH, 'dlio_sim.launch.py', 'exec'), ns)
            ld = ns['generate_launch_description']()
        self.assertEqual(ld.kind, 'LaunchDescription')
        actions = ld.a[0]
        self.assertEqual([x.kind for x in actions], ['DeclareLaunchArgument'] * 3 + ['SetRemap',
                                                                                 'IncludeLaunchDescription'])
        self.assertEqual([x.a for x in actions[:3]], [('namespace',), ('initial_pose_topic',), ('two_d_only',)])
        remap = actions[3]
        self.assertEqual(remap.kw['src'], 'dlio_map_node:__node')       # rcl node-specific: only that node
        self.assertEqual(remap.kw['dst'][0], 'dlio_map_node_')
        self.assertEqual(remap.kw['dst'][1].a, ('namespace',))
        inc = actions[4]
        self.assertEqual(inc.a[0].a, ('/home/swarm/code/dlio_ws/scripts/dlio_with_session.launch.py',))
        self.assertEqual([(k, v.a) for k, v in inc.kw['launch_arguments']],
                         [(k, (k,)) for k in ('namespace', 'initial_pose_topic', 'two_d_only')])
        self.assertIn('ros2 launch /rover_sim/run/dlio_sim.launch.py namespace:=JR81', sim.DLIO_SIM_LAUNCH)

    def test_dlio_scripts(self):
        self.assertEqual(sim.dlio_scripts_dir(PINS), REPO / '.cache' / 'src' / ('dlio_ws@' + 'a' * 40) / 'scripts')
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(sim, 'REPO', Path(d)):
                p = sim.dlio_scripts_problem(PINS)
                self.assertIn('run scripts/build_images.sh dlio-jr', p)
                self.assertIn('wait_zenoh_ready.py', p)
                scripts = sim.dlio_scripts_dir(PINS)
                scripts.mkdir(parents=True)
                for f in sim.DLIO_SCRIPTS:
                    (scripts / f).write_text('')
                self.assertIsNone(sim.dlio_scripts_problem(PINS))
                (scripts / 'map_odom_tf.py').unlink()
                self.assertIn('map_odom_tf.py', sim.dlio_scripts_problem(PINS))
                with mock.patch.object(sim, 'read_pins', lambda: PINS):
                    with self.assertRaisesRegex(sim.SimError, 'build_images.sh dlio-jr'):
                        sim.write_dlio_launch(SWAP, sim.RunContext(run_dir=Path(d)))
                    (scripts / 'map_odom_tf.py').write_text('')
                    sim.write_dlio_launch(SWAP, sim.RunContext(run_dir=Path(d)))
            self.assertEqual((Path(d) / 'dlio_sim.launch.py').read_text(), sim.DLIO_SIM_LAUNCH)
        # The pinned checkout of this repo's own pin, when build_images.sh has fetched it.
        pinned = sim.dlio_scripts_dir(sim.read_pins())
        if pinned.is_dir():
            self.assertIsNone(sim.dlio_scripts_problem(sim.read_pins()))

    def test_compose_mounts_pinned_scripts(self):
        doc = yaml.safe_load((REPO / 'docker' / 'compose.stack.yaml').read_text())
        for name, svc in doc['services'].items():
            mounts = [v for v in svc['volumes'] if isinstance(v, dict)]
            if name not in ('sim-dlio-jr', 'sim-dlio-rr'):
                self.assertEqual(mounts, [], name)
                continue
            self.assertEqual(len(mounts), 1, name)
            m = mounts[0]
            self.assertEqual((m['type'], m['target'], m['read_only']), ('bind', '/home/swarm/code/dlio_ws/scripts', True))
            self.assertEqual(m['bind'], {'create_host_path': False})     # never made as root when missing
            self.assertTrue(m['source'].startswith('../.cache/src/dlio_ws@${DLIO_SHA:?'), m['source'])
            self.assertTrue(m['source'].endswith('}/scripts'))
            self.assertEqual(sim.dlio_scripts_dir({'DLIO_SHA': 'X'}), REPO / '.cache' / 'src' / 'dlio_ws@X' / 'scripts')
            self.assertFalse(sim.docker_socket_mount(sim._volume_source(m)))

    def test_session_auto_arm(self):
        # Off by default: without mpc (Phase 5) nothing publishes cmd_vel_auto, and safety_node in
        # AUTO logs every 20 ms. Only the odom_eval session, which drives through it, arms.
        self.assertFalse(sim.validate_session(session()).auto_arm)
        self.assertTrue(sim.validate_session(session(auto_arm=True)).auto_arm)
        self.assertIn("auto_arm: 'no' is not true/false", errors_of(session(auto_arm='no')))
        self.assertIs(sim.validate_session(session()).to_json()['auto_arm'], False)
        self.assertEqual([n for n in ('smoke', 'multi_agent_swap', 'office_explore', 'odom_eval')
                          if sim.load_session(n).auto_arm], ['odom_eval'])
        st = sim.STACKS
        s = sim.validate_session(session(auto_arm=True))
        everything = [st[n] for n in ('zenoh', 'gazebo', 'drive', 'dlio')]
        self.assertTrue(sim.will_arm(s, everything))
        self.assertFalse(sim.will_arm(s, everything, no_arm=True))
        self.assertFalse(sim.will_arm(sim.validate_session(session(auto_arm=False)), everything))
        self.assertFalse(sim.will_arm(s, [st['zenoh'], st['gazebo'], st['drive']]))       # no DLIO, no arming
        self.assertFalse(sim.will_arm(s, [st['zenoh'], st['dlio']]))


class Arming(unittest.TestCase):
    def run_arm(self, world, service='auto_mode', gate='dlio', names=('JR81',), timeout='1.5'):
        calls = []
        out = run_with_fake_ros(sim._ARM_PY, sim.arm_args(service, gate, float(timeout), names)[4:], world, calls)
        return sim.parse_arm(out), calls, out

    def test_arms_once_dlio_registers_scans(self):
        world = safety_world(('JR81', 'DISARMED', 'now', True), ('RR83', 'DISARMED', 0, False),
                             ('JR82', 'DISARMED', 'now', False))
        res, calls, out = self.run_arm(world, names=('JR81', 'RR83', 'JR82'))
        self.assertEqual(calls, ['/JR81/auto_mode'])                 # never before DLIO is past its calibration
        self.assertEqual(res['JR81'], sim.ArmResult('JR81', 'ok', 'AUTO', 'rover switched to AUTO mode by '
                                                                          'auto_mode service'))
        self.assertEqual((res['RR83'].status, res['RR83'].mode), ('timeout', 'DISARMED'))
        self.assertIn('unstamped: no IMU', res['RR83'].message)
        self.assertIn('no scans yet', res['JR82'].message)
        self.assertTrue(out.startswith('[arm] JR81 RR83 JR82: /<ns>/auto_mode once DLIO registers scans'))

    def test_waits_for_the_seed_height(self):
        # DLIO registers scans unseeded too; a late seed leaves its odom at z 0.
        world = safety_world(('JR81', 'DISARMED', 'now', True), ('RR83', 'DISARMED', 'now', True))
        world['pubs']['/RR83/dlio/odom_node/odom']['z'] = 0.002
        world['pubs']['/JR81/dlio/odom_node/odom']['z'] = 0.36 + 0.09          # within ARM_Z_TOL_M
        res, calls, _ = self.run_arm(world, names=('JR81', 'RR83'))
        self.assertEqual(calls, ['/JR81/auto_mode'])
        self.assertEqual((res['RR83'].status, res['RR83'].mode), ('timeout', 'DISARMED'))
        self.assertIn('DLIO odom z 0.00, expected 0.40: the seed pose came too late', res['RR83'].message)

    def test_idempotent_and_disarm(self):
        res, calls, _ = self.run_arm(safety_world(('JR81', 'AUTO', 'now', True)))
        self.assertEqual((res['JR81'].status, res['JR81'].mode, calls), ('ok', 'AUTO', []))
        self.assertIn('already in AUTO', res['JR81'].message)
        res, calls, _ = self.run_arm(safety_world(('JR81', 'AUTO', None, False), ('RR83', 'DISARMED', None, False)),
                                     'disarm', 'none', ('JR81', 'RR83'))
        self.assertEqual(calls, ['/JR81/disarm', '/RR83/disarm'])      # no DLIO wait, and always called
        self.assertEqual([(r.status, r.mode) for r in res.values()], [('ok', 'DISARMED')] * 2)
        res, calls, _ = self.run_arm(safety_world(('JR81', 'DISARMED', None, False)), gate='none')
        self.assertEqual((res['JR81'].mode, calls), ('AUTO', ['/JR81/auto_mode']))      # --now

    def test_waits_for_safety_node(self):
        world = safety_world(('JR81', 'DISARMED', 'now', True), services={})
        res, calls, _ = self.run_arm(world)
        self.assertEqual((res['JR81'].status, calls), ('timeout', []))
        self.assertIn('/JR81/auto_mode is not available yet', res['JR81'].message)
        world = safety_world(('JR81', 'DISARMED', 'now', True))
        del world['pubs']['/JR81/mode']
        res, calls, _ = self.run_arm(world)
        self.assertIn('no /JR81/mode yet', res['JR81'].message)
        world = safety_world(('JR81', 'DISARMED', 'now', True))
        world['services']['/JR81/auto_mode'].update(success=False, sets=None, message='backup mode is active')
        res, calls, _ = self.run_arm(world, timeout='4')
        self.assertEqual((res['JR81'].status, res['JR81'].mode), ('refused', 'DISARMED'))

    def test_parse_and_report(self):
        out = ('[arm] JR81: waiting: x\n@@ JR81 ok mode=AUTO rover switched to AUTO mode by auto_mode service\n'
               '@@ RR83 timeout mode=None DLIO odom unstamped: no IMU has reached DLIO yet\n@@ junk\n')
        res = sim.parse_arm(out)
        self.assertEqual(list(res), ['JR81', 'RR83'])
        self.assertEqual(res['RR83'].message, 'DLIO odom unstamped: no IMU has reached DLIO yet')
        err, so = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(so):
            self.assertFalse(sim.report_arm(res, 'auto_mode'))
            self.assertTrue(sim.report_arm(OrderedDict([('JR81', res['JR81'])]), 'auto_mode'))
            self.assertFalse(sim.report_arm(OrderedDict([('JR81', res['JR81'])]), 'disarm'))   # mode must follow
        self.assertIn('auto_mode RR83: timeout', err.getvalue())
        self.assertIn('auto_mode JR81: ok (safety_node mode AUTO)', so.getvalue())

    def test_arm_args(self):
        args = sim.arm_args('auto_mode', 'dlio', 60, ['JR81', 'RR83'])
        self.assertEqual(args[:2], ['bash', '-c'])
        self.assertEqual(args[3:], ['arm', 'auto_mode', 'dlio', '60', '1', '0.1', 'JR81=0.36', 'RR83=0.4'])
        self.assertIn("'@@ %s ok mode=AUTO " + sim.ALREADY_AUTO + "'", sim._ARM_PY)   # report_interrupted reads it
        self.assertTrue(args[2].endswith('exec timeout -k 2 66 python3 -c "$SIM_ARM_PY" "$@"'))
        subprocess.run(['bash', '-n', '-c', args[2]], check=True)
        compile(sim._ARM_PY, '<sim_arm>', 'exec')
        # Ctrl-C finds it by the marker in its command line (python3 -c <script>).
        self.assertTrue(sim._ARM_PY.startswith('# ' + sim.ARM_MARKER))
        P = sim.Proc
        procs = [P(1, 0, 1, 1, '/sbin/docker-init -- sleep infinity'), P(7, 1, 7, 2, 'sleep infinity'),
                 P(40, 0, 40, 9, 'timeout -k 2 66 python3 -c ' + sim._ARM_PY.replace('\n', ' ') + ' auto_mode'),
                 P(41, 40, 40, 9, 'python3 -c ' + sim._ARM_PY.replace('\n', ' ') + ' auto_mode dlio 60 1 JR81'),
                 P(50, 0, 50, 9, 'python3 /home/swarm/code/rover_ws/install/lib/acl_ctrl_safety/safety_node')]
        self.assertEqual([p.pid for p in sim.stop_targets(procs, match=sim.ARM_MARKER)[0]], [40, 41])

    def test_arm_agents(self):
        seen = []

        def lines(name, args, timeout, env, on_line):
            seen.append((name, args, timeout, dict(env)))
            for l in ('[arm] JR81 RR83: ...', '@@ JR81 ok mode=AUTO switched', 'Traceback (most recent call last):',
                      '@@ JR99 ok mode=AUTO not asked for'):
                on_line(l)
            return 0

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'arm.log'
            out = io.StringIO()
            with mock.patch.object(sim, 'dexec_lines', lines), contextlib.redirect_stdout(out):
                res = sim.arm_agents(['JR81', 'RR83'], 'auto_mode', 'dlio', 30, log)
            text = log.read_text()
        self.assertEqual(seen[0][0], 'sim-drive')
        self.assertEqual(seen[0][1], sim.arm_args('auto_mode', 'dlio', 30, ['JR81', 'RR83']))
        self.assertEqual(seen[0][3], {'SIM_ARM_PY': sim._ARM_PY, 'RUST_LOG': 'zenoh=error'})
        self.assertEqual(list(res), ['JR81', 'RR83'])                 # the requested agents, in order
        self.assertEqual(res['JR81'].status, 'ok')
        self.assertEqual((res['RR83'].status, res['RR83'].mode), ('lost', '?'))
        self.assertIn('@@ JR81 ok mode=AUTO switched', text)
        self.assertIn('RR83: lost', text)
        self.assertIn('Traceback', out.getvalue())                  # anything else is shown, not swallowed

    def interrupt(self, names, disarm_too=False):
        """Ctrl-C during an auto_mode query that had armed JR81 (RR83 was in AUTO before);
        returns (the ArmInterrupted, the queries run, stop_processes calls)."""
        stopped, queries = [], []

        def lines(name, args, timeout, env, on_line):
            service, agents = args[4], [a.split('=')[0] for a in args[9:]]
            queries.append((service, agents))
            if service == 'auto_mode':
                on_line('@@ RR83 ok mode=AUTO ' + sim.ALREADY_AUTO)
                on_line('@@ JR81 ok mode=AUTO rover switched to AUTO mode by auto_mode service')
                raise KeyboardInterrupt
            if disarm_too:
                raise KeyboardInterrupt
            for n in agents:
                on_line('@@ %s ok mode=DISARMED rover switched to DISARMED mode by disarm service' % n)
            return 0

        with mock.patch.object(sim, 'dexec_lines', lines), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(sim, 'stop_processes', lambda n, **kw: stopped.append((n, kw)) or 1):
            with self.assertRaises(sim.ArmInterrupted) as cm:
                sim.arm_agents(names, 'auto_mode', 'dlio', 30)
        return cm.exception, queries, stopped

    def test_interrupt_fails_closed(self):
        e, queries, stopped = self.interrupt(['JR81', 'JR82', 'RR83'])
        self.assertEqual(stopped, [('sim-drive', {'match': sim.ARM_MARKER, 'grace_int': 2, 'grace_term': 2})])
        # JR82 too: the query may have armed it after the client's last line. RR83 was in AUTO before.
        self.assertEqual(queries, [('auto_mode', ['JR81', 'JR82', 'RR83']), ('disarm', ['JR81', 'JR82'])])
        self.assertEqual((e.service, list(e.seen), list(e.undo)), ('auto_mode', ['RR83', 'JR81'], ['JR81', 'JR82']))
        self.assertIsInstance(e, KeyboardInterrupt)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            sim.report_interrupted(e)
        self.assertIn('arming interrupted: JR81, JR82 DISARMED; RR83 was already in AUTO and stays so', err.getvalue())

        e, queries, stopped = self.interrupt(['JR81'], disarm_too=True)
        self.assertEqual((len(stopped), e.undo), (2, None))
        with contextlib.redirect_stderr(err):
            sim.report_interrupted(e)
        self.assertIn('so was the disarm that undoes it: JR81 may be in AUTO', err.getvalue())

        e.undo = OrderedDict([('JR81', sim.ArmResult('JR81', 'timeout', 'AUTO', 'no answer'))])
        with contextlib.redirect_stderr(err):
            sim.report_interrupted(e)
        self.assertIn('JR81 did not confirm DISARMED and may be in AUTO: scripts/sim.py disarm JR81', err.getvalue())
        d = sim.ArmInterrupted('disarm', ['JR81', 'RR83'], sim.parse_arm('@@ JR81 ok mode=DISARMED x'), None)
        with contextlib.redirect_stderr(err):
            sim.report_interrupted(d)
        self.assertIn('disarm interrupted: DISARMED JR81; not confirmed: RR83', err.getvalue())

    def test_dexec_lines_streams_and_times_out(self):
        with tempfile.TemporaryDirectory() as d:
            docker = Path(d) / 'docker'
            docker.write_text('#!/bin/sh\necho "args: $*"\necho one\nsleep 0.2\necho two >&2\n'
                              '[ "$4" = slow ] && sleep 30\nexit 3\n')
            docker.chmod(0o755)
            got = []
            with mock.patch.dict(os.environ, {'PATH': d + os.pathsep + os.environ['PATH']}):
                rc = sim.dexec_lines('sim-drive', ['fast'], 10, (('A', 'b'),), got.append)
                self.assertEqual(rc, 3)
                self.assertEqual(got, ['args: exec -e A=b sim-drive fast', 'one', 'two'])
                t = time.time()
                rc = sim.dexec_lines('sim-drive', ['slow'], 1, (), lambda l: None)
                self.assertLess(time.time() - t, 10)
                self.assertNotEqual(rc, 0)
                with self.assertRaises(sim.SimError):
                    sim.dexec_lines('acl-c2-zenoh', ['x'], 1, (), got.append)

    def test_session_agents(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            with self.assertRaisesRegex(sim.SimError, 'no running session'):
                sim.session_agents(run)
            (run / 'session.json').write_text(json.dumps(SWAP.to_json()))
            self.assertEqual(sim.session_agents(run), ['JR81', 'JR82', 'RR83', 'RR84'])
            self.assertEqual(sim.session_agents(run, ['RR84', 'JR81']), ['JR81', 'RR84'])
            with self.assertRaisesRegex(sim.SimError, 'JR02: not an agent of session multi_agent_swap'):
                sim.session_agents(run, ['JR02'])

    def cli(self, argv, isolation=(), results=None, project=('sim-drive', 'sim-gazebo'), raises=None):
        calls = []

        def arm_agents(*a, **k):
            calls.append(a)
            if raises is not None:
                raise raises
            return results
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'session.json').write_text(json.dumps(SWAP.to_json()))
            patches = [mock.patch.object(sim, 'running_sim_containers', lambda: list(project)),
                       mock.patch.object(sim, 'resolve_run_dir', lambda: run),
                       mock.patch.object(sim, 'check_isolation', lambda: list(isolation)),
                       mock.patch.object(sim, 'tmux_sessions', lambda: ['rsim-drive']),
                       mock.patch.object(sim, 'LOG_ROOT', run / 'logs'),
                       mock.patch.object(sim, 'arm_agents', arm_agents),
                       mock.patch.object(sim, 'dexec', lambda *a, **k: calls.append(('dexec',) + a))]
            for p in patches:
                p.start()
            out, err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = sim.main(argv)
            finally:
                for p in patches:
                    p.stop()
        return rc, calls, out.getvalue() + err.getvalue()

    def test_cli(self):
        ok = OrderedDict((n, sim.ArmResult(n, 'ok', 'AUTO', 'x')) for n in ('JR81', 'JR82', 'RR83', 'RR84'))
        rc, calls, _ = self.cli(['arm'], results=ok)
        self.assertEqual((rc, calls), (0, [(['JR81', 'JR82', 'RR83', 'RR84'], 'auto_mode', 'dlio', 60.0, None)]))
        rc, calls, _ = self.cli(['arm', 'RR83', '--now', '--timeout', '5'], results=OrderedDict([('RR83', ok['RR83'])]))
        self.assertEqual((rc, calls), (0, [(['RR83'], 'auto_mode', 'none', 5.0, None)]))
        dis = OrderedDict([('JR81', sim.ArmResult('JR81', 'ok', 'DISARMED', 'x'))])
        rc, calls, _ = self.cli(['disarm', 'JR81'], results=dis)
        self.assertEqual((rc, calls), (0, [(['JR81'], 'disarm', 'none', 60.0, None)]))
        rc, calls, _ = self.cli(['arm', 'JR81'], results=OrderedDict([('JR81', sim.ArmResult('JR81', 'timeout',
                                                                                             'DISARMED', 'x'))]))
        self.assertEqual(rc, 1)
        rc, calls, text = self.cli(['arm', 'JR02'])
        self.assertEqual((rc, calls), (1, []))
        self.assertIn('not an agent', text)
        # A sim-drive outside the compose project is not one check_isolation inspected.
        rc, calls, text = self.cli(['arm'], results=ok, project=('sim-gazebo',))
        self.assertEqual((rc, calls), (1, []))
        self.assertIn('sim-drive of project rover-sim-stack is not running', text)
        # Ctrl-C: the report of what it left armed, exit 130.
        undone = sim.ArmInterrupted('auto_mode', ['JR81'], OrderedDict(), OrderedDict(
            [('JR81', sim.ArmResult('JR81', 'ok', 'DISARMED', 'x'))]))
        rc, calls, text = self.cli(['arm', 'JR81'], raises=undone)
        self.assertEqual(rc, 130)
        self.assertIn('arming interrupted: JR81 DISARMED', text)

    def test_cli_fails_closed(self):
        for verb in ('arm', 'disarm'):
            rc, calls, text = self.cli([verb], isolation=['sim-drive reaches 10.0.0.1/7447 is open'])
            self.assertEqual((rc, calls), (3, []))                 # no ROS session at all
            self.assertIn('isolation: BROKEN', text)

    def test_up_arms_after_the_panes(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            ok = OrderedDict((a.name, sim.ArmResult(a.name, 'ok', 'AUTO', 'x')) for a in SWAP.agents)
            with mock.patch.object(sim, 'check_isolation', lambda: []), \
                    mock.patch.object(sim, 'arm_agents', lambda *a, **k: calls.append(a) or ok), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sim.arm_after_up(SWAP, Path(d)), 0)
            self.assertEqual(calls, [(['JR81', 'JR82', 'RR83', 'RR84'], 'auto_mode', 'dlio', sim.ARM_TIMEOUT_UP_S,
                                      Path(d) / 'arm.log')])
            with mock.patch.object(sim, 'check_isolation', lambda: ['bad']), \
                    mock.patch.object(sim, 'arm_agents', lambda *a, **k: calls.append(a)):
                with self.assertRaisesRegex(sim.SimError, 'NOT ARMING'):
                    sim.arm_after_up(SWAP, Path(d))
            self.assertEqual(len(calls), 1)
            undone = sim.ArmInterrupted('auto_mode', ['JR81'], OrderedDict(), OrderedDict(
                [('JR81', sim.ArmResult('JR81', 'ok', 'DISARMED', 'x'))]))
            for exc, says in ((undone, 'arming interrupted: JR81 DISARMED'), (KeyboardInterrupt, 'modes unknown')):
                with mock.patch.object(sim, 'check_isolation', lambda: []), \
                        mock.patch.object(sim, 'arm_agents', lambda *a, **k: (_ for _ in ()).throw(exc)), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(sim.arm_after_up(SWAP, Path(d)), 130)
                self.assertIn(says, err.getvalue())


class WaitSpawned(unittest.TestCase):
    """up waits for the spawn pane's unpause line, and fails when the pane does."""

    def run_wait(self, text, exited, timeout=5.0):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'gazebo' / 'spawn.log'
            if text is not None:
                log.parent.mkdir()
                log.write_text(text)
            with mock.patch.object(sim, 'spawn_pane_exited', lambda: exited), \
                    mock.patch.object(sim, 'SPAWN_PANE_GRACE_S', 0), mock.patch.object(sim.time, 'sleep', lambda s: None):
                return sim.wait_spawned(Path(d), timeout)

    def test_spawn_pane_found_by_slug(self):
        seen = []

        def tmux(*a, **k):
            seen.append(a)
            return types.SimpleNamespace(returncode=0, stdout=out)

        # The prompt retitled the spawn pane; its slug option still says which it is.
        for out, exited in (('\tdocker\nspawn\tdocker\nsim-tf\tdocker\n', False), ('spawn\tbash\n', True),
                            ('gzserver\tdocker\n', True)):
            with mock.patch.object(sim, 'tmux', tmux):
                self.assertIs(sim.spawn_pane_exited(), exited, out)
        self.assertEqual(seen[0], ('list-panes', '-s', '-t', '=rsim-gazebo', '-F',
                                   '#{@rsim_slug}\t#{pane_current_command}'))

    def test_waits_for_the_unpause_line(self):
        typed = sim.spawn_cmd(SWAP, sim.RunContext(run_dir=Path('/r')))
        done = typed + '\n[INFO] spawned\n[rover_sim] spawned JR81 JR82 RR83 RR84; physics unpaused at 04:54:23\n'
        self.assertGreaterEqual(self.run_wait(done, exited=False), 0)
        # The command line in the log (with $(date ...)) is not the pane's output.
        with self.assertRaisesRegex(sim.SimError, 'spawn pane failed'):
            self.run_wait(typed + '\nCaught exception in launch: JR81: spawn_entity failed (exit 124)\n', exited=True)
        with self.assertRaisesRegex(sim.SimError, 'not all spawned within 0 s'):
            self.run_wait(typed, exited=False, timeout=0)
        with self.assertRaisesRegex(sim.SimError, 'not all spawned'):
            self.run_wait(None, exited=False, timeout=0)              # no log yet

    def test_dry_run_says_so(self):
        out = subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up', 'odom_eval', '--dry-run'],
                             stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
        self.assertIn('after the panes: wait until the spawn pane has unpaused the world (up to %d s; fails if it '
                      'fails)' % sim.SPAWN_WAIT_S, out)
        self.assertLess(out.index('wait until the spawn pane'), out.index('after the panes: arm'))


class Phase3Cli(unittest.TestCase):
    def dry_run(self, *args):
        return subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up'] + list(args) + ['--dry-run'],
                              stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout

    def test_dry_run_shows_every_pane(self):
        out = self.dry_run('multi_agent_swap')
        for a in SWAP.agents:
            self.assertIn('  window %s\n    [dlio-%s] %s dlio odometry' % (a.name, a.name, a.name), out)
            for slug in ('seed-', 'map-odom-tf-', 'teleop-safety-', 'description-tf-', 'bridge-'):
                self.assertIn('[%s%s]' % (slug, a.name), out)
            self.assertIn('sim-dlio-%s bash -c' % a.cls, out)
        self.assertEqual(out.count('ros2 launch rover_control teleop.launch.yaml'), 4)
        self.assertEqual(out.count('dlio_sim.launch.py namespace:='), 4)
        self.assertIn('not arming: auto_arm is off', out)
        self.assertIn('-> /home/swarm/code/dlio_ws/scripts (ro)', out)
        self.assertIn('after the panes: arm JR81, JR82, RR83, RR84', self.dry_run('odom_eval'))
        self.assertIn('not arming: auto_arm needs', self.dry_run('odom_eval', '--no-arm'))
        self.assertIn('not arming: auto_arm needs', self.dry_run('odom_eval', '--stacks', 'zenoh,gazebo,drive'))

    def test_parser(self):
        p = sim.build_parser()
        a = p.parse_args(['arm', 'JR81', 'RR83', '--now'])
        self.assertEqual((a.cmd, a.agents, a.now, a.timeout), ('arm', ['JR81', 'RR83'], True, sim.ARM_TIMEOUT_S))
        d = p.parse_args(['disarm'])
        self.assertEqual((d.cmd, d.agents, d.timeout, hasattr(d, 'now')), ('disarm', [], sim.ARM_TIMEOUT_S, False))
        self.assertTrue(p.parse_args(['up', 'smoke', '--no-arm']).no_arm)


if __name__ == '__main__':
    unittest.main()
