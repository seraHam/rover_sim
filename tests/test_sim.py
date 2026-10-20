"""Unit tests for scripts/sim.py's pure logic (no docker, no tmux).

    python3 -m unittest discover -s tests -v
"""
import base64
import contextlib
import dataclasses
import fnmatch
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
import signal
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
                                            'elevation', 'mighty', 'rviz', 'goals'])
        self.assertTrue(all(s.implemented for s in sim.STACKS.values()))
        self.assertFalse(hasattr(sim.STACKS['goals'], 'late'))     # every stack starts paused
        self.assertEqual((sim.STACKS['mapping'].containers, sim.STACKS['elevation'].containers),
                         (('sim-mapping',), ('sim-elevation',)))
        self.assertEqual([sim.STACKS[n].containers for n in ('mighty', 'rviz', 'goals')],
                         [('sim-mighty',), ('sim-gazebo',), ('sim-mighty',)])
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
        # With the planner, goals: goal_relay.py's first drive (and RViz goals, if any).
        self.assertEqual([x.name for x in sim.select_stacks(s)], ['zenoh', 'gazebo', 'drive', 'frame_align', 'dlio',
                                                                  'mapping', 'mighty', 'goals'])
        self.assertEqual(sim.pending_stacks(s), [])
        gpu = sim.validate_session(session(mapping='gpu', rviz=True))
        # The elevation mapper needs the GPU: only mapping: gpu runs it. With RViz, its goals.
        self.assertEqual([x.name for x in sim.select_stacks(gpu)], ['zenoh', 'gazebo', 'drive', 'frame_align', 'dlio',
                                                                    'mapping', 'elevation', 'mighty', 'rviz', 'goals'])
        self.assertEqual(sim.pending_stacks(gpu), [])
        self.assertEqual(sim.session_profiles(gpu), ['gpu'])
        self.assertEqual(sim.session_profiles(s), [])
        # One agent: nothing to align.
        one = sim.validate_session(session(agents=[{'name': 'JR81', 'class': 'jr', 'pose': [0, 0, 0]}]))
        self.assertEqual([x.name for x in sim.select_stacks(one)], ['zenoh', 'gazebo', 'drive', 'dlio', 'mapping',
                                                                    'mighty', 'goals'])
        # No planner: neither mighty nor goals; a swap demo: goals without RViz.
        none = [x.name for x in sim.select_stacks(sim.validate_session(session(planner=False)))]
        self.assertEqual(('mighty' in none, 'goals' in none), (False, False))
        swap = session(demo='swap')
        swap['agents'][0]['goals'] = [[1.0, 0.0]]
        self.assertEqual([x.name for x in sim.select_stacks(sim.validate_session(swap))][-2:], ['mighty', 'goals'])

    def test_select_requested(self):
        s = sim.validate_session(session())
        self.assertEqual([x.name for x in sim.select_stacks(s, ['zenoh'])], ['zenoh'])
        self.assertEqual([x.name for x in sim.select_stacks(s, ['drive', 'gazebo'])], ['gazebo', 'drive'])
        self.assertEqual([x.name for x in sim.select_stacks(s, ['dlio', 'zenoh'])], ['zenoh', 'dlio'])
        self.assertEqual([x.name for x in sim.select_stacks(s, ['mapping', 'zenoh'])], ['zenoh', 'mapping'])
        self.assertEqual([x.name for x in sim.select_stacks(s, ['mighty', 'zenoh'])], ['zenoh', 'mighty'])
        with self.assertRaisesRegex(sim.SimError, r'goals stack does not apply .*rviz: False, planner: False'):
            sim.select_stacks(sim.validate_session(session(planner=False)), ['goals'])
        with self.assertRaisesRegex(sim.SimError, r'mighty stack does not apply .*planner: False'):
            sim.select_stacks(sim.validate_session(session(planner=False)), ['mighty'])
        with self.assertRaisesRegex(sim.SimError, r'elevation stack does not apply .*mapping: fallback'):
            sim.select_stacks(s, ['elevation'])
        with self.assertRaisesRegex(sim.SimError, 'unknown stack'):
            sim.select_stacks(s, ['nope'])
        one = sim.validate_session(session(agents=[{'name': 'JR81', 'class': 'jr', 'pose': [0, 0, 0]}]))
        with self.assertRaisesRegex(sim.SimError, r'frame_align stack does not apply .*1 agent'):
            sim.select_stacks(one, ['frame_align'])

    def test_generate_steps_once_each(self):
        st = sim.STACKS
        self.assertEqual(sim.generate_steps([st['zenoh'], st['gazebo'], st['drive'], st['frame_align'], st['dlio'],
                                             st['mapping'], st['elevation']]),
                         [sim.write_agent_files, sim.link_calib, sim.write_teleop_file, sim.write_dlio_launch,
                          sim.write_mapping_files, sim.check_elevation_launch, sim.check_kernel_cache])
        self.assertEqual(sim.generate_steps([st['elevation']]), [sim.check_elevation_launch, sim.check_kernel_cache])
        self.assertEqual(sim.generate_steps([st['drive']]), [sim.write_agent_files, sim.link_calib,
                                                             sim.write_teleop_file])
        self.assertEqual(sim.generate_steps([st['dlio']]), [sim.write_dlio_launch])
        self.assertEqual(sim.generate_steps([st['gazebo']]), [sim.write_agent_files])
        self.assertEqual(sim.generate_steps([st['zenoh']]), [])

    def test_every_stack_builds_panes(self):
        s = sim.validate_session(session(mapping='gpu', rviz=True))
        ctx = sim.RunContext(run_dir=Path('/tmp/x'))
        for st in sim.STACKS.values():
            self.assertTrue(st.panes(s, ctx), st.name)
        self.assertEqual(str(sim._not_yet('mighty')), 'the mighty stack lands in Phase 5 (mighty_node, '
                                                      'convert_odom_to_state, mpc, goal_selector per agent)')


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
            self.assertNotEqual(opposite[0].cls, a.cls)                         # crossings are cross-class
            # Across the centre, 20 deg past the opposite start on the same circle: no agent is
            # parked on it (mighty refuses a goal on an occupied cell), and no box is near.
            ((gx, gy),) = a.goals
            self.assertAlmostEqual(math.hypot(gx, gy), 4.0, delta=0.01)
            past = math.degrees(math.atan2(gy, gx) - math.atan2(opposite[0].y, opposite[0].x))
            self.assertAlmostEqual(math.remainder(past, 360.0), 20.0, delta=0.2, msg=a.name)
            self.assertGreater(min(math.hypot(gx - o.x, gy - o.y) for o in s.agents), 1.3, a.name)
        world = REPO / 'src' / 'rover_sim' / 'worlds' / 'arena.world'
        if world.is_file():
            boxes = world_boxes(world)
            for a in s.agents:
                self.assertGreater(min(box_distance(a.goals[0][0], a.goals[0][1], b) for b in boxes), 2.0, a.name)

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

    def z_of(el):
        p = el.find('pose')
        return float(p.text.split()[2]) if p is not None and p.text else 0.0

    out = []
    for m in ET.parse(str(path)).getroot().find('world').findall('model'):
        for link in m.findall('link'):
            lp = compose(pose_of(m), pose_of(link))
            for col in link.findall('collision'):
                size = col.find('geometry/box/size')
                if size is not None:
                    sx, sy, sz = (float(t) for t in size.text.split())
                    if z_of(m) + z_of(link) + z_of(col) - sz / 2 >= 1.0:
                        continue                  # over the robots (the office's sim-only ceiling)
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
        self.assertEqual(len(spawns), 4)
        self.assertLess(wait, spawns[0])
        # The world stays paused: `up` unpauses it once every wave has settled (bring_up).
        self.assertNotIn('unpause', cmd.replace('scripts/sim.py up unpauses it', ''))
        self.assertEqual(steps[spawns[-1] + 1], 'echo "[rover_sim] spawned JR81 JR82 RR83 RR84 into the paused world '
                                                'at $(date +%FT%T%z); scripts/sim.py up unpauses it"')
        self.assertEqual(steps[-1], 'exec sleep infinity')
        # Every CLI has a SIGKILL backstop: a hung call fails the pane, and up says so.
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
        self.assertEqual(sim.status_specs(['JR81', 'RR83'], {'dlio', 'drive', 'frame_align'}), [
            'odom:/JR81/dlio/odom_node/odom', 'count:/JR81/dlio/odom_node/pointcloud/deskewed', 'str:/JR81/mode',
            'odom:/RR83/dlio/odom_node/odom', 'count:/RR83/dlio/odom_node/pointcloud/deskewed', 'str:/RR83/mode'])
        self.assertEqual(sim.status_specs(['JR81'], ['zenoh']), [])
        for spec in sim.status_specs(['JR81'], ['gazebo', 'dlio', 'drive', 'mighty']):
            kind, topic = spec.split(':', 1)
            self.assertIn(kind, ('f64', 'str', 'odom', 'count', 'twist', 'bool'))
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
        self.assertEqual(sim._STATUS_SH, 'exec timeout -k 2 "$1" python3 -c "$SIM_STATUS_PY" %d 3 "${@:2}"'
                         % sim.ECHO_TIMEOUT_S)
        compile(sim._STATUS_PY, '<sim_status>', 'exec')
        self.assertNotIn('ros2 topic echo', sim._STATUS_PY)
        self.assertIn('LEAVE_GAP_S, LEAVE_PAUSE_S = %r, %r\n' % (sim.STATUS_LEAVE_GAP_S, sim.STATUS_LEAVE_PAUSE_S),
                      sim._STATUS_PY)
        # The timeout covers rclpy's start, the deadline and the paced leave of every subscription.
        self.assertEqual(sim.status_budget(0), sim.ECHO_TIMEOUT_S + 5)
        self.assertEqual(sim.status_budget(32), math.ceil(sim.ECHO_TIMEOUT_S + 4 + 32 * sim.STATUS_LEAVE_GAP_S
                                                          + sim.STATUS_LEAVE_PAUSE_S))
        out = subprocess.run(['bash', '-c', sim._STATUS_SH.replace('exec timeout -k 2', 'echo'), 'status', '15', 'a:/x',
                              'b:/y z'], stdout=subprocess.PIPE, universal_newlines=True, env={'SIM_STATUS_PY': 'P'}).stdout
        self.assertEqual(out, '15 python3 -c P 8 3 a:/x b:/y z\n')

    def test_query_leaves_one_subscription_at_a_time(self):
        # Undeclaring every subscription at once (destroy_node, or exiting) stalled the loaded graph;
        # the query destroys them one by one, STATUS_LEAVE_GAP_S apart, before its node, which has
        # no rosout and no parameter services to undeclare in a burst.
        world = {'pubs': {'/JR81/sim/rtf': {'hz': 50, 'value': 0.999}}}
        specs = ['f64:/JR81/sim/rtf', 'odom:/JR81/dlio/odom_node/odom', 'str:/JR81/mode']
        err = []
        t0 = time.time()
        run_with_fake_ros(sim._STATUS_PY, ['0.3', '0.1'] + specs, world, err=err)
        took = time.time() - t0
        self.assertEqual([l for l in err if l.startswith(('LEFT', 'DESTROYED'))],
                         ['LEFT /JR81/sim/rtf', 'LEFT /JR81/dlio/odom_node/odom', 'LEFT /JR81/mode',
                          "DESTROYED with 0 subscription(s) left, [('enable_rosout', False), "
                          "('start_parameter_services', False)]"])
        self.assertGreater(took, 0.3 + 3 * sim.STATUS_LEAVE_GAP_S + sim.STATUS_LEAVE_PAUSE_S)

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
        self.assertEqual(args[-4:], ['status', str(sim.status_budget(2)), 'f64:/JR81/sim/rtf',
                                     'f64:/JR81/sim/time_offset'])
        self.assertEqual(kw['timeout'], sim.status_budget(2) + 16)   # the container's own timeout first
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
            if name != 'sim-viz':                       # its own: VizCommand.test_docker_stop_is_a_ctrl_c
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
            run = [v for v in vols if isinstance(v, str) and re.search(r':/rover_sim/run(:ro)?$', v)]
            self.assertEqual(len(run), 1, name)
            self.assertEqual(run[0].endswith(':ro'), name in ('sim-viz', 'sim-mighty'), name)   # they only read it
            self.assertTrue(run[0].startswith('${ROVER_SIM_RUN:?'), name)
            self.assertEqual(svc['cpuset'], '${%s:-}' % sim.cpuset_var(name), name)
            for v in vols:
                self.assertFalse(sim.docker_socket_mount(sim._volume_source(v)), v)
        self.assertIn('/tmp/.X11-unix:/tmp/.X11-unix:rw', services['sim-gazebo']['volumes'])
        self.assertIn('/tmp/.X11-unix:/tmp/.X11-unix:ro', services['sim-viz']['volumes'])     # connect needs no write
        self.assertEqual(services['sim-elevation']['profiles'], ['gpu'])
        self.assertEqual({n: s['profiles'] for n, s in services.items() if 'profiles' in s},
                         {'sim-elevation': ['gpu'], 'sim-viz': ['viz']})
        dev = services['sim-elevation']['deploy']['resources']['reservations']['devices'][0]
        self.assertEqual((dev['driver'], dev['capabilities']), ('nvidia', ['gpu']))

    def test_elevation_kernel_cache(self):
        # cupy's compiled kernels outlive a recreated sim-elevation, as on the robot's disk; no
        # other service gets a named volume.
        named = {name: [v for v in svc['volumes'] if isinstance(v, str) and not v.startswith(('.', '/', '$'))]
                 for name, svc in self.doc['services'].items()}
        self.assertEqual(named.pop('sim-elevation'), ['cupy-kernel-cache:/root/.cupy'])
        self.assertEqual(set(map(tuple, named.values())), {()})
        self.assertEqual(self.doc['volumes'], {'cupy-kernel-cache': {'name': 'rover-sim-cupy-kernel-cache'}})

    def test_images_from_pin_file(self):
        want = {'sim-gazebo': 'GAZEBO_IMAGE', 'sim-drive': 'DRIVE_IMAGE', 'sim-dlio-jr': 'DLIO_JR_IMAGE',
                'sim-dlio-rr': 'DLIO_RR_IMAGE', 'sim-mapping': 'MAPPING_IMAGE',
                'sim-elevation': 'ELEVATION_IMAGE', 'sim-mighty': 'MIGHTY_IMAGE', 'sim-viz': 'GAZEBO_IMAGE'}
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
        self.assertEqual({v for k, v in b.items() if k not in ('sim-gazebo', 'sim-viz')}, {'6-19'})
        self.assertEqual(b['sim-viz'], '16-19')                         # off 2-5 and their siblings 12-15
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
        self.assertIn('before the panes: write_agent_files, link_calib, write_teleop_file, write_dlio_launch, '
                      'write_mapping_files, check_goal_relay', out)
        self.assertNotIn('not started yet (later phases)', out)
        self.assertIn('mapping: fallback (NOT PARITY', out)
        self.assertNotIn('stack elevation', out)                       # no GPU in the fallback
        self.assertIn('stack dlio -> tmux session rsim-dlio', out)
        # A planner session arms before the unpause, at once; odom_eval (no planner) after it, on DLIO.
        self.assertIn('after the last wave, before the unpause: arm JR81, RR83: /<ns>/auto_mode at once', out)
        self.assertIn('after the panes: arm JR81, JR82, RR83, RR84: /<ns>/auto_mode once its DLIO registers scans',
                      self.dry_run('odom_eval'))
        no_arm = subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up', 'smoke', '--no-arm',
                                 '--dry-run'], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
        self.assertIn('not arming: auto_arm needs the dlio and drive stacks, and mighty in a planner session (or '
                      '--no-arm was given)', no_arm)

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
def init(args=None, signal_handler_options=None):
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
import struct
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
            t = now if spec.get('stamp', 'now') == 'now' else spec['stamp']
            sec, nsec = int(t), int(round((t - int(t)) * 1e9)) % 1000000000
            return self.cb(b'\x00\x01\x00\x00' + struct.pack('<iI', sec, nsec) + b'\0' * 52)
        m, m.header, m.pose = Msg(), Msg(), Msg()
        m.data = spec.get('value')
        m.header.stamp = stamp(now if spec.get('stamp', 'now') == 'now' else spec['stamp'])
        m.pose.pose = Msg()
        m.pose.pose.position = Msg()
        m.pose.pose.position.z = spec.get('z', 0.0)
        m.linear, m.angular = Msg(), Msg()
        m.linear.x, m.angular.z = spec.get('v', 0.0), spec.get('w', 0.0)
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
    def __init__(self, name, **kw):
        self.subs, self.kw = [], kw
    def create_subscription(self, msg_type, topic, cb, qos, raw=False):
        self.subs.append(Sub(topic, cb, raw))
        return self.subs[-1]
    def destroy_subscription(self, sub):
        self.subs.remove(sub)
        print('LEFT %s' % sub.topic, file=sys.stderr)
    def create_client(self, srv_type, name):
        self.clients = getattr(self, 'clients', []) + [Client(name)]
        return self.clients[-1]
    def destroy_client(self, client):
        self.clients.remove(client)
        print('LEFT client %s' % client.name, file=sys.stderr)
    def destroy_node(self):
        print('CALLS %s' % ' '.join(rclpy.WORLD.get('calls', [])), file=sys.stderr)
        print('DESTROYED with %d subscription(s) left, %s' % (len(self.subs), sorted(self.kw.items())), file=sys.stderr)
        if getattr(self, 'clients', None):
            print('DESTROYED with %d client(s) left' % len(self.clients), file=sys.stderr)
''',
    'rclpy/signals.py': 'class SignalHandlerOptions:\n    NO = 0\n',
    'rclpy/qos.py': 'class ReliabilityPolicy:\n    BEST_EFFORT = 2\nclass QoSProfile:\n    def __init__(self, **kw):\n'
                    '        pass\n',
    'nav_msgs/__init__.py': '', 'nav_msgs/msg.py': 'class Odometry:\n    pass\nclass OccupancyGrid:\n    pass\n',
    'sensor_msgs/__init__.py': '', 'sensor_msgs/msg.py': 'class PointCloud2:\n    pass\n',
    'std_msgs/__init__.py': '', 'std_msgs/msg.py': 'class Float64:\n    pass\nclass String:\n    pass\n'
                                                   'class Bool:\n    pass\n',
    'geometry_msgs/__init__.py': '', 'geometry_msgs/msg.py': 'class Twist:\n    pass\n',
    'std_srvs/__init__.py': '', 'std_srvs/srv.py': 'class Trigger:\n    class Request:\n        pass\n',
}


def run_with_fake_ros(script, args, world, calls=None, err=None):
    """Runs an embedded rclpy script against FAKE_ROS; returns its stdout (and the services it
    called into `calls`, its stderr lines into `err`)."""
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
    if err is not None:
        err.extend(p.stderr.splitlines())
    return p.stdout


def dlio_session(ns, n=1):
    """A /<ns>/dlio/session id as dlio_with_session.launch.py makes one (start ns, host, pid)."""
    return '17600000%02d000000000-sim-dlio-%s-%d' % (n, ns[:2].lower(), 40 + n)


def safety_world(*agents, **over):
    """A world with each agent's safety_node (mode + both Trigger services) and its DLIO:
    agents are (ns, mode, odom stamp 'now' | 0 | None, scans?), DLIO's odom at the seed height
    and its session (dlio_session(ns)) announced at 1 Hz."""
    w = {'pubs': {}, 'services': {}}
    for ns, mode, stamp, scans in agents:
        w['pubs']['/%s/mode' % ns] = {'hz': 20, 'value': mode}
        if stamp is not None:
            w['pubs']['/%s/dlio/odom_node/odom' % ns] = {'hz': 100, 'stamp': stamp, 'z': sim.lidar_height_of(ns)}
            w['pubs']['/%s/dlio/session' % ns] = {'hz': 1, 'value': json.dumps(
                {'id': dlio_session(ns), 'start': 1760000000.0, 'ns': ns}, sort_keys=True)}
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
        self.assertEqual(by['dlio-RR83'].cmd, 'ros2 launch /rover_sim/run/dlio_sim.launch.py namespace:=RR83 '
                                              'initial_pose_topic:=world two_d_only:=false')
        self.assertEqual(by['dlio-RR83'].waits, sim.seed_wait(SWAP.agents[2]))
        # After the router's gate and before the node marker: `up` waits for DLIO to start.
        inner = inner_of(by['dlio-RR83'])
        self.assertLess(inner.index('wait_zenoh_ready.py'), inner.index('wait_for_seed.py'))
        self.assertTrue(inner.endswith(sim.seed_wait(SWAP.agents[2]) + ' && ' + sim.node_marker() + ' && '
                                       + by['dlio-RR83'].cmd))
        # DLIO starts only once a seed message has reached its container (not just once the seed
        # pane's publisher runs): this repo's wait_for_seed.py, mounted, on the agent's own topic.
        self.assertEqual(by['dlio-RR83'].waits, 'python3 /rover_sim/scripts/wait_for_seed.py RR83 --timeout %d'
                         % sim.SEED_WAIT_S)
        self.assertTrue((REPO / 'scripts' / sim.WAIT_FOR_SEED).is_file())
        self.assertGreaterEqual(sim.SEED_WAIT_S, 120)        # the paused bring-up takes minutes
        self.assertIn(' /RR83/world ', by['seed-RR83'].cmd)
        self.assertNotIn('pgrep', inner)
        self.assertFalse(hasattr(sim, 'SEED_LEAD_S'))
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
                if name not in ('sim-mapping', 'sim-elevation'):     # Phase4Mapping.test_compose_mounts
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
        # Off by default. The planner sessions arm (mpc publishes cmd_vel_auto from its start), and
        # so does odom_eval, whose drives publish it themselves.
        self.assertFalse(sim.validate_session(session()).auto_arm)
        self.assertTrue(sim.validate_session(session(auto_arm=True)).auto_arm)
        self.assertIn("auto_arm: 'no' is not true/false", errors_of(session(auto_arm='no')))
        self.assertIs(sim.validate_session(session()).to_json()['auto_arm'], False)
        shipped = ['smoke', 'multi_agent_swap', 'office_explore', 'odom_eval']
        self.assertEqual([n for n in shipped if sim.load_session(n).auto_arm], shipped)
        st = sim.STACKS
        s = sim.validate_session(session(auto_arm=True))
        everything = [st[n] for n in ('zenoh', 'gazebo', 'drive', 'dlio', 'mighty')]
        self.assertTrue(sim.will_arm(s, everything))
        # A planner session without its planner: nothing would send cmd_vel_auto, so no arming.
        self.assertFalse(sim.will_arm(s, everything[:-1]))
        self.assertTrue(sim.will_arm(dataclasses.replace(s, planner=False), everything[:-1]))
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
                                                                          'auto_mode service', dlio_session('JR81')))
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

    def test_z_check_binds_on_the_first_arm_of_a_dlio_session(self):
        zc = sim.arm_z_check
        # Within ARM_Z_TOL_M: nothing, whatever the session.
        self.assertEqual(zc('JR81', 0.40, 0.36, 0.1, None, None), (None, None))
        # Late-seed signature (odom anchored at 0), first arm: blamed on the seed.
        why, warn = zc('RR83', 0.002, 0.4, 0.1, 'S1', None)
        self.assertIsNone(warn)
        self.assertEqual(why, 'DLIO odom z 0.00, expected 0.40: the seed pose came too late; restart its dlio pane '
                              '(scripts/sim.py attach dlio: Ctrl-C, Up, Enter)')
        self.assertIn('the seed pose came too late', zc('RR83', -0.05, 0.4, 0.1, 'S2', 'S1')[0])   # after a restart
        # Off by more than the tolerance but not at 0: no seed blame, the offset and both ways out.
        why, warn = zc('JR81', 0.51, 0.36, 0.1, 'S1', None)
        self.assertIsNone(warn)
        self.assertEqual(why, "DLIO odom z 0.51, expected 0.36 (offset +0.15; first arm in this DLIO session): restart "
                              "its dlio pane (resets that agent's maps) or scripts/sim.py arm --no-z-check JR81")
        self.assertNotIn('too late', why)
        self.assertIn('(offset +0.15; DLIO restarted since its last arm)', zc('JR81', 0.51, 0.36, 0.1, 'S2', 'S1')[0])
        self.assertIn('(offset -0.21; no /JR81/dlio/session seen)', zc('JR81', 0.15, 0.36, 0.1, None, 'S1')[0])
        self.assertIn('no /JR81/dlio/session seen', zc('JR81', 0.51, 0.36, 0.1, None, None)[0])
        # Armed before in this very session: z drifts over a run, so a warning only.
        self.assertEqual(zc('JR81', 0.51, 0.36, 0.1, 'S1', 'S1'),
                         (None, 'odometry z drifted +0.15 m since its DLIO session started; not blocking'))
        # ... but z at 0 is the late seed, which no earlier arm excuses (a paused arm records the
        # session without any z check).
        self.assertIn('the seed pose came too late', zc('JR81', 0.0, 0.36, 0.1, 'S1', 'S1')[0])
        # --no-z-check: neither.
        self.assertEqual(zc('RR83', 0.0, 0.4, 0.1, None, None, check=False), (None, None))

    def test_dlio_session_id(self):
        sid = dlio_session('JR81')
        self.assertEqual(sim.dlio_session_id(json.dumps({'id': sid, 'start': 1.0, 'ns': 'JR81'})), sid)
        for bad in ('', 'not json', '[1]', '{"start": 1}', '{"id": 7}', '{"id": ""}', '{"id": "a b"}',
                    '{"id": " a"}', None):
            self.assertIsNone(sim.dlio_session_id(bad), bad)

    def test_parse_arm_sessions(self):
        res = sim.parse_arm('@@ JR81 session S-1\n@@ JR81 session S-2\n@@ JR81 ok mode=AUTO switched\n'
                            '@@ RR83 timeout mode=DISARMED DLIO odom z 0.51, expected 0.40 (offset +0.11; x)\n'
                            '@@ RR84 session S-9\n')
        self.assertEqual(list(res), ['JR81', 'RR83'])                # a session line alone is no result
        self.assertEqual((res['JR81'].session, res['JR81'].message), ('S-2', 'switched'))
        self.assertEqual(res['RR83'].session, '')

    def test_armed_sessions_record(self):
        R = sim.ArmResult
        res = OrderedDict([('JR81', R('JR81', 'ok', 'AUTO', 'switched', 'S-new')),
                           ('JR82', R('JR82', 'ok', 'AUTO', sim.ALREADY_AUTO, 'S-82')),    # not checked here
                           ('RR83', R('RR83', 'timeout', 'DISARMED', 'z', 'S-83')),
                           ('RR84', R('RR84', 'ok', 'AUTO', 'switched', ''))])             # no session seen
        self.assertEqual(sim.armed_sessions_after({'JR81': 'S-old', 'RR84': 'S-84'}, res),
                         {'JR81': 'S-new', 'RR84': 'S-84'})
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            self.assertEqual(sim.load_armed_sessions(run), {})
            sim.save_armed_sessions(run, {'JR81': 'S-1'})
            self.assertEqual(sim.load_armed_sessions(run), {'JR81': 'S-1'})
            self.assertEqual(sorted(p.name for p in run.iterdir()), ['armed_sessions.json'])   # no temp left
            for text in ('{broken', '["JR81"]'):
                (run / 'armed_sessions.json').write_text(text)
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(sim.load_armed_sessions(run), {})      # no record: every arm a first arm
                self.assertIn('first-arm z check', err.getvalue())
            (run / 'armed_sessions.json').write_text('{"JR81": "S-1", "RR83": 5}')
            self.assertEqual(sim.load_armed_sessions(run), {'JR81': 'S-1'})

    def drift_world(self, z=0.51, session=True):
        world = safety_world(('JR81', 'DISARMED', 'now', True))
        world['pubs']['/JR81/dlio/odom_node/odom']['z'] = z
        if not session:
            del world['pubs']['/JR81/dlio/session']
        return world

    def run_query(self, world, armed=None, z_check=True, service='auto_mode', gate='dlio', timeout=1.5):
        calls = []
        args = sim.arm_args(service, gate, timeout, ['JR81'], armed, z_check)[4:]
        out = run_with_fake_ros(sim._ARM_PY, args, world, calls)
        return sim.parse_arm(out)['JR81'], calls, out

    def test_rearm_in_the_same_session_only_warns(self):
        r, calls, out = self.run_query(self.drift_world(), {'JR81': dlio_session('JR81')})
        self.assertEqual((r.status, r.mode, r.session, calls), ('ok', 'AUTO', dlio_session('JR81'), ['/JR81/auto_mode']))
        self.assertEqual(out.count('[arm] JR81: WARNING: odometry z drifted +0.15 m since its DLIO session started; '
                                   'not blocking'), 1)
        # A new session (DLIO restarted), the first arm, or no session to compare: blocked.
        for armed, why in (({'JR81': dlio_session('JR81', 2)}, 'DLIO restarted since its last arm'),
                           (None, 'first arm in this DLIO session')):
            r, calls, out = self.run_query(self.drift_world(), armed)
            self.assertEqual((r.status, calls), ('timeout', []))
            self.assertIn('DLIO odom z 0.51, expected 0.36 (offset +0.15; %s)' % why, r.message)
            self.assertNotIn('WARNING', out)
        r, calls, _ = self.run_query(self.drift_world(session=False), {'JR81': dlio_session('JR81')})
        self.assertEqual((r.status, calls), ('timeout', []))
        self.assertIn('no /JR81/dlio/session seen', r.message)

    def test_no_z_check_and_disarm(self):
        r, calls, out = self.run_query(self.drift_world(z=0.0), z_check=False)
        self.assertEqual((r.status, r.session, calls), ('ok', dlio_session('JR81'), ['/JR81/auto_mode']))
        self.assertNotIn('WARNING', out)
        world = self.drift_world(z=0.0)
        world['pubs']['/JR81/mode']['value'] = 'AUTO'
        r, calls, _ = self.run_query(world, service='disarm', gate='none')
        self.assertEqual((r.status, r.mode, calls), ('ok', 'DISARMED', ['/JR81/disarm']))   # never a z check

    def test_waits_briefly_for_the_session_id(self):
        # z fine but no session: armed once SESSION_WAIT_S is over (nothing to record), not before.
        t = time.time()
        r, calls, _ = self.run_query(self.drift_world(z=0.36, session=False), timeout=8)
        self.assertEqual((r.status, r.session, calls), ('ok', '', ['/JR81/auto_mode']))
        self.assertGreater(time.time() - t, 3.0)
        self.assertIn('SESSION_WAIT_S, SESSION_WAIT_UP_S = 5.0, 2.0, 15.0, 3.0, 15.0', sim._ARM_PY)
        r, calls, _ = self.run_query(self.drift_world(z=0.36, session=False), timeout=1.5)
        self.assertEqual((r.status, calls), ('timeout', []))
        self.assertIn('no /JR81/dlio/session yet', r.message)

    def test_session_gate_records_without_the_dlio_checks(self):
        # up's paused arm: DLIO has no IMU yet (no odom, no scans), so nothing about them is
        # checked, but the session is waited for and reported, for the record.
        world = self.drift_world(z=0.0)
        del world['pubs']['/JR81/dlio/odom_node/odom']
        r, calls, out = self.run_query(world, gate='session')
        self.assertEqual((r.status, r.mode, r.session, calls), ('ok', 'AUTO', dlio_session('JR81'), ['/JR81/auto_mode']))
        self.assertNotIn('WARNING', out)
        r, calls, _ = self.run_query(self.drift_world(session=False), gate='session', timeout=1.5)
        self.assertEqual((r.status, calls), ('timeout', []))
        self.assertIn('no /JR81/dlio/session yet', r.message)

    def test_arm_agents_records_the_session(self):
        seen = []

        def lines(name, args, timeout, env, on_line):
            seen.append(args)
            for l in ('[arm] JR81: WARNING: odometry z drifted +0.15 m since its DLIO session started; not blocking',
                      '@@ JR81 session S-7', '@@ JR81 ok mode=AUTO switched',
                      '@@ RR83 session S-8', '@@ RR83 timeout mode=DISARMED z'):
                on_line(l)
            return 0

        with tempfile.TemporaryDirectory() as d:
            run, log = Path(d), Path(d) / 'arm.log'
            (run / 'armed_sessions.json').write_text('{"RR84": "S-4"}')
            err = io.StringIO()
            with mock.patch.object(sim, 'dexec_lines', lines), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(err):
                res = sim.arm_agents(['JR81', 'RR83'], 'auto_mode', 'dlio', 30, log, run_dir=run)
                self.assertEqual(seen[-1][-2:], ['JR81=0.36', 'RR83=0.4'])
                self.assertEqual(json.loads((run / 'armed_sessions.json').read_text()), {'JR81': 'S-7', 'RR84': 'S-4'})
                self.assertEqual(res['JR81'].session, 'S-7')
                self.assertIn('WARNING: JR81: odometry z drifted +0.15 m', err.getvalue())
                sim.arm_agents(['JR81', 'RR83'], 'auto_mode', 'dlio', 30, log, run_dir=run, z_check=False)
                self.assertEqual(seen[-1][9:], ['0', 'JR81=0.36=S-7', 'RR83=0.4'])
                # disarm and --now neither pass nor change the record.
                sim.arm_agents(['JR81'], 'disarm', 'none', 30, log, run_dir=run)
                sim.arm_agents(['JR81'], 'auto_mode', 'none', 30, log, run_dir=run)
                self.assertEqual([a[-1] for a in seen[-2:]], ['JR81=0.36', 'JR81=0.36'])
                # up's paused arm (gate session) records too.
                (run / 'armed_sessions.json').write_text('{}')
                sim.arm_agents(['JR81'], 'auto_mode', 'session', 30, log, run_dir=run)
                self.assertEqual(json.loads((run / 'armed_sessions.json').read_text()), {'JR81': 'S-7'})
                (run / 'armed_sessions.json').write_text('{"JR81": "S-7", "RR84": "S-4"}')
            self.assertEqual(json.loads((run / 'armed_sessions.json').read_text()), {'JR81': 'S-7', 'RR84': 'S-4'})
            text = log.read_text()
        self.assertEqual(text.count('NO z check (--no-z-check)'), 1)
        self.assertIn('@@ JR81 session S-7', text)

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
        self.assertEqual(args[3:], ['arm', 'auto_mode', 'dlio', '60', '1', '0.1', '1', 'JR81=0.36', 'RR83=0.4'])
        # A recorded DLIO session goes with its agent; --no-z-check is the 0.
        self.assertEqual(sim.arm_args('auto_mode', 'dlio', 60, ['JR81', 'RR83'], {'RR83': 'S-1', 'JR99': 'x'},
                                      z_check=False)[9:], ['0', 'JR81=0.36', 'RR83=0.4=S-1'])
        self.assertIn("'@@ %s ok mode=AUTO " + sim.ALREADY_AUTO + "'", sim._ARM_PY)   # report_interrupted reads it
        # rclpy's start, the 60 s, then the paced leave of 5 entities per agent.
        self.assertEqual(sim.arm_budget(60, 2), math.ceil(66 + 10 * sim.STATUS_LEAVE_GAP_S + sim.STATUS_LEAVE_PAUSE_S))
        self.assertEqual(sim.arm_budget(180, 4),
                         math.ceil(186 + 20 * sim.STATUS_LEAVE_GAP_S + sim.STATUS_LEAVE_PAUSE_S))
        self.assertTrue(args[2].endswith('exec timeout -k 2 %d python3 -c "$SIM_ARM_PY" "$@"' % sim.arm_budget(60, 2)))
        subprocess.run(['bash', '-n', '-c', args[2]], check=True)
        compile(sim._ARM_PY, '<sim_arm>', 'exec')
        # Ctrl-C finds it by the marker in its command line (python3 -c <script>).
        self.assertTrue(sim._ARM_PY.startswith('# ' + sim.ARM_MARKER))
        P = sim.Proc
        procs = [P(1, 0, 1, 1, '/sbin/docker-init -- sleep infinity'), P(7, 1, 7, 2, 'sleep infinity'),
                 P(40, 0, 40, 9, 'timeout -k 2 68 python3 -c ' + sim._ARM_PY.replace('\n', ' ') + ' auto_mode'),
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
        self.assertEqual(seen[0][2], sim.arm_budget(30, 2) + 14)          # after the container's own timeout
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
            service, agents = args[4], [a.split('=')[0] for a in args[10:]]
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
        # Its Ctrl-C ends the query's wait; it gets the time its paced leave takes (3 agents x 5 entities).
        self.assertEqual(stopped, [('sim-drive', {'match': sim.ARM_MARKER, 'grace_term': 2,
                                                  'grace_int': 15 * sim.STATUS_LEAVE_GAP_S + sim.STATUS_LEAVE_PAUSE_S + 2})])
        self.assertIn('SignalHandlerOptions.NO', sim._ARM_PY)
        self.assertIn('and not stop:', sim._ARM_PY)
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
        self.kw = []

        def arm_agents(*a, **k):
            calls.append(a)
            self.kw.append(k)
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

    def test_cli_z_check(self):
        ok = OrderedDict([('JR81', sim.ArmResult('JR81', 'ok', 'AUTO', 'x'))])
        rc, calls, text = self.cli(['arm', 'JR81', '--no-z-check'], results=ok)
        self.assertEqual((rc, calls), (0, [(['JR81'], 'auto_mode', 'dlio', 60.0, None)]))
        self.assertIs(self.kw[0]['z_check'], False)
        self.assertIsInstance(self.kw[0]['run_dir'], Path)              # where armed_sessions.json is kept
        self.assertIn('--no-z-check: arming without the DLIO odom height check', text)
        rc, calls, text = self.cli(['arm', 'JR81'], results=ok)
        self.assertIs(self.kw[0]['z_check'], True)
        self.assertNotIn('--no-z-check', text)
        # disarm has no such flag, and its query checks no z (gate none).
        dis = OrderedDict([('JR81', sim.ArmResult('JR81', 'ok', 'DISARMED', 'x'))])
        rc, calls, _ = self.cli(['disarm', 'JR81'], results=dis)
        self.assertEqual((rc, calls[0][1:3], self.kw[0]['z_check']), (0, ('disarm', 'none'), True))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                sim.build_parser().parse_args(['disarm', '--no-z-check'])

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
    """up waits for the spawn pane's last line, and fails when the pane does."""

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

    def test_waits_for_the_spawned_line(self):
        typed = sim.spawn_cmd(SWAP, sim.RunContext(run_dir=Path('/r')))
        done = typed + ('\n[INFO] spawned\n[rover_sim] spawned JR81 JR82 RR83 RR84 into the paused world at '
                        '2026-10-19T08:54:23+0000; scripts/sim.py up unpauses it\n')
        self.assertGreaterEqual(self.run_wait(done, exited=False), 0)
        # What the container's bash prints matches too.
        line = next(s for s in typed.split(' && ') if s.startswith('echo "[rover_sim] spawned'))
        out = subprocess.run(['bash', '-c', line], stdout=subprocess.PIPE, universal_newlines=True).stdout
        self.assertTrue(sim.SPAWNED_RE.search(out), out)
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
        self.assertIn('wave 0 settles once the spawn pane has spawned every agent into the paused world (up to '
                      '%d s; fails if it fails)' % sim.SPAWN_WAIT_S, out)
        self.assertIn('after each, wait until its panes have started their nodes', out)
        # The waves (spawn in the first), then the unpause, then arming.
        order = [out.index(t) for t in ('wave 0: zenoh, gazebo', 'wave 0 settles once the spawn pane',
                                        'wave 1: drive, frame_align, dlio', 'wave 2: mapping',
                                        'after the last wave: unpause physics', 'after the panes: arm')]
        self.assertEqual(order, sorted(order))
        self.assertIn("docker exec sim-gazebo bash -c '. /usr/share/gazebo/setup.sh && timeout -k 2 30 gz world -p 0",
                      out)


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
        self.assertIn('before the unpause: arm JR81, JR82, RR83, RR84', out)
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


# ---------------------------------------------------------------------------- Phase 4

OFFICE = sim.load_session('office_explore')
SMOKE = sim.load_session('smoke')
MAPPING_PINS = dict(PINS, MAPPING_SHA='b' * 40)
JR_REF = REPO / 'tools' / 'reference' / 'jr02'

# Synthetic robot mapping files with the robot files' layout (one /** block with nested groups,
# per-robot blocks after it); the values are made up.
SAMPLE_GM = """# test global_mapper file
/**:
  ros__parameters:
    z_ground: -1.0          # comment
    voxel_size: 0.1
    distance_grid_2d:
      agent_height: 2.0
      esdf_enabled: false
      truncation_distance: 15
    frame: odom
"""
SAMPLE_PMAP = """/**:
  ros__parameters:
    window_topic: live/occ_2d_topic
    publish_period_s: 2.0
/JR02/persistent_map_node:
  ros__parameters:
    corrected_path_topic: /x
"""
SAMPLE_GM_OVERLAY = """# test overlay: only the keys the sim changes
/**/global_mapper_ros:
  ros__parameters:
    z_ground: 0.15
    distance_grid_2d:
      agent_height: 0.55
      esdf_enabled: true
"""


def resolve_params(doc, fqn):
    """What a node gets from one params file under the rule rclpy 3.3.21 and rclcpp 16.0.19 apply
    (checked in throwaway containers): /**, then /**/<name>, then /<ns>/<name>."""
    out = {}
    for key in ('/**', '/**/' + fqn.rsplit('/', 1)[1], fqn):
        if key in doc:
            out.update(sim.flat_params(doc[key]['ros__parameters'] or {}))
    return out


def reference_launches():
    """JR02's mapping container: the `ros2 launch` command line of each pane (survey capture)."""
    out = {}
    for line in (JR_REF / 'processes' / 'docker_top_mapping.txt').read_text().splitlines():
        m = re.search(r'ros2 launch (\S+) (\S+) .*$', line)
        if m:
            out[m.group(2)] = m.group(0)
    return out


def checkout_file(rel):
    """A file of the pinned mapping_ws checkout (build_images.sh's), or None."""
    f = sim.mapping_checkout(sim.read_pins()) / rel
    return f if f.is_file() else None


class Phase4Mapping(unittest.TestCase):
    ctx = sim.RunContext(run_dir=Path(sim.RUN_DIR_DEFAULT))

    def test_class_settings_are_the_robots(self):
        env = dict(l.split('=', 1) for l in (JR_REF / 'rover.env').read_text().splitlines()
                   if '=' in l and not l.startswith('#'))
        jr = sim.MAPPER_SETTINGS['jr']
        self.assertEqual(jr.gm_params, env['MAPPER_PARAM_FILE'])
        self.assertEqual(jr.cloud, env['MAPPER_CLOUD_TOPIC'])
        self.assertEqual(env['MAPPER_FRAME'], 'JR02/odom')                 # global_frame:=<ns>/odom
        self.assertEqual((jr.tracker, jr.persistent_map), (env['MAPPER_OBSTACLE_TRACKER'] == 'true',
                                                           env['MAPPER_PERSISTENT_MAP'] == 'true'))
        self.assertEqual(env['MAPPER_D455'], 'false')                      # no secondary cloud arguments
        self.assertNotIn('MAP_FUSION_PARAM_FILE', env)
        self.assertEqual(jr.mf_params, 'map_fusion_junior.yaml')           # the JR default
        rr = sim.MAPPER_SETTINGS['rr']
        self.assertEqual((rr.gm_params, rr.mf_params), ('global_mapper_red_rover.yaml', 'map_fusion_red_rover.yaml'))
        self.assertTrue(rr.persistent_map)                                 # off by default on the robot
        self.assertEqual(list(sim.MAPPER_SETTINGS), list(sim.ROBOT_CLASSES))

    def test_gpu_panes_are_jr02s_command_lines(self):
        panes = sim.mapping_panes(SWAP, self.ctx)
        self.assertEqual([(w, [p.slug for p in ps]) for w, ps in sim.tmux_windows(panes)],
                         [(a.name, ['global-mapper-' + a.name, 'map-fusion-' + a.name, 'persistent-map-' + a.name])
                          for a in SWAP.agents])
        by = {p.slug: p for p in panes}
        ref = reference_launches()
        sub = lambda c: c.replace('JR02', 'JR81')
        self.assertEqual(by['global-mapper-JR81'].cmd, sub(ref['global_mapper_node.launch.py']))
        self.assertEqual(by['map-fusion-JR81'].cmd, sub(ref['map_fusion.launch.py']))
        # JR02's line but save_dir:=none (memory only; allowed_deviations.yaml persistent_map_node save_dir).
        robot_dir = 'save_dir:=/var/lib/rover/persistent_map/JR02'
        self.assertTrue(ref['persistent_map.launch.py'].endswith(' ' + robot_dir))
        in_memory = ref['persistent_map.launch.py'].replace(robot_dir, 'save_dir:=none')
        self.assertEqual(by['persistent-map-JR81'].cmd, sub(in_memory))
        self.assertEqual(by['persistent-map-RR84'].cmd, sub(in_memory).replace('JR81', 'RR84'))
        allowed = yaml.safe_load((JR_REF / 'allowed_deviations.yaml').read_text())
        self.assertIn('save_dir', allowed['all']['params']['persistent_map_node'])
        # The RR class: its own files, the same command otherwise.
        self.assertEqual(by['global-mapper-RR83'].cmd, sub(ref['global_mapper_node.launch.py']).replace(
            'JR81', 'RR83').replace('global_mapper_junior_dlio.yaml', 'global_mapper_red_rover.yaml'))
        self.assertEqual(by['map-fusion-RR84'].cmd, sub(ref['map_fusion.launch.py']).replace('JR81', 'RR84').replace(
            'map_fusion_junior.yaml', 'map_fusion_red_rover.yaml'))
        for p in panes:
            self.assertEqual((p.container, p.env, p.gate), ('sim-mapping', sim.robot_env(SWAP.agents[[
                a.name for a in SWAP.agents].index(p.window)]), True))
            inner = inner_of(p)
            self.assertIn('python3 /home/swarm/code/mapping_ws/install/share/rover_bringup/scripts/wait_zenoh_ready.py',
                          inner)
            self.assertTrue(inner.endswith(sim.node_marker() + ' && ' + p.cmd))
            subprocess.run(['bash', '-n', '-c', inner], check=True)
        # Only global_mapper waits for DLIO's odometry, as on the robot.
        self.assertEqual([p.slug for p in panes if p.waits], ['global-mapper-' + a.name for a in SWAP.agents])
        self.assertEqual(robot_gate(by['global-mapper-RR84'].waits), 'python3 /home/swarm/code/mapping_ws/install/'
                         'share/rover_bringup/scripts/wait_for_tf.py RR84/odom RR84/base_link')
        self.assertEqual(by['global-mapper-RR84'].waits, sim.verdict_gate(sim.tf_gate(SWAP.agents[3])))

    def test_tf_gate_is_the_robots(self):
        script = checkout_file('tmux/mapping_host_tmux.sh')
        if script is None:
            self.skipTest('no pinned mapping_ws checkout')
        line = next(l for l in script.read_text().splitlines() if 'wait_for_tf.py $FRAME' in l)
        gate = re.search(r'python3 \S+wait_for_tf.py \$FRAME \$ROBOT_NAME/base_link', line).group(0)
        a = SWAP.agents[0]
        self.assertEqual(sim.tf_gate(a), gate.replace('$WS', sim.MAPPING_WS).replace('$FRAME', 'JR81/odom').replace(
            '$ROBOT_NAME', 'JR81'))

    def test_fallback_panes(self):
        panes = sim.mapping_panes(SMOKE, self.ctx)
        self.assertEqual([(w, [p.slug for p in ps]) for w, ps in sim.tmux_windows(panes)],
                         [(a.name, ['global-mapper-' + a.name, 'grid-relay-' + a.name, 'persistent-map-' + a.name])
                          for a in SMOKE.agents])
        by = {p.slug: p for p in panes}
        gpu = {p.slug: p for p in sim.mapping_panes(dataclasses.replace(SMOKE, mapping='gpu'), self.ctx)}
        # The robot's command lines; only the param files differ.
        self.assertEqual(by['global-mapper-JR81'].cmd, gpu['global-mapper-JR81'].cmd.replace(
            '/mapping_config/global_mapper_junior_dlio.yaml', '/rover_sim/run/mapping/global_mapper_fallback_jr.yaml'))
        self.assertEqual(by['global-mapper-RR83'].cmd, gpu['global-mapper-RR83'].cmd.replace(
            '/mapping_config/global_mapper_red_rover.yaml', '/rover_sim/run/mapping/global_mapper_fallback_rr.yaml'))
        # persistent_map runs the robot's file: grid_relay.py publishes the live window it stitches.
        self.assertEqual(by['persistent-map-RR83'].cmd, gpu['persistent-map-RR83'].cmd)
        self.assertEqual(by['global-mapper-RR83'].waits, gpu['global-mapper-RR83'].waits)
        self.assertEqual(by['grid-relay-JR81'].cmd, 'python3 /rover_sim/scripts/grid_relay.py JR81')
        self.assertNotIn('map_fusion.launch.py', ' '.join(p.cmd for p in panes))
        self.assertIn('FALLBACK', by['global-mapper-JR81'].title)
        for p in panes:
            subprocess.run(['bash', '-n', '-c', inner_of(p)], check=True)

    def test_grid_relay_contract(self):
        """The pane's arguments are grid_relay.py's, and its node is one allowed_deviations.yaml
        lets the fallback run."""
        path = REPO / 'scripts' / 'grid_relay.py'
        if not path.is_file():
            self.skipTest('no scripts/grid_relay.py')
        spec = importlib.util.spec_from_file_location('grid_relay', str(path))
        relay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(relay)                                 # its pure part imports without rclpy
        allowed = yaml.safe_load((JR_REF / 'allowed_deviations.yaml').read_text())['mapping']['fallback']
        for a in SMOKE.agents:
            argv = shlex.split(sim.grid_relay_cmd(a))
            self.assertEqual(argv[:2], ['python3', '/rover_sim/scripts/grid_relay.py'])
            self.assertEqual(relay.parse_args(argv[2:]).ns, a.name)
            fqn = '/%s/%s' % (a.name, relay.node_name(a.name))
            self.assertTrue(any(fnmatch.fnmatch(fqn, pat.format(ns=a.name)) for pat in allowed['sim_only']), fqn)
        # map_fusion's and the elevation mapper's absence is allowed in the fallback, and nothing else.
        self.assertEqual(sorted(allowed['absent']), ['/{ns}/elevation_mapping_node', '/{ns}/map_fusion_node'])

    def test_parity_nodes_run_where_the_manifest_expects(self):
        nodes = yaml.safe_load((JR_REF / 'manifest.yaml').read_text())['nodes']
        launch_node = {'global_mapper_node.launch.py': 'global_mapper_ros', 'map_fusion.launch.py': 'map_fusion_node',
                       'persistent_map.launch.py': 'persistent_map_node',
                       'elevation_mapping_dlio.launch.py': 'elevation_mapping_node'}
        panes = sim.mapping_panes(SWAP, self.ctx) + sim.elevation_panes(SWAP, self.ctx)
        ran = {}
        for p in panes:
            launch = re.match(r'ros2 launch \S+ (\S+) ', p.cmd).group(1)
            ran['/%s/%s' % (re.search(r' quad:=(\S+)', p.cmd).group(1), launch_node[launch])] = p.container
        for name, e in nodes.items():
            if e['container'] in ('sim-mapping', 'sim-elevation'):
                for a in SWAP.agents:
                    self.assertEqual(ran.get(e['fqn'].replace('JR02', a.name)), e['container'], (name, a.name))

    def test_compose_mounts(self):
        doc = yaml.safe_load((REPO / 'docker' / 'compose.stack.yaml').read_text())
        want = {'sim-mapping': [('config/mapping', '/mapping_config')],
                'sim-elevation': [('config/elevation', '/usr/local/share/rover/elevation'),
                                  ('scripts/wait_zenoh_ready.py', '/usr/local/lib/rover/wait_zenoh_ready.py')]}
        for name, mounts in want.items():
            got = [v for v in doc['services'][name]['volumes'] if isinstance(v, dict)]
            self.assertEqual([(m['type'], m['read_only'], m['bind']) for m in got],
                             [('bind', True, {'create_host_path': False})] * len(mounts), name)
            for m, (rel, target) in zip(got, mounts):
                self.assertTrue(m['source'].startswith('../.cache/src/mapping_ws@${MAPPING_SHA:?'), m['source'])
                self.assertTrue(m['source'].endswith('}/' + rel), m['source'])
                self.assertEqual(m['target'], target)
            self.assertEqual(sorted((c, r) for c, r in sim.MAPPING_MOUNTS if c == name),
                             sorted((name, r) for r, _ in mounts))
        self.assertEqual(sim.ELEVATION_OVERLAY_IN, '/usr/local/share/rover/elevation')
        self.assertEqual(sim.CONTAINERS['sim-elevation'].probes[0], '/usr/local/lib/rover/wait_zenoh_ready.py')
        # rover_sim's own overlays reach every container with config/.
        self.assertTrue(sim.SIM_MAPPING_IN.startswith('/rover_sim/config/'))

    def test_mounts_problem(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(sim, 'REPO', Path(d)):
                p = sim.mapping_mounts_problem(MAPPING_PINS, ['sim-mapping', 'sim-elevation', 'sim-drive'])
                self.assertIn('run scripts/build_images.sh mapping', p)
                self.assertIn('config/elevation, config/mapping, scripts/wait_zenoh_ready.py', p)
                self.assertIsNone(sim.mapping_mounts_problem(MAPPING_PINS, ['sim-drive']))
                root = sim.mapping_checkout(MAPPING_PINS)
                (root / 'config' / 'mapping').mkdir(parents=True)
                self.assertIsNone(sim.mapping_mounts_problem(MAPPING_PINS, ['sim-mapping']))
                self.assertIn('config/elevation', sim.mapping_mounts_problem(MAPPING_PINS, ['sim-elevation']))
        self.assertEqual(sim.mapping_checkout({'MAPPING_SHA': 'X'}), REPO / '.cache' / 'src' / 'mapping_ws@X')

    def test_append_overlay(self):
        text = sim.append_overlay(SAMPLE_GM, 'gm.yaml', SAMPLE_GM_OVERLAY, 'ov.yaml', 'global_mapper_ros')
        self.assertIn(SAMPLE_GM.rstrip('\n'), text)                       # both verbatim, comments too
        self.assertTrue(text.rstrip('\n').endswith(SAMPLE_GM_OVERLAY.rstrip('\n')))
        self.assertLess(text.index('/**:'), text.index('/**/global_mapper_ros:'))
        doc = yaml.safe_load(text)
        got = resolve_params(doc, '/JR81/global_mapper_ros')
        want = sim.flat_params(yaml.safe_load(SAMPLE_GM)['/**']['ros__parameters'])
        want.update({'z_ground': 0.15, 'distance_grid_2d.agent_height': 0.55, 'distance_grid_2d.esdf_enabled': True})
        self.assertEqual(got, dict(want))
        self.assertEqual(resolve_params(doc, '/JR81/other_node')['z_ground'], -1.0)   # only that node
        # A persistent_map file with per-robot blocks keeps them.
        po = '/**/persistent_map_node:\n  ros__parameters:\n    window_topic: voxel/occ_2d_topic\n'
        pm = yaml.safe_load(sim.append_overlay(SAMPLE_PMAP, 'pm.yaml', po, 'po.yaml', 'persistent_map_node'))
        self.assertEqual(list(pm), ['/**', '/JR02/persistent_map_node', '/**/persistent_map_node'])
        self.assertEqual(resolve_params(pm, '/RR83/persistent_map_node')['window_topic'], 'voxel/occ_2d_topic')

    def test_append_overlay_refuses(self):
        def ov(params, key='/**/global_mapper_ros'):
            return yaml.safe_dump({key: {'ros__parameters': params}})
        for overlay, why in ((ov({'z_ground': 0.1}, key='/**'), 'one /\\*\\*/global_mapper_ros block'),
                             (ov({'z_ground': 0.1}) + ov({'a': 1}, key='/x/y'), 'one /\\*\\*/global_mapper_ros block'),
                             (ov({'z_ground': 0}), 'z_ground: double in gm.yaml, integer in the overlay'),
                             (ov({'distance_grid_2d': {'truncation_distance': 15.0}}), 'integer in gm.yaml, double'),
                             (ov({'distance_grid_2d': {'esdf_enabled': 'true'}}), 'bool in gm.yaml, string'),
                             (ov({'z_groud': 0.1}), 'z_groud: not in the /\\*\\* block of gm.yaml'),
                             (ov({'z_ground': None}), 'not a parameter value'),
                             (ov({'z_ground': float('nan')}), 'not a parameter value'),
                             (ov({}), 'sets no parameters'),
                             ('/**/global_mapper_ros:\n  ros__parameters: {z_ground: 1.0}\n  extra: 1\n', 'not a flat'),
                             ('a: [\n', 'ov.yaml')):
            with self.assertRaisesRegex(sim.SimError, why, msg=overlay):
                sim.append_overlay(SAMPLE_GM, 'gm.yaml', overlay, 'ov.yaml', 'global_mapper_ros')
        with self.assertRaisesRegex(sim.SimError, 'has a /\\*\\*/global_mapper_ros block of its own'):
            sim.append_overlay(SAMPLE_GM + ov({'z_ground': 0.1}), 'gm.yaml', ov({'z_ground': 0.2}), 'ov.yaml',
                               'global_mapper_ros')

    def test_shipped_overlays_on_the_pinned_files(self):
        if checkout_file('config/mapping/persistent_map.yaml') is None:
            self.skipTest('no pinned mapping_ws checkout')
        src = sim.mapping_checkout(sim.read_pins()) / 'config' / 'mapping'
        for a in SMOKE.agents:
            for node, base, overlay, out in sim.fallback_files(a):
                if not (sim.SIM_MAPPING_DIR / overlay).is_file():
                    self.skipTest('no config/mapping/%s yet' % overlay)
                base_text = (src / base).read_text()
                doc = yaml.safe_load(sim.append_overlay(base_text, base, (sim.SIM_MAPPING_DIR / overlay).read_text(),
                                                        overlay, node))
                fqn = '/%s/%s' % (a.name, node)
                got, robot = resolve_params(doc, fqn), resolve_params(yaml.safe_load(base_text), fqn)
                over = sim.flat_params(yaml.safe_load((sim.SIM_MAPPING_DIR / overlay).read_text())[
                    '/**/' + node]['ros__parameters'])
                self.assertEqual(sorted(got), sorted(robot), overlay)              # no new names, none lost
                for k, v in got.items():
                    self.assertEqual(v, over.get(k, robot[k]), (overlay, k))
        # No persistent_map overlay: the robot's window_topic is a topic grid_relay.py publishes.
        self.assertEqual(sorted(p.name for p in sim.SIM_MAPPING_DIR.iterdir()),
                         ['global_mapper_fallback_jr.yaml', 'global_mapper_fallback_rr.yaml'])
        window = resolve_params(yaml.safe_load((src / 'persistent_map.yaml').read_text()),
                                '/RR83/persistent_map_node')['window_topic']
        spec = importlib.util.spec_from_file_location('grid_relay', str(REPO / 'scripts' / 'grid_relay.py'))
        relay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(relay)
        self.assertIn(window, [dst for dst, _ in relay.OUTPUTS])

    def test_fallback_files(self):
        self.assertEqual(sim.fallback_files(SMOKE.agents[0]), [
            ('global_mapper_ros', 'global_mapper_junior_dlio.yaml', 'global_mapper_fallback_jr.yaml',
             'global_mapper_fallback_jr.yaml')])
        self.assertEqual(sim.fallback_files(SMOKE.agents[1])[0][1:3], ('global_mapper_red_rover.yaml',
                                                                      'global_mapper_fallback_rr.yaml'))

    def fake_repo(self, d, overlays=True):
        cfg = sim.mapping_checkout(MAPPING_PINS) / 'config' / 'mapping'
        cfg.mkdir(parents=True)
        for f in ('global_mapper_junior_dlio.yaml', 'global_mapper_red_rover.yaml'):
            (cfg / f).write_text(SAMPLE_GM)
        (cfg / 'persistent_map.yaml').write_text(SAMPLE_PMAP)
        (cfg / 'map_fusion_junior.yaml').write_text('/**:\n  ros__parameters: {a: 1}\n')
        own = Path(d) / 'config' / 'mapping'
        own.mkdir(parents=True)
        if overlays:
            for cls in ('jr', 'rr'):
                (own / ('global_mapper_fallback_%s.yaml' % cls)).write_text(SAMPLE_GM_OVERLAY)
        return own

    def test_config_problem_and_files(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as run:
            with mock.patch.object(sim, 'REPO', Path(d)), mock.patch.object(sim, 'read_pins', lambda: MAPPING_PINS):
                self.assertIn('run scripts/build_images.sh mapping', sim.mapping_config_problem(SMOKE, MAPPING_PINS))
                own = self.fake_repo(d)
                with mock.patch.object(sim, 'SIM_MAPPING_DIR', own):
                    self.assertIsNone(sim.mapping_config_problem(SMOKE, MAPPING_PINS))
                    # Parity needs both classes' map_fusion files: RR's is missing here.
                    gpu = dataclasses.replace(SMOKE, mapping='gpu')
                    self.assertIn('map_fusion_red_rover.yaml', sim.mapping_config_problem(gpu, MAPPING_PINS))
                    (own / 'global_mapper_fallback_rr.yaml').unlink()
                    self.assertIn('needs the overlay(s) global_mapper_fallback_rr.yaml',
                                  sim.mapping_config_problem(SMOKE, MAPPING_PINS))
                    with self.assertRaisesRegex(sim.SimError, 'global_mapper_fallback_rr.yaml'):
                        sim.write_mapping_files(SMOKE, sim.RunContext(run_dir=Path(run)))
                    (own / 'global_mapper_fallback_rr.yaml').write_text(SAMPLE_GM_OVERLAY)
                    out = Path(run) / 'mapping'
                    out.mkdir()
                    (out / 'stale.yaml').write_text('x')
                    with contextlib.redirect_stdout(io.StringIO()) as said:
                        sim.write_mapping_files(SMOKE, sim.RunContext(run_dir=Path(run)))
                    self.assertEqual(sorted(p.name for p in out.iterdir()), [
                        'global_mapper_fallback_jr.yaml', 'global_mapper_fallback_rr.yaml'])
                    self.assertIn('NOT PARITY', said.getvalue())
                    doc = yaml.safe_load((out / 'global_mapper_fallback_rr.yaml').read_text())
                    self.assertEqual(resolve_params(doc, '/RR83/global_mapper_ros')['z_ground'], 0.15)
                    self.assertIn('mapping_ws@bbbbbbb:config/mapping/global_mapper_red_rover.yaml verbatim',
                                  (out / 'global_mapper_fallback_rr.yaml').read_text())
                    # The parity panes read /mapping_config itself: nothing generated, stale files gone.
                    (Path(d) / '.cache' / 'src' / ('mapping_ws@' + 'b' * 40) / 'config' / 'mapping' /
                     'map_fusion_red_rover.yaml').write_text('/**:\n  ros__parameters: {a: 1}\n')
                    sim.write_mapping_files(gpu, sim.RunContext(run_dir=Path(run)))
                    self.assertEqual(list(out.iterdir()), [])

    def test_elevation_panes(self):
        panes = sim.elevation_panes(OFFICE, self.ctx)
        self.assertEqual([(w, [p.slug for p in ps]) for w, ps in sim.tmux_windows(panes)],
                         [(a.name, ['elevation-' + a.name]) for a in OFFICE.agents])
        p = panes[2]
        self.assertEqual((p.container, p.env, p.waits), ('sim-elevation', sim.robot_env(OFFICE.agents[2]), ''))
        # local_mapper_host_tmux.sh's line, for an elevation_mapping_cupy that takes param_overlays.
        self.assertEqual(p.cmd, "ros2 launch elevation_mapping_cupy elevation_mapping_dlio.launch.py quad:=RR83 "
                                "'extra_cloud_exclude:=livox/*' occupancy_grid_topic:=elev/occ_2d_topic "
                                "planning_occupancy_grid_topic:=elev/planning_occ_2d_topic "
                                "esdf_grid_topic:=elev/esdf_2d_topic")
        self.assertIn('extra_cloud_exclude:=livox/*', shlex.split(p.cmd))     # one argument, not globbed
        inner = inner_of(p)
        self.assertIn('if [ -f /usr/local/lib/rover/wait_zenoh_ready.py ]; then python3 '
                      '/usr/local/lib/rover/wait_zenoh_ready.py;', inner)
        subprocess.run(['bash', '-n', '-c', inner], check=True)
        out = subprocess.run(['bash', '-c', 'f() { printf "%s\\n" "$@"; }; f ' + p.cmd[len('ros2 launch '):]],
                             stdout=subprocess.PIPE, universal_newlines=True, cwd='/').stdout.split('\n')
        self.assertIn('extra_cloud_exclude:=livox/*', out)                    # as the container's bash passes it
        script = checkout_file('orin/local_mapper_host_tmux.sh')
        if script is not None:
            topics = re.search(r'^TOPIC_ARGS=" (.*)"$', script.read_text(), re.M).group(1).split()
            self.assertEqual(tuple(topics), sim.ELEVATION_TOPIC_ARGS)
            self.assertIn("OVERLAY_ARGS=\" 'extra_cloud_exclude:=livox/*'\"", script.read_text())
            # mapping_ws c1233f7 has overlays for neither class (only types/JK.yaml).
            self.assertEqual([sim.elevation_overlays(a, sim.read_pins()) for a in OFFICE.agents], [[]] * 4)

    def test_elevation_overlays(self):
        a, rr = OFFICE.agents[0], OFFICE.agents[2]
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(sim, 'REPO', Path(d)):
                root = sim.mapping_checkout(MAPPING_PINS) / 'config' / 'elevation'
                self.assertEqual(sim.elevation_overlays(a, MAPPING_PINS), [])
                for rel in ('types/JR.yaml', 'vehicles/JR02.yaml', 'vehicles/JR81.yaml', 'types/JK.yaml'):
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    (root / rel).write_text('node: {}\n')
                # The class's type file, then its reference robot's (JR02, which JR81 stands in for).
                self.assertEqual(sim.elevation_overlays(a, MAPPING_PINS),
                                 ['/usr/local/share/rover/elevation/types/JR.yaml',
                                  '/usr/local/share/rover/elevation/vehicles/JR02.yaml'])
                self.assertEqual(sim.elevation_overlays(rr, MAPPING_PINS), [])
        cmd = sim.elevation_cmd(a, ['/x/types/JR.yaml', '/x/vehicles/JR02.yaml'])
        args = shlex.split(cmd)
        self.assertEqual(args[5:7], ['extra_cloud_exclude:=livox/*',
                                     'param_overlays:=/x/types/JR.yaml,/x/vehicles/JR02.yaml'])
        self.assertEqual(tuple(args[-3:]), sim.ELEVATION_TOPIC_ARGS)          # last: nothing moves them back

    def test_dry_runs_show_every_mapping_pane(self):
        def dry(name):
            return subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up', name, '--dry-run'],
                                  stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
        office = dry('office_explore')
        self.assertIn('mapping: gpu (parity', office)
        for stack in ('mapping', 'elevation'):
            self.assertIn('stack %s -> tmux session rsim-%s' % (stack, stack), office)
        for a in OFFICE.agents:
            for slug in ('global-mapper-', 'map-fusion-', 'persistent-map-', 'elevation-'):
                self.assertIn('[%s%s]' % (slug, a.name), office)
        self.assertEqual(office.count('elevation_mapping_dlio.launch.py quad:='), 4)
        self.assertEqual(office.count('map_fusion.launch.py quad:='), 4)
        self.assertIn('elevation overlays: JR81 none; JR82 none; RR83 none; RR84 none', office)
        self.assertNotIn('grid_relay', office)
        smoke = dry('smoke')
        self.assertEqual(smoke.count('grid_relay.py '), 2)
        self.assertIn('fallback params: /rover_sim/run/mapping/global_mapper_fallback_rr.yaml = '
                      '/mapping_config/global_mapper_red_rover.yaml + '
                      '/rover_sim/config/mapping/global_mapper_fallback_rr.yaml (RR83)', smoke)
        self.assertNotIn('persistent_map_fallback', smoke)
        self.assertEqual(smoke.count('param_file:=/mapping_config/persistent_map.yaml save_dir:=none'), 2)
        self.assertNotIn('/var/lib/rover', smoke + office)
        self.assertNotIn('map_fusion.launch.py', smoke)


class PausedBringUp(unittest.TestCase):
    """The stacks start wave by wave into the paused world, then `up` unpauses it once."""
    ctx = sim.RunContext(run_dir=Path(sim.RUN_DIR_DEFAULT))

    def test_node_marker_in_every_pane(self):
        typed_lines = []
        for s in (SWAP, SMOKE, OFFICE):
            for st in sim.select_stacks(s):
                for p in st.panes(s, self.ctx):
                    inner = inner_of(p)
                    self.assertTrue(inner.endswith(sim.node_marker() + ' && ' + p.cmd), p.slug)
                    if p.waits:                                         # after the pane's own gates
                        self.assertIn(p.waits + ' && ' + sim.node_marker(), inner)
                    typed_lines.append(sim.pane_command(p))
        self.assertTrue(sim.NODE_STARTED_RE.search('[rover_sim] node starting at 2026-10-19T09:38:07+0000\r\n'))
        self.assertFalse(sim.NODE_STARTED_RE.search('[rover_sim] node starting at 09:38:07\r\n'))   # no date, no zone
        out = subprocess.run(['bash', '-c', sim.node_marker()], stdout=subprocess.PIPE, universal_newlines=True).stdout
        self.assertTrue(sim.NODE_STARTED_RE.search(out))
        for line in typed_lines:                                         # the typed line is not the output
            self.assertFalse(sim.NODE_STARTED_RE.search(line))

    def test_pane_commands(self):
        out = 'rsim-dlio\tdlio-JR81\tdocker\nrsim-dlio\tseed-JR81\tbash\nother\tx\tvim\nrsim-zenoh\t\tbash\n'
        with mock.patch.object(sim, 'tmux', lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=out)):
            self.assertEqual(sim.pane_commands(), {('rsim-dlio', 'dlio-JR81'): 'docker',
                                                   ('rsim-dlio', 'seed-JR81'): 'bash', ('rsim-zenoh', ''): 'bash'})
        # tmux could not list them: unknown, not "every pane ended".
        with mock.patch.object(sim, 'tmux', lambda *a, **k: types.SimpleNamespace(returncode=1, stdout='')):
            self.assertIsNone(sim.pane_commands())

    def test_wait_nodes_started(self):
        st = sim.STACKS
        mk = lambda slug: sim.Pane(slug, slug, 'sim-mapping', 'true')
        plan = [(st['dlio'], [mk('a'), mk('b')]), (st['mapping'], [mk('c'), mk('d')])]
        with tempfile.TemporaryDirectory() as d:
            for stack, slug, text in (('dlio', 'a', 'typed && echo "[rover_sim] node starting at $(date +%FT%T%z)" && x\n'
                                                    '[rover_sim] node starting at 2026-10-19T12:00:01+0000\n'),
                                      ('dlio', 'b', 'typed && ' + sim.node_marker() + '\n'),     # the typed line only
                                      ('mapping', 'd', 'waiting for zenoh router\n')):
                (Path(d) / stack).mkdir(exist_ok=True)
                (Path(d) / stack / (slug + '.log')).write_text(text)
            states = {('rsim-dlio', 'a'): 'docker', ('rsim-dlio', 'b'): 'bash', ('rsim-mapping', 'c'): 'docker',
                      ('rsim-mapping', 'd'): 'docker'}
            with mock.patch.object(sim, 'pane_commands', lambda: states), \
                    mock.patch.object(sim, 'SPAWN_PANE_GRACE_S', 0), mock.patch.object(sim.time, 'sleep', lambda s: None):
                waited, exited, missing = sim.wait_nodes_started(plan, Path(d), timeout=0)
                self.assertEqual((exited, missing), (['dlio/b'], ['mapping/c', 'mapping/d']))
                for slug in ('c', 'd'):
                    (Path(d) / 'mapping' / (slug + '.log')).write_text('[rover_sim] node starting at '
                                                                       '2026-10-19T12:00:09+0000\n')
                self.assertEqual(sim.wait_nodes_started(plan, Path(d), timeout=60)[1:], (['dlio/b'], []))
            # Before the grace, a pane at its bash prompt may simply not have started its docker exec.
            with mock.patch.object(sim, 'pane_commands', lambda: {}), \
                    mock.patch.object(sim.time, 'sleep', lambda s: None):
                self.assertEqual(sim.wait_nodes_started(plan[:1], Path(d), timeout=0)[1:], ([], ['dlio/b']))
            # While tmux cannot list the panes, none counts as ended: they are waited for.
            with mock.patch.object(sim, 'pane_commands', lambda: None), \
                    mock.patch.object(sim, 'SPAWN_PANE_GRACE_S', 0), mock.patch.object(sim.time, 'sleep', lambda s: None):
                self.assertEqual(sim.wait_nodes_started(plan[:1], Path(d), timeout=0)[1:], ([], ['dlio/b']))

    def test_wait_idle(self):
        t = [0.0]

        def run(samples, **kw):
            it = iter(samples)

            def sample():
                t[0] += 2.0
                return next(it)
            t[0] = 0.0

            def sleep(s):
                t[0] += s
            return sim.wait_idle(sample, (10.0, 100.0), clock=lambda: t[0], sleep=sleep, **kw)
        # A burst, one quiet sample, busy again, then two quiet in a row.
        self.assertEqual(run([(50, 900), (5, 90), (5, 150), (5, 90), (9.9, 99)], quiet=2, timeout=60), (10.0, (9.9, 99)))
        # Under the limit but still ramping up (smoke's two global_mappers: 53 % then 88 %) is not idle.
        self.assertEqual(run([(2, 53), (3, 88), (3, 60), (3, 58)], quiet=2, timeout=60), (8.0, (3, 58)))
        self.assertEqual(run([(3, 80), (3, 94)], quiet=2, timeout=60), (4.0, (3, 94)))       # within 15 % of a core
        self.assertEqual(run([(3, 80), (3, 96), (3, 97)], quiet=2, timeout=60), (6.0, (3, 97)))   # 16: a new run
        self.assertEqual(run([(50, 900)] * 40, quiet=2, timeout=20), (None, (50, 900)))
        # A sample that cannot be taken (a docker call timed out) is not quiet: it breaks the run, and
        # sampling goes on until the timeout rather than giving up on the settle.
        self.assertEqual(run([(3, 80), None, (3, 80), (3, 82)], quiet=2, timeout=60), (10.0, (3, 82)))
        self.assertEqual(run([None] * 10, quiet=2, timeout=20), (None, None))
        seen = []
        it = iter([None, (3, 80), (3, 80)])
        sim.wait_idle(lambda: next(it), (10.0, 100.0), clock=lambda: 0.0, on_sample=seen.append, sleep=lambda s: None)
        self.assertEqual(seen, [None, (3, 80), (3, 80)])                    # every sample is logged

    def test_load_sampler(self):
        clock, cpu = iter([100.0, 102.0, 104.0]), iter([10.0, 10.5, None])
        with mock.patch.object(sim.time, 'time', lambda: next(clock)), \
                mock.patch.object(sim, 'router_cpu_seconds', lambda: next(cpu)), \
                mock.patch.object(sim, 'docker_stats', lambda names: {'sim-drive': {'cpu': '12.50%'},
                                                                      'sim-mapping': {'cpu': '--'},
                                                                      'sim-dlio-jr': {'cpu': '30%'}}):
            s = sim.LoadSampler(['sim-drive', 'sim-mapping', 'sim-dlio-jr'])
            self.assertEqual(s(), (25.0, 42.5))
            self.assertIsNone(s())                                       # rmw_zenohd not readable

        def timed_out(*a, **k):
            raise sim.SimError('timed out after 20s: docker exec sim-drive python3 -c ...')
        clock, cpu = iter([100.0, 102.0, 104.0, 106.0]), iter([10.0, 10.5])
        with mock.patch.object(sim.time, 'time', lambda: next(clock)), \
                mock.patch.object(sim, 'router_cpu_seconds', lambda: next(cpu)), \
                mock.patch.object(sim, 'docker_stats', lambda names: {'sim-drive': {'cpu': '10%'}}):
            s = sim.LoadSampler(['sim-drive'])
            self.assertEqual(s(), (25.0, 10.0))
            with mock.patch.object(sim, 'router_cpu_seconds', timed_out):
                self.assertIsNone(s())
            with mock.patch.object(sim, 'docker_stats', timed_out), \
                    mock.patch.object(sim, 'router_cpu_seconds', lambda: 11.0):
                self.assertIsNone(s())
        with mock.patch.object(sim, 'router_cpu_seconds', timed_out):
            self.assertIsNone(sim.LoadSampler(['sim-drive']).prev[1])   # in __init__ too

    def test_unpause(self):
        calls = []

        def dexec(results):
            it = iter(results)

            def f(name, args, **kw):
                calls.append((name, args, kw))
                r = next(it)
                if isinstance(r, Exception):
                    raise r
                return r
            return f
        cmd = sim.unpause_cmd(self.ctx)
        # Gazebo transport, not gzserver's ROS services; not pinned to one core (gz then hangs).
        self.assertTrue(cmd.startswith('. /usr/share/gazebo/setup.sh && timeout -k 2 30 gz world -p 0 && '), cmd)
        self.assertNotIn('taskset', cmd)
        self.assertNotIn('ros2', cmd)
        subprocess.run(['bash', '-n', '-c', cmd], check=True)
        # It succeeds only once gz stats reports the world running.
        with tempfile.TemporaryDirectory() as d:
            gz = Path(d) / 'gz'
            gz.write_text('#!/bin/sh\n[ "$1" = stats ] && [ -e "%s/running" ] && echo "Factor[1.00] SimTime[1.0] '
                          'RealTime[1.0] Paused[F]"\nexit 0\n' % d)
            gz.chmod(0o755)
            fake = cmd.replace('. /usr/share/gazebo/setup.sh', 'PATH=%s:$PATH' % d)
            self.assertEqual(subprocess.run(['bash', '-c', fake], stdout=subprocess.PIPE).returncode, 1)
            (Path(d) / 'running').write_text('')
            self.assertEqual(subprocess.run(['bash', '-c', fake], stdout=subprocess.PIPE).returncode, 0)
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / sim.BRINGUP_LOG
            ok = FakeProc(0, 'Paused[F]\n')
            with mock.patch.object(sim, 'dexec', dexec([ok])):
                self.assertEqual(sim.unpause_physics(self.ctx, log), 1)
            self.assertEqual(calls[0][:2], ('sim-gazebo', ['bash', '-c', cmd]))
            self.assertGreater(calls[0][2]['timeout'], sim.UNPAUSE_CALL_S + 3 * 8)   # the container's backstops first
            with mock.patch.object(sim, 'dexec', dexec([FakeProc(1, 'gz stats: the world still reports no Paused[F]'),
                                                        ok])), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(sim.unpause_physics(self.ctx, log), 2)
            self.assertIn('unpause 1 of 2 failed (exit 1)', err.getvalue())
            with mock.patch.object(sim, 'dexec', dexec([FakeProc(124), sim.SimError('timed out after 70s')])), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(sim.SimError, 'physics was not unpaused'):
                    sim.unpause_physics(self.ctx, log)
            self.assertEqual(log.read_text().count('unpause '), 5)

    def test_waves(self):
        """Gazebo spawns on a quiet graph; DLIO's inputs come before DLIO's readers; elevation last."""
        st = sim.STACKS
        every = [(s, []) for s in st.values()]
        self.assertEqual([[s.name for s, _ in w] for w in sim.start_waves(every)],
                         [['zenoh', 'gazebo'], ['drive', 'frame_align', 'dlio'], ['mapping'], ['elevation'], ['mighty'],
                          ['rviz', 'goals']])
        plan = [(st[n], []) for n in ('elevation', 'zenoh', 'mapping', 'dlio')]
        self.assertEqual([[s.name for s, _ in w] for w in sim.start_waves(plan)],
                         [['zenoh'], ['dlio'], ['mapping'], ['elevation']])

    def test_settle_wave(self):
        with tempfile.TemporaryDirectory() as d:
            seen = []

            def run(nodes, idle, samples=((3.0, 80.0),)):
                del seen[:]

                def wait_idle(sample, limits, on_sample):
                    seen.append(('idle', limits))
                    for v in samples:
                        on_sample(v)
                    return idle
                fakes = {'wait_nodes_started': lambda wave, log_dir: seen.append(('nodes', [s.name for s, _ in wave]))
                         or nodes,
                         'running_sim_containers': lambda: ['sim-drive'], 'router_cpu_seconds': lambda: 1.0,
                         'wait_idle': wait_idle}
                with mock.patch.multiple(sim, **fakes), contextlib.redirect_stdout(io.StringIO()) as out, \
                        contextlib.redirect_stderr(io.StringIO()) as err:
                    ret = sim.settle_wave([(sim.STACKS['mapping'], [])], 4, Path(d))
                return ret, out.getvalue(), err.getvalue()
            ret, out, err = run((68.0, [], []), (4.0, (3.0, 80.0)))
            self.assertEqual(ret, [])
            self.assertEqual(seen, [('nodes', ['mapping']), ('idle', (sim.SETTLE_ROUTER_PCT, 200.0))])
            self.assertIn('mapping: all panes started their nodes after 68 s, idle 4 s later (router 3 %, sim '
                          'containers 80 % of a core)', out)
            self.assertEqual(err, '')
            log = (Path(d) / sim.BRINGUP_LOG).read_text()
            self.assertIn('mapping: load router 3.0 %, sim containers 80 % of a core', log)   # every sample
            # Neither a pane still at its gate nor busy (or unreadable) stacks hold `up` back for ever.
            ret, out, err = run((120.0, [], ['mapping/global-mapper-JR81']), (None, (40.0, 900.0)),
                                samples=((40.0, 900.0), None))
            self.assertEqual(ret, [])
            for want in ('still waiting on their gates after 120 s: mapping/global-mapper-JR81; going on',
                         'mapping: the stacks did not go idle (router 40 %, sim containers 900 % of a core'):
                self.assertIn(want, err)
            self.assertIn('mapping: load not readable', (Path(d) / sim.BRINGUP_LOG).read_text())
            # A pane that ended before its node started is returned at once, without the idle wait.
            ret, out, err = run((12.0, ['mapping/map-fusion-JR81'], []), (4.0, (3.0, 80.0)))
            self.assertEqual(ret, [('mapping/map-fusion-JR81', 'ended before its node started')])
            self.assertEqual(seen, [('nodes', ['mapping'])])

    def test_bring_up_order(self):
        order = []
        st = sim.STACKS
        plan = [(st[n], []) for n in ('zenoh', 'gazebo', 'drive', 'frame_align', 'dlio', 'mapping', 'elevation')]
        fakes = {'start_stack': lambda s, panes, log_dir: order.append('start ' + s.name) or 'rsim-' + s.name,
                 'wait_spawned': lambda log_dir: order.append('spawned') or 8.0,
                 'settle_wave': lambda wave, n, log_dir: order.append('settle ' + '+'.join(s.name for s, _ in wave))
                 or [],
                 'unpause_physics': lambda ctx, log: order.append('unpause') or 1,
                 'check_graph': lambda session, stacks: order.append('graph') or []}
        with tempfile.TemporaryDirectory() as d, mock.patch.multiple(sim, **fakes), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            sim.bring_up(OFFICE, plan, self.ctx, Path(d))
            log = (Path(d) / sim.BRINGUP_LOG).read_text()
        self.assertEqual(order, ['start zenoh', 'start gazebo', 'spawned', 'settle zenoh+gazebo',
                                 'start drive', 'start frame_align', 'start dlio', 'settle drive+frame_align+dlio',
                                 'start mapping', 'settle mapping', 'start elevation', 'settle elevation', 'graph',
                                 'unpause'])
        self.assertIn('physics unpaused at', out.getvalue())
        self.assertIn('wave 3: starting elevation', log)
        # Without gazebo in this `up`, there is no world to spawn into or unpause.
        del order[:]
        with tempfile.TemporaryDirectory() as d, mock.patch.multiple(sim, **fakes), \
                contextlib.redirect_stdout(io.StringIO()):
            sim.bring_up(OFFICE, [(st['zenoh'], []), (st['mapping'], [])], self.ctx, Path(d))
        self.assertEqual(order, ['start zenoh', 'settle zenoh', 'start mapping', 'settle mapping', 'graph'])
        # A pane that ended before its node started stops `up` after its wave: no later wave, no
        # unpause (the world stays paused, and nothing arms).
        del order[:]
        failing = dict(fakes, settle_wave=lambda wave, n, log_dir: order.append(
            'settle ' + '+'.join(s.name for s, _ in wave)) or ([('dlio/dlio-RR83', 'ended before its node started')]
                                                              if wave[-1][0].name == 'dlio' else []))
        with tempfile.TemporaryDirectory() as d, mock.patch.multiple(sim, **failing), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(sim.SimError, r'failed in wave 1: dlio/dlio-RR83 ended before its node started '
                                                      r'.*attach dlio\).*the world stays paused, and mapping, '
                                                      r'elevation did not start; every safety_node stays DISARMED'):
                sim.bring_up(OFFICE, plan, self.ctx, Path(d))
            self.assertIn('stopping: dlio/dlio-RR83 ended before its node started',
                          (Path(d) / sim.BRINGUP_LOG).read_text())
        self.assertEqual(order, ['start zenoh', 'start gazebo', 'spawned', 'settle zenoh+gazebo', 'start drive',
                                 'start frame_align', 'start dlio', 'settle drive+frame_align+dlio'])
        # The planner's waves come last, then RViz with the goals, all before the unpause.
        del order[:]
        full = plan + [(st[n], []) for n in ('mighty', 'rviz', 'goals')]
        with tempfile.TemporaryDirectory() as d, mock.patch.multiple(sim, **fakes), \
                contextlib.redirect_stdout(io.StringIO()):
            sim.bring_up(OFFICE, full, self.ctx, Path(d))
        self.assertEqual(order[-7:], ['start mighty', 'settle mighty', 'start rviz', 'start goals', 'settle rviz+goals',
                                      'graph', 'unpause'])
        # An edge missing from the graph (a declaration lost while starting) stops `up` before the
        # arm and the unpause.
        del order[:]
        armed = []
        lost = dict(fakes, check_graph=lambda session, stacks: order.append('graph') or [
            '/RR83/map_fusion_node is not subscribed to /RR83/elev/occ_2d_topic'])
        with tempfile.TemporaryDirectory() as d, mock.patch.multiple(sim, **lost), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(sim.SimError, r'(?s)graph check: 1 edge.*RR83/map_fusion_node is not subscribed '
                                                      r'to /RR83/elev/occ_2d_topic.*the world stays paused'):
                sim.bring_up(OFFICE, full, self.ctx, Path(d), before_unpause=lambda: armed.append(1) or 0)
        self.assertEqual((order[-1], armed), ('graph', []))

    def test_graph_edges(self):
        names = [s.name for s in sim.select_stacks(OFFICE, None)]
        edges = sim.graph_edges(OFFICE, names)
        self.assertIn(('/RR83/elev/occ_2d_topic', '/RR83/map_fusion_node'), edges)
        self.assertIn(('/JR81/cmd_vel_auto', '/JR81/safety_node'), edges)
        self.assertEqual(len(edges), len(sim.GRAPH_EDGES) * len(OFFICE.agents))
        # The fallback has no map_fusion and no elevation; without the planner, no planner edge.
        fb = sim.graph_edges(SMOKE, [s.name for s in sim.select_stacks(SMOKE, None)])
        self.assertFalse([e for e in fb if 'map_fusion_node' in e[1] or 'elevation' in e[1]])
        self.assertIn(('/JR81/live/occ_2d_topic', '/JR81/persistent_map_node'), fb)
        part = sim.graph_edges(OFFICE, ['zenoh', 'gazebo', 'drive', 'dlio'])
        self.assertEqual(sorted({t.split('/', 2)[2] for t, _ in part}), ['livox/lidar'])
        self.assertEqual(sim.graph_edges(OFFICE, ['zenoh']), [])
        # Every edge's ends run in the stacks it names (the subscriber's node is one of theirs).
        for topic, node, need, mode in sim.GRAPH_EDGES:
            self.assertTrue(set(need) <= set(sim.STACKS), topic)
            self.assertIn(mode, (None,) + sim.MAPPING_MODES, topic)

    def test_check_graph_reads_the_query(self):
        calls = []

        class P:
            returncode = 0
            stdout = ('@@ /JR81/livox/lidar /JR81/dlio_odom_node ok\n'
                      '@@ /RR83/livox/lidar /RR83/dlio_odom_node nosub\n')
        with mock.patch.object(sim, 'dexec', lambda name, args, **kw: calls.append((name, args, kw)) or P()):
            out = sim.check_graph(SMOKE, ['zenoh', 'gazebo', 'drive', 'dlio'])
        self.assertEqual(out, ['/RR83/dlio_odom_node is not subscribed to /RR83/livox/lidar'])
        name, args, kw = calls[0]
        self.assertEqual(name, sim.QUERY_CONTAINER)
        self.assertEqual(args[-2:], ['/JR81/livox/lidar|/JR81/dlio_odom_node', '/RR83/livox/lidar|/RR83/dlio_odom_node'])
        self.assertIn('SIM_GRAPH_PY', dict(kw['env']))
        # The query itself declares no publisher or subscription, so it has nothing to leave paced.
        self.assertNotIn('create_subscription', sim._GRAPH_PY)
        self.assertNotIn('create_publisher', sim._GRAPH_PY)
        self.assertIn('enable_rosout=False', sim._GRAPH_PY)
        P.stdout, P.returncode = '', 124
        with mock.patch.object(sim, 'dexec', lambda name, args, **kw: P()):
            self.assertEqual(sim.check_graph(SMOKE, ['gazebo', 'dlio']),
                             ['/JR81/dlio_odom_node on /JR81/livox/lidar: not checked (query exit 124)',
                              '/RR83/dlio_odom_node on /RR83/livox/lidar: not checked (query exit 124)'])

    def test_persistent_map_in_memory(self):
        # save_dir:=none in both modes: no run resumes another's map, nothing is written to
        # sim-mapping's layer, and no step has to delete anything there.
        for s in (SWAP, SMOKE):
            for p in sim.mapping_panes(s, self.ctx):
                if p.slug.startswith('persistent-map-'):
                    self.assertTrue(p.cmd.endswith(' save_dir:=none'), p.cmd)
                    self.assertIn('in memory', p.title)
        self.assertEqual(sim.STACKS['mapping'].generate, (sim.write_mapping_files,))
        self.assertFalse(hasattr(sim, 'clear_persistent_maps'))

    def test_check_elevation_launch(self):
        calls = []
        with mock.patch.object(sim, 'dexec', lambda name, args, **kw: calls.append((name, args, kw)) or FakeProc(
                0, '/ws/install/elevation_mapping_cupy/share/elevation_mapping_cupy/launch/'
                   'elevation_mapping_dlio.launch.py\n')), contextlib.redirect_stdout(io.StringIO()) as out:
            sim.check_elevation_launch(OFFICE, self.ctx)
        self.assertEqual(calls[0][:2], ('sim-elevation', ['bash', '-c', sim._ELEVATION_LAUNCH_SH, 'check', sim.SIM_WS]))
        self.assertIn('takes param_overlays and the grid topic arguments', out.getvalue())
        with mock.patch.object(sim, 'dexec', lambda name, args, **kw: FakeProc(1, '/x/e.launch.py declares no '
                                                                                     'param_overlays argument\n')):
            with self.assertRaisesRegex(sim.SimError, 'declares no param_overlays argument: the Orin would not pass '
                                                      'extra_cloud_exclude'):
                sim.check_elevation_launch(OFFICE, self.ctx)
        # The script itself, on launch files as the Orin's script greps them.
        with tempfile.TemporaryDirectory() as d:
            def check(text):
                f = Path(d) / 'install' / 'pkg' / 'share' / 'launch' / 'elevation_mapping_dlio.launch.py'
                if text is not None:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(text)
                p = subprocess.run(['bash', '-c', sim._ELEVATION_LAUNCH_SH, 'check', d], stdout=subprocess.PIPE,
                                   universal_newlines=True)
                return p.returncode, p.stdout.strip()
            self.assertEqual(check(None), (1, 'no elevation_mapping_dlio.launch.py under %s/install' % d))
            rc, out = check("DeclareLaunchArgument('occupancy_grid_topic', default_value='')\n")
            self.assertEqual(rc, 1)
            self.assertTrue(out.endswith('declares no param_overlays argument'), out)
            rc, out = check('DeclareLaunchArgument("param_overlays")\nDeclareLaunchArgument("occupancy_grid_topic")\n')
            self.assertEqual(rc, 0)
            self.assertTrue(out.endswith('elevation_mapping_dlio.launch.py'), out)

    def test_check_kernel_cache(self):
        # Only a warning: the run works, its first seconds of physics carry the compile burst.
        for rc, out, warned in ((0, '127\n', False), (0, '0\n', True), (1, '', True)):
            with mock.patch.object(sim, 'dexec', lambda name, args, **kw: FakeProc(rc, out)), \
                    contextlib.redirect_stdout(io.StringIO()) as o, contextlib.redirect_stderr(io.StringIO()) as e:
                sim.check_kernel_cache(OFFICE, self.ctx)
            self.assertEqual('cupy has no compiled kernels' in e.getvalue(), warned, (rc, out))
            self.assertEqual('cupy kernel cache: 127 compiled kernels' in o.getvalue(), not warned)
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, CUPY_CACHE_DIR=d)
            count = lambda: subprocess.run(['bash', '-c', sim._KERNEL_CACHE_SH], stdout=subprocess.PIPE, env=env,
                                           universal_newlines=True).stdout.strip()
            self.assertEqual(count(), '0')
            (Path(d) / 'a.cubin').write_text('')
            self.assertEqual(count(), '1')

    def test_fallback_up_needs_no_gpu(self):
        # mapping: fallback on a host without nvidia-smi: no gpu profile, no sim-elevation, and
        # gpu_available() is never asked.
        asked = {}
        with tempfile.TemporaryDirectory() as d:
            run, logs = Path(d) / 'run', Path(d) / 'logs'
            logs.mkdir()

            def no_gpu():
                asked['gpu'] = True
                return False

            def images(profiles, run_dir):
                asked['profiles'] = list(profiles)
                return OrderedDict((n, 'img') for n in sim.CONTAINERS if n != 'sim-elevation' or profiles)
            patches = {
                'gpu_available': no_gpu, 'compose_images': images,
                'check_images': lambda used: asked.setdefault('used', list(used)), 'image_provenance': lambda used: [],
                'mapping_mounts_problem': lambda pins, services: None, 'dlio_scripts_problem': lambda pins: None,
                'mapping_config_problem': lambda session, pins: None,
                'tmux_sessions': lambda: [], 'running_sim_containers': lambda: [], 'busy_containers': lambda n: [],
                'prepare_run_dir': lambda r: None,
                'sh': lambda args, **k: asked.setdefault('compose', list(args)),
                'compose_env': lambda run_dir: {}, 'wait_running': lambda s: None, 'ensure_run_dir': lambda r: None,
                'check_isolation': lambda: [], 'write_session_file': lambda s, r: None,
                'rotate_rtf_log': lambda r: None, 'clear_ros_logs': lambda n: None, 'make_log_dir': lambda: logs,
                'generate_steps': lambda stacks: [],
                'bring_up': lambda session, plan, ctx, log_dir, before_unpause=None: asked.setdefault(
                    'stacks', [s.name for s, _ in plan]) and (before_unpause() if before_unpause else 0),
                'arm_after_up': lambda session, log_dir, gate='dlio', run_dir=None: 0,
            }
            with mock.patch.multiple(sim, **patches), mock.patch.dict(os.environ, {'ROVER_SIM_RUN': str(run)}), \
                    mock.patch.object(sim.shutil, 'which', lambda tool: '/usr/bin/' + tool), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(sim.main(['up', 'smoke']), 0, err.getvalue())
                self.assertNotIn('gpu', asked)
                self.assertEqual(asked['profiles'], [])
                self.assertNotIn('sim-elevation', asked['used'])
                self.assertNotIn('--profile', asked['compose'])
                self.assertNotIn('elevation', asked['stacks'])
                # A gpu session on that host is refused before anything starts.
                asked.clear()
                self.assertEqual(sim.main(['up', 'office_explore']), 1)
                self.assertTrue(asked['gpu'])
                self.assertNotIn('compose', asked)
            self.assertIn('has mapping: gpu but this host has no nvidia-smi', err.getvalue())

    def test_up_runs_the_waves_before_arming(self):
        order = []
        with tempfile.TemporaryDirectory() as d:
            run, logs = Path(d) / 'run', Path(d) / 'logs'
            logs.mkdir()
            patches = {
                'compose_images': lambda profiles, run_dir: OrderedDict((n, 'img') for n in sim.CONTAINERS
                                                                        if n != 'sim-elevation'),
                'check_images': lambda used: None, 'image_provenance': lambda used: [],
                'mapping_mounts_problem': lambda pins, services: None, 'dlio_scripts_problem': lambda pins: None,
                'mapping_config_problem': lambda session, pins: None,
                'tmux_sessions': lambda: [], 'running_sim_containers': lambda: [], 'busy_containers': lambda n: [],
                'prepare_run_dir': lambda r: None, 'sh': lambda *a, **k: order.append('compose up'),
                'compose_env': lambda run_dir: {},
                'wait_running': lambda s: None, 'ensure_run_dir': lambda r: None, 'check_isolation': lambda: [],
                'write_session_file': lambda s, r: None, 'rotate_rtf_log': lambda r: None,
                'clear_ros_logs': lambda n: None, 'make_log_dir': lambda: logs,
                'generate_steps': lambda stacks: [lambda s, c: order.append('generate')],
                'bring_up': lambda session, plan, ctx, log_dir, before_unpause=None: order.append(
                    'bring_up ' + ','.join(s.name for s, _ in plan)) or (before_unpause() if before_unpause else 0),
                'arm_after_up': lambda session, log_dir, gate='dlio', run_dir=None: order.append('arm ' + gate) or 0,
            }
            with mock.patch.multiple(sim, **patches), mock.patch.dict(os.environ, {'ROVER_SIM_RUN': str(run)}), \
                    mock.patch.object(sim.shutil, 'which', lambda tool: '/usr/bin/' + tool), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                rc = sim.main(['up', 'odom_eval'])
        self.assertEqual(rc, 0, err.getvalue())
        # No planner: armed after the unpause, once DLIO tracks.
        self.assertEqual(order, ['compose up', 'generate', 'bring_up zenoh,gazebo,drive,frame_align,dlio,mapping',
                                 'arm dlio'])
        self.assertIn('mapping: fallback (NOT PARITY', err.getvalue())     # warned, not just printed


class Phase4Status(unittest.TestCase):
    def test_specs(self):
        # The grids only with --maps: the query's session ending stalled the loaded graph.
        self.assertEqual(sim.status_specs(['JR81'], ['mapping']), [])
        self.assertEqual(sim.status_specs(['JR81'], ['mapping'], maps=True),
                         ['grid:/JR81/occ_2d_topic', 'grid:/JR81/esdf_2d_topic', 'change:/JR81/persistent_map/grid'])
        self.assertEqual(sim.status_specs(['JR81'], ['dlio'], maps=True), ['odom:/JR81/dlio/odom_node/odom',
                                                                          'count:/JR81/dlio/odom_node/pointcloud/deskewed'])
        self.assertEqual(len(sim.status_specs(['JR81', 'RR83'], ['gazebo', 'dlio', 'drive', 'mapping'])), 2 * 5)
        self.assertEqual(len(sim.status_specs(['JR81', 'RR83'], ['gazebo', 'dlio', 'drive', 'mapping'], True)), 2 * 8)

    def test_map_lines(self):
        vals = sim.parse_status('@@ /JR81/occ_2d_topic 5.00 0.210 16\n@@ /JR81/esdf_2d_topic 5.00 2.400 16\n'
                                '@@ /JR81/persistent_map/grid 0.50 1.300 3\n@@ /RR83/occ_2d_topic 5.0 inf 16\n'
                                '@@ /RR83/esdf_2d_topic\n@@ /RR83/persistent_map/grid\n')
        self.assertEqual(sim.agent_lines(['JR81'], vals, {'mapping'}), ['JR81  '])     # not asked for
        lines = sim.agent_lines(['JR81', 'RR83'], vals, {'mapping'}, maps=True)
        self.assertEqual(lines[0], 'JR81  maps: occ_2d_topic 5.0 Hz (age 0.21 s), esdf_2d_topic 5.0 Hz (age 2.40 s, '
                                   'STALE), persistent_map/grid 0.5 Hz (age 1.30 s, on change)')
        self.assertEqual(lines[1], 'RR83  maps: occ_2d_topic 5.0 Hz, unstamped, esdf_2d_topic none, '
                                   'persistent_map/grid none (published on change)')
        # The persistent map is re-published only when it changed: an old one is not stale.
        old = sim.parse_status('@@ /JR81/persistent_map/grid 0.10 40.0 2\n')
        self.assertIn('persistent_map/grid 0.1 Hz (age 40.00 s, on change)',
                      sim.agent_lines(['JR81'], old, {'mapping'}, maps=True)[0])
        once = sim.parse_status('@@ /JR81/persistent_map/grid 0 1.84 1\n')    # one sample: no rate
        self.assertIn('persistent_map/grid once (age 1.84 s, on change)',
                      sim.agent_lines(['JR81'], once, {'mapping'}, maps=True)[0])
        both = sim.agent_lines(['JR81'], vals, {'drive', 'mapping'}, maps=True)[0]
        self.assertTrue(both.startswith('JR81  safety: no /JR81/mode'), both)

    def test_query_reads_grid_stamps_raw(self):
        world = {'pubs': {'/JR81/occ_2d_topic': {'hz': 20, 'stamp': 'now'},
                          '/JR81/persistent_map/grid': {'hz': 20, 'stamp': 0}}}
        specs = ['grid:/JR81/occ_2d_topic', 'grid:/JR81/persistent_map/grid', 'grid:/JR81/esdf_2d_topic']
        vals = sim.parse_status(run_with_fake_ros(sim._STATUS_PY, ['1.5', '0.5'] + specs, world))
        hz, age, n = vals['/JR81/occ_2d_topic']
        self.assertGreater(float(hz), 5)
        self.assertLess(float(age), 0.5)
        self.assertEqual(vals['/JR81/persistent_map/grid'][1], 'inf')
        self.assertEqual(vals['/JR81/esdf_2d_topic'], [])
        self.assertIn("raw=(k in RAW)", sim._STATUS_PY)                  # grids are never deserialised
        # A `change` grid is read like a grid, but the query does not wait for it: with only the
        # occupancy grid's window to fill, it ends long before its 5 s deadline.
        quiet = {'pubs': {'/JR81/occ_2d_topic': {'hz': 20, 'stamp': 'now'}}}
        t0 = time.time()
        vals = sim.parse_status(run_with_fake_ros(sim._STATUS_PY, ['5', '0.3', 'grid:/JR81/occ_2d_topic',
                                                                   'change:/JR81/persistent_map/grid'], quiet))
        self.assertLess(time.time() - t0, 4.0)
        self.assertEqual(vals['/JR81/persistent_map/grid'], [])
        self.assertGreater(float(vals['/JR81/occ_2d_topic'][0]), 5)
        changed = {'pubs': {'/JR81/persistent_map/grid': {'hz': 20, 'stamp': 'now'}}}
        vals = sim.parse_status(run_with_fake_ros(sim._STATUS_PY, ['1', '0.3', 'change:/JR81/persistent_map/grid'],
                                                  changed))
        self.assertLess(float(vals['/JR81/persistent_map/grid'][1]), 0.5)     # stamp read as a grid's

    def test_status_shows_maps_and_the_fallback_label(self):
        class Panes:
            returncode = 0
            stdout = 'rsim-mapping\tJR81\t0\tJR81 global_mapper\tdocker\nrsim-drive\tteleop\t0\tx\tdocker\n'

        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'session.json').write_text(json.dumps(SMOKE.to_json()))
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
                           '@@ /JR81/occ_2d_topic 5.0 0.2 15\n'))]
            for p in patches:
                p.start()
            try:
                texts = []
                for argv in (['status', '--maps'], ['status']):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        self.assertEqual(sim.cmd_status(sim.build_parser().parse_args(argv)), 0)
                    texts.append(out.getvalue())
            finally:
                for p in patches:
                    p.stop()
        self.assertEqual(asked, [sim.status_specs(['JR81', 'RR83'], {'drive', 'mapping'}, maps=True),
                                 sim.status_specs(['JR81', 'RR83'], {'drive', 'mapping'})])
        text, plain = texts
        self.assertIn('mapping: fallback (NOT PARITY', text)
        self.assertIn('maps from the fallback, NOT PARITY', text)
        self.assertIn('JR81  safety: no /JR81/mode (scripts/sim.py attach drive),  maps: occ_2d_topic 5.0 Hz '
                      '(age 0.20 s)', text)
        # Without --maps no grid is read, and the agent lines say where to get them.
        self.assertIn('; maps not read (status --maps)', plain)
        self.assertNotIn('maps: occ_2d_topic', plain)
        self.assertIn('mapping: fallback (NOT PARITY', plain)

    def test_mapping_label_only_when_mapping_runs(self):
        self.assertEqual(sim.mapping_label('fallback', False), 'not running (session mapping: fallback)')
        self.assertTrue(sim.mapping_label('fallback').startswith('fallback (NOT PARITY'))
        # odom_eval.yaml's own odometry-only line starts no mapping pane: no fallback label.
        out = subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up', 'odom_eval', '--stacks',
                              'zenoh,gazebo,drive,frame_align,dlio', '--dry-run'],
                             stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
        self.assertIn('mapping: not started (session mapping: fallback)', out)
        self.assertNotIn('NOT PARITY', out)
        self.assertNotIn('grid_relay', out)
        # status, with only the drive stack running.
        class Panes:
            returncode = 0
            stdout = 'rsim-drive\tteleop\t0\tx\tdocker\n'
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'session.json').write_text(json.dumps(SMOKE.to_json()))
            with mock.patch.multiple(sim, running_sim_containers=lambda: ['sim-drive'],
                                     docker_inspect=lambda kind, name: None, resolve_run_dir=lambda: Path(d),
                                     check_isolation=lambda: [], router_up=lambda: True,
                                     link_snapshot=lambda up: {'router': 1, 'p2p': 0, 'listen': 1, 'syn_sent': 0},
                                     tmux=lambda *a, **k: Panes(), status_values=lambda specs: {}), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                sim.cmd_status(sim.build_parser().parse_args(['status', '--maps']))
        self.assertIn('mapping: not running (session mapping: fallback)', out.getvalue())
        self.assertNotIn('NOT PARITY', out.getvalue())


# ---------------------------------------------------------------------------- Phase 5

def pinned_mighty_file(rel):
    """A file of the pinned mighty checkout (build_images.sh's), or None."""
    f = REPO / '.cache' / 'src' / ('mighty@' + sim.read_pins().get('MIGHTY_SHA', '')) / rel
    return f if f.is_file() else None


def goal_relay_module():
    path = REPO / 'scripts' / 'goal_relay.py'
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location('goal_relay', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                       # its pure part imports without rclpy
    return mod


def swap_session(**over):
    d = session(demo='swap', **over)
    d['agents'][0]['goals'] = [[2.0, 1.0]]
    d['agents'][1]['goals'] = [[0.0, 0.0]]
    return d


def robot_gate(waits):
    """The robot's own gate inside sim.verdict_gate (the pane goes on after its logged verdict)."""
    m = re.match(r'\{ g=/tmp/rsim-gate\.\$\$; (.*) 2>&1 \| tee \$g; ', waits)
    return m.group(1) if m else waits


def pane_line(p):
    """What the pane runs after the router's gate: its own gates (the robot's, unwrapped), then its command."""
    return (robot_gate(p.waits) + ' && ' if p.waits else '') + p.cmd


# Real pane output (Phase 4 log): the gate's warning in colour, the typed command line before it.
GATE_TIMEOUT = ("\x1b[33m[WARN] [1792407422.257998099] [wait_for_tf]: TIMEOUT after 60.0s; still missing "
                "[('JR81/map', 'JR81/odom')]; proceeding anyway (degraded)\x1b[0m\r\n")
GATE_OK = '[INFO] [1792407422.2] [wait_for_tf]: all transforms available; proceeding\r\n'
TYPED = "docker exec -it sim-mighty bash -c '... ros2 run mighty wait_for_tf.py $ROBOT_NAME/map $ROBOT_NAME/odom ...'\n"
MARKER = '[rover_sim] node starting at 2026-10-19T12:00:01+0000\r\n'


class Phase5Sessions(unittest.TestCase):
    def test_demo_and_planner(self):
        s = sim.validate_session(session())
        self.assertEqual((s.planner, s.demo), (True, 'none'))
        self.assertEqual(sim.validate_session(swap_session()).demo, 'swap')
        for bad, why in ((session(demo='tour'), "demo: 'tour' is not one of none, swap, explore"),
                         (session(planner='yes'), "planner: 'yes' is not true/false"),
                         (swap_session(planner=False), 'demo: swap needs the planner'),
                         (session(demo='swap'), 'demo: swap needs goals'),
                         (dict(swap_session(), demo='explore'), 'demo: explore takes no goals .*: JR81, RR83')):
            self.assertRegex(errors_of(bad), why)
        # An agent without goals in a swap stays put; one with goals is enough.
        d = swap_session()
        del d['agents'][1]['goals']
        self.assertEqual(sim.validate_session(d).agents[1].goals, ())
        j = sim.validate_session(swap_session(planner=True)).to_json()
        self.assertEqual((j['planner'], j['demo'], j['num_agents']), (True, 'swap', 83))
        self.assertNotIn('agents_override', j)

    def test_session_json_round_trip(self):
        for s in (SWAP, SMOKE, OFFICE, sim.load_session('odom_eval')):
            data = json.loads(json.dumps(dict(s.to_json(), started='2026-10-19T12:00:00+0000')))
            back = sim.session_from_json(data)
            self.assertEqual(back, s)
        over = sim.apply_agents_env(SWAP, 'JR81:jr,RR82:rr')
        self.assertEqual(sim.session_from_json(json.loads(json.dumps(over.to_json()))), over)
        with self.assertRaisesRegex(sim.SessionError, 'not a session file'):
            sim.session_from_json({'name': 'x'})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(sim.SimError, 'no running session'):
                sim.load_running_session(Path(d))
            sim.write_session_file(SMOKE, Path(d))
            self.assertEqual(sim.load_running_session(Path(d)), SMOKE)

    def test_shipped_sessions(self):
        smoke = SMOKE
        self.assertEqual((smoke.demo, smoke.auto_arm, smoke.planner), ('swap', True, True))
        jr, rr = smoke.agents
        # Each to 1 m beside the other's start (not onto it: it is parked there), mirrored, so the
        # two paths cross halfway.
        self.assertEqual((jr.goals, rr.goals), (((rr.x, rr.y + 1.0),), ((jr.x, jr.y + 1.0),)))
        self.assertEqual((SWAP.demo, SWAP.auto_arm), ('swap', True))
        self.assertEqual((OFFICE.demo, OFFICE.auto_arm, OFFICE.rviz), ('explore', True, False))
        oe = sim.load_session('odom_eval')
        self.assertEqual((oe.planner, oe.demo, oe.auto_arm), (False, 'none', True))
        # odom_eval.py --drive publishes cmd_vel_auto itself: no mpc there.
        self.assertNotIn('mighty', [s.name for s in sim.select_stacks(oe)])
        self.assertEqual([s.name for s in sim.select_stacks(OFFICE)][-2:], ['mighty', 'goals'])   # its first drive
        self.assertEqual([s.name for s in sim.select_stacks(SWAP)][-2:], ['mighty', 'goals'])
        # session.json carries what goal_relay.py reads.
        relay = goal_relay_module()
        if relay is not None:
            with tempfile.TemporaryDirectory() as d:
                sim.write_session_file(SWAP, Path(d))
                demo, goals = relay.load_session(str(Path(d) / sim.SESSION_FILE))
            self.assertEqual(demo, 'swap')
            self.assertEqual(goals, {a.name: [tuple(g) for g in a.goals] for a in SWAP.agents})


class AgentsEnv(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(sim.parse_agents_env(' JR81:jr, RR82:rr ,'), [('JR81', 'jr', None), ('RR82', 'rr', None)])
        self.assertEqual(sim.parse_agents_env('JR85:jr:1:-2.5:90'), [('JR85', 'jr', (1.0, -2.5, 90.0))])
        for bad, why in (('JR81', 'is not NAME:class'), ('JR81:jr:1:2', 'is not NAME:class'),
                         ('JR81:jr:a:2:3', 'must be numbers'), ('JR81:jr:nan:2:3', 'must be numbers'),
                         (' , ', 'names no agent')):
            with self.assertRaisesRegex(sim.SessionError, why, msg=bad):
                sim.parse_agents_env(bad)

    def test_fewer_take_the_sessions_places(self):
        s = sim.apply_agents_env(SWAP, 'JR81:jr,RR82:rr')
        self.assertEqual([(a.name, a.cls) for a in s.agents], [('JR81', 'jr'), ('RR82', 'rr')])
        for a, slot in zip(s.agents, SWAP.agents):
            self.assertEqual((a.x, a.y, a.yaw_deg, a.goals), (slot.x, slot.y, slot.yaw_deg, slot.goals))
        self.assertEqual((s.num_agents, s.agents_env, s.source), (82, 'JR81:jr,RR82:rr', SWAP.source))
        self.assertEqual(s.to_json()['agents_override'], 'JR81:jr,RR82:rr')
        # A pose of its own wins, and has no goals.
        s = sim.apply_agents_env(SWAP, 'JR81:jr,RR82:rr:1.0:1.0:0')
        self.assertEqual((s.agents[1].x, s.agents[1].y, s.agents[1].goals), (1.0, 1.0, ()))
        # validate_session's rules still hold.
        for bad, why in (('JR81:jr,RR81:rr', 'id 81 already used by JR81'), ('JR81:rr', 'needs the name prefix RR'),
                         ('JX81:jr', 'not a robot class')):
            with self.assertRaisesRegex(sim.SessionError, why):
                sim.apply_agents_env(SWAP, bad)

    def test_more_go_on_the_circle(self):
        names = ['JR81', 'RR82', 'JR83', 'RR84', 'JR85', 'RR86']
        s = sim.apply_agents_env(SWAP, ','.join('%s:%s' % (n, n[:2].lower()) for n in names))
        self.assertEqual([a.name for a in s.agents], names)
        self.assertEqual((s.agents[0].x, s.agents[0].y), (2.83, 2.83))          # where multi_agent_swap's first is
        boxes = world_boxes(REPO / 'src' / 'rover_sim' / 'worlds' / 'arena.world')
        for a in s.agents:
            self.assertAlmostEqual(math.hypot(a.x, a.y), 4.0, delta=0.01)
            facing = math.degrees(math.atan2(-a.y, -a.x))
            self.assertAlmostEqual(math.remainder(a.yaw_deg - facing, 360.0), 0.0, delta=0.1)
            self.assertEqual(a.goals, (tuple(sim.circle_goal('arena', (a.x, a.y), 6)),))   # past the far side
            self.assertGreater(min(box_distance(a.x, a.y, b) for b in boxes), 2.0, a.name)
        for k in range(1, sim.CIRCLE_MAX_AGENTS + 1):                         # 2.1 m apart or more
            pts = sim.circle_layout('arena', k)
            self.assertTrue(all(math.hypot(p[0] - q[0], p[1] - q[1]) > 2.0 for p in pts for q in pts if p is not q))
        self.assertIsNone(sim.circle_layout('arena', sim.CIRCLE_MAX_AGENTS + 1))
        # No swap goal on (or next to) another agent's start, for any count: mighty refuses a terminal
        # goal on an occupied cell. Four is multi_agent_swap's layout, goals included.
        for k in range(1, sim.CIRCLE_MAX_AGENTS + 1):
            pts = sim.circle_layout('arena', k)
            for p in pts:
                g = sim.circle_goal('arena', p, k)
                self.assertAlmostEqual(math.hypot(*g), 4.0, delta=0.01)
                self.assertGreaterEqual(min(math.hypot(g[0] - q[0], g[1] - q[1]) for q in pts),
                                        sim.SWAP_GOAL_CLEAR_M - 0.03, (k, p, g))   # poses and goals are cm-rounded
        four = sim.apply_agents_env(SWAP, 'JR81:jr,JR82:jr,RR83:rr,RR84:rr,JR85:jr')
        self.assertEqual(len(four.agents), 5)
        self.assertEqual([list(a.goals[0]) for a in SWAP.agents],
                         [sim.circle_goal('arena', (a.x, a.y), 4) for a in SWAP.agents])
        self.assertIsNone(sim.circle_layout('ACL_office_simple', 5))
        # Explore: no goals on the circle.
        explore = sim.validate_session(dict(session(world='arena', demo='explore')))
        self.assertTrue(all(not a.goals for a in sim.apply_agents_env(explore, 'JR81:jr,JR82:jr,JR83:jr').agents))

    def test_office_needs_poses_for_more(self):
        five = 'JR81:jr,JR82:jr,RR83:rr,RR84:rr,RR85:rr'
        with self.assertRaisesRegex(sim.SessionError, 'RR85 has no pose: session office_explore places 4 agent'
                                                      r'\(s\), and world ACL_office_simple has no default layout .* '
                                                      'give it one as RR85:rr:x:y:yaw_deg'):
            sim.apply_agents_env(OFFICE, five)
        s = sim.apply_agents_env(OFFICE, five + ':5.0:-2.0:90')
        self.assertEqual([(a.x, a.y) for a in s.agents[:4]], [(a.x, a.y) for a in OFFICE.agents])
        self.assertEqual((s.agents[4].x, s.agents[4].y, s.num_agents), (5.0, -2.0, 85))

    def test_up_and_dry_run(self):
        with mock.patch.dict(os.environ, {'ROVER_SIM_AGENTS': 'JR81:jr,RR82:rr'}):
            self.assertEqual([a.name for a in sim.load_up_session('multi_agent_swap').agents], ['JR81', 'RR82'])
        env = dict(os.environ)
        env.pop('ROVER_SIM_AGENTS', None)
        self.assertEqual(len(sim.load_up_session('multi_agent_swap').agents), 4)
        out = subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up', 'multi_agent_swap', '--dry-run'],
                             stdout=subprocess.PIPE, universal_newlines=True, check=True,
                             env=dict(env, ROVER_SIM_AGENTS='JR81:jr,RR82:rr')).stdout
        self.assertIn('agents from ROVER_SIM_AGENTS=JR81:jr,RR82:rr', out)
        self.assertIn('num_agents 82', out)
        self.assertEqual(out.count('frame_align_publisher.py'), 2)
        self.assertEqual(out.count('only_nodes:=mighty_node num_agents:=82'), 2)
        self.assertIn('  RR82  rr  id 82  pose (-2.83, 2.83, -45.0 deg)', out)
        self.assertNotIn('ROBOT_NAME=JR82', out)
        self.assertIn('/frame_align/JR81/JR82 (RR82)', out)                   # ego's prefix, the other's id


class MightyPanes(unittest.TestCase):
    ctx = sim.RunContext(run_dir=Path(sim.RUN_DIR_DEFAULT))

    def test_pane_lines_are_mighty_hw_sh(self):
        script = pinned_mighty_file('docker/mighty_hw.sh')
        if script is None:
            self.skipTest('no pinned mighty checkout')
        text = script.read_text()
        tf_gate = re.search(r"local tf_gate='([^']*)'", text).group(1)
        base = re.search(r"local launch_base='([^']*)'", text).group(1)
        titles = shlex.split(re.search(r'titles=\((.*)\)', text).group(1))
        lines = [c.replace('${tf_gate}', tf_gate).replace('${launch_base}', base)
                 for c in re.findall(r'"\$\(dx "(.*)"\)"', text)]
        self.assertEqual(len(lines), 4)
        panes = sim.mighty_panes(SWAP, self.ctx)
        for a in SWAP.agents:
            mine = [p for p in panes if p.window == a.name]
            self.assertEqual([p.title for p in mine], ['%s %s' % (a.name, t) for t in titles])
            # The robot's lines exactly, plus num_agents:= (the highest id).
            self.assertEqual([pane_line(p) for p in mine], [l + ' num_agents:=84' for l in lines])
        self.assertEqual(tf_gate, sim.MIGHTY_TF_GATE + ' && ')
        self.assertEqual(base, sim.MIGHTY_LAUNCH)
        # Its pane gate is the router's port, which every sim pane waits for too.
        self.assertIn('echo >/dev/tcp/127.0.0.1/${ZENOH_ROUTER_PORT}', text)

    def test_pane_lines_are_jr02s(self):
        """tools/reference/jr02 (tracked): JR02's own process list, names substituted."""
        top = (REPO / 'tools' / 'reference' / 'jr02' / 'processes' / 'docker_top_hw-mighty.txt').read_text()
        robot = dict((re.search(r'only_nodes:=(\S+)', l).group(1), l[l.index('ros2 launch'):].strip())
                     for l in top.splitlines() if 'onboard_mighty.launch.py' in l)
        self.assertEqual(sorted(robot), sorted(m.node for m in sim.MIGHTY_PANES))
        for p in sim.mighty_panes(SWAP, self.ctx):
            a = next(x for x in SWAP.agents if x.name == p.window)
            node = re.search(r'only_nodes:=(\S+)', p.cmd).group(1)
            self.assertEqual(p.cmd.replace('$ROBOT_NAME', a.name),
                             robot[node].replace('JR02', a.name) + ' num_agents:=%d' % SWAP.num_agents)

    def test_panes(self):
        panes = sim.mighty_panes(SWAP, self.ctx)
        self.assertEqual([(w, [p.slug for p in ps]) for w, ps in sim.tmux_windows(panes)],
                         [(a.name, ['mighty-node-' + a.name, 'odom-to-state-' + a.name, 'mpc-' + a.name,
                                    'goal-selector-' + a.name]) for a in SWAP.agents])
        by = {p.slug: p for p in panes}
        for a in SWAP.agents:
            for p in (by['mighty-node-' + a.name], by['mpc-' + a.name]):
                self.assertEqual((p.waits, p.strict_tf_gate), (sim.verdict_gate(sim.MIGHTY_TF_GATE), True))
            for p in (by['odom-to-state-' + a.name], by['goal-selector-' + a.name]):
                self.assertEqual((p.waits, p.strict_tf_gate, p.ready), ('', False, ''))
            self.assertEqual(by['mpc-' + a.name].ready, sim.MPC_READY)
            self.assertEqual(by['mighty-node-' + a.name].ready, '')
        for p in panes:
            a = next(x for x in SWAP.agents if x.name == p.window)
            self.assertEqual((p.container, p.env, p.gate), ('sim-mighty', sim.robot_env(a), True))
            self.assertIn(('ROBOT_NAME', a.name), p.env)
            self.assertNotIn('use_ground_robot', p.cmd)                # it switches the base config
            self.assertTrue(p.cmd.endswith(' num_agents:=84'))
            inner = inner_of(p)
            self.assertTrue(inner.endswith(sim.node_marker() + ' && ' + p.cmd))
            self.assertIn('until (exec 3<>/dev/tcp/sim-drive/7447', inner)
            subprocess.run(['bash', '-n', '-c', inner], check=True)
        # $ROBOT_NAME is the container's: the pane's env names the agent.
        p = by['mpc-RR83']
        env = dict(os.environ, **dict(p.env))
        out = subprocess.run(['bash', '-c', 'f() { printf "%s\\n" "$@"; }; f ' + p.cmd[len('ros2 launch '):]],
                             stdout=subprocess.PIPE, universal_newlines=True, env=env).stdout
        self.assertIn('namespace:=RR83', out.split('\n'))
        gate = subprocess.run(['bash', '-c', 'echo ' + robot_gate(p.waits)], stdout=subprocess.PIPE, universal_newlines=True,
                              env=dict(os.environ, **dict(p.env))).stdout
        self.assertEqual(gate, 'ros2 run mighty wait_for_tf.py RR83/map RR83/odom\n')
        # Fewer agents: the session's highest id.
        self.assertTrue(all(p.cmd.endswith(' num_agents:=83') for p in sim.mighty_panes(SMOKE, self.ctx)))

    def test_verdict_gate(self):
        # The robot's wait_for_tf.py gate, but its verdict decides, not its exit: a session that
        # fails to close after "proceeding" (rclpy.shutdown raised, 2026-10-20) must not stop the pane.
        lines = {'ok': '[INFO] [1.0] [wait_for_tf]: all transforms available; proceeding',
                 'degraded': ("\x1b[33m[WARN] [1.0] [wait_for_tf]: TIMEOUT after 60.0s; still missing x; "
                              "proceeding anyway (degraded)\x1b[0m"),
                 'none': '[INFO] [1.0] [wait_for_tf]: waiting for transforms'}
        for verdict, code, starts in (('ok', 0, True), ('ok', 1, True), ('degraded', 1, True), ('none', 1, False),
                                      ('none', 0, True)):
            cmd = 'bash -c %s' % shlex.quote('echo %s; exit %d' % (shlex.quote(lines[verdict]), code))
            r = subprocess.run(['bash', '-c', sim.verdict_gate(cmd) + ' && echo NODE'], stdout=subprocess.PIPE,
                               universal_newlines=True)
            self.assertEqual(('NODE' in r.stdout, r.returncode == 0), (starts, starts), (verdict, code, r.stdout))
            self.assertIn(lines[verdict], r.stdout)                      # the pane still shows the gate's output
            self.assertEqual('after its verdict' in r.stdout, code != 0 and starts, (verdict, code))
        self.assertTrue(sim.WAIT_FOR_TF_RE.search(lines['ok']) and sim.WAIT_FOR_TF_RE.search(lines['degraded']))

    def test_parity_contract(self):
        nodes = yaml.safe_load((JR_REF / 'manifest.yaml').read_text())['nodes']
        for name in ('mighty_node', 'goal_selector', 'mpc', 'convert_odom_to_state'):
            self.assertEqual(nodes[name]['container'], 'sim-mighty')
        allowed = yaml.safe_load((JR_REF / 'allowed_deviations.yaml').read_text())
        self.assertIn('num_agents', allowed['all']['params']['mighty_node'])
        self.assertEqual(sim.mighty_cmd('mpc', 84), sim.MIGHTY_LAUNCH + ' only_nodes:=mpc num_agents:=84')

    def test_launch_file_takes_these_arguments(self):
        launch = pinned_mighty_file('launch/onboard_mighty.launch.py')
        if launch is None:
            self.skipTest('no pinned mighty checkout')
        declared = set(re.findall(r"DeclareLaunchArgument\('(\w+)'", launch.read_text()))
        used = set(re.findall(r' (\w+):=', sim.mighty_cmd('mpc', 84)))
        self.assertEqual(used - declared, set())               # ros2 launch would silently ignore one
        self.assertIn('only_nodes', declared)
        keys = re.search(r'keyed = \{(.*?)\}', launch.read_text(), re.S).group(1)
        for m in sim.MIGHTY_PANES:
            self.assertIn("'%s'" % m.node, keys)
        # mpc's ready line is the one mpc_node logs after its NLP build.
        self.assertTrue(re.search(sim.MPC_READY, '[mpc_node-1] [INFO] [1.2] [JR81.mpc]: MPC controller started.'))


class TfGatesAndReady(unittest.TestCase):
    def test_node_started_and_gate_result(self):
        self.assertFalse(sim.node_started(TYPED))
        self.assertTrue(sim.node_started(TYPED + MARKER))
        self.assertFalse(sim.node_started(TYPED + MARKER, sim.MPC_READY))
        ready = '[mpc_node-1] [INFO] [1] [JR81.mpc]: MPC controller started.\n'
        self.assertTrue(sim.node_started(TYPED + MARKER + ready, sim.MPC_READY))
        # A restarted pane: the ready line must follow its last marker.
        self.assertFalse(sim.node_started(MARKER + ready + 'Ctrl-C\n' + MARKER, sim.MPC_READY))
        self.assertIsNone(sim.tf_gate_result(TYPED + MARKER))
        self.assertEqual(sim.tf_gate_result(TYPED + GATE_TIMEOUT + MARKER), 'degraded')
        self.assertEqual(sim.tf_gate_result(GATE_TIMEOUT + MARKER + GATE_OK + MARKER), 'ok')
        self.assertEqual(sim.tf_gate_result(GATE_OK + GATE_TIMEOUT), 'degraded')
        # mapping_ws's copy logs the same lines.
        copy = REPO / '.cache' / 'src' / ('mapping_ws@' + sim.read_pins().get('MAPPING_SHA', '')) / 'scripts' / \
            'wait_for_tf.py'
        mine = pinned_mighty_file('scripts/wait_for_tf.py')
        for f in (copy, mine):
            if f is not None and f.is_file():
                src = f.read_text()
                self.assertIn("'all transforms available; proceeding'", src)
                self.assertIn("proceeding anyway (degraded)'", src)

    def test_wait_nodes_started_waits_for_ready(self):
        st = sim.STACKS
        mpc = sim.Pane('m', 'mpc-JR81', 'sim-mighty', 'true', ready=sim.MPC_READY)
        plan = [(st['mighty'], [mpc])]
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'mighty').mkdir()
            log = Path(d) / 'mighty' / 'mpc-JR81.log'
            log.write_text(TYPED + MARKER)
            with mock.patch.object(sim, 'pane_commands', lambda: {('rsim-mighty', 'mpc-JR81'): 'docker'}), \
                    mock.patch.object(sim, 'SPAWN_PANE_GRACE_S', 0), mock.patch.object(sim.time, 'sleep', lambda s: None):
                self.assertEqual(sim.wait_nodes_started(plan, Path(d), timeout=0)[1:], ([], ['mighty/mpc-JR81']))
                log.write_text(TYPED + MARKER + 'MPC controller started.\n')
                self.assertEqual(sim.wait_nodes_started(plan, Path(d), timeout=0)[1:], ([], []))

    def test_settle_wave_stops_on_a_strict_gate(self):
        st = sim.STACKS
        strict = sim.Pane('p', 'mighty-node-JR81', 'sim-mighty', 'true', waits='gate', strict_tf_gate=True)
        loose = sim.Pane('g', 'global-mapper-JR81', 'sim-mapping', 'true', waits='gate')
        with tempfile.TemporaryDirectory() as d:
            for stack, slug in (('mighty', 'mighty-node-JR81'), ('mapping', 'global-mapper-JR81')):
                (Path(d) / stack).mkdir()
                (Path(d) / stack / (slug + '.log')).write_text(TYPED + GATE_TIMEOUT + MARKER)
            fakes = {'wait_nodes_started': lambda wave, log_dir: (61.0, [], []),
                     'wait_idle': lambda *a, **k: (4.0, (3.0, 80.0)), 'running_sim_containers': lambda: ['sim-drive'],
                     'router_cpu_seconds': lambda: 1.0}
            with mock.patch.multiple(sim, **fakes), contextlib.redirect_stdout(io.StringIO()):
                # The global_mapper gate always times out while paused: logged, not a failure.
                self.assertEqual(sim.settle_wave([(st['mapping'], [loose])], 4, Path(d)), [])
                ret = sim.settle_wave([(st['mighty'], [strict])], 4, Path(d))
            self.assertEqual(ret, [('mighty/mighty-node-JR81',
                                    'its TF gate timed out (wait_for_tf.py exits 0 and starts the node anyway)')])
            log = (Path(d) / sim.BRINGUP_LOG).read_text()
            self.assertIn('mapping/global-mapper-JR81: its wait_for_tf.py gate timed out and the node started '
                          'degraded\n', log)
            self.assertIn('mighty/mighty-node-JR81: its wait_for_tf.py gate timed out and the node started degraded '
                          '(strict: stopping)', log)
            (Path(d) / 'mighty' / 'mighty-node-JR81.log').write_text(TYPED + GATE_OK + MARKER)
            self.assertEqual(sim.degraded_tf_gates([(st['mighty'], [strict])], Path(d)), [])

    def test_start_stack_marks_strict_panes(self):
        calls = []

        class Done:
            stdout = '%1\n%2\n'
            returncode = 0
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(sim, 'tmux', lambda *a, **k: (calls.append(a), Done())[1]):
            sim.start_stack(sim.STACKS['mighty'], [sim.Pane('a', 'a', 'sim-mighty', 'true', strict_tf_gate=True),
                                                   sim.Pane('b', 'b', 'sim-mighty', 'true')], Path(d))
        self.assertIn(('set-option', '-p', '-t', '%1', '@rsim_tf_gate', 'strict'), calls)
        self.assertFalse([c for c in calls if '%2' in c and '@rsim_tf_gate' in c])


class RvizConfig(unittest.TestCase):
    ctx = sim.RunContext(run_dir=Path(sim.RUN_DIR_DEFAULT))

    def groups(self, s):
        doc = yaml.safe_load(sim.rviz_config(s))
        return doc, OrderedDict((g['Name'], g['Displays']) for g in doc['Visualization Manager']['Displays']
                                if g['Class'] == 'rviz_common/Group')

    def test_config(self):
        for s in (SWAP, SMOKE, OFFICE):
            text = sim.rviz_config(s)
            self.assertNotRegex(text, r'@[A-Z_]+@')
            self.assertTrue(text.startswith('# rover_sim RViz config for session %s (%s)' % (s.name, s.source)))
            doc, groups = self.groups(s)
            vm = doc['Visualization Manager']
            self.assertEqual(vm['Global Options']['Fixed Frame'], 'world')
            self.assertEqual(vm['Global Options']['Frame Rate'], 10)
            goal = [t for t in vm['Tools'] if t['Class'] == 'rviz_default_plugins/SetGoal']
            self.assertEqual([t['Topic']['Value'] for t in goal], ['/sim/goal'])
            self.assertEqual(list(groups), [a.name for a in s.agents])
            colors = set()
            for a in s.agents:
                displays = groups[a.name]
                topics = [d['Topic']['Value'] for d in displays if 'Topic' in d]
                self.assertTrue(all(t.startswith('/%s/' % a.name) for t in topics), topics)
                on = sorted(d['Topic']['Value'][len(a.name) + 2:] for d in displays if d['Enabled'] and 'Topic' in d)
                want = ['dlio/odom_node/pointcloud/deskewed', 'occ_2d_topic', 'persistent_map/grid', 'term_goal',
                        'traj_committed_colored'] + (['exploration/frontiers'] if s.demo == 'explore' else [])
                self.assertEqual(on, sorted(want), (s.name, a.name))
                axes = [d for d in displays if d['Class'] == 'rviz_default_plugins/Axes']
                self.assertEqual([d['Reference Frame'] for d in axes], ['%s/base_link' % a.name])
                for d in displays:
                    self.assertEqual(d['Enabled'], d['Value'], d['Name'])
                    if 'Topic' in d:
                        # the latched persistent map transient local, everything else volatile
                        self.assertEqual(d['Topic']['Durability Policy'],
                                         'Transient Local' if d['Topic']['Value'].endswith('persistent_map/grid')
                                         else 'Volatile', d['Name'])
                colors.add(next(d['Color'] for d in displays if d['Class'] == 'rviz_default_plugins/PointCloud2'))
                self.assertEqual(any('/elev/' in t for t in topics), s.mapping == 'gpu')
            self.assertEqual(len(colors), len(s.agents))
        # mighty's own names for what it publishes (mighty_node.cpp, goal_selector_node.cpp).
        src = pinned_mighty_file('src/mighty/mighty_node.cpp')
        gs = pinned_mighty_file('src/goal_selector/goal_selector_node.cpp')
        if src is not None and gs is not None:
            published = set(re.findall(r'"([\w/]+)"', src.read_text() + gs.read_text()))
            for name in ('traj_committed_colored', 'hgp_path_marker', 'point_G_term', 'term_goal',
                         'exploration/frontiers', 'exploration/current_goal'):
                self.assertIn(name, published)

    def test_view(self):
        self.assertEqual(sim.rviz_view(SWAP), (0.0, 0.0, 22.0))                # the arena's 16 m walls
        fx, fy, dist = sim.rviz_view(OFFICE)
        self.assertAlmostEqual(fx, sum(a.x for a in OFFICE.agents) / 4)
        self.assertGreater(dist, 1.25 * (max(a.x for a in OFFICE.agents) - min(a.x for a in OFFICE.agents)))

    def test_templates(self):
        self.assertEqual(sorted(p.name for p in sim.RVIZ_TEMPLATES.iterdir()), sorted(sim.RVIZ_PARTS))
        with self.assertRaisesRegex(sim.SimError, r'x.part: no value for @NS@'):
            sim.fill_template('a @NS@ b', {}, 'x.part')
        with tempfile.TemporaryDirectory() as d:
            for n in sim.RVIZ_PARTS:
                shutil.copy(str(sim.RVIZ_TEMPLATES / n), str(Path(d) / n))
            (Path(d) / 'footer.rviz.part').write_text((sim.RVIZ_TEMPLATES / 'footer.rviz.part').read_text()
                                                      + 'Extra: @OOPS@\n')
            with self.assertRaisesRegex(sim.SimError, 'footer.rviz.part: no value for @OOPS@'):
                sim.rviz_config(SWAP, Path(d))
            (Path(d) / 'agent.rviz.part').unlink()
            with self.assertRaisesRegex(sim.SimError, 'RViz template .*agent.rviz.part'):
                sim.rviz_config(SWAP, Path(d))

    def test_write_and_pane(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'rviz'
            out.mkdir()
            (out / 'stale.rviz').write_text('x')
            with contextlib.redirect_stdout(io.StringIO()):
                sim.write_rviz_config(SWAP, sim.RunContext(run_dir=Path(d)))
            self.assertEqual([p.name for p in out.iterdir()], ['session.rviz'])
            self.assertEqual((out / 'session.rviz').read_text(), sim.rviz_config(SWAP))
        s = sim.validate_session(session(rviz=True))
        (p,) = sim.rviz_panes(s, self.ctx)
        self.assertEqual((p.container, p.slug, p.window), ('sim-gazebo', 'rviz', 'rviz'))
        self.assertEqual(p.cmd, sim.pin_self('rest', self.ctx) + ' && rviz2 -d /rover_sim/run/rviz/session.rviz')
        self.assertIn(sim.RESTARTABLE['rviz'], p.cmd)
        subprocess.run(['bash', '-n', '-c', inner_of(p)], check=True)
        st = sim.STACKS['rviz']
        self.assertEqual((st.generate, st.wave, st.needs_display(s)), ((sim.write_rviz_config,), 5, True))


class GoalsStack(unittest.TestCase):
    ctx = sim.RunContext(run_dir=Path(sim.RUN_DIR_DEFAULT))

    def test_pane(self):
        (p,) = sim.goals_panes(SWAP, self.ctx)
        self.assertEqual((p.container, p.slug, p.window, p.env), ('sim-mighty', 'goal-relay', 'goals', ()))
        self.assertEqual(p.cmd, 'python3 /rover_sim/scripts/goal_relay.py --spawn /rover_sim/run/spawn.yaml '
                                '--session /rover_sim/run/session.json --mode swap')
        self.assertIn(sim.RESTARTABLE['goals'], p.cmd)
        subprocess.run(['bash', '-n', '-c', inner_of(p)], check=True)
        self.assertEqual([sim.goal_mode(s) for s in (SWAP, OFFICE, sim.validate_session(session()))],
                         ['swap', 'explore', 'rviz'])
        relay = goal_relay_module()
        if relay is None:
            self.skipTest('no scripts/goal_relay.py')
        args = relay.build_parser().parse_args(shlex.split(p.cmd)[2:])
        self.assertEqual((args.mode, args.spawn, args.session), ('swap', '/rover_sim/run/spawn.yaml',
                                                                 '/rover_sim/run/session.json'))
        self.assertEqual(relay.SESSION_DEMO_MODE, dict(sim.GOAL_MODES))
        self.assertEqual(relay.GOAL_TOPIC, sim.GOAL_TOPIC)
        self.assertEqual(relay.FIRST_DRIVE_M, sim.GOAL_RELAY_FIRST_DRIVE_M)
        self.assertEqual((relay.LEAVE_GAP_S, relay.LEAVE_PAUSE_S), (sim.STATUS_LEAVE_GAP_S, sim.STATUS_LEAVE_PAUSE_S))

    def test_check_goal_relay(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sim, 'REPO', Path(d)):
            with self.assertRaisesRegex(sim.SimError, 'no scripts/goal_relay.py'):
                sim.check_goal_relay(SWAP, self.ctx)
            (Path(d) / 'scripts').mkdir()
            (Path(d) / 'scripts' / 'goal_relay.py').write_text('')
            sim.check_goal_relay(SWAP, self.ctx)

    def test_up_starts_goals_paused(self):
        """goal_relay.py joins in the last paused wave and waits for AUTO itself: `up` starts nothing
        once physics runs, whether or not it arms."""
        order = []

        def up(argv, arm_rc):
            del order[:]
            with tempfile.TemporaryDirectory() as d:
                run, logs = Path(d) / 'run', Path(d) / 'logs'
                logs.mkdir()
                patches = {
                    'compose_images': lambda profiles, run_dir: OrderedDict((n, 'img') for n in sim.CONTAINERS),
                    'check_images': lambda used: None, 'image_provenance': lambda used: [],
                    'gpu_available': lambda: True,
                    'mapping_mounts_problem': lambda pins, services: None, 'dlio_scripts_problem': lambda pins: None,
                    'mapping_config_problem': lambda session, pins: None,
                    'tmux_sessions': lambda: [], 'running_sim_containers': lambda: [], 'busy_containers': lambda n: [],
                    'prepare_run_dir': lambda r: None, 'sh': lambda *a, **k: None, 'compose_env': lambda run_dir: {},
                    'wait_running': lambda s: None, 'ensure_run_dir': lambda r: None, 'check_isolation': lambda: [],
                    'write_session_file': lambda s, r: None, 'rotate_rtf_log': lambda r: None,
                    'clear_ros_logs': lambda n: None, 'make_log_dir': lambda: logs, 'generate_steps': lambda stacks: [],
                    'bring_up': lambda session, plan, ctx, log_dir, before_unpause=None: order.append(
                        'bring_up ' + ','.join(s.name for s, _ in plan)) or (before_unpause() if before_unpause else 0),
                    'arm_after_up': lambda session, log_dir, gate='dlio', run_dir=None: order.append('arm ' + gate) or arm_rc,
                }
                with mock.patch.multiple(sim, **patches), mock.patch.dict(os.environ, {'ROVER_SIM_RUN': str(run)}), \
                        mock.patch.object(sim.shutil, 'which', lambda tool: '/usr/bin/' + tool), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return sim.main(argv)
        # A planner session arms inside bring_up, before the unpause, without the DLIO gate but
        # recording each agent's DLIO session (gate session), so a later re-arm only warns on drift.
        self.assertEqual(up(['up', 'multi_agent_swap'], 0), 0)
        self.assertEqual(order, ['bring_up zenoh,gazebo,drive,frame_align,dlio,mapping,elevation,mighty,goals',
                                 'arm session'])
        self.assertEqual(up(['up', 'multi_agent_swap'], 1), 1)
        self.assertEqual(order[-1], 'arm session')
        self.assertEqual(up(['up', 'multi_agent_swap'], 130), 130)         # Ctrl-C while arming: nothing more
        self.assertEqual(order[-1], 'arm session')
        self.assertEqual(up(['up', 'multi_agent_swap', '--no-arm'], 0), 0)
        self.assertEqual(order, ['bring_up zenoh,gazebo,drive,frame_align,dlio,mapping,elevation,mighty,goals'])
        self.assertEqual(up(['up', 'office_explore'], 0), 0)                # explore: goal_relay's first drive
        self.assertEqual(order, ['bring_up zenoh,gazebo,drive,frame_align,dlio,mapping,elevation,mighty,goals',
                                 'arm session'])


class StartCommand(unittest.TestCase):
    def start(self, argv, isolation=(), project=('sim-drive', 'sim-gazebo', 'sim-mighty'), s=SWAP, nodes=(1.0, [], [])):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            run, logs = Path(d) / 'run', Path(d) / 'logs'
            run.mkdir()
            (logs / '20261019_120000' / 'rviz').mkdir(parents=True)
            (logs / '20261019_120000' / 'rviz' / 'rviz.log').write_text('old run\n')
            (logs / 'latest').symlink_to('20261019_120000')
            (run / 'session.json').write_text(json.dumps(s.to_json()))
            patches = {'running_sim_containers': lambda: list(project), 'resolve_run_dir': lambda: run,
                       'check_isolation': lambda: list(isolation), 'tmux_sessions': lambda: ['rsim-mighty'],
                       'LOG_ROOT': logs, 'grant_x_access': lambda: calls.append('xhost'),
                       'stop_processes': lambda name, **kw: calls.append(('stop', name, kw['match'])) or 1,
                       'start_stack': lambda st, panes, log_dir: calls.append(
                           ('start', st.name, [p.slug for p in panes], log_dir)) or 'rsim-' + st.name,
                       'wait_nodes_started': lambda plan, log_dir, timeout: nodes}
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.multiple(sim, **patches), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = sim.main(argv)
            files = sorted(str(p.relative_to(d)) for p in Path(d).rglob('*') if p.is_file())
        return rc, calls, out.getvalue() + err.getvalue(), files

    def test_rviz(self):
        rc, calls, text, files = self.start(['start', 'rviz'])
        self.assertEqual(rc, 0, text)
        self.assertEqual(calls[0], ('stop', 'sim-gazebo', 'rviz2 -d /rover_sim/run/rviz/session.rviz'))
        self.assertEqual(calls[1], 'xhost')
        self.assertEqual(calls[2][:3], ('start', 'rviz', ['rviz']))
        self.assertTrue(str(calls[2][3]).endswith('20261019_120000'))       # the running session's log dir
        self.assertIn('run/rviz/session.rviz', files)                      # the session has rviz: false
        self.assertTrue([f for f in files if re.match(r'logs/20261019_120000/rviz/rviz\.\d{8}_\d{6}\.log$', f)])
        self.assertIn('joins the running graph', text)

    def test_goals_and_refusals(self):
        rc, calls, text, _ = self.start(['start', 'goals'])
        self.assertEqual(rc, 0, text)
        self.assertEqual([c[:3] for c in calls], [('stop', 'sim-mighty', '/goal_relay.py '),
                                                  ('start', 'goals', ['goal-relay'])])
        rc, calls, text, _ = self.start(['start', 'goals'], s=sim.load_session('odom_eval'))
        self.assertEqual((rc, calls), (1, []))
        self.assertIn('runs no planner', text)
        rc, calls, text, _ = self.start(['start', 'rviz'], isolation=['network rover-sim is not internal'])
        self.assertEqual((rc, calls), (3, []))                               # no ROS session at all
        self.assertIn('isolation: BROKEN', text)
        rc, calls, text, _ = self.start(['start', 'rviz'], project=('sim-drive',))
        self.assertEqual((rc, calls), (1, []))
        self.assertIn('sim-gazebo of project rover-sim-stack is not running', text)
        rc, calls, text, _ = self.start(['start', 'goals'], nodes=(60.0, ['goals/goal-relay'], []))
        self.assertEqual(rc, 1)
        self.assertIn('ended before starting their node', text)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                sim.build_parser().parse_args(['start', 'mighty'])        # only the restartable ones


class Phase5Status(unittest.TestCase):
    def test_specs_and_lines(self):
        self.assertEqual(sim.status_specs(['JR81'], ['mighty']), ['twist:/JR81/cmd_vel_auto', 'bool:/JR81/map_hold'])
        vals = sim.parse_status('@@ /JR81/cmd_vel_auto 30.00 90 0.420 -0.100\n@@ /JR81/map_hold false\n'
                                '@@ /RR83/cmd_vel_auto\n@@ /RR83/map_hold true\n')
        lines = sim.agent_lines(['JR81', 'RR83', 'RR84'], vals, {'mighty'})
        self.assertEqual(lines[0], 'JR81  planner: cmd_vel_auto 30 Hz (v 0.42 m/s, w -0.10 rad/s), map_hold off')
        self.assertEqual(lines[1], 'RR83  planner: cmd_vel_auto: none in 8 s (mpc running? scripts/sim.py attach '
                                   'mighty), map_hold ON: mighty holds the agent, its maps are stale')
        self.assertIn('no map_hold in 8 s (mighty_node running?', lines[2])
        both = sim.agent_lines(['JR81'], dict(vals, **sim.parse_status('@@ /JR81/mode AUTO\n')), {'drive', 'mighty'})[0]
        self.assertTrue(both.startswith('JR81  safety AUTO,  planner: cmd_vel_auto 30 Hz'), both)

    def test_query_reads_twist_and_bool(self):
        world = {'pubs': {'/JR81/cmd_vel_auto': {'hz': 30, 'v': 0.25, 'w': 0.5},
                          '/JR81/map_hold': {'hz': 5, 'value': False}}}
        specs = ['twist:/JR81/cmd_vel_auto', 'bool:/JR81/map_hold', 'twist:/RR83/cmd_vel_auto', 'bool:/RR83/map_hold']
        vals = sim.parse_status(run_with_fake_ros(sim._STATUS_PY, ['1.5', '0.5'] + specs, world))
        hz, n, v, w = vals['/JR81/cmd_vel_auto']
        self.assertGreater(float(hz), 10)
        self.assertEqual((float(v), float(w)), (0.25, 0.5))
        self.assertEqual(vals['/JR81/map_hold'], ['false'])
        self.assertEqual((vals['/RR83/cmd_vel_auto'], vals['/RR83/map_hold']), ([], []))
        # A held agent's map_hold is one value, like the mode: the query does not wait for more.
        held = {'pubs': {'/JR81/map_hold': {'hz': 50, 'value': True}}}
        t0 = time.time()
        vals = sim.parse_status(run_with_fake_ros(sim._STATUS_PY, ['5', '0.3', 'bool:/JR81/map_hold'], held))
        self.assertLess(time.time() - t0, 4.0)
        self.assertEqual(vals['/JR81/map_hold'], ['true'])

    def test_status_shows_the_planner_and_a_degraded_gate(self):
        class Panes:
            returncode = 0
            stdout = ('rsim-mighty\tJR81\t0\tJR81 MIGHTY planner\tdocker\tmighty-node-JR81\tstrict\n'
                      'rsim-mighty\tJR81\t2\tJR81 MPC\tdocker\tmpc-JR81\tstrict\n'
                      'rsim-mighty\tJR81\t1\tJR81 convert_odom_to_state\tdocker\todom-to-state-JR81\t\n'
                      'rsim-drive\tteleop\t0\tx\tdocker\tteleop-safety-JR81\t\n')
        with tempfile.TemporaryDirectory() as d:
            run, logs = Path(d) / 'run', Path(d) / 'logs'
            run.mkdir()
            (logs / 'x' / 'mighty').mkdir(parents=True)
            (logs / 'latest').symlink_to('x')
            (logs / 'x' / 'mighty' / 'mighty-node-JR81.log').write_text(TYPED + GATE_TIMEOUT + MARKER)
            (logs / 'x' / 'mighty' / 'mpc-JR81.log').write_text(TYPED + GATE_OK + MARKER)
            (run / 'session.json').write_text(json.dumps(SMOKE.to_json()))
            asked = []
            with mock.patch.multiple(sim, running_sim_containers=lambda: ['sim-drive'], LOG_ROOT=logs,
                                     docker_inspect=lambda kind, name: None, resolve_run_dir=lambda: run,
                                     check_isolation=lambda: [], router_up=lambda: True,
                                     link_snapshot=lambda up: {'router': 1, 'p2p': 0, 'listen': 1, 'syn_sent': 0},
                                     tmux=lambda *a, **k: Panes(), status_values=lambda specs: asked.append(specs) or
                                     sim.parse_status('@@ /JR81/cmd_vel_auto 30.0 90 0.0 0.0\n'
                                                      '@@ /JR81/map_hold false\n')), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(sim.cmd_status(sim.build_parser().parse_args(['status'])), 0)
        text = out.getvalue()
        self.assertEqual(asked, [sim.status_specs(['JR81', 'RR83'], {'drive', 'mighty'})])
        self.assertIn('JR81 MIGHTY planner  (TF gate TIMED OUT: the node started without <ns>/map -> <ns>/odom)', text)
        self.assertIn('JR81 MPC  (TF gate passed)', text)
        self.assertIn('JR81 convert_odom_to_state\n', text)
        self.assertIn("planner: mpc's cmd_vel_auto, mighty_node's map_hold", text)
        self.assertIn('JR81  safety: no /JR81/mode (scripts/sim.py attach drive),  planner: cmd_vel_auto 30 Hz '
                      '(v 0.00 m/s, w 0.00 rad/s), map_hold off', text)


class ArmLeave(unittest.TestCase):
    def test_arm_query_leaves_one_entity_at_a_time(self):
        world = safety_world(('JR81', 'DISARMED', 'now', True), ('RR83', 'AUTO', 'now', True))
        err = []
        t0 = time.time()
        args = sim.arm_args('auto_mode', 'dlio', 1.5, ['JR81', 'RR83'])[4:]
        out = run_with_fake_ros(sim._ARM_PY, args, world, err=err)
        took = time.time() - t0
        self.assertEqual(sorted(sim.parse_arm(out)), ['JR81', 'RR83'])
        left = [l for l in err if l.startswith(('LEFT', 'DESTROYED'))]
        self.assertEqual(left, ['LEFT /JR81/mode', 'LEFT /JR81/dlio/odom_node/odom',
                                'LEFT /JR81/dlio/odom_node/pointcloud/deskewed', 'LEFT /JR81/dlio/session',
                                'LEFT /RR83/mode', 'LEFT /RR83/dlio/odom_node/odom',
                                'LEFT /RR83/dlio/odom_node/pointcloud/deskewed', 'LEFT /RR83/dlio/session',
                                'LEFT client /JR81/auto_mode', 'LEFT client /RR83/auto_mode',
                                "DESTROYED with 0 subscription(s) left, [('enable_rosout', False), "
                                "('start_parameter_services', False)]"])
        self.assertGreater(took, 10 * sim.STATUS_LEAVE_GAP_S + sim.STATUS_LEAVE_PAUSE_S)
        self.assertIn('LEAVE_GAP_S, LEAVE_PAUSE_S = %r, %r\n' % (sim.STATUS_LEAVE_GAP_S, sim.STATUS_LEAVE_PAUSE_S),
                      sim._ARM_PY)


class Phase5DryRuns(unittest.TestCase):
    def dry(self, name, *extra):
        env = dict(os.environ)
        env.pop('ROVER_SIM_AGENTS', None)
        return subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up', name, '--dry-run'] + list(extra),
                              stdout=subprocess.PIPE, universal_newlines=True, check=True, env=env).stdout

    def check_planner(self, out, s):
        for a in s.agents:
            self.assertIn('  window %s\n    [mighty-node-%s] %s MIGHTY planner\n' % (a.name, a.name, a.name), out)
            for slug in ('odom-to-state-', 'mpc-', 'goal-selector-'):
                self.assertIn('[%s%s]' % (slug, a.name), out)
            self.assertIn('-e ROBOT_NAME=%s sim-mighty bash -c' % a.name, out)
        self.assertEqual(out.count('ros2 launch mighty onboard_mighty.launch.py'), 4 * len(s.agents))
        self.assertEqual(out.count(' num_agents:=%d\'' % s.num_agents), 4 * len(s.agents))
        self.assertEqual(out.count(sim.MIGHTY_TF_GATE), 2 * len(s.agents))
        self.assertEqual(out.count('(ready: MPC controller started\\.)'), len(s.agents))
        self.assertIn('stack mighty -> tmux session rsim-mighty', out)
        # The planner's wave is the last before the unpause, after mapping.
        order = [out.index(t) for t in ('stack mapping ->', 'stack mighty ->', 'after the last wave: unpause')]
        self.assertEqual(order, sorted(order))

    def test_multi_agent_swap(self):
        out = self.dry('multi_agent_swap')
        self.check_planner(out, SWAP)
        self.assertIn('planner on, demo swap, num_agents 84', out)
        order = [out.index(t) for t in ('stack mighty ->', 'stack goals -> tmux session rsim-goals',
                                        '[goal-relay] goal_relay --mode swap',
                                        'after the last wave, before the unpause: arm JR81, JR82, RR83, RR84',
                                        'after the last wave: unpause',
                                        'goal_relay.py drives each agent in AUTO 1.2 m ahead in JOY mode first, then '
                                        'starts the swap\n')]
        self.assertEqual(order, sorted(order))
        self.assertIn('goal_relay.py --spawn /rover_sim/run/spawn.yaml --session /rover_sim/run/session.json '
                      '--mode swap', out)
        self.assertNotIn('stack rviz', out)
        self.assertIn('JR81  jr  id 81  pose (2.83, 2.83, -135.0 deg)  calibrated from JR02, lidar at 0.36 m  '
                      'goals (-1.69, -3.63)', out)
        # Without arming, goal_relay.py holds the first drive and the swap until scripts/sim.py arm.
        self.assertIn('goal_relay.py drives each agent in AUTO 1.2 m ahead in JOY mode first, then starts the swap '
                      '(after scripts/sim.py arm)', self.dry('multi_agent_swap', '--no-arm'))

    def test_office_explore_and_smoke(self):
        office = self.dry('office_explore')
        self.check_planner(office, OFFICE)
        self.assertIn('demo explore', office)
        self.assertIn('[goal-relay] goal_relay --mode explore (first drive, then RViz /sim/goal only)', office)
        smoke = self.dry('smoke')
        self.check_planner(smoke, SMOKE)
        self.assertIn('[goal-relay] goal_relay --mode swap', smoke)
        self.assertNotIn('onboard_mighty', self.dry('odom_eval'))

    def test_rviz_session(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'rv.yaml'
            data = yaml.safe_load((REPO / 'config' / 'sessions' / 'office_explore.yaml').read_text())
            data.update(name='rv', rviz=True)
            f.write_text(yaml.safe_dump(data))
            out = self.dry(str(f))
        self.assertIn('rviz: /rover_sim/run/rviz/session.rviz from rviz/templates/ on $DISPLAY', out)
        self.assertIn('write_rviz_config', out)
        order = [out.index(t) for t in ('stack mighty ->', 'stack rviz ->', 'stack goals ->', '--mode explore',
                                        'after the last wave: unpause')]
        self.assertEqual(order, sorted(order))
        self.assertIn("sim-gazebo bash -c '", out[out.index('[rviz]'):])
        self.assertIn('rviz2 -d /rover_sim/run/rviz/session.rviz', out)

    def test_up_rviz_flag(self):
        # --rviz is the session's rviz: true for one run: the same stack, last paused wave.
        out = self.dry('multi_agent_swap', '--rviz')
        self.assertIn('gui off, mapping gpu, rviz on', out)
        order = [out.index(t) for t in ('stack mighty ->', 'wave 5: rviz, goals', 'stack rviz ->', 'stack goals ->',
                                        'after the last wave: unpause')]
        self.assertEqual(order, sorted(order))
        self.assertIn('rviz2 -d /rover_sim/run/rviz/session.rviz', out)
        self.assertNotIn('stack rviz', self.dry('multi_agent_swap'))


def load_sim_viz():
    scripts = str(REPO / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import sim_viz
    return sim_viz


class VizCommand(unittest.TestCase):
    """`sim.py viz`: sim_viz.py in its own sim container (profile viz), never on the host."""

    def viz(self, argv, isolation=(), project=('sim-drive', 'sim-gazebo', 'sim-mighty'), s=SWAP, display=':1',
            spawn=True, nodes=(2.0, [], [])):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            run, logs = Path(d) / 'run', Path(d) / 'logs'
            run.mkdir()
            (logs / '20261019_120000').mkdir(parents=True)
            (logs / 'latest').symlink_to('20261019_120000')
            (run / 'session.json').write_text(json.dumps(s.to_json()))
            if spawn:
                (run / 'spawn.yaml').write_text('agents: []\n')
            up = list(project)

            def stop(name, **kw):
                calls.append(('stop', name, kw.get('match'), kw.get('grace_int')))
                return 1

            def compose(args, **kw):
                calls.append(('sh', list(args)))
                if 'up' in args:
                    up.append('sim-viz')
            patches = {'running_sim_containers': lambda: list(up), 'resolve_run_dir': lambda: run,
                       'check_isolation': lambda: calls.append('isolation') or list(isolation),
                       'tmux_sessions': lambda: ['rsim-mighty', 'rsim-viz'], 'LOG_ROOT': logs,
                       'grant_x_access': lambda: calls.append('xhost'), 'stop_processes': stop,
                       'tmux': lambda *a, **k: calls.append(('tmux',) + a),
                       'sh': compose, 'compose_env': lambda run_dir: {}, 'wait_running': lambda n: calls.append(
                           ('wait', list(n))),
                       'start_stack': lambda st, panes, log_dir: calls.append(('start', st.name, panes)) or 'rsim-viz',
                       'wait_nodes_started': lambda plan, log_dir, timeout: nodes}
            env = {'DISPLAY': display} if display else {}
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.multiple(sim, **patches), mock.patch.dict(os.environ, env), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                if not display:
                    os.environ.pop('DISPLAY', None)
                rc = sim.main(argv)
        return rc, calls, out.getvalue() + err.getvalue()

    def test_start(self):
        rc, calls, text = self.viz(['viz', '--agents', 'RR83', '--layers', 'default,map', '--follow'])
        self.assertEqual(rc, 0, text)
        kinds = [c if isinstance(c, str) else c[0] for c in calls]
        # The old viewer leaves first, then compose starts sim-viz alone, isolation is checked on it, X
        # granted, and only then the pane joins the graph.
        self.assertEqual(kinds, ['tmux', 'sh', 'wait', 'isolation', 'xhost', 'start'])
        compose = calls[1][1]
        self.assertEqual(compose[-3:], ['up', '-d', 'sim-viz'])
        self.assertIn('--profile', compose)
        self.assertEqual(compose[compose.index('--profile') + 1], 'viz')
        self.assertNotIn('gpu', compose)
        self.assertEqual(calls[2], ('wait', ['sim-viz']))
        pane = calls[-1][2][0]
        self.assertEqual((calls[-1][1], pane.container, pane.slug), ('viz', 'sim-viz', 'viz'))
        self.assertEqual(pane.cmd, 'python3 /rover_sim/scripts/sim_viz.py --spawn /rover_sim/run/spawn.yaml '
                                   '--agents RR83 --layers default,map --follow')
        line = sim.pane_command(pane)
        argv = shlex.split(line)
        self.assertEqual(argv[:4], ['docker', 'exec', '-it', 'sim-viz'])     # in its container, never the host
        self.assertNotIn('--network', line)
        self.assertIn('until (exec 3<>/dev/tcp/sim-drive/7447', argv[-1])     # gated on the router
        self.assertIn('joins the running graph', text)

    def test_snapshot(self):
        def run(rc=0, **kw):
            execs = []

            def dexec(name, args, check=True, timeout=None, **k):
                execs.append((name, list(args), timeout))
                return types.SimpleNamespace(returncode=rc, stdout='sim_viz: wrote it\n', stderr='')
            with mock.patch.object(sim, 'dexec', dexec):
                rc, calls, text = self.viz(['viz', '--snapshot', '/tmp/x/out.png', '--snapshot-after', '4',
                                            '--agents', 'JR81', '--follow'], display='', **kw)
            return rc, calls, text, execs
        rc, calls, text, execs = run()
        self.assertEqual(rc, 0, text)                                       # no DISPLAY needed
        kinds = [c if isinstance(c, str) else c[0] for c in calls]
        # sim-viz up (no viewer to stop: none ran), isolation, the exec, then docker cp to the host.
        self.assertEqual(kinds, ['sh', 'wait', 'isolation', 'sh'])
        self.assertEqual(calls[0][1][-3:], ['up', '-d', 'sim-viz'])
        self.assertEqual(calls[-1][1], ['docker', 'cp', 'sim-viz:/tmp/sim_viz_snapshot.png', '/tmp/x/out.png'])
        ((name, argv, timeout),) = execs
        self.assertEqual((name, argv[:2]), ('sim-viz', ['bash', '-c']))
        self.assertTrue(argv[2].startswith('source /opt/ros/humble/setup.bash && exec python3 '
                                           '/rover_sim/scripts/sim_viz.py --spawn /rover_sim/run/spawn.yaml'), argv[2])
        self.assertTrue(argv[2].endswith('--agents JR81 --follow --snapshot /tmp/sim_viz_snapshot.png --snapshot-after 4.0'),
                        argv[2])
        self.assertGreaterEqual(timeout, 4 + sim.VIZ_STOP_GRACE_S)
        self.assertIn('sim_viz: wrote it', text)
        # A running sim-viz is used as it is (a recreate would stop its viewer); broken isolation: no exec.
        rc, calls, text, execs = run(project=('sim-drive', 'sim-viz'))
        self.assertEqual([c if isinstance(c, str) else c[0] for c in calls], ['isolation', 'sh'])
        rc, calls, text, execs = run(isolation=['network rover-sim is not internal'])
        self.assertEqual((rc, execs), (3, []))
        self.assertIn('not taking the snapshot', text)
        rc, calls, text, execs = run(rc=1)
        self.assertEqual(rc, 1)
        self.assertNotIn('cp', [a for c in calls if not isinstance(c, str) for a in c[1]])

    def test_docker_stop_is_a_ctrl_c(self):
        # docker-init gives a stop's SIGTERM to the container's command alone, and the kernel SIGKILLs the
        # exec'd viewer once it exits: sim-viz's command traps it, Ctrl-Cs the viewer and waits for it.
        doc = yaml.safe_load((REPO / 'docker' / 'compose.stack.yaml').read_text())
        svc = doc['services']['sim-viz']
        cmd = svc['command']
        self.assertEqual(cmd[:2], ['bash', '-c'])
        self.assertIn('pkill -INT -f "[s]im_viz.py"', cmd[2])
        self.assertTrue(cmd[2].endswith('TERM; sleep infinity & wait'), cmd[2])
        self.assertEqual(svc['stop_grace_period'], '%ds' % sim.VIZ_STOP_GRACE_S)
        if not (shutil.which('pkill') and shutil.which('pgrep') and shutil.which('bash')):
            self.skipTest('no pkill/pgrep here')
        # The trap itself, on a stand-in named apart from any real viewer (the host sees the containers').
        with tempfile.TemporaryDirectory() as d:
            name = 'sim_viz_trap_%d.py' % os.getpid()
            fake = Path(d) / name
            fake.write_text('import signal, sys, time\n'
                            'signal.signal(signal.SIGINT, lambda *a: (time.sleep(1.0), open(sys.argv[1], "w")'
                            '.write("left"), sys.exit(0)))\n'
                            'time.sleep(30)\n')
            viewer = subprocess.Popen([sys.executable, str(fake), str(Path(d) / 'left')])
            # Its own group: the `sleep infinity &` it leaves behind (docker-init's exit reaps it there) goes too.
            init = subprocess.Popen(['bash', '-c', cmd[2].replace('[s]im_viz.py', '[s]' + name[1:])],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                time.sleep(0.5)
                t0 = time.time()
                init.terminate()
                self.assertEqual(init.wait(10), 0)
                self.assertGreaterEqual(time.time() - t0, 0.9)               # it waited for the leave
                self.assertEqual(viewer.wait(5), 0)
                self.assertEqual((Path(d) / 'left').read_text(), 'left')
            finally:
                viewer.kill()
                with contextlib.suppress(OSError):
                    os.killpg(init.pid, signal.SIGKILL)
                viewer.wait()
                init.wait()

    def test_stop_waits_for_the_paced_leave(self):
        rc, calls, text = self.viz(['viz', '--stop'], project=('sim-drive', 'sim-viz'))
        self.assertEqual(rc, 0, text)
        self.assertEqual(calls[0], ('stop', 'sim-viz', None, sim.VIZ_STOP_GRACE_S))
        self.assertEqual(calls[1], ('tmux', 'kill-session', '-t', '=rsim-viz'))
        self.assertEqual(len(calls), 2)                                     # no compose, no isolation, no ROS
        self.assertIn('left the graph one subscription at a time', text)
        sim_viz = load_sim_viz()
        self.assertGreaterEqual(sim.VIZ_STOP_GRACE_S, sim_viz.leave_seconds(sim.VIZ_MAX_PANELS * len(sim_viz.LAYERS))
                                + 10)

    def test_refusals(self):
        rc, calls, text = self.viz(['viz'], isolation=['network rover-sim is not internal'])
        self.assertEqual(rc, 3)
        self.assertNotIn('start', [c[0] for c in calls if not isinstance(c, str)])
        self.assertNotIn('xhost', calls)
        self.assertIn('isolation: BROKEN', text)
        for argv, kw, why in ((['viz'], {'project': ('sim-gazebo',)}, 'sim-drive of project rover-sim-stack'),
                              (['viz', '--agents', 'JR99'], {}, 'JR99: not an agent'),
                              (['viz', '--layers', 'bogus'], {}, 'unknown layer(s) bogus'),
                              (['viz'], {'display': ''}, 'headless: scripts/sim.py viz --snapshot <png>'),
                              (['viz'], {'spawn': False}, 'no spawn.yaml')):
            rc, calls, text = self.viz(argv, **kw)
            self.assertEqual((rc, calls), (1, []), why)
            self.assertIn(why, text)
        five = dataclasses.replace(SWAP, agents=tuple(sim.Agent('JR%d' % (81 + i), 'jr', float(i), 0.0, 0.0)
                                                       for i in range(5)))
        rc, calls, text = self.viz(['viz'], s=five)
        self.assertEqual((rc, calls), (1, []))
        self.assertIn('pick up to 4 with --agents', text)
        rc, calls, text = self.viz(['viz'], nodes=(60.0, ['viz/viz'], []))
        self.assertEqual(rc, 1)
        self.assertIn('ended before starting the viewer', text)

    def test_down_stops_the_viewer_first(self):
        order = []
        patches = {'running_sim_containers': lambda: ['sim-gazebo', 'sim-drive', 'sim-viz'],
                   'stop_processes': lambda name, **kw: order.append((name, kw.get('grace_int'))) or 1,
                   'tmux_sessions': lambda: [s for s in ('rsim-viz', 'rsim-gazebo') if ('kill', s) not in order],
                   'tmux': lambda *a, **k: order.append(('kill', a[2][1:])),
                   'parallel': lambda fn, items: order.append(('parallel', list(items))) or [fn(i) for i in items],
                   'resolve_run_dir': lambda: Path('/tmp/x'), 'compose_env': lambda r: {},
                   'sh': lambda args, **kw: order.append(('compose', list(args)))}
        with mock.patch.multiple(sim, **patches), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            sim.stop_run(remove_containers=True)
        self.assertEqual(order[0], ('sim-viz', sim.VIZ_STOP_GRACE_S))
        self.assertEqual(order[1], ('kill', 'rsim-viz'))
        self.assertEqual(order[2], ('parallel', ['sim-gazebo', 'sim-drive']))  # the rest after it has left
        self.assertEqual(order[3:5], [('sim-gazebo', 10.0), ('sim-drive', 10.0)])
        compose = order[-1][1]
        self.assertEqual(compose[-2:], ['down', '--remove-orphans'])
        self.assertEqual([compose[i + 1] for i, a in enumerate(compose) if a == '--profile'], ['gpu', 'viz'])

    def test_down_waits_for_goal_relays_leave(self):
        # sim-mighty's goal_relay.py leaves one entity at a time: 4 agents, ~17 s, longer than 10 s.
        seen = {}
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'session.json').write_text(json.dumps(SWAP.to_json()))
            patches = {'running_sim_containers': lambda: ['sim-mighty', 'sim-drive'],
                       'stop_processes': lambda name, **kw: seen.setdefault(name, kw.get('grace_int')) and 0,
                       'tmux_sessions': lambda: [], 'resolve_run_dir': lambda: Path(d)}
            with mock.patch.multiple(sim, **patches), contextlib.redirect_stdout(io.StringIO()):
                sim.stop_run()
        self.assertEqual(seen['sim-drive'], 10.0)
        self.assertEqual(seen['sim-mighty'], sim.goal_relay_leave_s(4))
        self.assertGreater(seen['sim-mighty'], (8 * 4 + 2) * sim.STATUS_LEAVE_GAP_S)

    def test_registry_and_isolation_cover_it(self):
        self.assertIn('sim-viz', sim.CONTAINERS)
        self.assertEqual(sim.sim_only('sim-viz'), 'sim-viz')
        self.assertNotIn('viz', sim.STACKS)                                 # `up` never starts it
        self.assertNotIn('sim-viz', {c for st in sim.STACKS.values() for c in st.containers})
        with mock.patch.object(sim, 'sh', lambda *a, **k: types.SimpleNamespace(
                stdout='sim-viz\nsim-drive\nacl-c2-zenoh\n', returncode=0)):
            self.assertEqual(sim.running_sim_containers(), ['sim-drive', 'sim-viz'])
        self.assertEqual(sim.build_parser().parse_args(['attach', 'viz']).stack, 'viz')

    def test_layers_match_the_viewer(self):
        sim_viz = load_sim_viz()
        self.assertEqual(sim.VIZ_LAYERS, tuple(sim_viz.LAYERS))
        self.assertEqual(sim.VIZ_MAX_PANELS, sim_viz.MAX_PANELS)
        self.assertEqual(sim.VIZ_SCRIPT, Path(sim_viz.__file__).name)

    def test_viz_cpus(self):
        self.assertEqual(sim.cpu_budget(12, sim.CONTAINERS)['sim-viz'], '6-7')    # 8-11 are 2-5's siblings
        self.assertEqual(sim.cpu_budget(32, sim.CONTAINERS)['sim-viz'], '28-31')
        no_smt = sim.cpu_budget(20, sim.CONTAINERS, lambda c: ())
        self.assertEqual(no_smt['sim-viz'], '16-19')
        self.assertEqual(sim.cpu_list([16, 17, 18, 19, 21, 3]), '3,16-19,21')
        self.assertEqual(sim.parse_cpu_list('2,12'), [2, 12])
        self.assertEqual(sim.parse_cpu_list('16-19\n'), [16, 17, 18, 19])
        self.assertEqual(sim.half_siblings(20)(2), {12})
        self.assertEqual(sim.half_siblings(20)(12), {2})
        self.assertEqual(sim.cpuset_var('sim-viz'), 'ROVER_SIM_CPUSET_VIZ')

    def test_compose_service(self):
        svc = yaml.safe_load((REPO / 'docker' / 'compose.stack.yaml').read_text())['services']['sim-viz']
        self.assertEqual(svc['profiles'], ['viz'])
        self.assertTrue(svc['image'].startswith('${GAZEBO_IMAGE:?'))
        self.assertEqual(svc['networks'], ['rover-sim'])
        self.assertNotIn('network_mode', svc)
        self.assertEqual(svc['environment']['DISPLAY'], '${DISPLAY:-}')
        self.assertIn('/tmp/.X11-unix:/tmp/.X11-unix:ro', svc['volumes'])
        for v in svc['volumes']:
            if v.split(':')[1].startswith('/rover_sim/'):
                self.assertTrue(v.endswith(':ro'), v)                       # zenoh, scripts, config, run
        grace = svc['stop_grace_period']
        self.assertTrue(grace.endswith('s') and float(grace[:-1]) >= sim.VIZ_STOP_GRACE_S)


if __name__ == '__main__':
    unittest.main()
