"""Unit tests for scripts/sim.py's pure logic (no docker, no tmux).

    python3 -m unittest discover -s tests -v
"""
import importlib.util
import ipaddress
import json
import os
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import OrderedDict
from pathlib import Path

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
        self.assertEqual(s.world, 'empty')
        self.assertEqual(s.mapping, 'fallback')
        self.assertFalse(s.rviz)
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
        self.assertEqual((s.mapping, s.rviz), ('gpu', False))

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
        e = errors_of(session(mapping='cpu', rviz='yes', world='bad world', extra=1))
        for want in ("mapping: 'cpu'", "rviz: 'yes'", "world: 'bad world'", 'unknown key(s): extra'):
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


class Stacks(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(list(sim.STACKS),
                         ['zenoh', 'gazebo', 'drive', 'dlio', 'mapping', 'elevation', 'mighty', 'rviz'])
        self.assertEqual([s.name for s in sim.STACKS.values() if s.implemented], ['zenoh'])
        for s in sim.STACKS.values():
            for c in s.containers:
                self.assertIn(c, sim.CONTAINERS)
        # Only stacks that open windows get X access (xhost +SI:localuser:root).
        self.assertEqual([s.name for s in sim.STACKS.values() if s.needs_display], ['rviz'])

    def test_select_default(self):
        s = sim.validate_session(session())
        self.assertEqual([x.name for x in sim.select_stacks(s)], ['zenoh'])
        self.assertEqual(sim.pending_stacks(s), ['gazebo (Phase 2)', 'drive (Phase 2)', 'dlio (Phase 3)',
                                                 'mapping (Phase 4)', 'mighty (Phase 5)'])
        gpu = sim.validate_session(session(mapping='gpu', rviz=True))
        self.assertIn('elevation (Phase 4)', sim.pending_stacks(gpu))
        self.assertIn('rviz (Phase 5)', sim.pending_stacks(gpu))
        self.assertEqual(sim.session_profiles(gpu), ['gpu'])
        self.assertEqual(sim.session_profiles(s), [])

    def test_select_requested(self):
        s = sim.validate_session(session())
        self.assertEqual([x.name for x in sim.select_stacks(s, ['zenoh'])], ['zenoh'])
        with self.assertRaisesRegex(sim.SimError, 'Phase 2'):
            sim.select_stacks(s, ['gazebo'])
        with self.assertRaisesRegex(sim.SimError, 'unknown stack'):
            sim.select_stacks(s, ['nope'])

    def test_placeholders_raise(self):
        s = sim.validate_session(session(mapping='gpu', rviz=True))
        ctx = sim.RunContext(run_dir=Path('/tmp/x'))
        for st in sim.STACKS.values():
            if not st.implemented:
                with self.assertRaisesRegex(sim.SimError, 'Phase %d' % st.phase):
                    st.panes(s, ctx)


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
            run = [v for v in vols if v.endswith(':/rover_sim/run')]
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
    def test_dry_run(self):
        out = subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up', 'smoke', '--dry-run'],
                             stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
        self.assertIn('stack zenoh -> tmux session rsim-zenoh', out)
        self.assertIn('rmw_zenohd', out)
        self.assertIn('not started yet (later phases): gazebo (Phase 2)', out)

    def test_bad_session_exit_code(self):
        p = subprocess.run([sys.executable, str(REPO / 'scripts' / 'sim.py'), 'up', 'nope', '--dry-run'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn('available: smoke', p.stderr)


if __name__ == '__main__':
    unittest.main()
