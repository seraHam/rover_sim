"""Unit tests for scripts/parity_check.py: the container guard, private-repo references, the
mapping mode from the running session, and the query session against a fake rclpy.

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, str(REPO / 'scripts' / (name + '.py')))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pc = load('parity_check')
sim = sys.modules.get('sim') or load('sim')


def git(*args, cwd=None):
    subprocess.run(['git'] + list(args), cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class ContainerGuard(unittest.TestCase):
    def test_same_registry_as_sim(self):
        self.assertEqual(list(pc.SIM_CONTAINERS), list(sim.CONTAINERS))

    def test_refuses_non_sim(self):
        for bad in ('acl-c2-master', 'acl-c2-zenoh', 'node-exporter', 'sim', 'sim-drive2', '', None):
            with self.assertRaises(SystemExit) as e:
                pc.sim_container(bad)
            self.assertEqual(e.exception.code, 2)
            with mock.patch.object(pc.subprocess, 'run', side_effect=AssertionError('docker called')):
                with self.assertRaises(SystemExit):
                    pc.Sim(5, 1).running(bad)          # before any docker call
                with self.assertRaises(SystemExit):
                    pc.Sim(5, 1).collect(bad, {'params': [], 'info': [], 'topics': []})
        self.assertEqual(pc.sim_container('sim-mighty'), 'sim-mighty')

    def test_cli_refuses_before_querying(self):
        for flag in ('--graph-container', '--container'):
            p = subprocess.run([sys.executable, str(REPO / 'scripts' / 'parity_check.py'), 'JR81', flag,
                                'acl-c2-master'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, timeout=60)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("refusing to query container 'acl-c2-master'", p.stderr)
            self.assertEqual(p.stdout, '')


class External(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = pc.CACHE
        pc.CACHE = self.tmp.name
        repo = os.path.join(self.tmp.name, 'src', 'cfgrepo@pinned')
        os.makedirs(repo)
        git('init', '-q', repo)
        Path(repo, 'cfg').mkdir()
        Path(repo, 'cfg', 'a.yaml').write_text('# robot copy\n/**:\n  ros__parameters: {k: 1.0e-9, s: x}\n')
        git('add', '-A', cwd=repo)
        git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'c', cwd=repo)
        self.sha = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
                                  universal_newlines=True).stdout.strip()

    def tearDown(self):
        pc.CACHE = self.old
        self.tmp.cleanup()

    def test_digest_is_of_content(self):
        a = pc.yaml_digest('/**:\n  ros__parameters: {k: 1.0e-9, s: x}\n')
        b = pc.yaml_digest('# launch rewrote me\n/**:\n  ros__parameters:\n    s: "x"\n    k: 1e-9\n')
        self.assertEqual(a, b)
        self.assertNotEqual(a, pc.yaml_digest('/**:\n  ros__parameters: {k: 2.0e-9, s: x}\n'))

    def test_fetch_from_cached_checkout(self):
        spec = {'url': 'git@example.invalid:x/cfgrepo.git', 'sha': self.sha, 'path': 'cfg/a.yaml',
                'yaml_sha256': pc.yaml_digest('/**: {ros__parameters: {s: x, k: 1.0e-9}}')}
        self.assertIn('robot copy', pc.fetch_external(spec))
        spec['yaml_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, "does not match the robot's copy"):
            pc.fetch_external(spec)


class Fixtures(unittest.TestCase):
    def test_no_private_copies(self):
        """Every file the manifest marks external is absent from tools/reference (public repo)."""
        import yaml
        ref = REPO / 'tools' / 'reference' / 'jr02'
        manifest = yaml.safe_load((ref / 'manifest.yaml').read_text())
        for rel, spec in manifest['external'].items():
            self.assertFalse((ref / rel).exists(), rel)
            self.assertRegex(spec['sha'], r'^[0-9a-f]{40}$', rel)
            self.assertRegex(spec['yaml_sha256'], r'^[0-9a-f]{64}$', rel)
        used = {f for n in manifest['nodes'].values() for f in self._files(n)}
        for rel in used - set(manifest['external']):
            self.assertTrue((ref / rel).is_file(), rel)

    @staticmethod
    def _files(node):
        p = node.get('params') or {}
        out = list(p.get('files') or [])
        for over in (p.get('by_class') or {}).values():
            out += over.get('files') or []
        return out

    def test_selftest(self):
        if not (REPO / '.cache' / 'src').is_dir():
            self.skipTest('no pinned checkouts (.cache/src): make images fetches them')
        for ns, extra in (('JR81', []), ('RR83', ['--mapping', 'fallback'])):
            p = subprocess.run([sys.executable, str(REPO / 'scripts' / 'parity_check.py'), ns, '--selftest'] + extra,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual(p.stdout.count('PASS'), 2, p.stdout)



class SessionMappingTest(unittest.TestCase):
    def run_dir(self, d, data):
        Path(d, 'session.json').write_text(json.dumps(data))
        return d

    def test_from_session_json(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_dir(d, {'name': 'smoke', 'mapping': 'fallback', 'agents': [{'name': 'JR81'}, {'name': 'RR83'}]})
            mode, note = pc.session_mapping('RR83', d)
            self.assertEqual(mode, 'fallback')
            self.assertIn('session smoke', note)
            self.assertNotIn('not one of', note)
            mode, note = pc.session_mapping('JR82', d)
            self.assertIn('JR82 is not one of its agents (JR81, RR83)', note)
            self.run_dir(d, {'name': 'x', 'agents': []})
            self.assertEqual(pc.session_mapping('JR81', d)[0], None)
        self.assertEqual(pc.session_mapping('JR81', '/nonexistent/run')[0], None)

    def test_main_takes_it(self):
        seen = []
        real = pc.Reference.__init__

        def spy(self, ref, ns, mapping):
            seen.append(mapping)
            real(self, ref, ns, mapping)
        with tempfile.TemporaryDirectory() as d:
            self.run_dir(d, {'name': 'smoke', 'mapping': 'fallback', 'agents': [{'name': 'JR81'}]})
            out = io.StringIO()
            with mock.patch.object(pc.Reference, '__init__', spy), contextlib.redirect_stdout(out):
                pc.main(['JR81', '--run-dir', d, '--show-ref', 'mpc'])
                pc.main(['JR81', '--run-dir', d, '--mapping', 'gpu', '--show-ref', 'mpc'])
                with mock.patch.dict(os.environ, {'ROVER_SIM_MAPPING': 'fallback'}):
                    pc.main(['JR81', '--run-dir', os.path.join(d, 'none'), '--show-ref', 'mpc'])
                pc.main(['JR81', '--run-dir', os.path.join(d, 'none'), '--show-ref', 'mpc'])
        self.assertEqual(seen, ['fallback', 'gpu', 'fallback', 'gpu'])


# ---------------------------------------------------------------------------- the query session, fake rclpy

class NS(types.SimpleNamespace):
    pass


class FakeRclpy:
    """What query() uses of rclpy, with a graph of two nodes, recording every call in order."""

    def __init__(self, answer=True):
        self.calls, self.answer = [], answer
        self.node = None
        rclpy = self

        class Node:
            def __init__(self, name, **kw):
                rclpy.node = self
                rclpy.calls.append(('Node', name, kw))

            def get_node_names_and_namespaces(self):
                return [('mighty_node', '/JR81'), ('safety_node', '/JR81'), ('sim_tf', '/')]

            def create_client(self, srv, name):
                rclpy.calls.append(('create_client', name))
                return NS(name=name, wait_for_service=lambda timeout_sec: rclpy.answer,
                          call_async=lambda req: rclpy.future(srv, req))

            def destroy_client(self, c):
                rclpy.calls.append(('destroy_client', c.name))

            def get_publisher_names_and_types_by_node(self, name, ns):
                return [('/%s/%s_out' % (ns.strip('/'), name), ['std_msgs/msg/String'])]

            def get_subscriber_names_and_types_by_node(self, name, ns):
                return [('/trajs', ['dynus_interfaces/msg/DynTraj'])]

            def get_publishers_info_by_topic(self, topic):
                return [NS(node_name='mighty_node', node_namespace='/JR81', topic_type='geometry_msgs/msg/PoseStamped',
                           qos_profile=NS(reliability=NS(name='RELIABLE'), durability=NS(name='VOLATILE')))]

            def get_subscriptions_info_by_topic(self, topic):
                return []

            def destroy_node(self):
                rclpy.calls.append(('destroy_node',))

        self.Node = Node

    def future(self, srv, req):
        if srv.__name__ == 'ListParameters':
            res = NS(result=NS(names=['num_agents', 'map_frame_id', 'v', 'unset']))
        else:
            self.calls.append(('get', tuple(req.names)))
            res = NS(values=[NS(type=2, integer_value=84), NS(type=4, string_value='JR81/map'),
                             NS(type=8, double_array_value=[0.5, 1.0]), NS(type=0)])
        return NS(done=lambda: True, result=lambda: res)

    def init(self, **kw):
        self.calls.append(('init',))

    def spin_once(self, node, timeout_sec=None):
        self.calls.append(('spin',))

    def spin_until_future_complete(self, node, fut, timeout_sec=None):
        pass

    def try_shutdown(self):
        self.calls.append(('shutdown',))


def srv(name):
    class Srv:
        class Request:
            def __init__(self):
                self.names = []
    Srv.__name__ = name
    return Srv


class QueryTest(unittest.TestCase):
    SPEC = {'params': ['/JR81/mighty_node', '/JR81/mpc'], 'info': ['/JR81/mighty_node', '/JR81/goal_selector'],
            'topics': ['/JR81/term_goal'], 'timeout': 3, 'spin': 0.0}

    def run_query(self, answer=True):
        fake = FakeRclpy(answer)
        mods = {'rclpy': fake, 'rclpy.node': NS(Node=fake.Node),
                'rcl_interfaces': types.ModuleType('rcl_interfaces'),
                'rcl_interfaces.srv': NS(ListParameters=srv('ListParameters'), GetParameters=srv('GetParameters'))}
        sleeps = []
        with mock.patch.dict(sys.modules, mods), mock.patch.object(pc.time, 'sleep', sleeps.append):
            out = pc.query(dict(self.SPEC), rclpy=fake)
        return out, fake.calls, sleeps

    def test_answers(self):
        out, calls, _ = self.run_query()
        self.assertEqual(out['nodes'], ['/JR81/mighty_node', '/JR81/safety_node', '/sim_tf'])
        self.assertEqual(out['params'], {'/JR81/mighty_node': {'flat': {
            'num_agents': 84, 'map_frame_id': 'JR81/map', 'v': [0.5, 1.0], 'unset': None}}})   # mpc not up
        self.assertEqual(out['info'], {'/JR81/mighty_node': {'publishers': {'/JR81/mighty_node_out': 'std_msgs/msg/String'},
                                                             'subscribers': {'/trajs': 'dynus_interfaces/msg/DynTraj'}}})
        self.assertEqual(out['topics']['/JR81/term_goal'], {'type': 'geometry_msgs/msg/PoseStamped', 'endpoints': [
            {'node': '/JR81/mighty_node', 'kind': 'PUBLISHER', 'reliability': 'RELIABLE', 'durability': 'VOLATILE'}]})
        self.assertIn(('get', ('num_agents', 'map_frame_id', 'v', 'unset')), calls)

    def test_leaves_gently(self):
        _, calls, sleeps = self.run_query()
        node = next(c for c in calls if c[0] == 'Node')
        self.assertEqual(node[2], {'enable_rosout': False, 'start_parameter_services': False})
        # each client goes right after its call, with spinning before the next one
        order = [c for c in calls if c[0] in ('create_client', 'destroy_client', 'destroy_node')]
        self.assertEqual([c[0] for c in order], ['create_client', 'destroy_client'] * 2 + ['destroy_node'])
        i = calls.index(order[1])
        self.assertEqual(calls[i + 1], ('spin',))
        self.assertEqual(sleeps, [pc.LEAVE_PAUSE_S])
        self.assertEqual(calls[-2:], [('destroy_node',), ('shutdown',)])

    def test_service_down(self):
        out, calls, _ = self.run_query(answer=False)
        self.assertIn('no /JR81/mighty_node/list_parameters service', out['params']['/JR81/mighty_node']['error'])
        self.assertEqual([c[0] for c in calls if 'client' in c[0]], ['create_client', 'destroy_client'])

    def test_query_main_refuses_outside_the_sim(self):
        environ = {k: v for k, v in os.environ.items() if k != 'ZENOH_SESSION_CONFIG_URI'}
        err = io.StringIO()
        with mock.patch.dict(os.environ, environ, clear=True), mock.patch.dict(sys.modules, {'rclpy': None}), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as e:
            pc.main(['--query', '{}'])
        self.assertEqual(e.exception.code, 2)
        self.assertIn('only inside a sim container', err.getvalue())


class CollectTest(unittest.TestCase):
    def test_docker_exec(self):
        seen = []

        def fake_run(cmd, **kw):
            seen.append(cmd)
            if cmd[1] == 'inspect':
                return NS(returncode=0, stdout='true rover-sim-stack\n')
            return NS(returncode=0, stdout='2026-10-19T10:00:00Z INFO zenoh noise\n' + pc.QUERY_MARK
                      + json.dumps({'nodes': ['/JR81/mpc'], 'params': {}, 'info': {}, 'topics': {}}) + '\n',
                      stderr='')
        with mock.patch.object(pc.subprocess, 'run', fake_run):
            out = pc.Sim(7, 1.5).collect('sim-drive', {'params': ['/JR81/mpc'], 'info': [], 'topics': []})
        self.assertEqual(out['nodes'], ['/JR81/mpc'])
        cmd = seen[-1]
        self.assertEqual(cmd[:4], ['docker', 'exec', 'sim-drive', 'bash'])
        self.assertIn('exec timeout -k 2 ', cmd[5])
        self.assertIn('python3 /rover_sim/scripts/parity_check.py --query "$1"', cmd[5])
        self.assertEqual(json.loads(cmd[7]), {'params': ['/JR81/mpc'], 'info': [], 'topics': [], 'timeout': 7,
                                             'spin': 1.5})

    def test_no_answer_is_an_error(self):
        def fake_run(cmd, **kw):
            if cmd[1] == 'inspect':
                return NS(returncode=0, stdout='true rover-sim-stack\n')
            return NS(returncode=124, stdout='', stderr='Terminated')
        with mock.patch.object(pc.subprocess, 'run', fake_run):
            out = pc.Sim(7, 1.5).collect('sim-drive', {'params': [], 'info': [], 'topics': []})
        self.assertIn('timed out', out['error'])

    def test_garbled_answer_is_an_error(self):
        def fake_run(cmd, **kw):
            if cmd[1] == 'inspect':
                return NS(returncode=0, stdout='true rover-sim-stack\n')
            return NS(returncode=0, stdout=pc.QUERY_MARK + '{"nodes": [2026-10-19T10:00 noise\n', stderr='')
        with mock.patch.object(pc.subprocess, 'run', fake_run):
            out = pc.Sim(7, 1.5).collect('sim-drive', {'params': [], 'info': [], 'topics': []})
        self.assertIn('unparsable answer', out['error'])

    def test_budget_covers_both_waits_of_both_calls(self):
        # call() may wait timeout for the service and timeout for the answer, twice per node.
        spec = {'params': ['/JR81/a', '/JR81/b'], 'timeout': 30, 'spin': 2.0}
        self.assertGreaterEqual(pc.query_budget(spec), 2 * 2 * (30 + 30) + 2.0 + 2 * 2 * pc.LEAVE_GAP_S)

    def test_planner_nodes_against_a_snapshot(self):
        """mighty_node's num_agents is ALLOWED, a goal_selector service error is an ERROR."""
        ref = pc.Reference('JR02', 'JR81', 'gpu')
        nodes = ref.nodes()
        flat = dict(ref.params(nodes['mighty_node'])[1], num_agents=84)
        snap = {'nodes': [nodes[n]['sim_fqn'] for n in ('mighty_node', 'goal_selector')],
                'params': {'/JR81/mighty_node': {'flat': flat}, '/JR81/goal_selector': {'error': 'did not answer'}},
                'info': {}, 'topics': {}}
        sim_ = pc.Sim(5, 1)
        sim_.running = lambda c: True
        sim_.collect = lambda c, spec: snap
        args = types.SimpleNamespace(container=None, graph_container=None, params=True, graph=False)
        _, results = pc.run_checks(ref, sim_, ['mighty_node', 'goal_selector', 'mpc'], args)
        by = {(r.section, r.name): r for r in results}
        self.assertEqual(by[('params', 'mighty_node')].status, pc.ALLOWED)
        self.assertIn('num_agents', by[('params', 'mighty_node')].lines[0][1])
        self.assertEqual(by[('params', 'goal_selector')].status, pc.ERROR)
        self.assertEqual(by[('nodes', 'mpc')].status, pc.NOT_RUNNING)


if __name__ == '__main__':
    unittest.main()
