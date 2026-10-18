"""Unit tests for scripts/parity_check.py: the container guard and private-repo references.

    python3 -m unittest discover -s tests -v
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
            with self.assertRaises(SystemExit):
                pc.Sim(5, 1).running(bad)          # before any docker call
            with self.assertRaises(SystemExit):
                pc.Sim(5, 1).ros2(bad, ['node', 'list'], 'x')
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


if __name__ == '__main__':
    unittest.main()
