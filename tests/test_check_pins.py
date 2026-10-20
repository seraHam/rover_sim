"""Unit tests for scripts/check_pins.py: parsing, the cross-pin checks and the robot comparison
(no ssh: the robot's output is given).

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('check_pins', str(REPO / 'scripts' / 'check_pins.py'))
cp = importlib.util.module_from_spec(_spec)
sys.modules['check_pins'] = cp
_spec.loader.exec_module(cp)

A, B, C, D = 'a' * 40, 'b' * 40, 'c' * 40, 'd' * 40
PINS = {'MIGHTY_SHA': '1' * 40, 'MAPPING_SHA': '2' * 40, 'DLIO_SHA': '3' * 40, 'DRIVE_SHA': '4' * 40,
        'ELEVATION_SHA': A, 'ROS2_NUMPY_SHA': B}


def repos(**versions):
    return 'repositories:\n' + ''.join('  %s:\n    type: git\n    url: https://x/%s.git\n    version: %s\n' % (k, k, v)
                                        for k, v in versions.items())


def cache(d, mighty=None, orin=None, dyn=C):
    """A .cache/src with the four pinned checkouts' pin files."""
    root = Path(d)
    m = root / ('mighty@' + PINS['MIGHTY_SHA'])
    m.mkdir()
    (m / 'mighty.repos').write_text(repos(**(mighty or {'DecompROS2': D, 'livox_ros_driver2': D, 'Livox-SDK2': D,
                                                        'dynus_interfaces': C, 'gazebo_ros_pkgs': '5' * 40})))
    mw = root / ('mapping_ws@' + PINS['MAPPING_SHA'])
    (mw / 'orin').mkdir(parents=True)
    (mw / 'docker').mkdir()
    (mw / 'orin' / 'orin.repos').write_text(repos(**(orin or {'elevation_mapping_cupy': A, 'ros2_numpy': B})))
    (mw / 'docker' / 'Dockerfile.mapping').write_text('ARG ACL_MAPPING_REF=%s\nARG MAP_FUSION_REF=%s\n'
                                                      'ARG DYNUS_INTERFACES_REF=%s\nARG WS\n' % ('6' * 40, '7' * 40, dyn))
    dl = root / ('dlio_ws@' + PINS['DLIO_SHA'])
    dl.mkdir()
    (dl / 'dlio.repos').write_text(repos(direct_lidar_inertial_odometry='8' * 40))
    return str(root)


class Parsing(unittest.TestCase):
    def test_env_repos_args(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'stacks.env')
            p.write_text('# comment\nMIGHTY_SHA=%s\n\nMIGHTY_IMAGE=rover-sim/mighty-hw:1111111\n' % ('1' * 40))
            self.assertEqual(cp.read_env(str(p)), {'MIGHTY_SHA': '1' * 40, 'MIGHTY_IMAGE': 'rover-sim/mighty-hw:1111111'})
        self.assertEqual(cp.repos_versions(repos(x=A, y=B)), {'x': A, 'y': B})
        self.assertEqual(cp.dockerfile_args('ARG A=1\nARG B\n  ARG C=no\nARG D=x y\n'), {'A': '1', 'D': 'x'})

    def test_same(self):
        self.assertTrue(cp.same(A, A[:7]))
        self.assertFalse(cp.same(A, B))
        self.assertFalse(cp.same('', ''))
        self.assertFalse(cp.same(A[:6], A))          # too short to name a commit

    def test_shipped_files_parse(self):
        pins = cp.read_env(str(REPO / 'docker' / 'stacks.env'))
        for k in ('MIGHTY_SHA', 'MAPPING_SHA', 'DLIO_SHA', 'DRIVE_SHA', 'ELEVATION_SHA', 'ROS2_NUMPY_SHA'):
            self.assertRegex(pins[k], r'^[0-9a-f]{40}$', k)
        sim = cp.repos_versions((REPO / 'rover_sim.sim.repos').read_text())
        for k in ('DecompROS2', 'livox_ros_driver2', 'Livox-SDK2', 'gazebo_ros_pkgs', 'livox_laser_simulation_ros2'):
            self.assertRegex(sim[k], r'^[0-9a-f]{40}$', k)


class Offline(unittest.TestCase):
    SIM = {'DecompROS2': D, 'livox_ros_driver2': D, 'Livox-SDK2': D, 'gazebo_ros_pkgs': '9' * 40}

    def test_all_ok(self):
        with tempfile.TemporaryDirectory() as d:
            out = cp.offline_checks(PINS, self.SIM, cache(d))
        self.assertEqual([s for s, _, _ in out if s != cp.INFO], [cp.OK] * 6)
        info = dict((w, t) for s, w, t in out if s == cp.INFO)
        self.assertIn('expected to differ', info['gazebo_ros_pkgs'])
        self.assertEqual(info['DLIO'], 'dlio_ws dlio.repos 8888888')
        self.assertEqual(info['mapping image'], 'acl-mapping 6666666, map_fusion 7777777')

    def test_differences(self):
        with tempfile.TemporaryDirectory() as d:
            out = cp.offline_checks(PINS, dict(self.SIM, DecompROS2=A), cache(d, orin={'elevation_mapping_cupy': B,
                                                                                       'ros2_numpy': B}, dyn=D))
        diffs = [w for s, w, _ in out if s == cp.DIFF]
        self.assertEqual(diffs, ['DecompROS2', 'elevation_mapping_cupy',
                                 'dynus_interfaces (mighty.repos vs Dockerfile.mapping)'])

    def test_missing_checkout(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(cp.SetupError, 'no checkout of mighty@1111111'):
                cp.offline_checks(PINS, self.SIM, d)


class Robot(unittest.TestCase):
    def test_revision_label_then_tag(self):
        self.assertEqual(cp.image_revision(['x:main-3cf7320'], {'org.opencontainers.image.revision': A}), (A, 'label'))
        self.assertEqual(cp.image_revision(['mighty-hw:main-3cf7320'], {}), ('3cf7320', 'tag mighty-hw:main-3cf7320'))
        self.assertEqual(cp.image_revision(['dlio:JR02-04c7239'], None)[0], '04c7239')
        self.assertEqual(cp.image_revision(['dlio:latest'], {}), (None, None))

    def test_robot_checks(self):
        pins = dict(PINS, MIGHTY_SHA='3cf7320' + '0' * 33, DLIO_SHA='04c7239' + '0' * 33)
        ps = 'hw-mighty\tmighty-hw:main-3cf7320\ndlio\tdlio:JR02-04c7239\nmapping\trover-mapping:x\nother\tfoo\n'
        inspect = {'mighty-hw:main-3cf7320': (['mighty-hw:main-3cf7320'], {}),
                   'dlio:JR02-04c7239': (['dlio:JR02-04c7239'], {'org.opencontainers.image.revision': B}),
                   'rover-mapping:x': (['rover-mapping:x'], {})}
        out = dict((w, (s, t)) for s, w, t in cp.robot_checks(pins, ps, inspect))
        self.assertEqual(out['hw-mighty'][0], cp.OK)
        self.assertEqual(out['dlio'][0], cp.DIFF)                # the label wins over the tag
        self.assertIn('(label)', out['dlio'][1])
        self.assertEqual(out['mapping'][0], cp.MISSING)          # no revision anywhere
        self.assertEqual(out['drive'], (cp.MISSING, 'not running on the robot'))

    def test_ssh_is_read_only_and_batch(self):
        calls = []

        answers = ['hw-mighty\tm:1\n', '["m:1"]\t{"org.opencontainers.image.revision": "%s"}\n' % A]

        def run(args, **kw):
            calls.append(args)
            return mock.Mock(returncode=0, stdout=answers[len(calls) - 1], stderr='')
        with mock.patch.object(cp.subprocess, 'run', run):
            ps, inspect = cp.read_robot('robot')
        self.assertEqual(inspect, {'m:1': (['m:1'], {'org.opencontainers.image.revision': A})})
        self.assertEqual(calls[0][:5], ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10'])
        # Only docker ps, then docker image inspect of what runs: nothing that writes.
        self.assertEqual([c[-1].split()[:2] for c in calls], [['docker', 'ps'], ['docker', 'image']])
        src = (REPO / 'scripts' / 'check_pins.py').read_text()
        self.assertNotIn('ros2 ', src.split('def ssh', 1)[1])     # no ros2 CLI on a robot, ever

    def test_main_offline(self):
        with tempfile.TemporaryDirectory() as d:
            c = cache(d)
            sim = repos(DecompROS2=D, livox_ros_driver2=D, **{'Livox-SDK2': D})
            with mock.patch.object(cp, 'CACHE', c), mock.patch.object(cp, 'read_env', lambda p: PINS), \
                    mock.patch.object(cp, 'read', lambda p: sim if p.endswith('rover_sim.sim.repos') else open(p).read()), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cp.main([]), 0)
        self.assertIn('check_pins: ok', out.getvalue())
        with mock.patch.object(cp, 'read_env', lambda p: (_ for _ in ()).throw(OSError('no stacks.env'))), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cp.main([]), 2)
        self.assertIn('no stacks.env', err.getvalue())


if __name__ == '__main__':
    unittest.main()
