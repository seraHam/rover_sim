"""Unit tests for scripts/wait_for_seed.py's pure logic (no rclpy).

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import math
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('wait_for_seed', str(REPO / 'scripts' / 'wait_for_seed.py'))
wait_for_seed = importlib.util.module_from_spec(_spec)
sys.modules['wait_for_seed'] = wait_for_seed
_spec.loader.exec_module(wait_for_seed)

sys.path.insert(0, str(REPO / 'scripts'))
import sim  # noqa: E402


class Args(unittest.TestCase):
    def test_defaults_and_timeout(self):
        a = wait_for_seed.parse_args(['JR81'])
        self.assertEqual((a.ns, a.timeout), ('JR81', wait_for_seed.TIMEOUT_S))
        self.assertGreaterEqual(wait_for_seed.TIMEOUT_S, 120)        # the paused bring-up takes minutes
        self.assertEqual(wait_for_seed.parse_args(['RR83', '--timeout', '5']).timeout, 5.0)

    def test_rejects(self):
        for argv in ([], ['jr81'], ['JR81/x'], ['JR81', '--timeout', '0'], ['JR81', '--timeout', 'nan']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                wait_for_seed.parse_args(argv)
            self.assertEqual(cm.exception.code, 2, argv)

    def test_sim_pane_line_parses(self):
        # The DLIO pane's own command line (sim.py seed_wait) is one this parser takes.
        line = sim.seed_wait(sim.Agent('RR83', 'rr', 0.0, 0.0, 0.0))
        words = line.split()
        self.assertEqual(words[:2], ['python3', '/rover_sim/scripts/wait_for_seed.py'])
        args = wait_for_seed.parse_args(words[2:])
        self.assertEqual((args.ns, args.timeout), ('RR83', float(sim.SEED_WAIT_S)))


class Seed(unittest.TestCase):
    def test_the_seed_panes_message_is_taken(self):
        # What sim.py's seed pane publishes, as ros2 topic pub reads its YAML.
        a = sim.Agent('JR81', 'jr', 0.0, 0.0, 0.0)
        msg = yaml.safe_load(shlex.split(sim.seed_pose_cmd(a))[-1])
        self.assertIsNone(wait_for_seed.seed_problem(msg['header']['frame_id'], float(msg['pose']['position']['z'])))

    def test_problems(self):
        self.assertIn("frame_id 'map'", wait_for_seed.seed_problem('map', 0.36))
        self.assertIn("frame_id ''", wait_for_seed.seed_problem('', 0.36))
        self.assertIn('not finite', wait_for_seed.seed_problem('world', math.nan))
        self.assertIn('not finite', wait_for_seed.seed_problem('world', math.inf))
        self.assertIsNone(wait_for_seed.seed_problem('world', 0.0))       # z 0 is a seed (the robot's own choice)


class Lines(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(wait_for_seed.topic('JR81'), '/JR81/world')
        self.assertEqual(wait_for_seed.node_name('JR81'), 'rover_sim_wait_for_seed_JR81')
        self.assertNotEqual(wait_for_seed.node_name('JR81'), wait_for_seed.node_name('JR82'))
        self.assertTrue(wait_for_seed.waiting_line('JR81', 0).startswith(
            '[rover_sim] waiting for the JR81 seed pose on /JR81/world ...'))
        self.assertIn('(10 s; last one not taken: z nan is not finite)',
                      wait_for_seed.waiting_line('JR81', 10.2, 'z nan is not finite'))
        self.assertEqual(wait_for_seed.received_line('JR81', 0.36, 5.04),
                         '[rover_sim] JR81 seed pose received on /JR81/world after 5.0 s: z 0.360')
        t = wait_for_seed.timeout_line('JR81', 300)
        self.assertIn('no JR81 seed pose on /JR81/world after 300 s', t)
        self.assertIn('Not starting DLIO', t)
        self.assertIn("only rejected ones: frame_id 'map'", wait_for_seed.timeout_line('JR81', 5, "frame_id 'map'"))

    def test_imports_without_rclpy(self):
        # In a fresh interpreter: other test modules in the same run (the images have ROS) may
        # already have imported rclpy, which says nothing about this module.
        code = ('import sys; sys.path.insert(0, %r); import wait_for_seed; '
                'sys.exit(1 if "rclpy" in sys.modules else 0)' % str(REPO / 'scripts'))
        self.assertEqual(subprocess.run([sys.executable, '-c', code]).returncode, 0)


if __name__ == '__main__':
    unittest.main()
