"""Unit tests for scripts/grid_relay.py's pure logic (no rclpy).

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('grid_relay', str(REPO / 'scripts' / 'grid_relay.py'))
grid_relay = importlib.util.module_from_spec(_spec)
sys.modules['grid_relay'] = grid_relay
_spec.loader.exec_module(grid_relay)

Endpoint = namedtuple('Endpoint', 'node_namespace node_name')


def grid(x=-10.05, y=-10.15, res=0.1, stamp='mapper-now'):
    """A duck-typed nav_msgs/OccupancyGrid: what the relay touches."""
    return SimpleNamespace(header=SimpleNamespace(stamp=stamp, frame_id='JR81/odom'),
                           info=SimpleNamespace(resolution=res, width=200, height=200,
                                                origin=SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=0.15))),
                           data=b'')


class ShiftTest(unittest.TestCase):
    def test_half_cell_onto_the_voxel_lattice(self):
        info = grid().info
        grid_relay.shift_origin(info, 0.5)
        self.assertAlmostEqual(info.origin.position.x, -10.0)
        self.assertAlmostEqual(info.origin.position.y, -10.1)
        self.assertEqual(info.origin.position.z, 0.15)

    def test_zero_is_a_pure_relay(self):
        info = grid().info
        grid_relay.shift_origin(info, 0.0)
        self.assertEqual((info.origin.position.x, info.origin.position.y), (-10.05, -10.15))


class RelayStateTest(unittest.TestCase):
    def test_nothing_before_the_first_cloud(self):
        s = grid_relay.RelayState()
        s.on_grid(grid_relay.OCC_IN, grid())
        self.assertEqual(s.tick(), [])
        s.on_cloud('cloud-1')
        self.assertEqual([t for t, _ in s.tick()], ['occ_2d_topic', 'live/occ_2d_topic'])

    def test_outputs_carry_the_cloud_stamp_and_the_shift(self):
        s = grid_relay.RelayState(0.5)
        s.on_cloud('cloud-7')
        occ, esdf = grid(), grid()
        s.on_grid(grid_relay.OCC_IN, occ)
        s.on_grid(grid_relay.ESDF_IN, esdf)
        out = s.tick()
        self.assertEqual([t for t, _ in out], ['occ_2d_topic', 'live/occ_2d_topic', 'esdf_2d_topic'])
        self.assertIs(out[0][1], occ)
        self.assertIs(out[1][1], occ)            # the live window is the same grid
        self.assertIs(out[2][1], esdf)
        for _, m in out:
            self.assertEqual(m.header.stamp, 'cloud-7')
            self.assertAlmostEqual(m.info.origin.position.x, -10.0)

    def test_each_grid_once(self):
        # A message is shifted in place, so sending it again would shift it twice.
        s = grid_relay.RelayState(0.5)
        s.on_cloud('c')
        m = grid()
        s.on_grid(grid_relay.OCC_IN, m)
        self.assertEqual(len(s.tick()), 2)
        self.assertEqual(s.tick(), [])
        self.assertAlmostEqual(m.info.origin.position.x, -10.0)

    def test_newest_grid_wins_between_ticks(self):
        s = grid_relay.RelayState()
        s.on_cloud('c')
        old, new = grid(), grid()
        s.on_grid(grid_relay.OCC_IN, old)
        s.on_grid(grid_relay.OCC_IN, new)
        self.assertTrue(all(m is new for _, m in s.tick()))

    def test_frozen_cloud_stamp_is_passed_on(self):
        # DLIO down: global_mapper republishes with now(), the relay keeps the last cloud's
        # stamp, so mighty's map-age hold sees the data age as on the robot.
        s = grid_relay.RelayState()
        s.on_cloud('last-cloud')
        for _ in range(3):
            s.on_grid(grid_relay.OCC_IN, grid(stamp='mapper-now'))
            self.assertEqual({m.header.stamp for _, m in s.tick()}, {'last-cloud'})


class ContractTest(unittest.TestCase):
    def test_others(self):
        eps = [Endpoint('/JR81', 'grid_relay_JR81'), Endpoint('/JR81', 'map_fusion_node'),
               Endpoint('/', 'rogue'), Endpoint('/JR81/', 'grid_relay_JR81')]
        self.assertEqual(grid_relay.other_publishers(eps, 'JR81', 'grid_relay_JR81'),
                         ['/JR81/map_fusion_node', '/rogue'])

    def test_only_me(self):
        self.assertEqual(grid_relay.other_publishers([Endpoint('/JR81', 'grid_relay_JR81')], 'JR81',
                                                     'grid_relay_JR81'), [])


class ContractNamesTest(unittest.TestCase):
    def test_outputs_are_map_fusions(self):
        # map_fusion's output names on the robot; mighty_node reads occ_2d_topic and
        # esdf_2d_topic, persistent_map_node live/occ_2d_topic.
        self.assertEqual(dict(grid_relay.OUTPUTS), {'occ_2d_topic': 'voxel/occ_2d_topic',
                                                    'live/occ_2d_topic': 'voxel/occ_2d_topic',
                                                    'esdf_2d_topic': 'voxel/esdf_2d_topic'})
        self.assertEqual(grid_relay.CLOUD_IN, 'dlio/odom_node/pointcloud/deskewed')
        self.assertEqual(grid_relay.node_name('RR83'), 'grid_relay_RR83')

    def test_qos(self):
        self.assertEqual(grid_relay.IN_QOS, {'reliability': 'best_effort', 'depth': 1})
        self.assertEqual(grid_relay.OUT_QOS, {'reliability': 'reliable', 'depth': 10})


class ArgsTest(unittest.TestCase):
    def test_defaults(self):
        a = grid_relay.parse_args(['JR81'])
        self.assertEqual((a.ns, a.rate, a.origin_offset_cells), ('JR81', 5.0, 0.5))

    def test_refused(self):
        for argv in (['jr81'], ['JR8'], ['JR81', '--rate', '0'], ['JR81', '--rate', '100'],
                     ['JR81', '--origin-offset-cells', '2'], []):
            with self.assertRaises(SystemExit, msg=argv), contextlib.redirect_stderr(io.StringIO()):
                grid_relay.parse_args(argv)


if __name__ == '__main__':
    unittest.main()
