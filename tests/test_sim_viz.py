"""Unit tests for scripts/sim_viz.py: the marker store, frames, views, layout and bounds without
rclpy, numpy or cv2 (the host's python), the drawing and the paced leave where numpy and cv2 are
installed (the gazebo image).

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import math
import os
import signal
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
_spec = importlib.util.spec_from_file_location('sim_viz', str(REPO / 'scripts' / 'sim_viz.py'))
viz = importlib.util.module_from_spec(_spec)
sys.modules['sim_viz'] = viz
_spec.loader.exec_module(viz)
import sim_tf  # noqa: E402

HAVE_CV = viz.cv2 is not None
NS = types.SimpleNamespace


def pt(x, y, z=0.0):
    return NS(x=x, y=y, z=z)


def quat(yaw):
    return NS(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def rgba(r, g, b, a=1.0):
    return NS(r=r, g=g, b=b, a=a)


def marker(type_=viz.LINE_STRIP, action=viz.ADD, ns='a', id_=0, frame='JR81/map', points=(), colors=(),
           color=None, scale=(0.1, 0.1, 0.1), x=0.0, y=0.0, z=0.0, yaw=0.0, lifetime=0.0, text=''):
    return NS(type=type_, action=action, ns=ns, id=id_, header=NS(frame_id=frame), points=list(points),
              colors=list(colors), color=color or rgba(1.0, 0.0, 0.0), scale=NS(x=scale[0], y=scale[1], z=scale[2]),
              pose=NS(position=pt(x, y, z), orientation=quat(yaw)),
              lifetime=NS(sec=int(lifetime), nanosec=int(round((lifetime % 1) * 1e9))), text=text)


SPAWNS = [sim_tf.SpawnPose('JR81', -4.5, 4.0, -math.pi / 4), sim_tf.SpawnPose('RR83', 13.0, -14.5, math.pi / 2)]


def close(test, a, b, places=6):
    test.assertEqual(len(a), len(b))
    for u, v in zip(a, b):
        test.assertAlmostEqual(u, v, places=places)


class Frames(unittest.TestCase):
    def test_aff(self):
        a = viz.Aff.rigid(1.0, 2.0, math.pi / 2)
        close(self, a.apply(1.0, 0.0), (1.0, 3.0))
        b = viz.Aff.rigid(0.0, 0.0, 0.0, 2.0)
        close(self, (a @ b).apply(1.0, 0.0), (1.0, 4.0))        # b first, then a
        self.assertAlmostEqual(a.yaw(), math.pi / 2)
        self.assertEqual(viz.Aff().rows(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_frame_table(self):
        frames = viz.frame_table(SPAWNS)
        close(self, viz.resolve_frame(frames, 'world').apply(3.0, 4.0), (3.0, 4.0))
        m = viz.resolve_frame(frames, 'JR81/map')
        self.assertIs(viz.resolve_frame(frames, 'JR81/odom'), m)        # map -> odom is the identity
        self.assertIsNone(viz.resolve_frame(frames, '/JR81/map'))       # tf2 rejects a leading /, RViz2 too
        close(self, m.apply(0.0, 0.0), (-4.5, 4.0))
        s = math.sqrt(0.5)
        close(self, m.apply(1.0, 0.0), (-4.5 + s, 4.0 - s))              # spawned facing -45 deg
        close(self, viz.resolve_frame(frames, 'RR83/map').apply(1.0, 0.0), (13.0, -13.5))
        for bad in ('', 'map', 'JR81/base_link', 'JR82/map', 'odom'):
            self.assertIsNone(viz.resolve_frame(frames, bad), bad)

    def test_marker_points_apply_the_pose_then_the_frame(self):
        frames = viz.frame_table(SPAWNS)
        m = marker(points=[pt(1.0, 0.0), pt(0.0, 2.0)], x=1.0, y=1.0, yaw=math.pi / 2)
        close(self, sum(viz.marker_points(m, viz.Aff()), ()), (1.0, 2.0, -1.0, 1.0))
        world = viz.marker_points(m, viz.resolve_frame(frames, 'RR83/map'))
        # RR83/map: x along world y. (1, 2) in its map -> (13 - 2, -14.5 + 1).
        close(self, world[0], (11.0, -13.5))
        # A pose tilted 90 deg about x puts a point's z where its y was: z is not simply dropped.
        tilt = marker(points=[pt(0.0, 0.0, 1.0)])
        tilt.pose.orientation = NS(x=math.sqrt(0.5), y=0.0, z=0.0, w=math.sqrt(0.5))
        close(self, viz.marker_points(tilt, viz.Aff())[0], (0.0, -1.0))
        zero = marker(points=[pt(1.0, 0.0)])
        zero.pose.orientation = NS(x=0.0, y=0.0, z=0.0, w=0.0)       # RViz: treated as the identity
        close(self, viz.marker_points(zero, viz.Aff())[0], (1.0, 0.0))

    def test_yaw_of(self):
        self.assertAlmostEqual(viz.yaw_of(quat(1.2)), 1.2)


class Store(unittest.TestCase):
    T, U = '/JR81/hgp_path_marker', '/JR81/original_hgp_path_marker'

    def keys(self, s, topic, now):
        return [(m.ns, m.id) for m in s.markers(topic, now)[0]]

    def test_add_modify_delete(self):
        s = viz.MarkerStore()
        s.apply(self.T, [marker(ns='line', id_=1), marker(ns='dots', id_=2, type_=viz.SPHERE_LIST)], 0.0)
        s.apply(self.T, [marker(ns='line', id_=1, points=[pt(0, 0), pt(1, 1)])], 0.1)     # MODIFY = ADD
        self.assertEqual(self.keys(s, self.T, 0.2), [('line', 1), ('dots', 2)])         # first-add order kept
        self.assertEqual(len(s.markers(self.T, 0.2)[0][0].points), 2)
        s.apply(self.T, [marker(action=viz.DELETE, ns='line', id_=2)], 0.3)             # no such (ns, id)
        self.assertEqual(len(self.keys(s, self.T, 0.3)), 2)
        s.apply(self.T, [marker(action=viz.DELETE, ns='line', id_=1)], 0.4)
        self.assertEqual(self.keys(s, self.T, 0.4), [('dots', 2)])
        s.apply(self.T, [marker(action=1)], 0.5)                                        # deprecated / unknown
        self.assertEqual(s.bad_actions, {self.T: 1})

    def test_mighty_clear_then_add_keeps_the_path(self):
        # mighty's hgp arrays alternate: 2 DELETEs (ns '', frame '') then the line + dots (ns set).
        s = viz.MarkerStore()
        for k in range(4):
            now = 0.1 * k
            s.apply(self.T, [marker(action=viz.DELETE, ns='', id_=50001, frame=''),
                             marker(action=viz.DELETE, ns='', id_=50002, frame='')], now)
            s.apply(self.T, [marker(ns='global_path_line', id_=50001, lifetime=10.0),
                             marker(ns='global_path_dots', id_=50002, type_=viz.SPHERE_LIST, lifetime=10.0)], now)
        self.assertEqual(self.keys(s, self.T, 0.5), [('global_path_line', 50001), ('global_path_dots', 50002)])

    def test_deleteall_clears_only_its_topic(self):
        s = viz.MarkerStore()
        s.apply(self.T, [marker(ns='a', id_=1), marker(ns='b', id_=1)], 0.0)
        s.apply(self.U, [marker(ns='a', id_=1)], 0.0)
        s.apply(self.T, [marker(action=viz.DELETEALL, ns=''), marker(ns='c', id_=3)], 1.0)   # frontiers' shape
        self.assertEqual(self.keys(s, self.T, 1.0), [('c', 3)])
        self.assertEqual(self.keys(s, self.U, 1.0), [('a', 1)])

    def test_lifetime(self):
        s = viz.MarkerStore()
        s.apply(self.T, [marker(id_=1, lifetime=2.0), marker(id_=2, lifetime=0.0)], 0.0)
        s.apply(self.T, [], 1.9)
        self.assertEqual(self.keys(s, self.T, 1.9), [('a', 1), ('a', 2)])
        s.apply(self.T, [], 2.1)
        self.assertEqual(self.keys(s, self.T, 2.1), [('a', 2)])                         # 0: forever
        s.apply(self.T, [marker(id_=1, lifetime=2.0)], 2.2)                             # re-added: from now
        self.assertEqual(self.keys(s, self.T, 4.1), [('a', 2), ('a', 1)])

    def test_non_finite_markers_are_rejected_as_in_rviz(self):
        s = viz.MarkerStore()
        s.apply(self.T, [marker(ns='l', id_=1, points=[pt(0, 0), pt(1, 1)])], 0.0)
        nan, inf = float('nan'), float('inf')
        bad = [marker(ns='l', id_=1, points=[pt(0, 0), pt(nan, 1)]),         # replaces (l, 1): it is deleted
               marker(ns='l', id_=2, x=inf), marker(ns='l', id_=3, scale=(nan, 0.1, 0.1)),
               marker(ns='l', id_=4, color=rgba(1, nan, 0)),
               marker(ns='l', id_=5, points=[pt(0, 0)], colors=[rgba(0, 0, 0, inf)])]
        bad[1].pose.orientation = NS(x=0.0, y=0.0, z=0.0, w=1.0)
        s.apply(self.T, bad + [marker(ns='l', id_=6, points=[pt(0, 0), pt(1e300, -1e300)])], 0.1)   # huge: finite
        self.assertEqual(self.keys(s, self.T, 0.2), [('l', 6)])
        self.assertEqual(s.invalid, {self.T: 5})
        q = marker()
        q.pose.orientation = NS(x=0.0, y=0.0, z=0.0, w=nan)
        self.assertFalse(viz.marker_ok(q))

    def test_stale_fades_then_drops(self):
        s = viz.MarkerStore()
        s.apply(self.T, [marker(id_=1)], 0.0)
        self.assertEqual(s.markers(self.T, viz.STALE_S - 0.1)[1], 1.0)
        ms, fade = s.markers(self.T, viz.STALE_S + 0.1)
        self.assertEqual((len(ms), fade), (1, viz.STALE_FADE))
        self.assertEqual(s.markers(self.T, viz.DROP_S + 0.1)[0], [])
        self.assertEqual(s.markers('/never', 100.0), ([], 1.0))
        # Any array refreshes the topic, an empty one too (mighty's clear arrays).
        s.apply(self.T, [marker(id_=1)], 100.0)
        s.apply(self.T, [], 104.0)
        self.assertEqual(s.markers(self.T, 108.0)[1], 1.0)


class Views(unittest.TestCase):
    def test_world_view_letterboxes(self):
        v = viz.world_view((-10.0, -5.0, 10.0, 5.0), 400, 400)
        self.assertAlmostEqual(v.res, 0.05)                             # x decides: 20 m over 400 px
        a = v.aff()
        close(self, a.apply(0.0, 0.0), (199.5, 199.5))                  # pixel centres are integers
        close(self, a.apply(-10.0, 0.0), (-0.5, 199.5))                 # the left edge of pixel 0
        u, w = a.apply(0.0, 1.0)
        self.assertLess(w, 199.5)                                       # y up
        close(self, v.extent(), (-10.0, -10.0, 10.0, 10.0))

    def test_follow_view(self):
        v = viz.follow_view(3.0, 4.0, 15.0, 1.0, 900, 600)
        self.assertAlmostEqual(v.res, 15.0 / 600)                       # the shorter side spans follow_m
        close(self, v.aff().apply(3.0, 4.0), (449.5, 299.5))
        self.assertAlmostEqual(viz.follow_view(0, 0, 15.0, 2.0, 900, 600).res, 7.5 / 600)

    def test_grid_spacing(self):
        self.assertEqual(viz.grid_spacing(0.025), 2.0)
        self.assertEqual(viz.grid_spacing(0.06), 5.0)
        self.assertEqual(viz.grid_spacing(0.005), 0.5)
        # A runaway pose's view: still a few dozen lines a frame, not one per 100 m.
        for res in (10.0, 1e5, 1e18):
            sp = viz.grid_spacing(res)
            self.assertGreaterEqual(sp / res, 60.0)
            self.assertLessEqual(1600 * res / sp, 1600 / 60.0 + 1)

    def test_far_points_do_not_grow_the_view(self):
        spawns = [sim_tf.SpawnPose('JR81', 0.0, 0.0, 0.0), sim_tf.SpawnPose('RR83', 30.0, 0.0, 0.0)]
        self.assertTrue(viz.near_spawns((25.0, 150.0), spawns))
        self.assertFalse(viz.near_spawns((1e21, 0.0), spawns))
        self.assertFalse(viz.near_spawns((0.0, -viz.GROW_MAX_M - 1.0), spawns))

    def test_grid_aff(self):
        frames = viz.frame_table(SPAWNS)
        info = NS(resolution=0.1, width=150, height=190, origin=NS(position=pt(-7.5, -9.0), orientation=quat(0.0)))
        v = viz.world_view((-20.0, -20.0, 20.0, 20.0), 400, 400)       # 0.1 m/px
        m = viz.grid_aff(info, viz.resolve_frame(frames, 'world'), v)
        close(self, m.apply(0, 0), v.aff().apply(-7.45, -8.95))          # cell (0, 0)'s centre
        close(self, viz.grid_outline(m, 150, 190)[0], v.aff().apply(-7.5, -9.0))
        close(self, viz.grid_outline(m, 150, 190)[2], v.aff().apply(7.5, 10.0))
        # In an agent's odom frame: the spawn pose turns it.
        m = viz.grid_aff(info, viz.resolve_frame(frames, 'RR83/odom'), v)
        close(self, viz.grid_outline(m, 150, 190)[0], v.aff().apply(13.0 + 9.0, -14.5 - 7.5))


class Layout(unittest.TestCase):
    def test_grid_shape(self):
        self.assertEqual([viz.grid_shape(n) for n in (1, 2, 3, 4)], [(1, 1), (1, 2), (2, 2), (2, 2)])
        for n in (0, 5):
            with self.assertRaises(ValueError):
                viz.grid_shape(n)

    def test_panel_size(self):
        rows, cols, w, h = viz.panel_size(1, 1600, 1000, 30)
        self.assertEqual((rows, cols, w), (1, 1, 1600))
        self.assertEqual(h, 1000 - 30 - (viz.HEADER_ROWS * viz.ROW_H + 8))
        self.assertEqual(viz.panel_size(2, 1600, 1000, 30)[:3], (1, 2, 800))
        self.assertEqual(viz.panel_size(3, 1600, 1000, 30)[:3], (2, 2, 800))

    def test_bounds(self):
        self.assertEqual(viz.view_bounds({}, SPAWNS, [-1, -2, 3, 4]), (-1.0, -2.0, 3.0, 4.0))
        with self.assertRaises(ValueError):
            viz.view_bounds({}, SPAWNS, [3, -2, -1, 4])
        # spawn.yaml's walls are sim.py world_walls's [xmin, ymin, xmax, ymax]
        self.assertEqual(viz.view_bounds({'walls': [-5, -6, 5, 6]}, SPAWNS), (-6.0, -7.0, 6.0, 7.0))
        m = viz.SPAWN_MARGIN_M
        self.assertEqual(viz.view_bounds({'world': 'ACL_office_simple'}, SPAWNS),
                         (-4.5 - m, -14.5 - m, 13.0 + m, 4.0 + m))

    def test_grow_bounds(self):
        self.assertIsNone(viz.grow_bounds(None, []))
        b = viz.grow_bounds(None, [(-5.1, -18.4), (0.5, 0.5)])
        self.assertEqual(b, (-8.0, -22.0, 4.0, 4.0))                 # margin 2, edges on 2 m steps
        self.assertEqual(viz.grow_bounds(b, [(0.0, 0.0), (1.9, -19.9)]), b)        # inside: unchanged
        self.assertEqual(viz.grow_bounds(b, [(0.0, 0.0)], margin=10.0), (-10.0, -22.0, 10.0, 10.0))  # grows only
        self.assertEqual(viz.grow_bounds((-1, -1, 1, 1), [(12.0, 0.0)], margin=1.0, step=1.0), (-1, -1, 13.0, 1))

    def test_cell_aff(self):
        info = NS(resolution=0.1, width=150, height=190, origin=NS(position=pt(-7.5, -9.0), orientation=quat(0.0)))
        m = viz.cell_aff(info, viz.Aff())
        close(self, m.apply(0, 0) + m.apply(150, 190), (-7.5, -9.0, 7.5, 10.0))

    def test_layers_and_agents(self):
        self.assertEqual(viz.parse_layers('default'), list(viz.DEFAULT_LAYERS))
        self.assertNotIn('map', viz.DEFAULT_LAYERS)                     # the megabyte map: asked for only
        self.assertEqual(viz.parse_layers('pose,occ'), ['occ', 'pose'])
        self.assertEqual(viz.parse_layers('default,map')[-1], 'map')
        self.assertEqual(viz.parse_layers('all'), list(viz.LAYERS))
        for bad in ('', 'occ,bogus'):
            with self.assertRaises(ValueError):
                viz.parse_layers(bad)
        self.assertEqual([p.name for p in viz.pick_agents(SPAWNS, ['RR83'])], ['RR83'])
        self.assertEqual(len(viz.pick_agents(SPAWNS, [])), 2)
        with self.assertRaisesRegex(ValueError, 'not in spawn.yaml'):
            viz.pick_agents(SPAWNS, ['JR99'])
        many = [sim_tf.SpawnPose('JR%d' % (81 + i), float(i), 0.0, 0.0) for i in range(5)]
        with self.assertRaisesRegex(ValueError, 'pick up to 4 with --agents'):
            viz.pick_agents(many, [])
        self.assertEqual(len(viz.pick_agents(many, ['JR81', 'JR85'])), 2)

    def test_topics(self):
        self.assertEqual([viz.topic_of('JR81', lay) for lay in viz.LAYERS.values()],
                         ['/JR81/occ_2d_topic', '/JR81/original_hgp_path_marker', '/JR81/hgp_path_marker',
                          '/JR81/traj_committed_colored', '/JR81/exploration/frontiers', '/JR81/dlio/odom_node/pose',
                          '/JR81/persistent_map/grid'])
        self.assertEqual(sorted(viz.DRAW_ORDER), sorted(viz.LAYERS))


class Text(unittest.TestCase):
    def test_rate(self):
        r = viz.Rate()
        self.assertIsNone(r.age(0.0))
        for k in range(11):
            r.tick(k * 0.2)
        self.assertAlmostEqual(r.hz(2.0), 5.0)
        self.assertAlmostEqual(r.age(2.5), 0.5)
        self.assertEqual(r.hz(2.0 + viz.RATE_WINDOW_S + 1), 0.0)       # silent for the window: 0 Hz
        self.assertEqual(r.count, 11)
        polled = viz.Rate()                                              # the pose: 10 a poll, 10 polls a second
        for k in range(21):
            polled.tick(k * 0.1, 10)
        self.assertAlmostEqual(polled.hz(2.0), 100.0)
        self.assertEqual(polled.count, 210)
        self.assertAlmostEqual(polled.age(2.05), 0.05)

    def test_polled_pose(self):
        panel = viz.AgentPanel(sim_tf.SpawnPose('JR81', 1.0, 2.0, 0.0), 'jr', viz.PANEL_COLORS[0], ['pose'])
        for k in range(12):
            panel.pose_sink(b'raw%d' % k)
        seen = []
        panel.take_pose(5.0, lambda raw: seen.append(raw) or pose_msg(0.0, 0.0, 0.0))
        self.assertEqual((seen, panel.rates['pose'].count, panel.pose_new), ([b'raw11'], 12, 0))   # newest only
        panel.take_pose(5.1, seen.append)                                # nothing new: nothing deserialized
        self.assertEqual(len(seen), 1)

        def broken(raw):
            raise ValueError('bad buffer')
        panel.pose_sink(b'x')
        panel.take_pose(5.2, broken)
        self.assertEqual(panel.invalid, {'pose': 1})
        frames = viz.frame_table([panel.spawn])
        panel.msgs['pose'] = pose_msg(float('nan'), 0.0, 0.0)
        self.assertEqual(panel.pose_world(frames), (1.0, 2.0, 0.0))     # the spawn pose
        self.assertEqual(panel.pose_world(frames), (1.0, 2.0, 0.0))
        self.assertEqual(panel.invalid, {'pose': 2})                     # counted once per message
        self.assertEqual(panel.world_points(frames), [])

    def test_layer_text(self):
        occ = viz.LAYERS['occ']
        self.assertEqual(viz.layer_text(1, occ, True, True, 5.0, 0.25), '1 occ 5.0 Hz 0.2 s')
        self.assertEqual(viz.layer_text(1, occ, True, False, 5.0, 12.0), '1 occ (hidden) 5.0 Hz 12 s')
        self.assertEqual(viz.layer_text(7, viz.LAYERS['map'], False, True, 0, None), '7 map: not subscribed')
        self.assertEqual(viz.layer_text(2, occ, True, True, 0.0, None), '2 occ 0.0 Hz none yet')


class GraphSafety(unittest.TestCase):
    def test_leave_pacing_is_goal_relays(self):
        import goal_relay
        self.assertIs(viz.goal_relay.leave_gently, goal_relay.leave_gently)
        self.assertEqual((goal_relay.LEAVE_GAP_S, goal_relay.LEAVE_PAUSE_S), (0.5, 0.5))
        self.assertEqual(viz.leave_seconds(6), 3.5)

    def test_refuses_another_graph_before_rclpy(self):
        with tempfile.TemporaryDirectory() as d:
            spawn = Path(d) / 'spawn.yaml'
            spawn.write_text('agents:\n- {name: JR81, class: jr, x: 0.0, y: 0.0, yaw: 0.0}\n')
            env = dict(os.environ)
            env.pop('ZENOH_SESSION_CONFIG_URI', None)
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules, {'rclpy': None}), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(viz.main(['sim_viz.py', '--spawn', str(spawn)]), 2)
            self.assertIn('refusing to join another graph', err.getvalue())
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(viz.main(['sim_viz.py', '--spawn', str(spawn), '--agents', 'RR83']), 2)
                self.assertEqual(viz.main(['sim_viz.py', '--spawn', str(Path(d) / 'none.yaml')]), 2)
            self.assertIn('not in spawn.yaml', err.getvalue())


# ---------------------------------------------------------------------------- with numpy and cv2

def occ_grid(frame='JR81/odom', w=40, h=30, res=0.1, ox=-2.0, oy=-1.5, occupied=()):
    data = [0] * (w * h)
    for i, j in occupied:
        data[j * w + i] = 100
    data[-1] = -1
    return NS(header=NS(frame_id=frame), data=data,
              info=NS(resolution=res, width=w, height=h, origin=NS(position=pt(ox, oy), orientation=quat(0.0))))


def pose_msg(x, y, yaw, frame='JR81/map'):
    return NS(header=NS(frame_id=frame), pose=NS(position=pt(x, y), orientation=quat(yaw)))


@unittest.skipUnless(HAVE_CV, 'numpy and cv2 are not installed (the gazebo image has them)')
class Drawing(unittest.TestCase):
    def setUp(self):
        self.spawns = [sim_tf.SpawnPose('JR81', 0.0, 0.0, 0.0)]
        self.frames = viz.frame_table(self.spawns)
        self.store = viz.MarkerStore()
        self.panel = viz.AgentPanel(self.spawns[0], 'jr', viz.PANEL_COLORS[0], list(viz.LAYERS))
        self.view = viz.world_view((-5.0, -5.0, 5.0, 5.0), 400, 400)   # 0.025 m/px, world = JR81/map
        self.shown = set(viz.LAYERS)

    def px(self, x, y):
        u, v = self.view.aff().apply(x, y)
        return int(round(v)), int(round(u))

    def test_truth_odometry_in_the_header(self):
        # odometry: truth (spawn.yaml; NOT PARITY): the panel says so next to the agent's name.
        texts = []
        with mock.patch.object(viz, 'put', lambda img, text, *a, **k: texts.append(text)):
            viz.draw_header(self.panel, 400, self.shown, False, 1.0, 15.0, self.view, 0.0)
            truth = viz.AgentPanel(self.spawns[0], 'jr', viz.PANEL_COLORS[0], list(viz.LAYERS), 'truth')
            viz.draw_header(truth, 400, self.shown, False, 1.0, 15.0, self.view, 0.0)
        self.assertIn('JR81 (JR)', texts)
        self.assertIn('JR81 (JR, TRUTH odom: not parity)', texts)

    def draw(self, now=0.0):
        return self.panel.draw(self.view, self.shown, self.store, self.frames, now)

    def test_occ_grid(self):
        self.panel.msgs['occ'] = occ_grid(occupied=[(10, 10)])
        img = self.draw()
        self.assertEqual(tuple(img[self.px(-0.95, -0.45)]), viz.OCC_OCC)          # cell (10, 10)'s centre
        self.assertEqual(tuple(img[self.px(0.5, 0.5)]), viz.OCC_FREE)
        self.assertEqual(tuple(img[self.px(3.0, 3.0)]), viz.BG)                   # outside the window
        self.shown.discard('occ')
        self.assertEqual(tuple(self.draw()[self.px(0.5, 0.5)]), viz.BG)          # hidden, still stored

    def test_line_and_per_vertex_colours(self):
        self.store.apply('/JR81/hgp_path_marker', [marker(points=[pt(-2, 0), pt(2, 0)], scale=(0.1, 0, 0))], 0.0)
        traj = marker(type_=viz.LINE_LIST, points=[pt(-2, 2), pt(0, 2), pt(0, 2), pt(2, 2)],
                      colors=[rgba(0, 0, 1)] * 2 + [rgba(0, 1, 0)] * 2, color=rgba(0, 0, 0, 0.0), scale=(0.15, 0, 0))
        self.store.apply('/JR81/traj_committed_colored', [traj], 0.0)
        img = self.draw()
        self.assertEqual(tuple(img[self.px(0.0, 0.0)]), (0, 0, 255))              # red, BGR
        self.assertEqual(tuple(img[self.px(-1.0, 2.0)]), (255, 0, 0))             # per-vertex blue
        self.assertEqual(tuple(img[self.px(1.0, 2.0)]), (0, 255, 0))              # though the marker's a = 0
        self.assertEqual(self.panel.stats['drawn'], 2)

    def test_alpha_fade_and_counts(self):
        sq = marker(type_=viz.CUBE, x=1.0, y=1.0, scale=(1.0, 1.0, 1.0), color=rgba(1, 1, 1, 0.5))
        self.store.apply('/JR81/exploration/frontiers', [
            sq, marker(type_=viz.MESH, id_=1), marker(type_=viz.TRIANGLES, id_=2),
            marker(id_=3, frame='JR81/base_link', points=[pt(0, 0), pt(1, 1)])], 0.0)
        img = self.draw()
        half = tuple(img[self.px(1.0, 1.0)])
        self.assertTrue(all(abs(c - (70 + 255) / 2.0) <= 2 for c in half), half)   # 50 % over the background
        st = self.panel.stats
        self.assertEqual((st['drawn'], st['skipped'], st['frames']), (1, 2, {'JR81/base_link': 1}))
        faded = tuple(self.draw(viz.STALE_S + 1)[self.px(1.0, 1.0)])
        self.assertTrue(all(c < half[0] - 10 for c in faded), faded)

    def test_shapes_points_text(self):
        ms = [marker(type_=viz.SPHERE, id_=1, x=-2, y=-2, scale=(0.5, 0.5, 0.5), color=rgba(0, 1, 0)),
              marker(type_=viz.CYLINDER, id_=2, x=2, y=-2, scale=(0.5, 0.5, 0.5), color=rgba(0, 1, 0)),
              marker(type_=viz.ARROW, id_=3, x=-2, y=2, scale=(1.0, 0.1, 0.1), color=rgba(0, 1, 0)),
              marker(type_=viz.ARROW, id_=4, points=[pt(1, 2), pt(2, 2)], scale=(0.1, 0.2, 0.2), color=rgba(0, 1, 0)),
              marker(type_=viz.POINTS, id_=5, points=[pt(0, -3)], scale=(0.2, 0.2, 0), color=rgba(0, 1, 0)),
              marker(type_=viz.CUBE_LIST, id_=6, points=[pt(0, 3)], scale=(0.3, 0.3, 0.3), color=rgba(0, 1, 0)),
              marker(type_=viz.SPHERE_LIST, id_=7, points=[pt(-3, 0)], scale=(0.3, 0.3, 0.3), color=rgba(0, 1, 0)),
              marker(type_=viz.TEXT, id_=8, x=3, y=0, text='id=4\n1.0s/9.0s', color=rgba(0, 0, 0))]
        self.store.apply('/JR81/exploration/frontiers', ms, 0.0)
        img = self.draw()
        for x, y in ((-2, -2), (2, -2), (-1.6, 2), (1.5, 2), (0, -3), (0, 3), (-3, 0)):
            self.assertEqual(tuple(img[self.px(x, y)]), (0, 255, 0), (x, y))
        r, c = self.px(3, 0)
        box = img[r - 15:r + 15, c - 30:c + 30].reshape(-1, 3)
        self.assertTrue(((box < 40).all(axis=1)).any() and ((box == 255).all(axis=1)).any())   # text and halo
        self.assertEqual(self.panel.stats['drawn'], 8)

    def test_pose_and_follow(self):
        self.panel.msgs['pose'] = pose_msg(1.0, -1.0, 0.0)
        img = self.draw()
        self.assertEqual(tuple(img[self.px(0.85, -1.0)]), viz.PANEL_COLORS[0])    # behind the heading line
        self.assertEqual(tuple(img[self.px(1.4, -1.0)]), (0, 0, 0))               # the heading line, +x
        v = viz.Viewer([self.panel], self.frames, self.store, (-5, -5, 5, 5), (600, 500), True, 15.0, viz.LAYERS)
        view = v.view_of(self.panel)
        self.assertEqual((view.cx, view.cy), (1.0, -1.0))
        self.assertFalse(v.key(ord('+')))
        self.assertAlmostEqual(v.zoom, 1.25)
        v.key(ord('f'))
        self.assertFalse(v.follow)
        v.key(ord('1'))
        self.assertNotIn('occ', v.shown)
        self.assertTrue(v.key(ord('q')) and v.key(27))
        frame = v.render(0.0)
        self.assertEqual(frame.shape[1], 600)
        self.assertLessEqual(frame.shape[0], 500)

    def test_world_view_fits_what_was_seen(self):
        self.panel.msgs['pose'] = pose_msg(0.0, -18.0, 0.0)
        mostly_unknown = occ_grid(ox=-7.5, oy=-28.0, w=150, h=190)
        mostly_unknown.data = [-1] * (150 * 190)
        mostly_unknown.data[150 * 100 + 75] = 0                                   # one known cell: (0.0, -18.0)
        self.panel.msgs['occ'] = mostly_unknown
        close(self, sum(self.panel.world_points(self.frames)[1:], ()), (0.0, -18.0, 0.1, -17.9))
        v = viz.Viewer([self.panel], self.frames, self.store, (-12.0, -12.0, 12.0, 12.0), (600, 500), False, 15.0,
                       viz.LAYERS, grow=True)
        v.render(0.0)
        self.assertEqual(v.bounds, (-2.0, -20.0, 4.0, 2.0))        # pose, known cell and spawn (0, 0), +/- 2 m
        self.panel.msgs['pose'] = pose_msg(9.0, -1.0, 0.0)
        v.render(0.0)
        self.assertEqual(v.bounds, (-2.0, -20.0, 12.0, 2.0))       # grown, never shrunk
        fixed = viz.Viewer([self.panel], self.frames, self.store, (-12.0, -12.0, 12.0, 12.0), (600, 500), False,
                           15.0, viz.LAYERS)
        fixed.render(0.0)
        self.assertEqual(fixed.bounds, (-12.0, -12.0, 12.0, 12.0))

    def test_known_box_of_a_turned_grid(self):
        # A 1 m x 1 m grid, all known, in the map of an agent spawned at 45 deg: its world box is the
        # turned square's (1.41 m across), not its index box's.
        spawns = [sim_tf.SpawnPose('JR81', 0.0, 0.0, math.pi / 4)]
        panel = viz.AgentPanel(spawns[0], 'jr', viz.PANEL_COLORS[0], ['occ'])
        panel.msgs['occ'] = occ_grid(frame='JR81/map', w=10, h=10, ox=0.0, oy=0.0)
        panel.msgs['occ'].data = [0] * 100
        (x0, y0), (x1, y1) = panel.world_points(viz.frame_table(spawns))
        close(self, (x0, y0, x1, y1), (-math.sqrt(0.5), 0.0, math.sqrt(0.5), math.sqrt(2.0)))

    def test_bad_messages_do_not_end_the_session(self):
        nan, inf = float('nan'), float('inf')
        self.store.apply('/JR81/hgp_path_marker', [marker(points=[pt(0, 0), pt(nan, 1)])], 0.0)
        self.store.apply('/JR81/traj_committed_colored', [marker(type_=viz.LINE_LIST, points=[pt(0, 0), pt(1, 0)],
                                                                 scale=(1e6, 0, 0))], 0.0)    # metres wide
        short = occ_grid()
        short.data = short.data[:-5]                                       # not width x height
        self.panel.msgs['occ'] = short
        self.panel.msgs['map'] = occ_grid(frame='JR81/map', res=nan)
        for pose in (pose_msg(nan, 0.0, 0.0), pose_msg(inf, 1.0, 0.0)):
            self.panel.msgs['pose'] = pose
            for follow, grow in ((False, False), (False, True), (True, False)):
                v = viz.Viewer([self.panel], self.frames, self.store, (-5, -5, 5, 5), (600, 500), follow, 15.0,
                               viz.LAYERS, grow=grow)
                img = v.render(0.0)
                self.assertEqual(img.shape[1], 600)
                if follow:
                    view = v.view_of(self.panel)
                    self.assertEqual((view.cx, view.cy), (0.0, 0.0))     # the spawn pose
        self.assertEqual(self.panel.errors, 0, self.panel.last_error)
        self.assertEqual(self.panel.invalid, {'occ': 1, 'map': 1, 'pose': 2})
        self.assertEqual(self.panel.invalid_count(self.store), 5)          # and the NaN path
        self.assertEqual(self.panel.stats['drawn'], 1)                      # the wide line, capped
        self.assertTrue(any('invalid 5' in line for line in v.summary(0.0)))

    def test_a_failing_panel_is_counted_not_fatal(self):
        spawns = [sim_tf.SpawnPose('JR81', 0.0, 0.0, 0.0), sim_tf.SpawnPose('RR83', 5.0, 0.0, 0.0)]
        panels = [viz.AgentPanel(q, 'jr', viz.PANEL_COLORS[i], ['pose']) for i, q in enumerate(spawns)]
        v = viz.Viewer(panels, viz.frame_table(spawns), self.store, (-5, -5, 10, 5), (800, 500), False, 15.0,
                       ['pose'], grow=True)
        with mock.patch.object(panels[0], 'draw', side_effect=ValueError('cannot convert float NaN to integer')), \
                mock.patch.object(panels[0], 'world_points', side_effect=OverflowError('inf')):
            img = v.render(0.0)
        self.assertEqual(img.shape[1], 800)
        self.assertEqual((panels[0].errors, panels[1].errors), (2, 0))
        self.assertIn('ValueError: cannot convert', panels[0].last_error)
        self.assertIn('draw errors 2', v.summary(0.0)[0])

    def test_legend_pose_colour(self):
        one = viz.Viewer([self.panel], self.frames, self.store, (-5, -5, 5, 5), (600, 500), False, 15.0, viz.LAYERS)
        rows = [t for row in viz.legend_rows(600, self.panel.color) for t in row]
        self.assertIn(('  ' + viz.POSE_LEGEND, self.panel.color), [r[:2] for r in rows])
        self.assertEqual(one.legend.shape[1], 600)

    def test_render_layouts(self):
        for n in (1, 2, 3, 4):
            spawns = [sim_tf.SpawnPose('JR%d' % (81 + i), float(i), 0.0, 0.0) for i in range(n)]
            panels = [viz.AgentPanel(p, 'jr', viz.PANEL_COLORS[i], list(viz.DEFAULT_LAYERS))
                      for i, p in enumerate(spawns)]
            v = viz.Viewer(panels, viz.frame_table(spawns), viz.MarkerStore(), viz.view_bounds({}, spawns), (800, 600),
                           False, 15.0, viz.DEFAULT_LAYERS)
            img = v.render(0.0)
            self.assertEqual(img.shape[1], 800, n)
            self.assertEqual(len(v.summary(0.0)), n)


def ros_msgs():
    try:
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError:
        return None
    return NS(Point=Point, ColorRGBA=ColorRGBA, Marker=Marker, MarkerArray=MarkerArray)


@unittest.skipUnless(HAVE_CV and ros_msgs(), 'needs numpy, cv2 and the ROS messages (the gazebo image has them)')
class RealMessages(unittest.TestCase):
    """goal_selector's frontier array (goal_selector_node.cpp publishFrontierMarkers) and mighty's path
    arrays, as the real message classes carry them."""

    def test_frontiers_and_paths(self):
        R = ros_msgs()

        def col(r, g, b, a):
            return R.ColorRGBA(r=r, g=g, b=b, a=a)

        def p(x, y, z=0.1):
            return R.Point(x=x, y=y, z=z)

        def mk(type_, ns, id_, **kw):
            m = R.Marker(type=type_, ns=ns, id=id_, action=R.Marker.ADD)
            m.header.frame_id = 'JR81/map'
            m.pose.orientation.w = 1.0
            for k, v in kw.items():
                setattr(m, k, v)
            return m
        delete_all = R.Marker(action=R.Marker.DELETEALL)
        delete_all.header.frame_id = 'JR81/map'
        cyan, grey = col(0.0, 0.8, 1.0, 0.9), col(0.6, 0.6, 0.6, 0.7)
        centroids = mk(R.Marker.SPHERE_LIST, 'frontier_centroids', 0, points=[p(1.0, 1.0), p(-1.0, 1.0)],
                       colors=[cyan, grey])
        centroids.scale.x = centroids.scale.y = centroids.scale.z = 0.3
        cells = mk(R.Marker.POINTS, 'frontier_cells', 0, points=[p(2.5, -1.0), p(-2.5, -1.0)], colors=[cyan, grey])
        cells.scale.x = cells.scale.y = 0.08
        label = mk(R.Marker.TEXT_VIEW_FACING, 'frontier_labels', 0, text='id=4', color=col(0.0, 0.0, 0.0, 1.0))
        label.pose.position.x, label.pose.position.y, label.scale.z = 0.0, -2.5, 0.4
        line = mk(R.Marker.LINE_STRIP, 'frontier_goal_line', 0, points=[p(0.5, 0.0), p(0.5, 2.0)],
                  color=col(1.0, 1.0, 0.0, 0.9))
        line.scale.x = 0.15
        frontiers = R.MarkerArray(markers=[delete_all, label, centroids, cells, line])
        clear = R.MarkerArray(markers=[R.Marker(action=R.Marker.DELETE, id=50001),
                                       R.Marker(action=R.Marker.DELETE, id=50002)])
        path = mk(R.Marker.LINE_STRIP, 'global_path_line', 50001, points=[p(-3.0, 0.0), p(-3.0, 2.0)],
                  color=col(1.0, 0.0, 0.0, 1.0))
        path.scale.x = 0.03
        path.lifetime.sec = 10
        spawns = [sim_tf.SpawnPose('JR81', 0.0, 0.0, 0.0)]
        frames, store = viz.frame_table(spawns), viz.MarkerStore()
        panel = viz.AgentPanel(spawns[0], 'jr', viz.PANEL_COLORS[0], list(viz.LAYERS))
        panel.store_markers('frontiers', store)(frontiers)
        hgp = panel.store_markers('hgp', store)
        hgp(clear)
        hgp(R.MarkerArray(markers=[path]))
        hgp(clear)                                                                  # the next cycle's
        view = viz.world_view((-4.0, -4.0, 4.0, 4.0), 400, 400)                    # 0.02 m/px
        img = panel.draw(view, set(viz.LAYERS), store, frames, time.monotonic())
        a = view.aff()

        def at(x, y):
            u, v = a.apply(x, y)
            return tuple(int(c) for c in img[int(round(v)), int(round(u))])
        bg = viz.BG
        self.assertEqual(panel.stats['drawn'], 5)
        for x, y, c, alpha in ((1.0, 1.0, (255, 204, 0), 0.9), (-1.0, 1.0, (153, 153, 153), 0.7),
                               (2.5, -1.0, (255, 204, 0), 0.9), (-2.5, -1.0, (153, 153, 153), 0.7),
                               (0.5, 1.0, (0, 255, 255), 0.9)):          # off the 2 m grid lines
            want = tuple(alpha * k + (1 - alpha) * b for k, b in zip(c, bg))
            got = at(x, y)
            self.assertTrue(all(abs(g - w) <= 3 for g, w in zip(got, want)), (x, y, got, want))
        self.assertEqual(at(-3.0, 1.0), (0, 0, 255))                               # the path survived the clears
        self.assertEqual(panel.rates['hgp'].count, 3)


class FakeHandle:
    """rclpy's subscription handle as goal_relay.Poller takes from it: queued raw samples."""

    def __init__(self, queue):
        self.queue = queue

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def take_message(self, msg_type, raw):
        return (self.queue.pop(0), None) if self.queue else None


class FakeSub:
    def __init__(self, topic, msg_type=None, raw=False):
        self.topic, self.msg_type, self.raw = topic, msg_type, raw
        self.handle = FakeHandle(list(FakeNode.queued.get(topic, ())))


class FakeNode:
    log = []
    queued = {}                 # topic -> raw samples a polled subscription finds

    def __init__(self, name, namespace=None, enable_rosout=True, start_parameter_services=True):
        FakeNode.log.append(('node', name, namespace, enable_rosout, start_parameter_services))

    def create_subscription(self, msg_type, topic, cb, qos, raw=False):
        FakeNode.log.append(('sub', topic, qos.reliability, qos.depth, raw))
        return FakeSub(topic, msg_type, raw)

    def destroy_subscription(self, sub):
        FakeNode.log.append(('destroy', sub.topic))

    def destroy_node(self):
        FakeNode.log.append(('destroy_node',))


def fake_ros(spin):
    qos = types.ModuleType('rclpy.qos')
    qos.HistoryPolicy = NS(KEEP_LAST=1)
    qos.ReliabilityPolicy = NS(RELIABLE='reliable', BEST_EFFORT='best_effort')
    qos.QoSProfile = lambda **k: NS(**k)
    rclpy = types.ModuleType('rclpy')
    rclpy.init = lambda **k: FakeNode.log.append(('init', k.get('signal_handler_options')))
    rclpy.spin_once = spin
    rclpy.try_shutdown = lambda: FakeNode.log.append(('shutdown',))
    node = types.ModuleType('rclpy.node')
    node.Node = FakeNode
    signals = types.ModuleType('rclpy.signals')
    signals.SignalHandlerOptions = NS(NO='no')
    utilities = types.ModuleType('rclpy.utilities')
    utilities.remove_ros_args = lambda argv: argv
    serialization = types.ModuleType('rclpy.serialization')
    serialization.deserialize_message = lambda raw, cls: pose_msg(float(raw.decode()), 0.0, 0.0)
    mods = {'rclpy': rclpy, 'rclpy.node': node, 'rclpy.qos': qos, 'rclpy.signals': signals,
            'rclpy.utilities': utilities, 'rclpy.serialization': serialization}
    for pkg, names in (('geometry_msgs', ('PoseStamped',)), ('nav_msgs', ('OccupancyGrid',)),
                       ('visualization_msgs', ('MarkerArray',))):
        m = types.ModuleType(pkg + '.msg')
        for n in names:
            setattr(m, n, type(n, (), {}))
        mods[pkg], mods[pkg + '.msg'] = types.ModuleType(pkg), m
    return mods


class FakeDisplay:
    """Display's interface, scripted: events() hands out the next item of script (a list of keys, or
    None: the window is gone) and then no keys."""
    script, ready_error, log = [], None, []

    def __init__(self, shape):
        self.busy, self.shape = False, shape
        FakeDisplay.log.append(('display', shape[1]))

    def wait_ready(self, timeout, stopped):
        FakeDisplay.log.append(('ready', FakeDisplay.ready_error))
        return FakeDisplay.ready_error

    def events(self):
        return FakeDisplay.script.pop(0) if FakeDisplay.script else []

    def show(self, img):
        FakeDisplay.log.append(('show', img.shape))
        FakeNode.log.append(('show',))

    def how(self):
        return 'closed'

    def close(self):
        FakeDisplay.log.append(('close',))
        FakeNode.log.append(('close',))

    def reap(self, timeout=2.0):
        FakeDisplay.log.append(('reap',))
        return 0


@unittest.skipUnless(HAVE_CV, 'numpy and cv2 are not installed (the gazebo image has them)')
class Leave(unittest.TestCase):
    """Every exit leaves one subscription at a time: the snapshot's end, an error, SIGTERM, q, the
    window closing or its process dying."""

    def run_main(self, spin, extra=(), window=False):
        FakeNode.log, FakeDisplay.log = [], []
        sleeps = []
        with tempfile.TemporaryDirectory() as d:
            spawn = Path(d) / 'spawn.yaml'
            spawn.write_text('agents:\n- {name: JR81, class: jr, x: 0.0, y: 0.0, yaw: 0.0}\n'
                             '- {name: RR83, class: rr, x: 2.0, y: 0.0, yaw: 0.0}\n')
            argv = ['sim_viz.py', '--spawn', str(spawn)] + ([] if window else [
                '--snapshot', str(Path(d) / 'out.png'), '--snapshot-after', '0.05']) + list(extra)
            old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                with mock.patch.dict(sys.modules, fake_ros(spin)), \
                        mock.patch.dict(os.environ, {'ZENOH_SESSION_CONFIG_URI': '/rover_sim/zenoh/session.json5'}), \
                        mock.patch.object(viz.goal_relay.time, 'sleep', lambda s: sleeps.append(s)), \
                        mock.patch.object(viz, 'Display', FakeDisplay), \
                        contextlib.redirect_stdout(io.StringIO()) as out, \
                        contextlib.redirect_stderr(io.StringIO()) as err:
                    try:
                        rc = viz.main(argv)
                    except RuntimeError as e:
                        rc = e
                wrote = (Path(d) / 'out.png').is_file()
            finally:
                for s, h in old.items():
                    signal.signal(s, h)
        return rc, wrote, sleeps, out.getvalue() + err.getvalue()

    def check_leave(self, sleeps, n, name='sim_viz_snapshot'):
        log = [e for e in FakeNode.log if e[0] not in ('show', 'close')]
        self.assertEqual(log[0], ('init', 'no'))                                  # no rclpy signal handler
        self.assertEqual(log[1], ('node', name, '/sim', False, False))
        self.assertEqual(log[2], ('node', name + '_poll', '/sim', False, False))  # the polled pose's
        destroys = [i for i, e in enumerate(log) if e[0] == 'destroy']
        self.assertEqual(len(destroys), n)
        self.assertEqual(sleeps, [0.5] * n + [0.5])                               # one at a time, then a pause
        self.assertEqual(log[destroys[-1] + 1][0], 'spin')
        self.assertEqual([e[0] for e in log[-3:]], ['destroy_node', 'destroy_node', 'shutdown'])
        # The polled pose subscriptions go first (goal_relay all_entities), then the rest.
        self.assertTrue(all(t.endswith('/dlio/odom_node/pose') for _, t in [log[i] for i in destroys[:2]]))

    def spin(self, *a, **k):
        FakeNode.log.append(('spin',))

    def test_snapshot_end(self):
        rc, wrote, sleeps, out = self.run_main(self.spin)
        self.assertEqual(rc, 0)
        self.assertTrue(wrote)
        subs = [e for e in FakeNode.log if e[0] == 'sub']
        self.assertEqual(len(subs), 2 * len(viz.DEFAULT_LAYERS))                 # not the map
        rel = {e[1].split('/', 2)[2]: e[2:] for e in subs if e[1].startswith('/JR81/')}
        self.assertEqual(rel['hgp_path_marker'], ('reliable', 10, False))
        self.assertEqual(rel['exploration/frontiers'], ('reliable', 10, False))
        self.assertEqual(rel['occ_2d_topic'], ('best_effort', 1, False))
        self.assertEqual(rel['dlio/odom_node/pose'], ('best_effort', viz.POSE_DEPTH, True))     # polled, raw
        self.check_leave(sleeps, len(subs))
        self.assertIn('left the graph: 12 subscription(s)', out)
        self.assertEqual(FakeDisplay.log, [])                                     # a snapshot opens no window

    def test_snapshot_reads_the_polled_pose(self):
        FakeNode.queued = {'/JR81/dlio/odom_node/pose': [b'%d' % k for k in range(25)]}
        try:
            rc, _, _, out = self.run_main(self.spin, ['--agents', 'JR81'])
        finally:
            FakeNode.queued = {}
        self.assertEqual(rc, 0)
        self.assertIn('pose 25 msgs', out)                                        # all counted
        self.assertIn('view centre', out)

    def test_spin_failing_all_through_the_leave(self):
        def always(*a, **k):
            FakeNode.log.append(('spin',))
            raise RuntimeError('callback keeps failing')
        rc, wrote, sleeps, _ = self.run_main(always)
        self.assertIsInstance(rc, RuntimeError)
        self.check_leave(sleeps, 2 * len(viz.DEFAULT_LAYERS))                    # still all, still paced

    def test_window_q(self):
        FakeDisplay.script, FakeDisplay.ready_error = [[], [ord('2')], [ord('q')]], None
        rc, _, sleeps, out = self.run_main(self.spin, window=True)
        self.assertEqual(rc, 0, out)
        self.assertEqual(FakeDisplay.log[:2], [('display', 1600), ('ready', None)])
        self.assertIn(('show', FakeDisplay.log[2][1]), FakeDisplay.log)
        self.assertEqual(FakeDisplay.log[-2:], [('close',), ('reap',)])
        self.check_leave(sleeps, 2 * len(viz.DEFAULT_LAYERS), name='sim_viz')
        # The window closes first, then the node leaves.
        names = [e[0] for e in FakeNode.log]
        self.assertLess(names.index('close'), names.index('destroy'))

    def test_window_gone(self):
        FakeDisplay.script, FakeDisplay.ready_error = [[], None], None           # closed, or its process died
        rc, _, sleeps, out = self.run_main(self.spin, window=True)
        self.assertEqual(rc, 0)
        self.assertIn('the window is gone (closed)', out)
        self.check_leave(sleeps, 2 * len(viz.DEFAULT_LAYERS), name='sim_viz')

    def test_no_display_no_join(self):
        FakeDisplay.script, FakeDisplay.ready_error = [], 'the window process ended (exit 2) before its window opened'
        rc, _, sleeps, out = self.run_main(self.spin, window=True)
        self.assertEqual(rc, 2)
        self.assertEqual(FakeNode.log, [('close',)])                              # no rclpy.init, no node
        self.assertEqual(sleeps, [])
        self.assertIn('not joining the graph', out)
        self.assertEqual(FakeDisplay.log[-2:], [('close',), ('reap',)])

    def test_error(self):
        def boom(*a, **k):
            FakeNode.log.append(('spin',))
            if sum(1 for e in FakeNode.log if e[0] == 'spin') == 1:
                raise RuntimeError('boom')
        rc, wrote, sleeps, _ = self.run_main(boom, ['--layers', 'all'])
        self.assertIsInstance(rc, RuntimeError)
        self.assertFalse(wrote)
        self.check_leave(sleeps, 2 * len(viz.LAYERS))

    def test_sigterm_mid_run_and_again_while_leaving(self):
        def term(*a, **k):
            FakeNode.log.append(('spin',))
            os.kill(os.getpid(), signal.SIGTERM)          # first: stop; later ones: ignored while leaving
        rc, wrote, sleeps, out = self.run_main(term)
        self.assertEqual((rc, wrote), (128 + signal.SIGTERM, False))     # stopped before its frame
        self.assertIn('no frame written', out)
        self.check_leave(sleeps, 2 * len(viz.DEFAULT_LAYERS))
        self.assertIn('still leaving', out)


if __name__ == '__main__':
    unittest.main()
