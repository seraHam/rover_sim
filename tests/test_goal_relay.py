"""Unit tests for scripts/goal_relay.py: frames, the agent choice, the legs, and its node against
a fake rclpy (no ROS needed).

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
_spec = importlib.util.spec_from_file_location('goal_relay', str(REPO / 'scripts' / 'goal_relay.py'))
gr = importlib.util.module_from_spec(_spec)
sys.modules['goal_relay'] = gr
_spec.loader.exec_module(gr)
import livox_sim_bridge  # noqa: E402
import sim  # noqa: E402
import sim_tf  # noqa: E402

SWAP = {  # multi_agent_swap.yaml's starts, each sent to the opposite start (the session offsets them)
    'JR81': ((2.83, 2.83, -135.0), [(-2.83, -2.83)]),
    'JR82': ((-2.83, 2.83, -45.0), [(2.83, -2.83)]),
    'RR83': ((-2.83, -2.83, 45.0), [(2.83, 2.83)]),
    'RR84': ((2.83, -2.83, 135.0), [(-2.83, 2.83)]),
}


def spawn_data(agents=SWAP):
    return {'world': 'arena', 'walls': [-8.0, -8.0, 8.0, 8.0], 'agents': [
        {'name': n, 'class': n[:2].lower(), 'id': int(n[2:]), 'x': p[0], 'y': p[1], 'yaw': math.radians(p[2]),
         'yaw_deg': p[2], 'lidar_height': 0.36} for n, (p, _) in agents.items()]}


def session_data(agents=SWAP, demo='swap'):
    return {'name': 'multi_agent_swap', 'world': 'arena', 'demo': demo, 'agents': [
        {'name': n, 'class': n[:2].lower(), 'id': int(n[2:]), 'pose': list(p), 'goals': [list(g) for g in goals]}
        for n, (p, goals) in agents.items()]}


def poses(agents=SWAP):
    return {p.name: p for p in sim_tf.spawn_poses(spawn_data(agents))}


class FramesTest(unittest.TestCase):
    def test_inverse_and_heading(self):
        for p in poses().values():
            for x, y in ((0.0, 0.0), (1.5, -2.0), (-7.0, 3.3)):
                mx, my = gr.to_map(p, x, y)
                wx, wy = gr.to_world(p, mx, my)
                self.assertAlmostEqual(wx, x, places=9)
                self.assertAlmostEqual(wy, y, places=9)
            self.assertEqual(gr.to_map(p, p.x, p.y), (0.0, 0.0))     # the spawn pose is the map origin
        # JR81 faces the centre: the opposite corner is 8 m straight ahead in JR81/map.
        mx, my = gr.to_map(poses()['JR81'], -2.83, -2.83)
        self.assertAlmostEqual(mx, 2 * 2.83 * math.sqrt(2), places=6)
        self.assertAlmostEqual(my, 0.0, places=9)

    def test_matches_sim_tf(self):
        """to_world is sim_tf's world -> <ns>/map (translation, then yaw about z)."""
        for p in poses().values():
            _, child, t, q = sim_tf.world_to_map(p)
            yaw = 2 * math.atan2(q[2], q[3])
            wx, wy = gr.to_world(p, 1.0, 2.0)
            self.assertAlmostEqual(wx, t[0] + math.cos(yaw) * 1.0 - math.sin(yaw) * 2.0)
            self.assertAlmostEqual(wy, t[1] + math.sin(yaw) * 1.0 + math.cos(yaw) * 2.0)


class GoalChoiceTest(unittest.TestCase):
    names = {'JR81', 'RR83'}

    def test_frame_agent(self):
        for f in ('world', '', '/world', None):
            self.assertIsNone(gr.frame_agent(f, self.names))
        self.assertEqual(gr.frame_agent('JR81/map', self.names), 'JR81')
        self.assertEqual(gr.frame_agent('/RR83/odom', self.names), 'RR83')
        with self.assertRaisesRegex(gr.GoalError, 'no sim agent JR02'):
            gr.frame_agent('JR02/map', self.names)
        for f in ('map', 'JR81/base_link', 'odom'):
            with self.assertRaisesRegex(gr.GoalError, "send it in 'world'"):
                gr.frame_agent(f, self.names)

    def test_nearest(self):
        truth = {'JR81': (2.0, 0.0), 'RR83': (-2.0, 0.0)}
        self.assertEqual(gr.nearest_agent(truth, 1.0, 5.0), 'JR81')
        self.assertEqual(gr.nearest_agent(truth, -0.5, 0.0), 'RR83')
        self.assertEqual(gr.nearest_agent(truth, 0.0, 3.0), 'JR81')      # a tie goes by name
        self.assertIsNone(gr.nearest_agent({}, 0.0, 0.0))

    def test_resolve(self):
        sp = poses()
        truth = {'JR81': (2.8, 2.8), 'RR83': (-2.8, -2.8)}
        ns, mx, my, how = gr.resolve_goal('world', -2.83, -2.0, sp, truth)
        self.assertEqual((ns, how), ('RR83', 'nearest'))
        self.assertEqual((round(mx, 6), round(my, 6)),
                         tuple(round(v, 6) for v in gr.to_map(sp['RR83'], -2.83, -2.0)))
        ns, mx, my, how = gr.resolve_goal('world', -2.83, -2.0, sp, truth, selected='JR81')
        self.assertEqual((ns, how), ('JR81', 'selected'))
        self.assertAlmostEqual(gr.to_world(sp['JR81'], mx, my)[0], -2.83)
        self.assertEqual(gr.resolve_goal('JR82/map', 1.0, 2.0, sp, {}), ('JR82', 1.0, 2.0, 'frame'))
        with self.assertRaisesRegex(gr.GoalError, 'no ground truth'):
            gr.resolve_goal('world', 0.0, 0.0, sp, {})

    def test_selection(self):
        self.assertEqual(gr.parse_selection(' JR81 ', self.names), 'JR81')
        for t in ('', 'nearest', 'NONE', None):
            self.assertIsNone(gr.parse_selection(t, self.names))
        with self.assertRaisesRegex(gr.GoalError, 'no sim agent'):
            gr.parse_selection('JR02', self.names)


class StopsTest(unittest.TestCase):
    def test_swap_session(self):
        sp = poses()
        stops = gr.swap_stops(sp['JR81'], SWAP['JR81'][1], 0.5)
        self.assertEqual([s[0] for s in stops], ['start', 'goal1'])
        self.assertEqual(stops[0][1], (0.0, 0.0))
        self.assertAlmostEqual(stops[1][1][0], 8.004, places=3)

    def test_odom_eval_session_returns_to_start(self):
        """config/sessions/odom_eval.yaml ends its goals at the start: the cycle goes there anyway."""
        goals = [(4.95, 0.71), (7.07, 2.83), (4.95, 4.95), (2.83, 2.83)]
        stops = gr.swap_stops(poses()['JR81'], goals, 0.5)
        self.assertEqual([s[0] for s in stops], ['start', 'goal1', 'goal2', 'goal3'])

    def test_degenerate(self):
        sp = poses()['JR81']
        self.assertEqual(gr.swap_stops(sp, [], 0.5), [])
        self.assertEqual(gr.swap_stops(sp, [(2.9, 2.9)], 0.5), [])           # at the start
        stops = gr.swap_stops(sp, [(0.0, 0.0), (0.1, 0.1), (-2.0, -2.0)], 0.5)
        self.assertEqual([s[0] for s in stops], ['start', 'goal1', 'goal3'])  # goal2 sits on goal1

    def test_timeout(self):
        self.assertEqual(gr.leg_timeout(8.0), 30.0 + 6.0 * 8.0)
        self.assertEqual(gr.leg_timeout(8.0, fixed=20), 20.0)


class LegsTest(unittest.TestCase):
    def legs(self, laps=0, reached_by='estimate', ns='JR81'):
        sp = poses()[ns]
        return gr.AgentLegs(ns, sp, gr.swap_stops(sp, SWAP[ns][1], 0.5), laps, 0.5, reached_by), sp

    @staticmethod
    def sends(acts):
        return [v for k, v in acts if k == 'send']

    @staticmethod
    def logs(acts):
        return [v for k, v in acts if k == 'log']

    def test_cycle(self):
        a, sp = self.legs(laps=1)
        goal = a.stops[1][1]
        acts = a.start(100.0, (0.0, 0.0))
        self.assertEqual(self.sends(acts), [goal])
        self.assertEqual(a.start(100.5, (0.0, 0.0)), [])                    # once
        self.assertEqual(a.update(100.5, (1.0, 0.0), None), [])
        self.assertTrue(a.on_commit(*goal))                                 # goal_selector's echo
        self.assertEqual(a.update(103.0, (2.0, 0.0), None), [])            # echoed: no resend
        self.assertFalse(a.on_commit(5.0, 5.0))                             # exploration elsewhere
        acts = a.update(104.0, (3.0, 0.0), None)                           # >= 2 s since the send
        self.assertEqual(self.sends(acts), [goal])
        self.assertEqual(a.update(105.0, (3.5, 0.0), None), [])            # at most every 2 s
        truth_w = gr.to_world(sp, goal[0] - 0.3, 0.1)
        acts = a.update(112.0, (goal[0] - 0.45, 0.0), truth_w)             # inside goal_radius
        (rec,) = self.logs(acts)
        self.assertEqual((rec['result'], rec['from'], rec['to'], rec['leg'], rec['lap']),
                         ('reached', 'start', 'goal1', 1, 1))
        self.assertEqual((rec['duration_s'], rec['est_dist'], rec['resends'], rec['foreign_commits']),
                         (12.0, 0.45, 1, 1))
        self.assertAlmostEqual(rec['truth_dist'], math.hypot(0.3, 0.1), places=3)
        self.assertEqual(rec['target_world'], [-2.83, -2.83])
        self.assertEqual(self.sends(acts), [(0.0, 0.0)])                   # the next leg at once
        acts = a.update(150.0, (0.2, 0.1), None)
        (rec,) = self.logs(acts)
        self.assertEqual((rec['result'], rec['from'], rec['to'], rec['leg']), ('reached', 'goal1', 'start', 2))
        self.assertEqual(self.sends(acts), [])                              # 1 lap: done
        self.assertIn('swap done after 2 legs', acts[-1][1])
        self.assertEqual(a.update(160.0, (0.0, 0.0), None), [])

    def test_timeout_moves_on(self):
        a, _ = self.legs()
        a.start(0.0, (0.0, 0.0))
        a.on_commit(*a.stops[1][1])
        limit = gr.leg_timeout(8.004)
        self.assertEqual(a.update(limit - 0.1, (1.0, 0.0), None), [])
        acts = a.update(limit + 0.05, (1.0, 0.0), None)
        (rec,) = self.logs(acts)
        self.assertEqual((rec['result'], rec['timeout_s']), ('timeout', round(limit, 1)))
        self.assertEqual(self.sends(acts), [(0.0, 0.0)])

    def test_no_estimate_waits_for_the_timeout(self):
        a, _ = self.legs()
        a.start(0.0, (0.0, 0.0))
        a.on_commit(*a.stops[1][1])
        self.assertEqual(self.logs(a.update(10.0, None, None)), [])

    def test_reached_by_truth(self):
        a, sp = self.legs(reached_by='truth')
        a.start(0.0, (0.0, 0.0))
        goal = a.stops[1][1]
        self.assertEqual(self.logs(a.update(5.0, goal, None)), [])          # the estimate alone: no
        rec, = self.logs(a.update(6.0, (0.0, 0.0), gr.to_world(sp, goal[0], goal[1] + 0.2)))
        self.assertEqual((rec['result'], rec['reached_by'], rec['truth_dist']), ('reached', 'truth', 0.2))

    def test_rviz_goal_interrupts_and_resumes(self):
        a, _ = self.legs()
        a.start(0.0, (0.0, 0.0))
        goal = a.stops[1][1]
        acts = a.manual(3.0, (1.0, 1.0), (1.0, 0.0))
        (rec,) = self.logs(acts)
        self.assertEqual((rec['result'], rec['kind']), ('preempted', 'swap'))
        self.assertEqual(self.sends(acts), [(1.0, 1.0)])
        acts = a.update(9.0, (1.1, 0.9), None)
        (rec,) = self.logs(acts)
        self.assertEqual((rec['kind'], rec['from'], rec['to'], rec['result']), ('rviz', 'here', 'rviz', 'reached'))
        self.assertIsNone(rec['leg'])
        self.assertEqual(self.sends(acts), [goal])                          # the interrupted leg again
        self.assertEqual((a.leg['from'], a.leg['to'], a.leg['leg']), ('rviz', 'goal1', 1))
        rec, = self.logs(a.update(30.0, (goal[0] - 0.1, 0.0), None))
        self.assertEqual(rec['leg'], 1)
        self.assertEqual(a.leg['leg'], 2)

    def test_rviz_goal_before_the_cycle(self):
        a, _ = self.legs()
        a.manual(0.0, (1.0, 1.0), (0.0, 0.0))
        self.assertEqual(a.start(1.0, (0.5, 0.5)), [])                      # the rviz goal keeps running
        self.assertEqual(a.leg['kind'], 'rviz')
        acts = a.update(4.0, (1.0, 1.0), None)
        self.assertEqual(self.sends(acts), [a.stops[1][1]])
        self.assertEqual((a.leg['from'], a.leg['leg']), ('rviz', 1))

    def test_rviz_only_agent(self):
        sp = poses()['RR83']
        a = gr.AgentLegs('RR83', sp)
        self.assertFalse(a.cycling)
        self.assertEqual(a.start(0.0, (0.0, 0.0)), [])
        a.manual(0.0, (2.0, 0.0), None, gr.to_world(sp, 0.0, 0.0))
        self.assertEqual(a.leg['deadline'], 30.0 + 12.0)                    # timeout from the truth
        rec, = self.logs(a.update(20.0, (1.8, 0.0), None))
        self.assertEqual(rec['kind'], 'rviz')
        self.assertIsNone(a.leg)                                            # then goal_selector's
        self.assertFalse(a.on_commit(5.0, 5.0))

    def test_leg_lines(self):
        a, _ = self.legs()
        a.start(0.0, (0.0, 0.0))
        rec, = self.logs(a.update(10.0, a.stops[1][1], None))
        line = gr.leg_line(rec)
        self.assertEqual(gr.parse_leg_line(line), rec)
        ros = '[goal_relay.py-1] [INFO] [1760000000.123456789] [sim.goal_relay]: ' + line + '\r\n'
        self.assertEqual(gr.parse_leg_line(ros), rec)
        self.assertIsNone(gr.parse_leg_line('LEG {not json}'))
        self.assertIsNone(gr.parse_leg_line('LEG {"no": "agent"}'))
        self.assertIsNone(gr.parse_leg_line('COMMIT {"agent": "JR81"}'))
        self.assertEqual(json.loads(line[4:]), rec)


class ConfigTest(unittest.TestCase):
    def test_goal_radius_of_the_robot_config(self):
        """The reference copy of hw_mighty_ground_robot.yaml (byte-identical to mighty 3cf7320 and dbc69ea)."""
        path = REPO / 'tools' / 'reference' / 'jr02' / 'config' / 'mighty' / 'hw_mighty_ground_robot.yaml'
        self.assertEqual(gr.load_goal_radius(str(path)), 0.5)
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, 'x.yaml')
            for text in ('mighty_node: {ros__parameters: {}}\n', 'mighty_node: {ros__parameters: {goal_radius: 0}}\n',
                         '{}\n'):
                Path(bad).write_text(text)
                with self.assertRaisesRegex(gr.GoalError, 'no positive mighty_node goal_radius'):
                    gr.load_goal_radius(bad)

    def test_pick_mode(self):
        self.assertEqual(gr.pick_mode('swap', 'none', {}), 'swap')
        self.assertEqual(gr.pick_mode('auto', 'swap', {}), 'swap')
        self.assertEqual(gr.pick_mode('auto', 'explore', {}), 'explore')
        self.assertEqual(gr.pick_mode('auto', 'none', {'JR81': [(1, 1)]}), 'rviz')
        self.assertEqual(gr.pick_mode('auto', None, {'JR81': [(1, 1)]}), 'swap')
        self.assertEqual(gr.pick_mode('auto', None, {'JR81': []}), 'rviz')

    def test_plan_agents(self):
        sp = poses()
        goals = {n: g for n, (_, g) in SWAP.items()}
        legs = gr.plan_agents(sp, goals, ['JR81', 'RR83'], 'swap', 0, 0.5, 'estimate', None)
        self.assertEqual(sorted(legs), ['JR81', 'RR83'])
        self.assertTrue(all(a.cycling for a in legs.values()))
        self.assertFalse(any(a.cycling for a in gr.plan_agents(sp, goals, list(sp), 'rviz', 0, 0.5, 'estimate',
                                                                None).values()))
        with self.assertRaisesRegex(gr.GoalError, 'no sim agent JR02'):
            gr.plan_agents(sp, goals, ['JR81', 'JR02'], 'swap', 0, 0.5, 'estimate', None)
        with self.assertRaisesRegex(gr.GoalError, 'no served agent has a session goal'):
            gr.plan_agents(sp, {}, ['JR81'], 'swap', 0, 0.5, 'estimate', None)


# ---------------------------------------------------------------------------- the node, fake rclpy

class NS(types.SimpleNamespace):
    pass


def pose_stamped():
    return NS(header=NS(stamp=None, frame_id=''), pose=NS(position=NS(x=0.0, y=0.0, z=0.0),
                                                           orientation=NS(x=0.0, y=0.0, z=0.0, w=0.0)))


class FakeEntity:
    def __init__(self, kind, msg_type, topic, cb, qos, raw):
        self.kind, self.type, self.topic, self.cb, self.qos, self.raw = kind, msg_type, topic, cb, qos, raw
        self.sent, self.subscribers, self.queue = [], 1, []
        self.msg_type, entity = msg_type, self

        class Handle:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def take_message(self, msg_type, raw):
                return (entity.queue.pop(0), 'info') if entity.queue else None
        self.handle = Handle()

    def publish(self, m):
        self.sent.append(m)

    def get_subscription_count(self):
        return self.subscribers


class FakeClient:
    def __init__(self, name):
        self.kind, self.topic, self.ready, self.requests = 'client', name, True, []

    def service_is_ready(self):
        return self.ready

    def call_async(self, req):
        self.requests.append(req)
        return NS(done=lambda: True)


def twist():
    return NS(linear=NS(x=0.0, y=0.0, z=0.0), angular=NS(x=0.0, y=0.0, z=0.0))


class FakeNode:
    def __init__(self, *a, **kw):
        self.args, self.kw = a, kw
        self.entities, self.calls, self.logs = [], [], []

    def get_logger(self):
        return NS(info=lambda m: self.logs.append(('info', m)), warn=lambda m: self.logs.append(('warn', m)),
                  error=lambda m: self.logs.append(('error', m)))

    def get_clock(self):
        return NS(now=lambda: NS(to_msg=lambda: 'stamp'))

    def create_publisher(self, msg_type, topic, qos):
        e = FakeEntity('pub', msg_type, topic, None, qos, False)
        self.entities.append(e)
        return e

    def create_subscription(self, msg_type, topic, cb, qos, raw=False):
        e = FakeEntity('sub', msg_type, topic, cb, qos, raw)
        self.entities.append(e)
        return e

    def create_timer(self, period, cb):
        return NS(cancel=lambda: self.calls.append('cancel'), period=period, cb=cb)

    def create_client(self, srv_type, name):
        e = FakeClient(name)
        self.entities.append(e)
        return e

    def destroy_client(self, e):
        self.calls.append(('destroy_client', e.topic))

    def destroy_subscription(self, e):
        self.calls.append(('destroy_subscription', e.topic))

    def destroy_publisher(self, e):
        self.calls.append(('destroy_publisher', e.topic))

    def destroy_node(self):
        self.calls.append('destroy_node')

    def topic(self, name):
        (e,) = [e for e in self.entities if e.topic == name]
        return e


class Policy(types.SimpleNamespace):
    pass


QOS = types.SimpleNamespace(
    QoSProfile=lambda **kw: dict(kw), HistoryPolicy=Policy(KEEP_LAST='keep_last'),
    ReliabilityPolicy=Policy(RELIABLE='reliable', BEST_EFFORT='best_effort'),
    DurabilityPolicy=Policy(VOLATILE='volatile'))
FAKE_MODULES = {
    'rclpy': types.ModuleType('rclpy'), 'rclpy.qos': QOS,
    'geometry_msgs': types.ModuleType('geometry_msgs'), 'geometry_msgs.msg': NS(PoseStamped=pose_stamped, Twist=twist),
    'nav_msgs': types.ModuleType('nav_msgs'), 'nav_msgs.msg': NS(Odometry='Odometry'),
    'std_msgs': types.ModuleType('std_msgs'), 'std_msgs.msg': NS(String='String'),
    'std_srvs': types.ModuleType('std_srvs'), 'std_srvs.srv': NS(Trigger=NS(Request=lambda: 'trigger'))}


def odom_cdr(x, y, yaw, stamp_ns=1790000000 * 10 ** 9, z=0.0):
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    return livox_sim_bridge.odom_build(stamp_ns, 'world', 'JR81/chassis', (x, y, z) + q + (0.0,) * 78)


class FirstDriveTest(unittest.TestCase):
    """From AUTO: JOY, straight ahead until the estimate moved dist, zeros, AUTO again."""

    def test_the_drive(self):
        fd = gr.FirstDrive('JR81', 1.2, 0.3, z_expect=0.36)
        self.assertEqual(fd.step(0.0, 'DISARMED', (0.0, 0.0, 0.36)), [])     # never armed here
        self.assertEqual(fd.step(0.1, 'AUTO', None), [])                     # no DLIO estimate yet
        self.assertEqual(fd.step(0.2, 'AUTO', (0.0, 0.0, 0.36)), [('call', 'joy_mode')])
        self.assertEqual(fd.step(0.3, 'AUTO', (0.0, 0.0, 0.36)), [])         # mode change pending
        self.assertEqual(fd.step(3.3, 'AUTO', (0.0, 0.0, 0.36)), [('call', 'joy_mode')])   # again after RETRY_S
        self.assertEqual(fd.step(3.4, 'JOY', (0.0, 0.0, 0.36)), [('vel', 0.3)])
        self.assertEqual(fd.step(5.0, 'JOY', (0.6, 0.0, 0.36)), [('vel', 0.3)])
        self.assertEqual(fd.step(5.1, 'JOY', None), [('vel', 0.0)])          # stale estimate: hold
        self.assertEqual(fd.step(7.5, 'JOY', (1.21, 0.0, 0.36)), [('vel', 0.0)])
        self.assertEqual(fd.state, 'stopping')
        self.assertEqual(fd.step(7.7, 'JOY', (1.22, 0.0, 0.36)), [('vel', 0.0)])
        self.assertEqual(fd.step(8.1, 'JOY', (1.22, 0.0, 0.36)), [('call', 'auto_mode')])
        self.assertFalse(fd.done)
        ((kind, rec),) = fd.step(8.3, 'AUTO', (1.22, 0.0, 0.36))
        self.assertEqual((kind, rec['agent'], rec['moved'], rec['wanted']), ('log', 'JR81', 1.21, 1.2))
        self.assertTrue(fd.done)
        self.assertEqual(fd.step(9.0, 'AUTO', (1.22, 0.0, 0.36)), [])
        self.assertTrue(gr.first_line(rec).startswith('FIRST {'))

    def test_off_late_seed_interrupted_and_timeout(self):
        self.assertTrue(gr.FirstDrive('JR81', 0.0, 0.3).done)
        fd = gr.FirstDrive('RR83', 1.2, 0.3, z_expect=0.40)
        ((kind, text),) = fd.step(0.0, 'AUTO', (0.0, 0.0, 0.0))             # seeded too late: z 0
        self.assertEqual(kind, 'info')
        self.assertIn('expected 0.40', text)
        self.assertEqual(fd.step(1.0, 'AUTO', (0.0, 0.0, 0.0)), [])          # said once, no drive
        fd = gr.FirstDrive('RR83', 1.2, 0.3)
        fd.step(0.0, 'AUTO', (0.0, 0.0, 0.4))
        fd.step(0.1, 'JOY', (0.0, 0.0, 0.4))
        self.assertEqual(fd.step(0.5, 'DISARMED', (0.1, 0.0, 0.4))[0][0], 'info')
        self.assertEqual(fd.state, 'wait')                                  # starts over once armed again
        fd.step(1.0, 'AUTO', (0.1, 0.0, 0.4))
        fd.step(1.1, 'JOY', (0.1, 0.0, 0.4))
        self.assertEqual(fd.step(1.1 + 1.2 / 0.3 + fd.EXTRA_S, 'JOY', (0.3, 0.0, 0.4)), [('vel', 0.0)])
        self.assertEqual(fd.state, 'stopping')                              # blocked: it ends where it is

    def test_restarted_relay_does_not_drive_again(self):
        # `sim.py start goals` mid-run: the agent is metres from its spawn, in AUTO; no JOY, no drive.
        fd = gr.FirstDrive('JR81', 1.2, 0.3, z_expect=0.36)
        ((kind, text),) = fd.step(0.0, 'AUTO', (4.0, -2.5, 0.51))      # and DLIO's z has drifted
        self.assertEqual(kind, 'info')
        self.assertIn('from its spawn already: no first drive', text)
        self.assertTrue(fd.done)
        fd = gr.FirstDrive('JR81', 1.2, 0.3, z_expect=0.36)
        self.assertEqual(fd.step(0.0, 'AUTO', (0.2, 0.1, 0.36)), [('call', 'joy_mode')])   # parked: still driven

    def test_never_re_arms_a_disarmed_agent(self):
        def drive_to(state):
            fd = gr.FirstDrive('RR83', 1.2, 0.3)
            fd.step(0.0, 'AUTO', (0.0, 0.0, 0.4))
            if state != 'to_joy':
                fd.step(0.1, 'JOY', (0.0, 0.0, 0.4))
                fd.step(5.0, 'JOY', (1.2, 0.0, 0.4))
            if state == 'to_auto':
                fd.step(5.6, 'JOY', (1.2, 0.0, 0.4))
            self.assertEqual(fd.state, state)
            return fd
        for state in ('to_joy', 'stopping', 'to_auto'):
            fd = drive_to(state)
            out = fd.step(20.0, 'DISARMED', (1.2, 0.0, 0.4))           # well past RETRY_S
            self.assertEqual([k for k, _ in out], ['info'], state)        # no ('call', ...): not re-armed
            self.assertEqual(fd.state, 'wait', state)
            self.assertEqual(fd.step(30.0, 'DISARMED', (1.2, 0.0, 0.4)), [], state)
        # Re-armed by the operator after it drove: it counts as driven.
        fd = drive_to('to_auto')
        fd.step(20.0, 'DISARMED', (1.2, 0.0, 0.4))
        self.assertEqual(fd.step(21.0, 'AUTO', (1.2, 0.0, 0.4))[0][0], 'info')
        self.assertTrue(fd.done)

    def test_time_limit_without_an_estimate(self):
        fd = gr.FirstDrive('JR81', 1.2, 0.3)
        fd.step(0.0, 'AUTO', (0.0, 0.0, 0.36))
        fd.step(0.1, 'JOY', (0.0, 0.0, 0.36))
        self.assertEqual(fd.step(1.0, 'JOY', None), [('vel', 0.0)])       # DLIO gone: hold still
        self.assertEqual(fd.step(0.1 + 1.2 / 0.3 + fd.EXTRA_S, 'JOY', None), [('vel', 0.0)])
        self.assertEqual(fd.state, 'stopping')                            # not held in JOY for good
        self.assertEqual((fd.record['moved'], fd.record['to_map']), (None, None))

    def test_setup_reads_the_seed_heights(self):
        with tempfile.TemporaryDirectory() as d:
            sp = os.path.join(d, 'spawn.yaml')
            data = spawn_data()
            data['agents'][0]['lidar_height'] = 0.40
            del data['agents'][1]['lidar_height']
            Path(sp).write_text(yaml.safe_dump(data))
            heights = gr.lidar_heights(sp)
        names = [a['name'] for a in data['agents']]
        self.assertEqual(heights, dict([(names[0], 0.40)] + [(n, 0.36) for n in names[2:]]))
        fds = gr.first_drives(['JR81', 'RR83'], {'JR81': 0.36}, 1.2, 0.3)
        self.assertEqual((fds['JR81'].z_expect, fds['RR83'].z_expect, fds['RR83'].dist), (0.36, None, 1.2))


class ShippedFirstDrives(unittest.TestCase):
    def test_clear_ahead(self):
        """Every shipped planner session's first drive (straight ahead from the start) stays clear of
        the world's boxes and ends well apart from the other agents."""
        import demo_eval as de
        fps = {'jr': (-0.27, 0.18, -0.20, 0.20), 'rr': (-0.26, 0.26, -0.21, 0.21)}   # the spawned URDFs, rounded up
        n = 0
        for f in sorted((REPO / 'config' / 'sessions').glob('*.yaml')):
            s = yaml.safe_load(f.read_text())
            if s.get('planner') is False:
                continue
            boxes = de.world_boxes((de.WORLDS_DIR / (s['world'] + '.world')).read_text())
            ends = []
            for a in s['agents']:
                x, y, t = a['pose'][0], a['pose'][1], math.radians(a['pose'][2])
                for k in range(13):
                    d = (gr.FIRST_DRIVE_M + 0.1) * k / 12
                    poly = de.rect_poly(x + d * math.cos(t), y + d * math.sin(t), t, fps[a['class']])
                    clear = min(de.poly_distance(poly, de.box_poly(b)) for b in boxes)
                    self.assertGreater(clear, 0.3, (f.name, a['name'], d))
                ends.append((x + gr.FIRST_DRIVE_M * math.cos(t), y + gr.FIRST_DRIVE_M * math.sin(t)))
                n += 1
            for i, e in enumerate(ends):
                for o in ends[i + 1:]:
                    self.assertGreater(math.hypot(e[0] - o[0], e[1] - o[1]), 1.5, f.name)
        self.assertGreaterEqual(n, 8)


class RelayTest(unittest.TestCase):
    def relay(self, mode='swap', poller=None, first=0.0):
        sp = poses()
        goals = {n: g for n, (_, g) in SWAP.items()}
        legs = gr.plan_agents(sp, goals, ['JR81', 'RR83'], mode, 0, 0.5, 'estimate', None)
        node = FakeNode()
        with mock.patch.dict(sys.modules, FAKE_MODULES):
            r = gr.Relay(node, sp, legs, mode, ready_timeout=180.0, poller=poller,
                         first=gr.first_drives(legs, {}, first, 0.3))
        return r, node

    def test_entity_count_is_the_leaves(self):
        # sim.py waits this long for the paced leave (goal_relay_leave_s) before it SIGTERMs.
        r, _ = self.relay(first=1.2)
        self.assertEqual(len(r.all_entities()), gr.ENTITIES_PER_AGENT * 2 + gr.ENTITIES_ROOT)
        self.assertEqual((gr.ENTITIES_PER_AGENT, gr.ENTITIES_ROOT), sim.GOAL_RELAY_ENTITIES)
        self.assertGreaterEqual(sim.goal_relay_leave_s(4), (8 * 4 + 2) * gr.LEAVE_GAP_S + gr.LEAVE_PAUSE_S)

    def test_first_drive_before_the_swap(self):
        """Each agent drives off its spot in JOY mode, through safety_node, before any leg starts."""
        r, node = self.relay(first=1.2)
        topics = {e.topic for e in node.entities}
        self.assertTrue({'/JR81/joy_teleop/cmd_vel', '/JR81/joy_mode', '/JR81/auto_mode', '/RR83/auto_mode'} <= topics)
        self.assertEqual(len(r.all_entities()), len(node.entities))           # all left gently
        x, stamp = {'JR81': 0.0, 'RR83': 0.0}, [1790000000 * 10 ** 9]

        def tick(mode, dt=0.0):
            stamp[0] += int(dt * 1e9)
            for ns in x:
                node.topic('/%s/mode' % ns).cb(NS(data=mode))
                node.topic('/%s/dlio/odom_node/odom' % ns).cb(odom_cdr(x[ns], 0.0, 0.0, stamp_ns=stamp[0]))
            r.tick()
        tick('AUTO')
        tick('AUTO', dt=3.0)                     # DLIO still in its IMU calibration (odom on a wall timer)
        self.assertEqual(node.topic('/JR81/joy_mode').requests, [])
        tick('AUTO', dt=1.5)
        self.assertEqual(len(node.topic('/JR81/joy_mode').requests), 1)
        tick('JOY')
        (v,) = node.topic('/JR81/joy_teleop/cmd_vel').sent
        self.assertEqual(v.linear.x, 0.3)
        self.assertEqual(node.topic('/JR81/term_goal_rviz').sent, [])        # no goal while it drives
        self.assertEqual([fd.state for fd in r.first.values()], ['driving', 'driving'])
        self.assertIn('first drive: JR81=wait RR83=wait', [m for _, m in node.logs if 'swap waits' in m][-1])
        x.update(JR81=1.25, RR83=1.25)
        tick('JOY')
        with mock.patch.object(gr.time, 'time', lambda: r.first['JR81'].t + 1.0):
            r._first_drives(r.first['JR81'].t + 1.0)
        self.assertEqual(len(node.topic('/RR83/auto_mode').requests), 1)
        with mock.patch.object(gr.time, 'time', lambda: r.first['JR81'].t + 1.1):
            tick('AUTO')
        self.assertTrue(all(fd.done for fd in r.first.values()))
        self.assertTrue(any(m.startswith('FIRST {') for _, m in node.logs))
        self.assertEqual(len(node.topic('/JR81/term_goal_rviz').sent), 1)    # the swap's first leg

    def test_endpoints(self):
        r, node = self.relay()
        topics = {e.topic: e for e in node.entities}
        self.assertEqual(set(topics), {
            '/JR81/term_goal_rviz', '/JR81/ground_truth/odom_wall', '/JR81/dlio/odom_node/odom', '/JR81/mode',
            '/JR81/term_goal', '/RR83/term_goal_rviz', '/RR83/ground_truth/odom_wall', '/RR83/dlio/odom_node/odom',
            '/RR83/mode', '/RR83/term_goal', '/sim/goal', '/sim/goal_agent'})
        # goal_selector's critical_qos on term_goal_rviz; ground truth and DLIO serialized, latest only
        self.assertEqual(topics['/JR81/term_goal_rviz'].qos,
                         {'history': 'keep_last', 'depth': 10, 'reliability': 'reliable', 'durability': 'volatile'})
        self.assertTrue(topics['/JR81/dlio/odom_node/odom'].raw and topics['/RR83/ground_truth/odom_wall'].raw)
        self.assertEqual(topics['/JR81/dlio/odom_node/odom'].qos['reliability'], 'best_effort')
        self.assertEqual(len(r.all_entities()), len(node.entities))

    def feed(self, node, ns, x, y, yaw=0.0):
        node.topic('/%s/ground_truth/odom_wall' % ns).cb(odom_cdr(x, y, yaw))

    def test_swap_starts_together_once_ready(self):
        r, node = self.relay()
        sp = poses()
        for ns in ('JR81', 'RR83'):
            node.topic('/%s/dlio/odom_node/odom' % ns).cb(odom_cdr(0.0, 0.0, 0.0))
        node.topic('/JR81/mode').cb(NS(data='AUTO'))
        r.tick()
        self.assertEqual(node.topic('/JR81/term_goal_rviz').sent, [])        # RR83 is still DISARMED
        self.assertIn('swap waits for RR83', node.logs[-1][1])
        node.topic('/RR83/mode').cb(NS(data='AUTO'))
        r.tick()
        (m,) = node.topic('/JR81/term_goal_rviz').sent
        self.assertEqual(m.header.frame_id, 'JR81/map')
        self.assertAlmostEqual(m.pose.position.x, 8.004, places=3)
        self.assertAlmostEqual(m.pose.position.y, 0.0, places=9)
        self.assertEqual((m.pose.position.z, m.pose.orientation.w), (0.0, 1.0))
        self.assertEqual(len(node.topic('/RR83/term_goal_rviz').sent), 1)
        # goal_selector echoes it; a COMMIT line says whose it is
        echo = pose_stamped()
        echo.pose.position.x, echo.header.stamp = m.pose.position.x, NS(sec=5, nanosec=7)
        node.topic('/JR81/term_goal').cb(echo)
        commit = json.loads(node.logs[-1][1][len('COMMIT '):])
        self.assertEqual((commit['agent'], commit['ours'], commit['stamp']), ('JR81', True, 5000000007))
        self.assertEqual(commit['world'], [-2.83, -2.83])
        self.assertAlmostEqual(gr.to_world(sp['JR81'], m.pose.position.x, m.pose.position.y)[0], -2.83)

    def test_rviz_goal_to_the_nearest(self):
        r, node = self.relay(mode='rviz')
        self.feed(node, 'JR81', 2.83, 2.83)
        self.feed(node, 'RR83', -2.83, -2.83)
        g = pose_stamped()
        g.header.frame_id, g.pose.position.x, g.pose.position.y = 'world', 2.0, 2.0
        node.topic('/sim/goal').cb(g)
        (m,) = node.topic('/JR81/term_goal_rviz').sent
        wx, wy = gr.to_world(poses()['JR81'], m.pose.position.x, m.pose.position.y)
        self.assertEqual((round(wx, 6), round(wy, 6)), (2.0, 2.0))
        node.topic('/sim/goal_agent').cb(NS(data='RR83'))
        node.topic('/sim/goal').cb(g)
        self.assertEqual(len(node.topic('/RR83/term_goal_rviz').sent), 1)
        node.topic('/sim/goal_agent').cb(NS(data='JR02'))
        self.assertEqual(node.logs[-1][0], 'warn')
        g.header.frame_id = 'JR02/map'
        node.topic('/sim/goal').cb(g)
        self.assertIn('refused', node.logs[-1][1])
        self.assertEqual(len(node.topic('/JR81/term_goal_rviz').sent) + len(node.topic('/RR83/term_goal_rviz').sent), 2)

    def test_no_subscriber_warns_once(self):
        r, node = self.relay(mode='rviz')
        node.topic('/JR81/term_goal_rviz').subscribers = 0
        g = pose_stamped()
        g.header.frame_id = 'JR81/map'
        for _ in range(3):
            node.topic('/sim/goal').cb(g)
        self.assertEqual(sum(1 for k, m in node.logs if k == 'warn' and 'no subscriber' in m), 1)

    def test_leave_one_at_a_time(self):
        poller = gr.Poller(FakeNode, 'goal_relay_poll', '/sim')
        r, node = self.relay(poller=poller)
        spins = []
        with mock.patch.object(gr.time, 'sleep', lambda s: spins.append(('sleep', s))):
            gr.leave_gently(r.all_entities(), [poller.node, node], lambda: spins.append('spin'))
        destroyed = [c for n in (poller.node, node) for c in n.calls if isinstance(c, tuple)]
        self.assertEqual([t for _, t in destroyed], [e.topic for _, _, e in r.all_entities()])
        self.assertEqual(len(destroyed), 8 + 4)                             # 4 of them polled
        self.assertEqual((poller.node.calls[-1], node.calls[-1]), ('destroy_node', 'destroy_node'))
        self.assertEqual(spins[:2], ['spin', ('sleep', gr.LEAVE_GAP_S)])
        self.assertEqual(spins[-1], ('sleep', gr.LEAVE_PAUSE_S))
        self.assertEqual(spins.count('spin'), len(r.all_entities()))

    def test_polled_ground_truth_and_estimate(self):
        poller = gr.Poller(FakeNode, 'goal_relay_poll', '/sim')
        r, node = self.relay(poller=poller)
        self.assertEqual(poller.node.kw, {'namespace': '/sim', 'enable_rosout': False, 'start_parameter_services': False})
        polled = {e.topic: e for e in poller.node.entities}
        self.assertEqual(sorted(polled), ['/JR81/dlio/odom_node/odom', '/JR81/ground_truth/odom_wall',
                                          '/RR83/dlio/odom_node/odom', '/RR83/ground_truth/odom_wall'])
        self.assertNotIn('/JR81/ground_truth/odom_wall', {e.topic for e in node.entities})
        q = polled['/JR81/ground_truth/odom_wall']
        q.queue = [odom_cdr(1.0, 0.0, 0.0), odom_cdr(2.83, 2.83, 0.0)]      # only the newest is used
        r.tick()
        self.assertEqual(r.truth['JR81'][:2], (2.83, 2.83))
        self.assertEqual(q.queue, [])


class MainTest(unittest.TestCase):
    """main() refuses what is not the sim before rclpy opens a session."""

    def run_main(self, argv, env='/rover_sim/zenoh/session.json5', spawn=None, session=None):
        inits = []
        rclpy = types.ModuleType('rclpy')
        rclpy.init = lambda **k: inits.append(k)
        fake = {'rclpy': rclpy, 'rclpy.utilities': NS(remove_ros_args=lambda a: list(a)),
                'rclpy.node': NS(Node=FakeNode), 'rclpy.signals': NS(SignalHandlerOptions=NS(NO='no'))}
        with tempfile.TemporaryDirectory() as d:
            sp, se = os.path.join(d, 'spawn.yaml'), os.path.join(d, 'session.json')
            Path(sp).write_text(yaml.safe_dump(spawn or spawn_data()))
            Path(se).write_text(json.dumps(session or session_data()))
            environ = {k: v for k, v in os.environ.items() if k != 'ZENOH_SESSION_CONFIG_URI'}
            if env:
                environ['ZENOH_SESSION_CONFIG_URI'] = env
            err = io.StringIO()
            with mock.patch.dict(sys.modules, fake), mock.patch.dict(os.environ, environ, clear=True), \
                    contextlib.redirect_stderr(err):
                code = gr.main(['goal_relay.py', '--spawn', sp, '--session', se, '--goal-radius', '0.5'] + argv)
        return code, err.getvalue(), inits

    def test_refusals(self):
        for env, argv, says in ((None, [], 'not in a rover_sim container'),
                                ('/etc/zenoh.json5', [], 'not in a rover_sim container'),
                                ('/rover_sim/zenoh/session.json5', ['--agents', 'JR81,JR02'], 'no sim agent JR02'),
                                ('/rover_sim/zenoh/session.json5', ['--laps', '-1'], '--laps >= 0')):
            code, err, inits = self.run_main(argv, env=env)
            self.assertEqual(code, 2, err)
            self.assertIn(says, err)
            self.assertEqual(inits, [])
        code, err, inits = self.run_main(['--mode', 'swap'], session=session_data(
            {n: (p, []) for n, (p, _) in SWAP.items()}, demo='none'))
        self.assertIn('no served agent has a session goal', err)
        self.assertEqual(inits, [])

    def test_needs_a_goal_radius(self):
        with tempfile.TemporaryDirectory() as d:
            args = gr.build_parser().parse_args(['--spawn', os.path.join(d, 'none.yaml')])
            with mock.patch.dict(os.environ, {'ZENOH_SESSION_CONFIG_URI': '/rover_sim/zenoh/session.json5'}):
                with self.assertRaisesRegex(gr.GoalError, 'none.yaml'):
                    gr.setup(args)


if __name__ == '__main__':
    unittest.main()
