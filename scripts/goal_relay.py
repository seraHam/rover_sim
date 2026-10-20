#!/usr/bin/env python3
"""Goals for the sim's agents: RViz goals in the world frame, and the swap demo's legs.

  goal_relay.py [--mode auto|swap|rviz|explore] [--agents JR81,RR83] [--laps N]
                [--spawn /rover_sim/run/spawn.yaml] [--session /rover_sim/run/session.json]
                [--reached-by estimate|truth] [--goal-radius m] [--leg-timeout s]

One process for the session, node /sim/goal_relay, in sim-mighty. Every goal goes to an
agent's goal_selector as the robot's operator sends one: /<ns>/term_goal_rviz, a
geometry_msgs/PoseStamped with goal_selector's QoS (reliable, volatile, keep last 10). goal_selector
ignores frame_id and takes x, y as <ns>/map coordinates, so the relay sends them there (frame
<ns>/map, z 0, orientation identity). goal_selector commits it as a manual goal, which preempts
exploration, and republishes it on term_goal for mighty_node.

RViz (always on): /sim/goal, the PoseStamped of RViz's 2D Goal Pose tool. Its header.frame_id
picks the agent:
  world (or empty)       the agent selected on /sim/goal_agent, else the agent nearest the goal
                         by ground truth (ground_truth/odom_wall); the goal goes from world to
                         <ns>/map through the agent's spawn pose (spawn.yaml), what sim_tf.py
                         publishes as world -> <ns>/map
  <ns>/map, <ns>/odom    that agent, the goal already in its map (map -> odom is identity)
  anything else          refused
/sim/goal_agent (std_msgs/String) selects the agent for world goals: '<ns>', or '' / 'nearest'
to go back to the nearest agent. A goal for an agent that is cycling the swap interrupts its leg
(logged as preempted) and runs as an rviz leg under the same rules; the interrupted leg then
starts again.

--mode auto (the default) takes the session's demo from session.json: swap, explore, or none
(rviz); a session.json without one gives swap when it has goals.

First drive (every mode, --first-drive, default 1.2 m): an agent in AUTO whose DLIO is past its
IMU calibration is switched to safety_node's JOY mode, driven straight ahead on
/<ns>/joy_teleop/cmd_vel (the gamepad's path) and handed back to AUTO, as an operator drives a
rover before autonomy; nothing else is sent to it before that. From a cold start mighty plans no
path over floor the rover never saw (FirstDrive). One FIRST {json} line per agent.

Swap demo (--mode swap): each agent drives start -> goal 1 -> ... -> goal N -> start, round and
round (--laps, 0 = until stopped). start is the spawn pose; goals are the session's, in the world
frame. The agents start together, once each is in AUTO (safety_node's mode), has a DLIO estimate,
has had its first drive, and its goal_selector subscribes to term_goal_rviz. A leg ends
  reached   when the agent's own estimate (DLIO odom: <ns>/odom, the same as <ns>/map) is within
            goal_radius of the goal, mighty_node's own REACHED test (mighty.cpp needReplan: xy
            distance from the state to the terminal goal < goal_radius; mighty_node's goal_radius
            from the planner config, hw_mighty_ground_robot.yaml); --reached-by truth uses ground
            truth instead;
  timeout   after 30 s + 6 s per metre of straight-line distance (3x the time at 0.5 m/s, mighty's
            ground_robot_v_max), or --leg-timeout, and the next leg starts anyway.
The next leg's goal goes out the moment a leg ends, and agents do not wait for each other:
hw_goal_selector.yaml has exploration on, so an agent without a manual goal is sent to a frontier
within a second. For the same reason, while a leg runs, a goal_selector commitment anywhere else
(term_goal; e.g. exploration after the stuck watchdog released the goal of an agent waiting for
another) gets the leg's goal sent again, at most every 2 s, and so does a goal that goal_selector
did not echo on term_goal within 2 s. After the last lap the agents are goal_selector's again.

Explore demo (--mode explore) and --mode rviz: nothing to send. goal_selector explores by itself
(exploration.enabled true, from its first frontier) and pauses for a manual goal until it is
reached or released. The relay relays RViz goals and logs every commitment it sees.

Every leg ends with one line, LEG followed by a JSON object (parse_leg_line; demo_eval.py reads
them from this pane's log):
  agent leg lap kind (swap|rviz) from to result (reached|timeout|preempted) t_start t_end
  duration_s (wall clock, s) target_map target_world est_dist truth_dist (m, xy, at the end)
  reached_by timeout_s resends foreign_commits
and every goal_selector commitment with COMMIT {agent, stamp, map, world, ours}.

Only sim agents get goals: the zenoh session config must be the sim's (ZENOH_SESSION_CONFIG_URI,
config/sim.env) and every agent must be in the sim's spawn.yaml, both checked before rclpy opens a
session. The node has no rosout publisher and no parameter services, and on exit (Ctrl-C,
SIGTERM) it leaves the graph one entity at a time (scripts/sim.py STATUS_LEAVE_GAP_S): with
rmw_zenoh 0.1.9 a session that undeclares many entities at once stalls delivery for everyone.

The pure parts (frames, the agent choice, the legs) import without rclpy;
tests/test_goal_relay.py covers them.
"""

import argparse
import json
import math
import os
import re
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import drive_pattern  # noqa: E402  require_sim: the sim's zenoh session only
import livox_sim_bridge  # noqa: E402  odom_parse: serialized Odometry, no rclpy message objects
import sim_tf  # noqa: E402  spawn.yaml parsing

NODE_NAMESPACE, NODE_NAME = '/sim', 'goal_relay'
GOAL_TOPIC, SELECT_TOPIC = '/sim/goal', '/sim/goal_agent'
WORLD = 'world'
# mighty's planner config as the mighty-hw image installs it (mighty_node's goal_radius).
PLANNER_CONFIG = '/home/swarm/code/mighty_ws/install/mighty/share/mighty/config/hw_mighty_ground_robot.yaml'
LEG_TIMEOUT_BASE_S = 30.0
LEG_TIMEOUT_S_PER_M = 6.0          # 3x the time at mighty's ground_robot_v_max (0.5 m/s)
RESEND_S = 2.0                     # min gap between sends of one leg's goal
ECHO_TOL_M = 0.05                  # a term_goal this close to the leg's goal is goal_selector's echo
FRESH_S = 1.0                      # older estimates and ground truth are not used
LEAVE_GAP_S, LEAVE_PAUSE_S = 0.5, 0.5   # scripts/sim.py STATUS_LEAVE_GAP_S, STATUS_LEAVE_PAUSE_S
# Graph entities: per agent term_goal_rviz, joy_teleop/cmd_vel, the truth and estimate polls, mode,
# term_goal and the joy_mode/auto_mode clients; /sim/goal and /sim/goal_agent once (sim.py
# GOAL_RELAY_ENTITIES sizes `down`'s and `start goals`' wait for the leave from these).
ENTITIES_PER_AGENT, ENTITIES_ROOT = 8, 2
FIRST_DRIVE_M, FIRST_DRIVE_SPEED = 1.2, 0.3   # past an RR's 0.77 m blind floor, slowly (FirstDrive)
# DLIO publishes odom on a wall timer from its start, stamped with its last IMU sample (0 before
# the first): an odom stamp this far past the first non-zero one is past its 3 s IMU calibration.
DLIO_SETTLED_S = 4.0
FIRST_RE = re.compile(r'\bFIRST (\{.*\})\s*$')
LEG_RE = re.compile(r'\bLEG (\{.*\})\s*$')


class GoalError(ValueError):
    pass


# ---------------------------------------------------------------------------- frames

def to_map(p, x, y):
    """World (x, y) in the map frame of the agent spawned at p (sim_tf.SpawnPose)."""
    dx, dy = x - p.x, y - p.y
    c, s = math.cos(p.yaw), math.sin(p.yaw)
    return c * dx + s * dy, -s * dx + c * dy


def to_world(p, x, y):
    c, s = math.cos(p.yaw), math.sin(p.yaw)
    return p.x + c * x - s * y, p.y + s * x + c * y


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def quat_yaw(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# ---------------------------------------------------------------------------- RViz goals

def frame_agent(frame_id, names):
    """The agent a goal's frame names: None for world, ns for '<ns>/map' or '<ns>/odom'."""
    f = (frame_id or '').strip('/')
    if f in ('', WORLD):
        return None
    ns, _, rest = f.partition('/')
    if rest not in ('map', 'odom'):
        raise GoalError("frame %r: send it in 'world', or in '<ns>/map' for one agent" % frame_id)
    if ns not in names:
        raise GoalError('frame %r: no sim agent %s (spawn.yaml has %s)' % (frame_id, ns, ', '.join(sorted(names))))
    return ns


def nearest_agent(truth, x, y):
    """The agent whose ground truth {ns: (x, y)} is nearest (x, y), or None."""
    best = min(truth.items(), key=lambda kv: (dist(kv[1], (x, y)), kv[0]), default=None)
    return best[0] if best else None


def resolve_goal(frame_id, x, y, spawns, truth, selected=None):
    """(ns, x_map, y_map, how) of a /sim/goal pose; how = frame | selected | nearest."""
    ns = frame_agent(frame_id, spawns)
    if ns:
        return ns, x, y, 'frame'
    if selected:
        ns, how = selected, 'selected'
    else:
        ns, how = nearest_agent(truth, x, y), 'nearest'
        if ns is None:
            raise GoalError('no ground truth from any agent yet: cannot pick the nearest one')
    mx, my = to_map(spawns[ns], x, y)
    return ns, mx, my, how


def parse_selection(text, names):
    """/sim/goal_agent's text: an agent name, or None for the nearest agent."""
    t = (text or '').strip()
    if t.lower() in ('', 'nearest', 'none'):
        return None
    if t not in names:
        raise GoalError('/sim/goal_agent %r: no sim agent of that name (spawn.yaml has %s)'
                        % (t, ', '.join(sorted(names))))
    return t


# ---------------------------------------------------------------------------- legs

def swap_stops(spawn, goals, radius):
    """[(label, (x_map, y_map))] of one agent's cycle: start (its spawn pose, the map origin),
    then its session goals (world frame). A goal within radius of the stop before it would be
    reached at once, so it is dropped, and so is a last goal back at the start (the cycle returns
    there anyway)."""
    stops = [('start', (0.0, 0.0))]
    for i, g in enumerate(goals, 1):
        xy = to_map(spawn, float(g[0]), float(g[1]))
        if dist(xy, stops[-1][1]) >= radius:
            stops.append(('goal%d' % i, xy))
    if len(stops) > 2 and dist(stops[-1][1], stops[0][1]) < radius:
        stops.pop()
    return stops if len(stops) >= 2 else []


def leg_timeout(d, fixed=None):
    return float(fixed) if fixed else LEG_TIMEOUT_BASE_S + LEG_TIMEOUT_S_PER_M * d


def leg_line(rec):
    return 'LEG ' + json.dumps(rec, sort_keys=True)


def parse_leg_line(line):
    """The record of a LEG line (also with a ROS log prefix), else None."""
    m = LEG_RE.search(line)
    if not m:
        return None
    try:
        rec = json.loads(m.group(1))
    except ValueError:
        return None
    return rec if isinstance(rec, dict) and 'agent' in rec else None


def _r(v, n=3):
    return None if v is None else round(v, n)


class AgentLegs:
    """One agent's legs: its swap cycle (stops, laps) and its RViz goals. Pure: every method
    takes the wall time and measurements (estimate in <ns>/map, truth in world) and returns
    actions, [('send', (x_map, y_map))], [('log', record)] and [('info', text)]."""

    def __init__(self, ns, spawn, stops=(), laps=0, goal_radius=0.5, reached_by='estimate',
                 fixed_timeout=None, resend_s=RESEND_S):
        self.ns, self.spawn, self.stops, self.laps = ns, spawn, list(stops), laps
        self.radius, self.reached_by, self.fixed_timeout, self.resend_s = (
            goal_radius, reached_by, fixed_timeout, resend_s)
        self.leg = None             # the running leg
        self.n = 0                  # swap legs started
        self.resume = None          # the swap leg an rviz goal interrupted
        self.started = self.finished = False
        self.records = []

    @property
    def cycling(self):
        return len(self.stops) >= 2

    def _begin(self, now, kind, frm, to, target, origin, index=None):
        """origin: where the agent is (map frame), for the leg's timeout; None = unknown."""
        d = dist(origin, target) if origin is not None else 0.0
        self.leg = {'kind': kind, 'from': frm, 'to': to, 'target': tuple(target), 't0': now,
                    'deadline': now + leg_timeout(d, self.fixed_timeout), 'sent': now, 'ours_last': False,
                    'resends': 0, 'foreign': 0, 'index': index}
        return [('send', tuple(target))]

    def _swap_leg(self, now, origin, i, frm=None):
        k = len(self.stops)
        a, b = self.stops[i % k], self.stops[(i + 1) % k]
        acts = self._begin(now, 'swap', frm or a[0], b[0], b[1], origin, index=i)
        self.leg.update(leg=i + 1, lap=i // k + 1)
        return acts

    def _next_swap(self, now, origin):
        if self.laps and self.n >= self.laps * len(self.stops):
            if self.finished:
                return []
            self.finished = True
            return [('info', '%s: swap done after %d legs; goal_selector has it from here' % (self.ns, self.n))]
        i, self.n = self.n, self.n + 1
        return self._swap_leg(now, origin, i)

    def _origin(self, est, truth):
        if est is not None:
            return est
        return to_map(self.spawn, *truth) if truth is not None else None

    def start(self, now, est, truth=None):
        """Begin the swap cycle (once). With an rviz goal running, it begins after that."""
        if not self.cycling or self.started:
            return []
        self.started = True
        if self.leg is not None:
            self.resume, self.n = 0, 1
            return []
        return self._next_swap(now, self._origin(est, truth))

    def _dists(self, est, truth):
        t = self.leg['target']
        d_est = dist(est, t) if est is not None else None
        d_truth = dist(to_map(self.spawn, *truth), t) if truth is not None else None
        return d_est, d_truth

    def _end(self, now, result, est, truth):
        g = self.leg
        d_est, d_truth = self._dists(est, truth)
        rec = {'agent': self.ns, 'kind': g['kind'], 'from': g['from'], 'to': g['to'], 'result': result,
               't_start': round(g['t0'], 3), 't_end': round(now, 3), 'duration_s': round(now - g['t0'], 3),
               'target_map': [_r(v) for v in g['target']],
               'target_world': [_r(v) for v in to_world(self.spawn, *g['target'])],
               'est_dist': _r(d_est), 'truth_dist': _r(d_truth), 'reached_by': self.reached_by,
               'timeout_s': round(g['deadline'] - g['t0'], 1), 'resends': g['resends'],
               'foreign_commits': g['foreign'], 'leg': g.get('leg'), 'lap': g.get('lap')}
        self.leg = None
        self.records.append(rec)
        return [('log', rec)]

    def _after(self, now, origin, interrupted):
        """What follows a leg that ended: the swap leg an rviz goal interrupted, else the next."""
        if interrupted is not None:
            self.resume = None
            return self._swap_leg(now, origin, interrupted, frm='rviz')
        return self._next_swap(now, origin) if self.started else []

    def manual(self, now, target, est, truth=None):
        """An RViz goal: it replaces the running leg; a swap leg it interrupts runs again after it."""
        acts = []
        if self.leg is not None:
            if self.leg['kind'] == 'swap':
                self.resume = self.leg['index']
            acts += self._end(now, 'preempted', est, truth)
        return acts + self._begin(now, 'rviz', 'here', 'rviz', target, self._origin(est, truth))

    def update(self, now, est, truth):
        """est: (x, y) in <ns>/map or None; truth: (x, y) in world or None."""
        if self.leg is None:
            return []
        d_est, d_truth = self._dists(est, truth)
        d = d_est if self.reached_by == 'estimate' else d_truth
        result = ('reached' if d is not None and d < self.radius else
                  'timeout' if now >= self.leg['deadline'] else None)
        if result:
            interrupted = self.resume if self.leg['kind'] == 'rviz' else None
            return self._end(now, result, est, truth) + self._after(now, self._origin(est, truth), interrupted)
        g = self.leg
        if not g['ours_last'] and now - g['sent'] >= self.resend_s:
            g['sent'] = now
            g['resends'] += 1
            return [('send', g['target'])]
        return []

    def on_commit(self, x, y):
        """goal_selector committed (x, y) (term_goal). True when it is the running leg's goal."""
        if self.leg is None:
            return False
        ours = dist((x, y), self.leg['target']) <= ECHO_TOL_M
        self.leg['ours_last'] = ours
        if not ours:
            self.leg['foreign'] += 1
        return ours


# ---------------------------------------------------------------------------- first drive

class FirstDrive:
    """The operator's first drive of one agent, before any goal: from AUTO, safety_node's JOY mode
    (/<ns>/joy_mode), straight ahead `dist` m at `speed` on /<ns>/joy_teleop/cmd_vel (the gamepad's
    path through safety_node), stop, then AUTO again (/<ns>/auto_mode).

    Why: a parked rover has never seen the floor in front of it (the MID-360 is tilted down: its
    lowest beam meets the floor 0.5 m ahead of a JR, 0.77 m ahead of an RR) or beside and behind
    it, and mighty cuts every path where it first meets 3 unknown cells, then backs it off
    0.5 m: from a cold start the path is one point and the rover waits, for goals and frontiers
    alike. On the robot the operator has driven it to where autonomy starts, so that floor was
    seen. Pure: step() takes the wall time, safety_node's mode and the DLIO estimate (x, y, z) in
    <ns>/map (None if stale) and returns [('call', 'joy_mode'|'auto_mode'), ('vel', v),
    ('log', record), ('info', text)]. An agent that is not in AUTO is never armed here, and one the
    operator takes out of the drive's modes (disarmed) is left alone. An agent already off its spot
    (AWAY_M from the <ns>/map origin, its spawn: a relay restarted mid-run) is not driven again."""

    RETRY_S = 3.0                      # a mode request with no mode change is sent again
    AWAY_M = 0.3                       # this far from the spawn: it has been driven already
    STOP_S = 0.5                       # zeros before AUTO (safety_node smooths JOY commands, tau 0.2 s)
    EXTRA_S = 10.0                     # beyond dist / speed, then the drive ends where it is

    def __init__(self, ns, dist, speed, z_expect=None, z_tol=0.15):
        self.ns, self.dist, self.speed, self.z_expect, self.z_tol = ns, dist, speed, z_expect, z_tol
        self.state = 'done' if dist <= 0 else 'wait'
        self.t = self.start = None
        self.t_drive = None
        self.warned = False
        self.record = None

    @property
    def done(self):
        return self.state == 'done'

    def step(self, now, mode, est):
        s = self.state
        if s == 'done':
            return []
        if s == 'wait':
            if mode != 'AUTO' or est is None:
                return []
            if math.hypot(est[0], est[1]) > self.AWAY_M:
                self.state = 'done'
                return [('info', '%s: %.2f m from its spawn already: no first drive (driven before this relay '
                                 'started)' % (self.ns, math.hypot(est[0], est[1])))]
            if self.z_expect is not None and abs(est[2] - self.z_expect) > self.z_tol:
                if self.warned:
                    return []
                self.warned = True
                return [('info', '%s: DLIO z %.2f, expected %.2f: no first drive (its seed came late? restart '
                                 'its dlio pane)' % (self.ns, est[2], self.z_expect))]
            self.state, self.t = 'to_joy', now
            return [('call', 'joy_mode')]
        if s == 'to_joy':
            if mode == 'JOY' and est is not None:
                self.state, self.start, self.t_drive = 'driving', est, now
                return [('vel', self.speed)]
            if mode not in ('AUTO', 'JOY'):    # the operator disarmed it meanwhile: not ours to arm
                self.state = 'wait'
                return [('info', '%s: first drive given up (mode %s)' % (self.ns, mode))]
            if now - self.t >= self.RETRY_S:
                self.t = now
                return [('call', 'joy_mode')]
            return []
        if s == 'driving':
            if mode != 'JOY':                  # someone took it over (disarmed, AUTO): start again later
                self.state = 'wait'
                return [('info', '%s: first drive interrupted (mode %s)' % (self.ns, mode))]
            late = now - self.t_drive >= self.dist / self.speed + self.EXTRA_S
            if est is None and not late:
                return [('vel', 0.0)]          # no estimate: hold still meanwhile, within the time limit
            moved = dist(est, self.start) if est is not None else None
            if late or moved >= self.dist:
                self.state, self.t = 'stopping', now
                self.record = {'agent': self.ns, 'from_map': [_r(v) for v in self.start[:2]],
                               'to_map': [_r(v) for v in est[:2]] if est is not None else None,
                               'moved': _r(moved) if moved is not None else None,
                               'duration_s': round(now - self.t_drive, 2), 'wanted': self.dist}
                return [('vel', 0.0)]
            return [('vel', self.speed)]
        if s == 'stopping':
            if mode != 'JOY':                  # disarmed (or taken over) while stopping: leave it so
                self.state = 'wait'
                return [('info', '%s: first drive interrupted while stopping (mode %s)' % (self.ns, mode))]
            if now - self.t < self.STOP_S:
                return [('vel', 0.0)]
            self.state, self.t = 'to_auto', now
            return [('call', 'auto_mode')]
        if s == 'to_auto':
            if mode == 'AUTO':
                self.state = 'done'
                return [('log', self.record)]
            if mode != 'JOY':                  # disarmed before AUTO came back: not ours to arm
                self.state = 'wait'
                return [('info', '%s: first drive left in mode %s (not re-armed)' % (self.ns, mode))]
            if now - self.t >= self.RETRY_S:
                self.t = now
                return [('call', 'auto_mode')]
            return []
        return []


def first_line(rec):
    return 'FIRST ' + json.dumps(rec, sort_keys=True)


# ---------------------------------------------------------------------------- config

def load_session(path):
    """(demo or None, {ns: [(x, y), ...]}) from sim.py's <run>/session.json (world frame goals)."""
    with open(path) as f:
        data = json.load(f)
    goals = {}
    for a in data.get('agents') or []:
        goals[a['name']] = [(float(g[0]), float(g[1])) for g in a.get('goals') or []]
    return data.get('demo'), goals


def load_goal_radius(path):
    """mighty_node's goal_radius from a planner config (hw_mighty_ground_robot.yaml)."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    v = ((data.get('mighty_node') or {}).get('ros__parameters') or {}).get('goal_radius')
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not v > 0:
        raise GoalError('%s: no positive mighty_node goal_radius' % path)
    return float(v)


MODES = ('auto', 'swap', 'rviz', 'explore')
SESSION_DEMO_MODE = {'swap': 'swap', 'explore': 'explore', 'none': 'rviz'}


def pick_mode(mode, session_demo, goals):
    """--mode auto: the session's demo (sim.py DEMOS), else swap when it gives goals."""
    if mode != 'auto':
        return mode
    if session_demo in SESSION_DEMO_MODE:
        return SESSION_DEMO_MODE[session_demo]
    return 'swap' if any(goals.values()) else 'rviz'


def plan_agents(spawns, goals, names, mode, laps, radius, reached_by, fixed_timeout):
    """{ns: AgentLegs} for the served agents; refuses names the sim did not spawn."""
    unknown = [n for n in names if n not in spawns]
    if unknown:
        raise GoalError('no sim agent %s in spawn.yaml (it has %s)' % (', '.join(unknown), ', '.join(sorted(spawns))))
    out = {}
    for ns in names:
        stops = swap_stops(spawns[ns], goals.get(ns) or [], radius) if mode == 'swap' else []
        out[ns] = AgentLegs(ns, spawns[ns], stops, laps, radius, reached_by, fixed_timeout)
    if mode == 'swap' and not any(a.cycling for a in out.values()):
        raise GoalError('swap, but no served agent has a session goal away from its start')
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--mode', choices=MODES, default='auto',
                    help="auto (default): the session's demo; swap; rviz or explore: RViz goals only")
    ap.add_argument('--agents', default='', help='comma-separated; default every agent in spawn.yaml')
    ap.add_argument('--laps', type=int, default=0, help='swap cycles per agent; 0 = until stopped')
    ap.add_argument('--spawn', default='/rover_sim/run/spawn.yaml')
    ap.add_argument('--session', default='/rover_sim/run/session.json')
    ap.add_argument('--reached-by', choices=('estimate', 'truth'), default='estimate')
    ap.add_argument('--goal-radius', type=float, default=None, help="m; default mighty_node's (--planner-config)")
    ap.add_argument('--planner-config', default=PLANNER_CONFIG)
    ap.add_argument('--leg-timeout', type=float, default=None,
                    help='s per leg; default %.0f s + %.0f s/m' % (LEG_TIMEOUT_BASE_S, LEG_TIMEOUT_S_PER_M))
    ap.add_argument('--ready-timeout', type=float, default=180.0,
                    help='s to wait for every agent to be ready before the ready ones start alone')
    ap.add_argument('--first-drive', type=float, default=FIRST_DRIVE_M,
                    help='m each agent drives straight ahead in JOY mode once in AUTO, before any goal '
                         '(default %.1f; 0 = none): see FirstDrive' % FIRST_DRIVE_M)
    ap.add_argument('--first-drive-speed', type=float, default=FIRST_DRIVE_SPEED, help='m/s')
    return ap


def lidar_heights(path):
    """{ns: base_link height above the floor} from sim.py's spawn.yaml (DLIO's seed z), where given."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {a['name']: float(a['lidar_height']) for a in data.get('agents') or []
            if isinstance(a, dict) and isinstance(a.get('lidar_height'), (int, float))}


def first_drives(names, heights, dist, speed):
    return {ns: FirstDrive(ns, dist, speed, heights.get(ns)) for ns in names}


def setup(args):
    """(spawns, mode, {ns: AgentLegs}, goal_radius, {ns: FirstDrive}): everything checked before
    rclpy starts."""
    drive_pattern.require_sim()
    try:
        poses = sim_tf.load_spawn(args.spawn)
        heights = lidar_heights(args.spawn)
    except (OSError, sim_tf.SpawnError) as e:
        raise GoalError(str(e))
    spawns = {p.name: p for p in poses}
    names = [n for n in args.agents.split(',') if n] or [p.name for p in poses]
    try:
        session_demo, goals = load_session(args.session)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
        if args.mode in ('auto', 'swap'):
            raise GoalError('%s: %s' % (args.session, e))
        session_demo, goals = None, {}
    radius = args.goal_radius
    if radius is None:
        try:
            radius = load_goal_radius(args.planner_config)
        except (OSError, ValueError) as e:
            raise GoalError('%s (pass --goal-radius)' % e)
    if not radius > 0 or args.laps < 0:
        raise GoalError('--goal-radius must be > 0 and --laps >= 0')
    if args.first_drive < 0 or not args.first_drive_speed > 0:
        raise GoalError('--first-drive must be >= 0 and --first-drive-speed > 0')
    mode = pick_mode(args.mode, session_demo, {n: goals.get(n) for n in names})
    legs = plan_agents(spawns, goals, names, mode, args.laps, radius, args.reached_by, args.leg_timeout)
    return spawns, mode, legs, radius, first_drives(legs, heights, args.first_drive, args.first_drive_speed)


# ---------------------------------------------------------------------------- ROS

def leave_gently(entities, nodes, spin_once, gap=LEAVE_GAP_S, pause=LEAVE_PAUSE_S):
    """Undeclare one entity at a time, spinning in between, then close the nodes: rmw_zenoh 0.1.9
    stalls delivery graph-wide for ~45 ms per entity a session undeclares at once (Phase 4).
    entities: [(node, kind, entity)], kind publisher | subscription."""
    for node, kind, e in entities:
        try:
            getattr(node, 'destroy_' + kind)(e)
        except Exception:  # noqa: BLE001  leaving: one failure must not keep the rest declared
            pass
        spin_once()
        time.sleep(gap)
    time.sleep(pause)
    for node in nodes:
        node.destroy_node()


class Poller:
    """Subscriptions read on a timer rather than one callback per message: they live on a second
    node that no executor spins, and poll() takes what their queues hold, as rclpy's executor does
    (Executor._take_subscription). rclpy costs ~0.3 ms a callback (measured: 12 % of a core for 400
    Odometry/s), so a 100 Hz topic polled at 10 Hz costs a tenth of that. latest: only the newest
    message (keep a depth-1 queue); else every one since the last poll (size the depth for that)."""

    def __init__(self, node_cls, name, namespace):
        self.node = node_cls(name, namespace=namespace, enable_rosout=False, start_parameter_services=False)
        self.subs = []

    def add(self, msg_type, topic, qos, handler, raw=True, latest=True):
        sub = self.node.create_subscription(msg_type, topic, lambda m: None, qos, raw=raw)
        self.subs.append((sub, handler, latest))
        return sub

    def poll(self, which=None):
        for sub, handler, latest in self.subs:
            if which is not None and sub not in which:
                continue
            last = None
            while True:
                with sub.handle:
                    got = sub.handle.take_message(sub.msg_type, sub.raw)
                if got is None:
                    break
                if latest:
                    last = got[0]
                else:
                    handler(got[0])
            if last is not None:
                handler(last)

    def entities(self):
        return [(self.node, 'subscription', sub) for sub, _, _ in self.subs]


class Relay:
    """The node's subscriptions and publishers, feeding AgentLegs."""

    def __init__(self, node, spawns, legs, mode, ready_timeout, poller=None, first=None):
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Odometry
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self.node, self.spawns, self.legs, self.mode_name, self.poller = node, spawns, legs, mode, poller
        self.PoseStamped, self.Twist, self.Trigger = PoseStamped, Twist, Trigger
        self.first = first or {}
        self.log = node.get_logger()
        self.truth, self.est, self.mode, self.first_stamp = {}, {}, {}, {}
        self.selected = None
        self.entities = []
        self.unheard = set()            # agents whose term_goal_rviz had no subscriber at the last send
        self.t_begin, self.ready_timeout = time.time(), ready_timeout
        self.cycle_started, self.last_wait_log = False, 0.0
        # goal_selector's critical_qos for term_goal_rviz; the rest best effort, latest only, so no
        # publisher ever waits for this reader.
        critical = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        latest = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pubs = {}
        for ns in legs:
            pub = node.create_publisher(PoseStamped, '/%s/term_goal_rviz' % ns, critical)
            self.pubs[ns] = self._add('publisher', pub)
            # ground truth and DLIO come at 100 Hz: read the newest at each tick (Poller), serialized
            for store, topic in ((self.truth, 'ground_truth/odom_wall'), (self.est, 'dlio/odom_node/odom')):
                if poller is not None:
                    poller.add(Odometry, '/%s/%s' % (ns, topic), latest, self._on_odom(store, ns))
                else:
                    self._sub(Odometry, '/%s/%s' % (ns, topic), self._on_odom(store, ns), latest, raw=True)
            self._sub(String, '/%s/mode' % ns, self._on_mode(ns), latest)
            self._sub(PoseStamped, '/%s/term_goal' % ns, self._on_commit(ns), critical)
        # The first drive goes through safety_node like the operator's gamepad: its services and
        # teleop_twist_joy's output topic (safety_node's subscription QoS: depth 1, reliable).
        self.joy_pubs, self.clients, self.calls, self.srv_warned = {}, {}, {}, {}
        for ns, fd in self.first.items():
            if fd.done:
                continue
            self.joy_pubs[ns] = self._add('publisher', node.create_publisher(Twist, '/%s/joy_teleop/cmd_vel' % ns, 1))
            for srv in ('joy_mode', 'auto_mode'):
                self.clients[ns, srv] = self._add('client', node.create_client(Trigger, '/%s/%s' % (ns, srv)))
        self._sub(PoseStamped, GOAL_TOPIC, self._on_goal, critical)
        self._sub(String, SELECT_TOPIC, self._on_select, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.timer = node.create_timer(0.1, self.tick)
        cyc = [ns for ns, a in legs.items() if a.cycling]
        self.log.info('mode %s; agents %s%s; goals on %s (world, or <ns>/map), agent choice on %s'
                      % (mode, ', '.join(legs), '; cycling ' + ', '.join(
                          '%s %s' % (ns, '->'.join(s[0] for s in legs[ns].stops)) for ns in cyc) if cyc else '',
                         GOAL_TOPIC, SELECT_TOPIC))

    def _add(self, kind, e):
        self.entities.append((self.node, kind, e))
        return e

    def all_entities(self):
        """Every entity, the polled subscriptions first, for leave_gently."""
        return (self.poller.entities() if self.poller else []) + self.entities

    def _sub(self, msg_type, topic, cb, qos, raw=False):
        return self._add('subscription', self.node.create_subscription(msg_type, topic, cb, qos, raw=raw))

    def _on_odom(self, store, ns):
        def f(buf):
            try:
                sec, nsec, _, _, v = livox_sim_bridge.odom_parse(buf)
            except ValueError:
                return
            stamp = sec + nsec * 1e-9
            if stamp > 0:
                self.first_stamp.setdefault((id(store), ns), stamp)
            store[ns] = (v[0], v[1], quat_yaw(*v[3:7]), time.time(), v[2], stamp)
        return f

    def _tracking(self, ns, now):
        """(x, y, z) of a fresh DLIO estimate past its IMU calibration (DLIO_SETTLED_S), else None."""
        v = self.est.get(ns)
        first = self.first_stamp.get((id(self.est), ns))
        if v is None or first is None or now - v[3] > FRESH_S or v[5] - first < DLIO_SETTLED_S:
            return None
        return v[0], v[1], v[4]

    def _on_mode(self, ns):
        def f(m):
            self.mode[ns] = m.data
        return f

    def _fresh(self, store, ns, now):
        v = store.get(ns)
        return (v[0], v[1]) if v is not None and now - v[3] <= FRESH_S else None

    def _on_commit(self, ns):
        def f(m):
            x, y = m.pose.position.x, m.pose.position.y
            ours = self.legs[ns].on_commit(x, y)
            wx, wy = to_world(self.spawns[ns], x, y)
            stamp = m.header.stamp.sec * 1000000000 + m.header.stamp.nanosec
            self.log.info('COMMIT ' + json.dumps({'agent': ns, 'stamp': stamp, 'map': [_r(x), _r(y)],
                                                  'world': [_r(wx), _r(wy)], 'ours': ours}, sort_keys=True))
        return f

    def _on_select(self, m):
        try:
            self.selected = parse_selection(m.data, self.spawns.keys() & self.legs.keys())
        except GoalError as e:
            self.log.warn(str(e))
            return
        self.log.info('world goals go to %s' % (self.selected or 'the nearest agent'))

    def _on_goal(self, m):
        now = time.time()
        x, y = m.pose.position.x, m.pose.position.y
        truth = {ns: v for ns in self.legs for v in [self._fresh(self.truth, ns, now)] if v is not None}
        served = {ns: self.spawns[ns] for ns in self.legs}
        try:
            ns, mx, my, how = resolve_goal(m.header.frame_id, x, y, served, truth, self.selected)
        except GoalError as e:
            self.log.warn('goal (%.2f, %.2f) in %r refused: %s' % (x, y, m.header.frame_id, e))
            return
        self.log.info('rviz goal (%.2f, %.2f) in %s -> %s (%s): (%.2f, %.2f) in %s/map'
                      % (x, y, m.header.frame_id or WORLD, ns, how, mx, my, ns))
        self._run(ns, self.legs[ns].manual(now, (mx, my), self._fresh(self.est, ns, now),
                                           self._fresh(self.truth, ns, now)))

    def _ready(self, ns, now):
        return (self.mode.get(ns) == 'AUTO' and self._fresh(self.est, ns, now) is not None
                and self.pubs[ns].get_subscription_count() > 0 and (ns not in self.first or self.first[ns].done))

    def _first_drives(self, now):
        for ns, fd in self.first.items():
            if fd.done:
                continue
            for kind, arg in fd.step(now, self.mode.get(ns), self._tracking(ns, now)):
                if kind == 'vel':
                    m = self.Twist()
                    m.linear.x = float(arg)
                    self.joy_pubs[ns].publish(m)
                elif kind == 'call':
                    client = self.clients[ns, arg]
                    if not client.service_is_ready():
                        if now - self.srv_warned.get(ns, 0.0) > 10.0:
                            self.srv_warned[ns] = now
                            self.log.warn('/%s/%s is not available: the first drive waits' % (ns, arg))
                        continue                       # its state asks again after RETRY_S
                    self.calls[ns, arg] = client.call_async(self.Trigger.Request())
                    self.log.info('%s: first drive: %s' % (ns, arg))
                elif kind == 'log':
                    self.log.info(first_line(arg))
                else:
                    self.log.info(arg)

    def tick(self):
        if self.poller is not None:
            self.poller.poll()
        now = time.time()
        self._first_drives(now)
        self._start_cycles(now)
        for ns, a in self.legs.items():
            self._run(ns, a.update(now, self._fresh(self.est, ns, now), self._fresh(self.truth, ns, now)))

    def _start_cycles(self, now):
        """Every cycling agent starts together once all are ready; after --ready-timeout the
        ready ones start alone, and the others as each gets ready."""
        waiting = [ns for ns, a in self.legs.items() if a.cycling and not a.started]
        if not waiting:
            return
        ready = [ns for ns in waiting if self._ready(ns, now)]
        late = now - self.t_begin > self.ready_timeout
        if not self.cycle_started and not (ready and (len(ready) == len(waiting) or late)):
            if now - self.last_wait_log > 10.0:
                self.last_wait_log = now
                self.log.info('swap waits for %s (safety mode: %s; DLIO estimate: %s; goal_selector subscribed: %s; '
                              'first drive: %s)' % (
                                  ', '.join(ns for ns in waiting if ns not in ready),
                                  ' '.join('%s=%s' % (ns, self.mode.get(ns, '-')) for ns in waiting),
                                  ' '.join(ns for ns in waiting if self._fresh(self.est, ns, now)) or 'none',
                                  ' '.join(ns for ns in waiting if self.pubs[ns].get_subscription_count() > 0)
                                  or 'none',
                                  ' '.join('%s=%s' % (ns, self.first[ns].state) for ns in waiting if ns in self.first)
                                  or 'none'))
            return
        if not self.cycle_started and len(ready) < len(waiting):
            self.log.warn('starting without %s: not ready after %.0f s (AUTO, a DLIO estimate and goal_selector '
                          'subscribed to term_goal_rviz)' % (', '.join(ns for ns in waiting if ns not in ready),
                                                             self.ready_timeout))
        self.cycle_started = True
        for ns in ready:
            self._run(ns, self.legs[ns].start(now, self._fresh(self.est, ns, now), self._fresh(self.truth, ns, now)))

    def _run(self, ns, actions):
        for kind, v in actions:
            if kind == 'send':
                self._send(ns, v)
            elif kind == 'log':
                self.log.info(leg_line(v))
            else:
                self.log.info(v)

    def _send(self, ns, xy):
        m = self.PoseStamped()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = '%s/map' % ns
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(xy[0]), float(xy[1]), 0.0
        m.pose.orientation.w = 1.0
        if self.pubs[ns].get_subscription_count() == 0:
            if ns not in self.unheard:
                self.log.warn('/%s/term_goal_rviz has no subscriber (goal_selector not up?): (%.2f, %.2f) is lost; '
                              'a running leg sends its goal again every %.0f s' % (ns, xy[0], xy[1], RESEND_S))
            self.unheard.add(ns)
        else:
            self.unheard.discard(ns)
        self.pubs[ns].publish(m)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    try:
        from rclpy.utilities import remove_ros_args
        rest = remove_ros_args(argv)[1:]
    except ImportError:                    # no ROS here: only the refusals can run
        rest = argv[1:]
    args = build_parser().parse_args(rest)
    try:                                   # all of it before rclpy opens a session
        spawns, mode, legs, radius, first = setup(args)
    except (GoalError, drive_pattern.PlanError) as e:
        print('goal_relay: %s' % e, file=sys.stderr)
        return 2

    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    # No rclpy SIGINT handler: Ctrl-C raises here with the context still up, for the paced leave.
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    node = Node(NODE_NAME, namespace=NODE_NAMESPACE, enable_rosout=False, start_parameter_services=False)
    poller = Poller(Node, NODE_NAME + '_poll', NODE_NAMESPACE)
    relay = None
    try:
        relay = Relay(node, spawns, legs, mode, args.ready_timeout, poller, first)
        node.get_logger().info('goal_radius %.2f m (%s), reached by %s, leg timeout %s; first drive %s'
                               % (radius, 'given' if args.goal_radius else args.planner_config, args.reached_by,
                                  '%.0f s' % args.leg_timeout if args.leg_timeout else
                                  '%.0f s + %.0f s/m' % (LEG_TIMEOUT_BASE_S, LEG_TIMEOUT_S_PER_M),
                                  '%.1f m at %.1f m/s in JOY mode, for agents in AUTO' % (
                                      args.first_drive, args.first_drive_speed) if args.first_drive else 'off'))
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        # The leave is paced (about 17 s with 4 agents: ENTITIES_PER_AGENT, ENTITIES_ROOT); a second
        # Ctrl-C or the SIGTERM after it must not cut it short and drop the rest at once (the stall
        # it exists to avoid). sim.py waits it out (goal_relay_leave_s); SIGKILL still ends it.
        def leaving(signum, _frame):
            print('goal_relay: signal %d while leaving the graph: still leaving, one entity at a time' % signum,
                  flush=True)
        signal.signal(signal.SIGINT, leaving)
        signal.signal(signal.SIGTERM, leaving)
        if relay is not None:
            relay.timer.cancel()
        leave_gently(relay.all_entities() if relay else poller.entities(), [poller.node, node],
                     lambda: rclpy.spin_once(node, timeout_sec=0))
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
