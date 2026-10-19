#!/usr/bin/env python3
"""Drive one agent along a scripted path, closed loop on the sim's ground truth.

  drive_pattern.py --ns JR81 --pattern square|loop --size m [--turn left|right]
  drive_pattern.py --ns JR81 --pattern waypoints --waypoints 'x,y;x,y;...'
      [--speed 0.6] [--omega 0.5] [--accel 0.5] [--alpha 1.0] [--topic cmd_vel_auto]
      [--spawn /rover_sim/run/spawn.yaml] [--teleop /rover_sim/run/teleop_sim.yaml]
      [--from-here] [--timeout s]

Publishes geometry_msgs/Twist on /<ns>/<topic> at --rate Hz. The default cmd_vel_auto goes
through safety_node in AUTO, exactly as mpc's commands do on the robot (cmd_vel would bypass
it): arm the agent first, with `scripts/sim.py arm <ns>` on the host (which checks the sim's
isolation and waits for DLIO) or a session with auto_arm. The feedback is
/<ns>/ground_truth/odom_wall, not DLIO, so the driven path is the intended one whatever DLIO
estimates (odom_eval.py grades DLIO against the same truth). Only sim agents are driven: the
agent must be in the sim's spawn.yaml, the zenoh session config must be the sim's
(ZENOH_SESSION_CONFIG_URI, config/sim.env), and nothing is published before ground truth,
which only the sim has, arrives.

Patterns, all in the world frame and anchored at the agent's spawn pose (spawn.yaml), or at
its current pose with --from-here; an agent away from its spawn pose drives there first:
  square     a closed square of side --size, forward first, turning --turn in place at each
             corner, and finishing with a turn back to the spawn heading
  loop       a closed circle of diameter --size, forward first towards --turn, driven
             without stopping (curvature instead of in-place turns, so at most
             0.8 * omega * r fast), then the same final turn
  waypoints  --waypoints, each reached by a turn in place and a straight leg; ends there
--size has no default, as no one size fits every world and start. A plan that comes within
0.5 m (WALL_MARGIN) of the world's walls (spawn.yaml walls, from src/rover_sim/config/
worlds.yaml) is refused; obstacles inside them are not checked. The plan's "50 m square"
check is --pattern waypoints with the odom_eval session's goals repeated 4 times (48 m, see
config/sessions/odom_eval.yaml): a single 12.5 m square fits only the empty world, where
DLIO cannot track.

Speeds ramp at --accel / --alpha and stop on a profile, so the plant (Gazebo's diff drive,
which slews v at the class's linear_acceleration and does not slew w) follows the commands.
--speed, --accel and --omega are clamped to the class limits in src/rover_sim/config/
classes.yaml, and --speed and --omega to the agent's safety_node lin_vel_max / ang_vel_max
(--teleop), at which safety_node saturates cmd_vel_auto. At the end it publishes zeros for
--stop-secs, then goes silent.

Exit 0 when the pattern completes, 2 on a setup error (no ground truth, bad arguments),
3 on --timeout (by default twice the planned time plus 30 s).

The planner and the controller (Driver) are pure and import without rclpy;
tests/test_drive_pattern.py drives them against a unicycle model. odom_eval.py --drive runs
the same DriveRunner in its own node.
"""

import argparse
import math
import os
import re
import sys
import time
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERNS = ('square', 'loop', 'waypoints')
PREFIX_CLASS = {'JR': 'jr', 'RR': 'rr'}
SIM_SESSION_URI = '/rover_sim/zenoh/session.json5'   # every sim container's (config/sim.env)
WALL_MARGIN = 0.5      # m: the larger robot's half-diagonal (an RR's, 0.35) plus tracking error
SAFETY_DEFAULTS = {'lin_vel_max': 1.0, 'ang_vel_max': 1.0}   # safety_node's declare_parameter defaults

# A plan is a list of segments, each driven to a stop before the next one starts.
Turn = namedtuple('Turn', 'yaw')                       # turn in place to a world yaw
Line = namedtuple('Line', 'x0 y0 x1 y1')               # straight leg, stopping at (x1, y1)
Arc = namedtuple('Arc', 'cx cy r theta0 sweep')        # circle about (cx, cy) from polar angle
                                                       # theta0 by sweep rad (> 0 = ccw)
Limits = namedtuple('Limits', 'v_max accel omega_max')
Pose2 = namedtuple('Pose2', 'x y yaw')


class PlanError(ValueError):
    pass


def wrap(a):
    return math.remainder(a, 2.0 * math.pi)


# ---------------------------------------------------------------------------- limits

def classes_candidates(explicit=None):
    """Where classes.yaml may be: --classes, this checkout (host, or sim-gazebo's mount),
    <run>/classes.yaml, the installed rover_sim share (gazebo image)."""
    if explicit:
        return [explicit]
    out = [os.path.join(HERE, '..', 'src', 'rover_sim', 'config', 'classes.yaml'),
           '/rover_sim/run/classes.yaml']
    try:
        from ament_index_python.packages import get_package_share_directory
        out.append(os.path.join(get_package_share_directory('rover_sim'), 'config', 'classes.yaml'))
    except Exception:            # not sourced, or no rover_sim in this image
        pass
    return out


def load_classes(explicit=None):
    """(path, {class: {...}}) of the first readable classes.yaml; '-' reads stdin."""
    import yaml
    if explicit == '-':
        return '<stdin>', yaml.safe_load(sys.stdin)
    for p in classes_candidates(explicit):
        if os.path.isfile(p):
            with open(p) as f:
                return os.path.normpath(p), yaml.safe_load(f)
    raise PlanError('classes.yaml not found (looked in %s); pass --classes PATH, or --classes - '
                    'with the file on stdin' % ', '.join(classes_candidates(explicit)))


def class_limits(classes, cls):
    c = (classes or {}).get(cls)
    if not isinstance(c, dict):
        raise PlanError('classes.yaml has no class %r' % cls)
    try:
        lim = Limits(float(c['v_max']), float(c['linear_acceleration']), float(c['omega_max']))
    except (KeyError, TypeError, ValueError) as e:
        raise PlanError('classes.yaml %s: missing or bad drive limit (%s)' % (cls, e))
    if not all(v > 0 and math.isfinite(v) for v in lim):
        raise PlanError('classes.yaml %s: limits must be > 0: %s' % (cls, lim))
    return lim


def clamp_to(name, value, limit, warn, what='the class limit'):
    if value > limit:
        warn('%s %.3g is over %s %.3g: using %.3g' % (name, value, what, limit, limit))
        return limit
    return value


def params_key_matches(key, fqn):
    """rcl's matching of a params-file node key ('/**', '/**/name', '/ns/*', '/ns/name')."""
    key = key if key.startswith('/') else '/' + key
    rx = ''.join({'**': '.*', '*': '[^/]*'}.get(t, re.escape(t)) for t in re.split(r'(\*\*|\*)', key))
    return re.fullmatch(rx.replace('/.*/', '(?:/.*)?/'), fqn) is not None


def safety_limits(doc, ns):
    """(lin_vel_max, ang_vel_max) that /<ns>/safety_node gets from a parsed teleop file: the
    ros__parameters of every node key that matches it, applied in file order as rcl does,
    over safety_node's own defaults."""
    lim = dict(SAFETY_DEFAULTS)
    for key, block in (doc or {}).items():
        if isinstance(key, str) and isinstance(block, dict) and params_key_matches(key, '/%s/safety_node' % ns):
            params = block.get('ros__parameters') or {}
            for k in lim:
                if k in params:
                    try:
                        lim[k] = float(params[k])
                    except (TypeError, ValueError):
                        raise PlanError('teleop file %s: %s %r is not a number' % (key, k, params[k]))
    return lim['lin_vel_max'], lim['ang_vel_max']


def load_safety_limits(path, ns):
    """safety_limits from the file at path, or None when there is no such file."""
    import yaml
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return safety_limits(yaml.safe_load(f), ns)
    except (OSError, yaml.YAMLError) as e:
        raise PlanError('%s: %s' % (path, e))


# ---------------------------------------------------------------------------- planning

def parse_waypoints(text):
    """'x,y;x,y' -> [(x, y)]."""
    pts = []
    for i, item in enumerate(t for t in (text or '').replace(' ', '').split(';') if t):
        try:
            x, y = (float(v) for v in item.split(','))
        except ValueError:
            raise PlanError('waypoint %d %r is not x,y' % (i, item))
        if not (math.isfinite(x) and math.isfinite(y)):
            raise PlanError('waypoint %d %r is not finite' % (i, item))
        pts.append((x, y))
    if not pts:
        raise PlanError('--waypoints needs at least one x,y')
    return pts


def plan_square(p, size, left=True):
    if not size > 0:
        raise PlanError('--size must be > 0')
    s = 1.0 if left else -1.0
    corners = [(p.x, p.y)]
    for k in range(3):
        h = p.yaw + s * k * math.pi / 2.0
        x, y = corners[-1]
        corners.append((x + size * math.cos(h), y + size * math.sin(h)))
    corners.append((p.x, p.y))
    segs = []
    for k in range(4):
        (x0, y0), (x1, y1) = corners[k], corners[k + 1]
        segs.append(Line(x0, y0, x1, y1))
        segs.append(Turn(wrap(p.yaw + s * (k + 1) * math.pi / 2.0)))
    return segs                               # the last turn is back to p.yaw


def plan_loop(p, size, left=True):
    if not size > 0:
        raise PlanError('--size must be > 0')
    s = 1.0 if left else -1.0
    r = size / 2.0
    cx, cy = p.x - s * r * math.sin(p.yaw), p.y + s * r * math.cos(p.yaw)
    theta0 = math.atan2(p.y - cy, p.x - cx)
    return [Arc(cx, cy, r, theta0, s * 2.0 * math.pi), Turn(wrap(p.yaw))]


def plan_waypoints(p, pts):
    segs, (x, y) = [], (p.x, p.y)
    for (wx, wy) in pts:
        if math.hypot(wx - x, wy - y) < 1e-3:
            continue
        segs.append(Turn(math.atan2(wy - y, wx - x)))
        segs.append(Line(x, y, wx, wy))
        x, y = wx, wy
    return segs


def plan_approach(cur, anchor, pos_tol=0.05, yaw_tol=math.radians(3.0)):
    """Segments that bring an agent at cur to the anchor pose (none if it is there)."""
    segs = []
    if math.hypot(anchor.x - cur.x, anchor.y - cur.y) > pos_tol:
        segs = plan_waypoints(cur, [(anchor.x, anchor.y)])
        segs.append(Turn(wrap(anchor.yaw)))
    elif abs(wrap(anchor.yaw - cur.yaw)) > yaw_tol:
        segs.append(Turn(wrap(anchor.yaw)))
    return segs


def need_size(pattern, size):
    if pattern in ('square', 'loop') and size is None:
        raise PlanError('--pattern %s needs --size (m): there is no default, as no one size fits every world '
                        '(the arena\'s walls are at +-8 m)' % pattern)


def plan(pattern, anchor, size=None, left=True, waypoints=None):
    need_size(pattern, size)
    if pattern == 'square':
        return plan_square(anchor, size, left)
    if pattern == 'loop':
        return plan_loop(anchor, size, left)
    if pattern == 'waypoints':
        return plan_waypoints(anchor, waypoints or [])
    raise PlanError('unknown pattern %r' % pattern)


def plan_outside(segs, walls, margin=WALL_MARGIN):
    """The first point a plan drives to that is within margin of walls [xmin, ymin, xmax, ymax]
    (or past them), or None. Only where segments end and how far arcs reach count: a leg
    from a start near a wall to a point inside the box moves away from it."""
    xmin, ymin, xmax, ymax = walls[0] + margin, walls[1] + margin, walls[2] - margin, walls[3] - margin
    for sg in segs:
        if isinstance(sg, Line):
            pts = [(sg.x1, sg.y1)]
        elif isinstance(sg, Arc):
            pts = [(sg.cx - sg.r, sg.cy), (sg.cx + sg.r, sg.cy), (sg.cx, sg.cy - sg.r), (sg.cx, sg.cy + sg.r)]
        else:
            continue
        for x, y in pts:
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                return (x, y)
    return None


def plan_extent(segs, start_yaw):
    """(path length m, total in-place turning rad) of a plan driven from start_yaw."""
    dist, turn, yaw = 0.0, 0.0, start_yaw
    for sg in segs:
        if isinstance(sg, Turn):
            turn += abs(wrap(sg.yaw - yaw))
            yaw = sg.yaw
        elif isinstance(sg, Line):
            dist += math.hypot(sg.x1 - sg.x0, sg.y1 - sg.y0)
            yaw = math.atan2(sg.y1 - sg.y0, sg.x1 - sg.x0)
        else:
            dist += abs(sg.sweep) * sg.r
            yaw = wrap(sg.theta0 + sg.sweep + math.copysign(math.pi / 2.0, sg.sweep))
    return dist, turn


# ---------------------------------------------------------------------------- control

class Driver:
    """Segment-by-segment controller: feed it the truth pose, get (v, w) back.

    Lines are tracked by cross-track and heading error with progress measured along the
    planned line, so corners stay where the plan put them; arcs the same about their centre.
    Speeds follow a stopping profile (sqrt(2 a d), at PROFILE of the ramp rate, for margin
    against command latency) with a small creep floor, and every command is rate-limited
    to accel / alpha. After each segment it commands zero until the agent is still."""

    PROFILE = 0.8        # fraction of accel / alpha the stopping profile plans with
    ARC_W_SHARE = 0.8    # of the omega clamp an arc's feed-forward v / r may use (the rest corrects)
    K_CROSS = 1.0        # rad of heading correction per atan(m) of cross-track error
    K_HEAD = 2.0         # 1/s heading-error gain
    V_CREEP, W_CREEP = 0.05, 0.05
    STILL_V, STILL_W = 0.02, 0.02
    SETTLE_MAX = 2.0     # s of zero commands at most before moving on regardless
    SETTLE_BLIND = 0.5   # s of zero commands when no measured twist is given

    def __init__(self, segments, speed, accel, omega, alpha, pos_tol=0.01, yaw_tol=math.radians(1.0)):
        if not (speed > 0 and accel > 0 and omega > 0 and alpha > 0):
            raise PlanError('speed, accel, omega and alpha must be > 0')
        self.segs = list(segments)
        self.speed, self.accel, self.omega, self.alpha = speed, accel, omega, alpha
        self.pos_tol, self.yaw_tol = pos_tol, yaw_tol
        self.i = 0
        self.settle_since = None
        self.v = self.w = 0.0
        self.t_prev = None
        self.arc_phi = None              # unwrapped polar angle on the current arc
        self.log = []                    # (t, segment index, what) for reports

    @property
    def done(self):
        return self.i >= len(self.segs)

    def _ramp(self, cur, target, rate, dt):
        step = rate * dt
        return cur + max(-step, min(step, target - cur))

    def _stop_speed(self, dist, vmax, rate, creep):
        if dist <= 0:
            return 0.0
        return max(creep, min(vmax, math.sqrt(2.0 * self.PROFILE * rate * dist)))

    def arc_speed(self, r):
        """Cruise speed on an arc of radius r: past omega * r its feed-forward yaw rate alone
        would hit the omega clamp, and the agent would drift off the circle."""
        return min(self.speed, self.ARC_W_SHARE * self.omega * r)

    def duration(self, start_yaw):
        """Seconds the plan takes at cruise speed and turn rate (ramps and settling excluded)."""
        t, yaw = 0.0, start_yaw
        for sg in self.segs:
            if isinstance(sg, Turn):
                t += abs(wrap(sg.yaw - yaw)) / self.omega
                yaw = sg.yaw
            elif isinstance(sg, Line):
                t += math.hypot(sg.x1 - sg.x0, sg.y1 - sg.y0) / self.speed
                yaw = math.atan2(sg.y1 - sg.y0, sg.x1 - sg.x0)
            else:
                t += abs(sg.sweep) * sg.r / self.arc_speed(sg.r)
                yaw = wrap(sg.theta0 + sg.sweep + math.copysign(math.pi / 2.0, sg.sweep))
        return t

    def _line(self, sg, x, y, yaw):
        lx, ly = sg.x1 - sg.x0, sg.y1 - sg.y0
        length = math.hypot(lx, ly)
        ux, uy = lx / length, ly / length
        dx, dy = x - sg.x0, y - sg.y0
        remaining = length - (dx * ux + dy * uy)
        if remaining <= self.pos_tol:
            return None
        cross = dx * uy - dy * ux                       # > 0: right of the line
        herr = wrap(math.atan2(uy, ux) + math.atan(self.K_CROSS * cross) - yaw)
        v = self._stop_speed(remaining - self.pos_tol / 2.0, self.speed, self.accel, self.V_CREEP)
        if abs(herr) > math.radians(30.0):              # far off heading: turn first
            v = 0.0
        return v, max(-self.omega, min(self.omega, self.K_HEAD * herr))

    def _arc(self, sg, x, y, yaw):
        phi = math.atan2(y - sg.cy, x - sg.cx)
        if self.arc_phi is None:
            self.arc_phi = sg.theta0 + wrap(phi - sg.theta0)
        else:
            self.arc_phi += wrap(phi - self.arc_phi)
        s = 1.0 if sg.sweep > 0 else -1.0
        remaining = sg.r * (abs(sg.sweep) - s * (self.arc_phi - sg.theta0))
        if remaining <= self.pos_tol:
            return None
        radial = math.hypot(x - sg.cx, y - sg.cy) - sg.r    # > 0: outside
        herr = wrap(phi + s * math.pi / 2.0 + s * math.atan(self.K_CROSS * radial) - yaw)
        v = self._stop_speed(remaining - self.pos_tol / 2.0, self.arc_speed(sg.r), self.accel, self.V_CREEP)
        if abs(herr) > math.radians(30.0):
            v = 0.0
        w = s * self.v / sg.r + self.K_HEAD * herr
        return v, max(-self.omega, min(self.omega, w))

    def _turn(self, sg, yaw):
        err = wrap(sg.yaw - yaw)
        if abs(err) <= self.yaw_tol:
            return None
        return 0.0, math.copysign(self._stop_speed(abs(err), self.omega, self.alpha, self.W_CREEP), err)

    def step(self, t, x, y, yaw, v_meas=None, w_meas=None):
        """(v, w) to command at time t (s, any monotonic clock) for the truth pose."""
        dt = 0.0 if self.t_prev is None else max(0.0, min(0.2, t - self.t_prev))
        self.t_prev = t
        target = (0.0, 0.0)
        while not self.done:
            if self.settle_since is not None:
                still = (v_meas is not None and w_meas is not None
                         and abs(v_meas) < self.STILL_V and abs(w_meas) < self.STILL_W)
                waited = t - self.settle_since
                if (still and abs(self.v) < 1e-9 and abs(self.w) < 1e-9) or waited >= self.SETTLE_MAX or \
                        (v_meas is None and waited >= self.SETTLE_BLIND):
                    self.settle_since = None
                    self.i += 1
                    self.arc_phi = None
                    continue
                break
            sg = self.segs[self.i]
            if isinstance(sg, Turn):
                cmd = self._turn(sg, yaw)
            elif isinstance(sg, Line):
                cmd = self._line(sg, x, y, yaw)
            else:
                cmd = self._arc(sg, x, y, yaw)
            if cmd is None:
                self.log.append((t, self.i, 'reached'))
                self.settle_since = t
                self.v = self.w = 0.0        # the profile has already brought both near zero
                break
            target = cmd
            break
        self.v = self._ramp(self.v, target[0], self.accel, dt)
        self.w = self._ramp(self.w, target[1], self.alpha, dt)
        if self.done:
            self.v = self.w = 0.0
        return self.v, self.w


# ---------------------------------------------------------------------------- spawn

def spawn_walls(data, source='spawn.yaml'):
    """(world, [xmin, ymin, xmax, ymax] or None) from parsed spawn.yaml (sim.py spawn_table)."""
    world = data.get('world') if isinstance(data, dict) else None
    w = data.get('walls') if isinstance(data, dict) else None
    if w is None:
        return world, None
    try:
        box = [float(v) for v in w]
    except (TypeError, ValueError):
        box = []
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[0] >= box[2] or box[1] >= box[3]:
        raise PlanError('%s: walls %r is not [xmin, ymin, xmax, ymax]' % (source, w))
    return world, box


def spawn_entry(data, ns, source='spawn.yaml'):
    """(Pose2, class, lidar_height) of agent ns in parsed spawn.yaml (sim.py spawn_table)."""
    agents = data.get('agents') if isinstance(data, dict) else None
    if not isinstance(agents, list):
        raise PlanError('%s: expected {agents: [...]}' % source)
    for a in agents:
        if isinstance(a, dict) and a.get('name') == ns:
            try:
                pose = Pose2(float(a['x']), float(a['y']), float(a['yaw']))
                h = float(a['lidar_height'])
            except (KeyError, TypeError, ValueError) as e:
                raise PlanError('%s: %s: missing or bad x/y/yaw/lidar_height (%s)' % (source, ns, e))
            if not all(math.isfinite(v) for v in pose + (h,)):
                raise PlanError('%s: %s: non-finite pose' % (source, ns))
            return pose, a.get('class'), h
    raise PlanError('%s: no agent %s' % (source, ns))


def load_spawn(path):
    import yaml
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise PlanError('%s: %s' % (path, e))


def load_spawn_entry(path, ns):
    return spawn_entry(load_spawn(path), ns, path)


def ns_class(ns):
    return PREFIX_CLASS.get(ns[:2])


# ---------------------------------------------------------------------------- ROS

def require_sim():
    """Commands go only to sim agents, on the sim's own zenoh graph: checked before any session
    is opened, so a process configured for another graph never joins it."""
    uri = os.environ.get('ZENOH_SESSION_CONFIG_URI')
    if uri != SIM_SESSION_URI:
        raise PlanError('not in a rover_sim container (ZENOH_SESSION_CONFIG_URI=%r, the sim\'s is %s): '
                        'refusing to publish commands' % (uri, SIM_SESSION_URI))


def join_negative_values(argv, opts=('--waypoints',)):
    """['--waypoints', '-2,0;...'] -> ['--waypoints=-2,0;...']: argparse takes a value that
    starts with '-' (and is not a plain number) for an option."""
    out, it = [], iter(argv)
    for a in it:
        if a in opts:
            v = next(it, None)
            out.append(a if v is None else '%s=%s' % (a, v))
        else:
            out.append(a)
    return out


def build_parser(prog=None):
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split('\n\n')[0])
    add_drive_args(ap)
    return ap


def add_drive_args(ap, with_ns=True):
    if with_ns:
        ap.add_argument('--ns', required=True, help='agent namespace, e.g. JR81')
        ap.add_argument('--spawn', default='/rover_sim/run/spawn.yaml')
    ap.add_argument('--pattern', choices=PATTERNS, default='square')
    ap.add_argument('--size', type=float, default=None, help='square side / loop diameter, m (no default)')
    ap.add_argument('--speed', type=float, default=0.6, help='cruise speed, m/s')
    ap.add_argument('--omega', type=float, default=0.5, help='in-place turn rate, rad/s')
    ap.add_argument('--accel', type=float, default=0.5, help='linear ramp, m/s^2')
    ap.add_argument('--alpha', type=float, default=1.0, help='angular ramp, rad/s^2')
    ap.add_argument('--turn', choices=('left', 'right'), default='left')
    ap.add_argument('--waypoints', default='', help="world-frame 'x,y;x,y;...' (--pattern waypoints)")
    ap.add_argument('--topic', default='cmd_vel_auto', help='relative to /<ns>, or absolute')
    ap.add_argument('--rate', type=float, default=20.0, help='command rate, Hz')
    ap.add_argument('--from-here', action='store_true', help='anchor the pattern at the current pose')
    ap.add_argument('--classes', default=None, help="classes.yaml ('-' = stdin)")
    ap.add_argument('--teleop', default='/rover_sim/run/teleop_sim.yaml',
                    help="safety_node's params file (its lin/ang_vel_max clamp --speed and --omega)")
    ap.add_argument('--wait', type=float, default=15.0, help='s to wait for ground truth')
    ap.add_argument('--timeout', type=float, default=None, help='s; default 2x the planned time + 30')
    ap.add_argument('--stop-secs', type=float, default=1.0, help='s of zero commands at the end')
    return ap


class DriveRunner:
    """Drives args' pattern from inside an rclpy node: its own odom_wall subscription,
    Twist publisher and timer. finished is None while running, then the exit code."""

    def __init__(self, node, args, ns, spawn_path):
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self.node, self.args, self.ns = node, args, ns
        self.Twist = Twist
        self.log = node.get_logger()
        self.finished = None
        self.armed = True                       # False: hold off (odom_eval waits for DLIO)
        self.driver = None
        self.truth = None                       # (recv mono, x, y, yaw, vx, wz)
        self.t_start = time.monotonic()
        self.t_stop = None
        self.moved_check = None
        self.stats = {}
        if not args.rate > 0:
            raise PlanError('--rate must be > 0')

        # Sim agents only: on the sim's graph, and one the sim spawned, even with --from-here.
        require_sim()
        spawn = load_spawn(spawn_path)
        anchor, cls, _ = spawn_entry(spawn, ns, spawn_path)
        self.anchor = None if args.from_here else anchor
        self.world, self.walls = spawn_walls(spawn, spawn_path)
        cls = cls or ns_class(ns)
        if cls is None:
            raise PlanError('no class for %s (spawn.yaml has none and the prefix is unknown)' % ns)
        need_size(args.pattern, args.size)
        path, classes = load_classes(args.classes)
        lim = class_limits(classes, cls)
        warn = self.log.warn
        v_lim, w_lim, v_from, w_from = lim.v_max, lim.omega_max, 'the class limit', 'the class limit'
        safety = load_safety_limits(args.teleop, ns)
        if safety is None:
            warn('no %s: --speed and --omega are clamped to the class limits only' % args.teleop)
        else:
            if safety[0] < v_lim:
                v_lim, v_from = safety[0], "/%s/safety_node's lin_vel_max" % ns
            if safety[1] < w_lim:
                w_lim, w_from = safety[1], "/%s/safety_node's ang_vel_max" % ns
        self.speed = clamp_to('--speed', args.speed, v_lim, warn, v_from)
        self.accel = clamp_to('--accel', args.accel, lim.accel, warn)
        self.omega = clamp_to('--omega', args.omega, w_lim, warn, w_from)
        self.pts = parse_waypoints(args.waypoints) if args.pattern == 'waypoints' else None
        if self.anchor is not None or self.pts:          # what can be checked before the agent's pose is known
            self._check_walls(plan(args.pattern, self.anchor or Pose2(0.0, 0.0, 0.0), args.size,
                                   args.turn == 'left', self.pts))
        self.log.info('%s (%s): %s, limits from %s%s: v_max %.2f, accel %.2f, omega_max %.2f; driving at '
                      '%.2f m/s, %.2f m/s^2, %.2f rad/s, %.2f rad/s^2 on %s'
                      % (ns, cls, args.pattern, path, '' if safety is None else ' and ' + args.teleop,
                         v_lim, lim.accel, w_lim, self.speed, self.accel, self.omega, args.alpha, self.topic_name()))

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        node.create_subscription(Odometry, '/%s/ground_truth/odom_wall' % ns, self._on_truth, qos)
        self.pub = node.create_publisher(Twist, self.topic_name(), 10)
        self.timer = node.create_timer(1.0 / args.rate, self._tick)

    def topic_name(self):
        t = self.args.topic
        return t if t.startswith('/') else '/%s/%s' % (self.ns, t)

    def _on_truth(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.truth = (time.monotonic(), p.x, p.y, yaw, m.twist.twist.linear.x, m.twist.twist.angular.z)

    def _check_walls(self, segs):
        if not self.walls:
            return
        bad = plan_outside(segs, self.walls)
        if bad:
            raise PlanError('the plan drives to (%.2f, %.2f), within %.1f m of the walls of %s (spawn.yaml walls '
                            '%s): use a smaller --size, or waypoints inside them'
                            % (bad[0], bad[1], WALL_MARGIN, self.world, self.walls))

    def arm(self):
        self.armed = True
        self.t_start = time.monotonic()

    def stop_now(self, secs=0.25):
        """Zero commands, once it has commanded anything: never to an agent it has not driven."""
        if self.driver is None:
            return
        for _ in range(max(1, int(secs * self.args.rate))):
            self._publish(0.0, 0.0)
            time.sleep(1.0 / self.args.rate)

    def _publish(self, v, w):
        m = self.Twist()
        m.linear.x, m.angular.z = float(v), float(w)
        self.pub.publish(m)

    def _start(self, now):
        _, x, y, yaw = self.truth[:4]
        cur = Pose2(x, y, yaw)
        # waypoints are absolute, so they start from wherever the agent is
        anchor = cur if self.anchor is None or self.args.pattern == 'waypoints' else self.anchor
        segs = plan_approach(cur, anchor) + plan(self.args.pattern, anchor, self.args.size,
                                                 self.args.turn == 'left', self.pts)
        if not segs:
            raise PlanError('nothing to drive: the plan is empty')
        self._check_walls(segs)
        dist, turn = plan_extent(segs, yaw)
        self.driver = Driver(segs, self.speed, self.accel, self.omega, self.args.alpha)
        planned = self.driver.duration(yaw) + len(segs) * (Driver.SETTLE_BLIND + 1.0)
        self.deadline = now + (self.args.timeout if self.args.timeout else 2.0 * planned + 30.0)
        self.stats = {'segments': len(segs), 'planned_length_m': dist, 'planned_turn_rad': turn,
                      'start': [x, y, yaw], 'anchor': list(anchor)}
        self.log.info('start (%.3f, %.3f, %.1f deg), anchor (%.3f, %.3f, %.1f deg): %d segments, %.2f m, '
                      '%.0f deg of turns, ~%.0f s (timeout in %.0f s)'
                      % (x, y, math.degrees(yaw), anchor.x, anchor.y, math.degrees(anchor.yaw), len(segs),
                         dist, math.degrees(turn), planned, self.deadline - now))
        self.moved_check = (now + 3.0, x, y, yaw)

    def _tick(self):
        now = time.monotonic()
        if self.finished is not None or not self.armed:
            return
        if self.t_stop is not None:                      # zero commands, then silence
            if now - self.t_stop < self.args.stop_secs:
                self._publish(0.0, 0.0)
            else:
                self.timer.cancel()
                self.finished = self.stats['code']
            return
        if self.truth is None:
            if now - self.t_start > self.args.wait:
                self.log.error('no /%s/ground_truth/odom_wall within %.0f s' % (self.ns, self.args.wait))
                self.stats['code'] = 2
                self.finished = 2
            return
        if self.driver is None:
            try:
                self._start(now)
            except PlanError as e:
                self.log.error(str(e))
                self.stats['code'] = 2
                self.finished = 2
                self.timer.cancel()
                return
        if now - self.truth[0] > 0.3:                    # stale truth: hold still, then ramp again
            self.driver.v = self.driver.w = 0.0
            self._publish(0.0, 0.0)
            return
        _, x, y, yaw, vx, wz = self.truth
        v, w = self.driver.step(now, x, y, yaw, vx, wz)
        self._publish(v, w)
        if self.moved_check and now > self.moved_check[0]:
            _, x0, y0, yaw0 = self.moved_check
            if math.hypot(x - x0, y - y0) < 0.02 and abs(wrap(yaw - yaw0)) < math.radians(1.0):
                self.log.warn('commanding motion for 3 s but the agent has not moved: is /%s/safety_node in '
                              'AUTO? (on the host: scripts/sim.py arm %s)' % (self.ns, self.ns))
            self.moved_check = None
        if self.driver.done or now > self.deadline:
            code = 0 if self.driver.done else 3
            self.stats.update(code=code, elapsed_s=now - self.t_start, end=[x, y, yaw])
            if code:
                self.log.error('timeout after %.0f s in segment %d of %d'
                               % (now - self.t_start, self.driver.i + 1, len(self.driver.segs)))
            else:
                self.log.info('done in %.1f s at (%.3f, %.3f, %.1f deg)'
                              % (now - self.t_start, x, y, math.degrees(yaw)))
            self.t_stop = now
            self._publish(0.0, 0.0)


def main(argv=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    argv = sys.argv if argv is None else argv
    args = build_parser().parse_args(join_negative_values(rclpy.utilities.remove_ros_args(argv)[1:]))
    try:                                   # DriveRunner checks again; this is before rclpy opens a session
        require_sim()
        spawn_entry(load_spawn(args.spawn), args.ns, args.spawn)
        need_size(args.pattern, args.size)
    except PlanError as e:
        print('drive_pattern: %s' % e, file=sys.stderr)
        return 2
    # No rclpy SIGINT handler: Ctrl-C then raises here with the context still up, so the
    # agent is left with a zero command rather than its last one.
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    node = Node('drive_pattern_%s' % args.ns)
    code, runner = 2, None
    try:
        runner = DriveRunner(node, args, args.ns, args.spawn)
        while rclpy.ok() and runner.finished is None:
            rclpy.spin_once(node, timeout_sec=0.05)
        code = runner.finished if runner.finished is not None else 2
    except PlanError as e:
        node.get_logger().error(str(e))
    except KeyboardInterrupt:
        code = 130
        if runner is not None:
            runner.stop_now()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
