#!/usr/bin/env python3
"""Live 2D view of the sim agents' planner map, global paths, committed trajectory and frontiers.

  sim_viz.py [--spawn /rover_sim/run/spawn.yaml] [--agents JR81,JR82] [--layers occ,hgp,...]
             [--follow] [--follow-m 15] [--bounds XMIN YMIN XMAX YMAX] [--size W H] [--hz 5]
             [--snapshot out.png [--snapshot-after S]]

scripts/sim.py viz starts it in the sim-viz container (a window on $DISPLAY) and `sim.py viz --stop`
and `sim.py down` stop it. --snapshot opens no window: it listens for S seconds, writes one frame to
a PNG and exits (tests, headless checks; `sim.py viz --snapshot PNG` runs it in sim-viz).

One panel per agent: one agent fills the window, two sit side by side, three or four make a 2x2;
of more, --agents picks up to four. Every panel is drawn in the world frame through the agent's
spawn pose (world -> <ns>/map, as sim_tf.py publishes it; <ns>/odom is <ns>/map), x to the right
and y up like Gazebo's top view, so all panels share one orientation. The world view is --bounds,
else spawn.yaml's walls, else the spawn poses +/- SPAWN_MARGIN_M until data comes, then the box round
the spawn poses, the DLIO poses and the grids' known cells, grown in GROW_STEP_M steps, never shrunk.
`f` switches every panel to a follow view, --follow-m metres across round the agent's DLIO pose (+ and
- zoom), and back.

Layers (--layers picks the ones subscribed, once at start; the number key shows or hides one):
  1 occ        /<ns>/occ_2d_topic              the planner's map window: free white, occupied red,
                                               the window outlined in cyan
  2 orig_hgp   /<ns>/original_hgp_path_marker  mighty's original global path (orange line + dots)
  3 hgp        /<ns>/hgp_path_marker           the global path it plans on (red line + dots)
  4 traj       /<ns>/traj_committed_colored    the committed trajectory, coloured by speed (jet)
  5 frontiers  /<ns>/exploration/frontiers     goal_selector: frontier cells, goals (cyan active,
                                               grey dormant), their ids, the robot -> goal line
  6 pose       /<ns>/dlio/odom_node/pose       DLIO's estimate: a disc with a heading line
  7 map        /<ns>/persistent_map/grid       light grey free, black occupied; not subscribed by
                                               default (it can be megabytes): --layers default,map
The header shows each layer's rate and the age of its last message (amber once older than STALE_S).

Markers follow RViz's MarkerArray rules, per topic: kept by (ns, id); ADD/MODIFY replaces, DELETE
removes that (ns, id), DELETEALL clears the topic's markers; a marker with a lifetime expires that
long after it arrived (0: never). On top of RViz: a topic that has sent nothing for STALE_S is drawn
faded, and after DROP_S its markers are dropped, so a stopped planner's last path does not look
current. Drawn top-down (z ignored, the marker pose applied to its points): ARROW, CUBE, SPHERE,
CYLINDER, LINE_STRIP, LINE_LIST, CUBE_LIST, SPHERE_LIST, POINTS, TEXT_VIEW_FACING; per-vertex colours
where given, else the marker's; alpha blended; line width from scale.x (at least 1 px). MESH_RESOURCE
and TRIANGLE_LIST are skipped and counted. Frames: world, and every spawn.yaml agent's <ns>/map and
<ns>/odom; a marker or grid in any other frame is counted as an unsupported frame, not drawn ('/JR81/map'
too: tf2 rejects a leading /). As in RViz, a marker with a NaN or inf in it deletes its (ns, id), a grid
whose data does not match width x height is not drawn, and both are counted as invalid; so is a
non-finite DLIO pose (the follow view then centres on the spawn pose). A panel whose drawing fails
anyway counts the error and draws empty: the session goes on.

Graph safety (rmw_zenoh 0.1.9 stalls delivery for every agent when a session undeclares many entities
at once, which once made the parked DLIOs diverge): no rosout, no parameter services, subscriptions
made once at start (the 100 Hz pose on a second node, read every POLL_S as goal_relay.py's Poller
does); on every exit (q, Esc, closing the window, Ctrl-C, SIGTERM, the snapshot's end, an error) it
leaves one subscription at a time (goal_relay.py leave_gently), and further signals do not cut that
short. The window lives in a child process with no ROS (sim_viz.py --window): GDK _exit()s a process
that loses its X connection, so a restarted display or a logout ends only the child, and the node
still leaves gently. The window opens before the node joins: no display, no join. Never SIGKILL it
while a session runs (a docker stop of sim-viz is a Ctrl-C: compose.stack.yaml).

The pure parts (marker store, frames, views, layout, bounds) import without rclpy, numpy or cv2;
tests/test_sim_viz.py covers them, and the drawing where cv2 is installed.
"""

import argparse
import math
import mmap
import os
import select
import signal
import subprocess
import sys
import time
from collections import OrderedDict, deque, namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import drive_pattern  # noqa: E402  SIM_SESSION_URI: the sim's zenoh session only
import goal_relay  # noqa: E402  leave_gently and its pacing
import sim_tf  # noqa: E402  spawn.yaml parsing

try:
    import cv2
    import numpy as np
except ImportError:             # the host's python: the pure parts only
    cv2 = np = None

NODE_NAMESPACE, NODE_NAME = '/sim', 'sim_viz'
WIN = 'rover_sim viz'
WORLD = 'world'

Layer = namedtuple('Layer', 'key topic kind label default')
LAYERS = OrderedDict((lay.key, lay) for lay in (
    Layer('occ', 'occ_2d_topic', 'grid', 'occ', True),
    Layer('orig_hgp', 'original_hgp_path_marker', 'markers', 'orig HGP', True),
    Layer('hgp', 'hgp_path_marker', 'markers', 'HGP', True),
    Layer('traj', 'traj_committed_colored', 'markers', 'traj', True),
    Layer('frontiers', 'exploration/frontiers', 'markers', 'frontiers', True),
    Layer('pose', 'dlio/odom_node/pose', 'pose', 'pose', True),
    Layer('map', 'persistent_map/grid', 'grid', 'map', False),
))
DEFAULT_LAYERS = tuple(k for k, lay in LAYERS.items() if lay.default)
# Bottom to top. The fat committed trajectory goes under the 3 cm paths, which start at the robot.
DRAW_ORDER = ('map', 'occ', 'frontiers', 'traj', 'orig_hgp', 'hgp', 'pose')

# visualization_msgs/Marker
ARROW, CUBE, SPHERE, CYLINDER, LINE_STRIP, LINE_LIST, CUBE_LIST, SPHERE_LIST, POINTS, TEXT, MESH, TRIANGLES = \
    range(12)
ADD, DELETE, DELETEALL = 0, 2, 3          # MODIFY is ADD
SKIPPED_TYPES = (MESH, TRIANGLES)

STALE_S = 5.0          # a marker topic silent this long is drawn faded (frontiers come every 2 s)
STALE_FADE = 0.35
DROP_S = 30.0          # and its markers are dropped after this
RATE_WINDOW_S = 10.0
POLL_S = 0.1           # the pose is read this often (Poller): ~10 samples a poll at 100 Hz
POSE_DEPTH = 32        # room for them between polls, best effort: the header shows the true rate
MAX_PANELS = 4
MAX_LINE_PX = 255      # cv2 refuses thicker (32767); a marker metres wide is not a line anyway
WINDOW_FLAG = '--window'   # the child process that owns the window (Display)
WINDOW_READY_S = 20.0
WINDOW_POLL_MS = 10
SPAWN_MARGIN_M = 12.0  # the view round the spawn poses when spawn.yaml has no walls
GROW_MARGIN_M, GROW_STEP_M = 2.0, 2.0   # then fits what the agents have seen, and only grows
# Points this far from every spawn do not grow the view: a diverged DLIO (1e21 m in a Phase 5 run)
# would otherwise widen it for good, and the frame with it (the sim's worlds are tens of metres).
GROW_MAX_M = 200.0
WALL_MARGIN_M = 1.0
OCC_THRESHOLD = 50     # OccupancyGrid: 0-49 free, 50-100 occupied, -1 unknown

# BGR
PANEL_COLORS = ((255, 160, 40), (60, 200, 60), (40, 140, 255), (220, 80, 220))
BG, GRIDLINE = (70, 70, 70), (88, 88, 88)
MAP_FREE, MAP_OCC = (200, 200, 200), (0, 0, 0)
OCC_FREE, OCC_OCC, OCC_OUTLINE = (255, 255, 255), (60, 60, 215), (220, 220, 0)
HEAD_BG, LEGEND_BG = (35, 35, 35), (25, 25, 25)
TEXT_ON, TEXT_OFF, TEXT_STALE = (235, 235, 235), (120, 120, 120), (0, 170, 255)
HEADER_ROWS, ROW_H = 4, 18
FONT_SCALE = 0.45
SHIFT = 3              # cv2 sub-pixel bits for lines and shapes


# ---------------------------------------------------------------------------- frames

class Aff:
    """A 2D affine map (x, y) -> (a x + b y + c, d x + e y + f); p @ q applies q first."""
    __slots__ = ('a', 'b', 'c', 'd', 'e', 'f')

    def __init__(self, a=1.0, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0):
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f

    @classmethod
    def rigid(cls, x, y, yaw, scale=1.0):
        co, si = math.cos(yaw) * scale, math.sin(yaw) * scale
        return cls(co, -si, x, si, co, y)

    def __matmul__(self, o):
        return Aff(self.a * o.a + self.b * o.d, self.a * o.b + self.b * o.e, self.a * o.c + self.b * o.f + self.c,
                   self.d * o.a + self.e * o.d, self.d * o.b + self.e * o.e, self.d * o.c + self.e * o.f + self.f)

    def apply(self, x, y):
        return self.a * x + self.b * y + self.c, self.d * x + self.e * y + self.f

    def yaw(self):
        return math.atan2(self.d, self.a)

    def rows(self):
        return [[self.a, self.b, self.c], [self.d, self.e, self.f]]


def frame_table(spawns):
    """frame id -> Aff (world <- frame): world, and each agent's <ns>/map and <ns>/odom (the identity
    in the sim: dlio_ws map_odom_tf.py)."""
    out = {WORLD: Aff()}
    for p in spawns:
        out[p.name + '/map'] = out[p.name + '/odom'] = Aff.rigid(p.x, p.y, p.yaw)
    return out


def resolve_frame(frames, frame_id):
    """Aff (world <- frame_id), or None for a frame the viewer cannot place. No leading / stripped:
    tf2 rejects such ids ('frame_ids cannot start with a /'), so RViz2 draws nothing there either."""
    return frames.get(frame_id)


def _finite(*vals):
    return all(math.isfinite(v) for v in vals)


def pose_ok(pose):
    p, q = pose.position, pose.orientation
    return _finite(p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def marker_ok(m):
    """RViz's validateFloats: pose, scale, colour, points and colours all finite."""
    s, c = m.scale, m.color
    return (pose_ok(m.pose) and _finite(s.x, s.y, s.z, c.r, c.g, c.b, c.a)
            and all(_finite(q.x, q.y, q.z) for q in m.points) and all(_finite(q.r, q.g, q.b, q.a) for q in m.colors))


def grid_ok(msg):
    """What RViz's map display needs of an OccupancyGrid: a finite origin, a resolution > 0 and
    width x height cells of data."""
    i = msg.info
    return (pose_ok(i.origin) and math.isfinite(i.resolution) and i.resolution > 0
            and len(msg.data) == i.width * i.height)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat_matrix(q):
    """Rotation matrix rows of a quaternion, normalised; an all-zero one is the identity (RViz warns
    and does the same)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    s = 2.0 / n
    return ((1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)),
            (s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)),
            (s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)))


def pose_aff(pose):
    """The marker pose seen from above: (x, y, z) -> its frame's (x, y), as an Aff on (x, y) plus the
    z column (a tilted pose moves a point sideways by its z)."""
    r, t = quat_matrix(pose.orientation), pose.position
    return Aff(r[0][0], r[0][1], t.x, r[1][0], r[1][1], t.y), (r[0][2], r[1][2])


def marker_points(m, to_out):
    """The marker's points through its pose and to_out (an Aff on its frame's x, y): [(x, y)]."""
    aff, (zx, zy) = pose_aff(m.pose)
    full = to_out @ aff
    out = []
    for p in m.points:
        x, y = full.apply(p.x, p.y)
        out.append((x + to_out.a * zx * p.z + to_out.b * zy * p.z, y + to_out.d * zx * p.z + to_out.e * zy * p.z))
    return out


# ---------------------------------------------------------------------------- markers

Entry = namedtuple('Entry', 'marker t_recv expires')


class MarkerStore:
    """RViz's MarkerArray semantics, one marker table per topic, keyed by (ns, id)."""

    def __init__(self):
        self.topics = OrderedDict()     # topic -> OrderedDict((ns, id) -> Entry), in first-add order
        self.last = {}                  # topic -> arrival of its last array (monotonic s)
        self.bad_actions = {}           # topic -> markers with an unknown action
        self.invalid = {}               # topic -> ADDs with a NaN or inf (RViz: rejected, (ns, id) deleted)

    def apply(self, topic, markers, now):
        table = self.topics.setdefault(topic, OrderedDict())
        self.last[topic] = now
        for m in markers:
            if m.action == DELETEALL:
                table.clear()
            elif m.action == DELETE:
                table.pop((m.ns, m.id), None)
            elif m.action == ADD:
                if not marker_ok(m):
                    table.pop((m.ns, m.id), None)
                    self.invalid[topic] = self.invalid.get(topic, 0) + 1
                    continue
                life = m.lifetime.sec + m.lifetime.nanosec * 1e-9
                table[(m.ns, m.id)] = Entry(m, now, now + life if life > 0 else None)
            else:
                self.bad_actions[topic] = self.bad_actions.get(topic, 0) + 1

    def prune(self, now):
        for topic, table in self.topics.items():
            if now - self.last.get(topic, now) > DROP_S:
                table.clear()
                continue
            for k in [k for k, e in table.items() if e.expires is not None and e.expires <= now]:
                del table[k]

    def stale(self, topic, now):
        return topic in self.last and now - self.last[topic] > STALE_S

    def markers(self, topic, now):
        """The live markers of a topic, oldest first, and the alpha factor they are drawn with."""
        self.prune(now)
        table = self.topics.get(topic) or {}
        return [e.marker for e in table.values()], (STALE_FADE if self.stale(topic, now) else 1.0)


# ---------------------------------------------------------------------------- views and layout

class View:
    """World (m) -> panel pixels: (cx, cy) at the panel's centre, res m/px, x right, y up."""

    def __init__(self, cx, cy, res, w, h):
        self.cx, self.cy, self.res, self.w, self.h = cx, cy, res, w, h

    def aff(self):
        """For cv2: integer pixel coordinates are pixel centres."""
        r = self.res
        return Aff(1.0 / r, 0.0, self.w / 2.0 - self.cx / r - 0.5, 0.0, -1.0 / r, self.h / 2.0 + self.cy / r - 0.5)

    def extent(self):
        hw, hh = self.w * self.res / 2.0, self.h * self.res / 2.0
        return self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh


def world_view(bounds, w, h):
    xmin, ymin, xmax, ymax = bounds
    res = max((xmax - xmin) / float(w), (ymax - ymin) / float(h))
    return View((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, res, w, h)


def follow_view(x, y, follow_m, zoom, w, h):
    return View(x, y, follow_m / zoom / float(min(w, h)), w, h)


def grid_spacing(res, min_px=60.0):
    """Grid line spacing (m) at least min_px apart: at any scale the frame has a few dozen lines."""
    for s in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        if s / res >= min_px:
            return s
    s = 100.0
    while s / res < min_px:                # a huge view (a runaway pose): never millions of lines
        s *= 10.0
    return s


def near_spawns(q, spawns, limit=GROW_MAX_M):
    """Whether world point q is within limit of any spawn pose (what may grow the view)."""
    return any(math.hypot(q[0] - p.x, q[1] - p.y) <= limit for p in spawns)


def grid_shape(n):
    """(rows, cols) of the panels for n agents."""
    if not 1 <= n <= MAX_PANELS:
        raise ValueError('%d panels: the viewer shows 1 to %d (--agents)' % (n, MAX_PANELS))
    return {1: (1, 1), 2: (1, 2)}.get(n, (2, 2))


def panel_size(n, width, height, legend_h, header_h=HEADER_ROWS * ROW_H + 8):
    """(rows, cols, panel width, map height under its header) for a width x height window."""
    rows, cols = grid_shape(n)
    return rows, cols, width // cols, max(50, (height - legend_h) // rows - header_h)


def view_bounds(spawn_data, spawns, override=None, margin=SPAWN_MARGIN_M):
    """(xmin, ymin, xmax, ymax) of the world view: --bounds, else spawn.yaml's walls (sim.py
    world_walls, the outer walls' inner faces) plus WALL_MARGIN_M, else the spawn poses +/- margin."""
    if override:
        xmin, ymin, xmax, ymax = (float(v) for v in override)
        if not (xmin < xmax and ymin < ymax):
            raise ValueError('--bounds %s: expected XMIN YMIN XMAX YMAX with XMIN < XMAX, YMIN < YMAX' % (override,))
        return xmin, ymin, xmax, ymax
    walls = (spawn_data or {}).get('walls')
    if walls:
        xmin, ymin, xmax, ymax = (float(v) for v in walls)
        return xmin - WALL_MARGIN_M, ymin - WALL_MARGIN_M, xmax + WALL_MARGIN_M, ymax + WALL_MARGIN_M
    xs, ys = [p.x for p in spawns], [p.y for p in spawns]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def grow_bounds(bounds, points, margin=GROW_MARGIN_M, step=GROW_STEP_M):
    """bounds (None: none yet) grown, never shrunk, to keep every point margin inside, its edges on
    multiples of step: the scale changes now and then, not with every move."""
    if not points:
        return bounds
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    fit = (math.floor((min(xs) - margin) / step) * step, math.floor((min(ys) - margin) / step) * step,
           math.ceil((max(xs) + margin) / step) * step, math.ceil((max(ys) + margin) / step) * step)
    if bounds is None:
        return fit
    return min(bounds[0], fit[0]), min(bounds[1], fit[1]), max(bounds[2], fit[2]), max(bounds[3], fit[3])


def cell_aff(info, world_from_frame):
    """OccupancyGrid cell units (cell (i, j) spans [i, i+1] x [j, j+1]) -> world."""
    o = info.origin
    return world_from_frame @ Aff.rigid(o.position.x, o.position.y, yaw_of(o.orientation), info.resolution)


def grid_aff(info, world_from_frame, view):
    """OccupancyGrid cell index (column, row; integers at cell centres) -> panel pixels."""
    return view.aff() @ cell_aff(info, world_from_frame) @ Aff(1.0, 0.0, 0.5, 0.0, 1.0, 0.5)


def grid_outline(m, width, height):
    """The grid's outer corners through m (a grid_aff)."""
    return [m.apply(x, y) for x, y in ((-0.5, -0.5), (width - 0.5, -0.5), (width - 0.5, height - 0.5),
                                       (-0.5, height - 0.5))]


# ---------------------------------------------------------------------------- rates and text

class Rate:
    """Arrivals over the last RATE_WINDOW_S; tick(now, n): n of them seen at once (a poll)."""

    def __init__(self):
        self.times = deque()            # (t, n)
        self.count = self.in_window = 0

    def tick(self, now, n=1):
        self.times.append((now, n))
        self.count += n
        self.in_window += n

    def _trim(self, now):
        while self.times and self.times[0][0] < now - RATE_WINDOW_S:
            self.in_window -= self.times.popleft()[1]

    def hz(self, now):
        self._trim(now)
        t = self.times
        return (self.in_window - t[0][1]) / (t[-1][0] - t[0][0]) if len(t) > 1 and t[-1][0] > t[0][0] else 0.0

    def age(self, now):
        return now - self.times[-1][0] if self.times else None


def fmt_age(age):
    if age is None:
        return 'none yet'
    return '%.1f s' % age if age < 10 else '%.0f s' % age


def layer_text(num, layer, subscribed, shown, hz, age):
    """A layer's header item: '1 occ 5.0 Hz 0.2 s'."""
    if not subscribed:
        return '%d %s: not subscribed' % (num, layer.label)
    return '%d %s%s %.1f Hz %s' % (num, layer.label, '' if shown else ' (hidden)', hz, fmt_age(age))


def parse_layers(text):
    """--layers: comma-separated keys of LAYERS; 'default' and 'all' stand for those sets."""
    out = []
    for k in (t.strip() for t in text.split(',')):
        if not k:
            continue
        keys = DEFAULT_LAYERS if k == 'default' else tuple(LAYERS) if k == 'all' else (k,)
        for key in keys:
            if key not in LAYERS:
                raise ValueError('unknown layer %r (layers: %s, default, all)' % (key, ', '.join(LAYERS)))
            if key not in out:
                out.append(key)
    if not out:
        raise ValueError('--layers: no layer given')
    return [k for k in LAYERS if k in out]


def pick_agents(spawns, requested):
    """The spawn.yaml agents to show, in spawn.yaml's order."""
    names = [p.name for p in spawns]
    if requested:
        bad = [n for n in requested if n not in names]
        if bad:
            raise ValueError('%s: not in spawn.yaml (agents: %s)' % (', '.join(bad), ', '.join(names)))
        picked = [p for p in spawns if p.name in requested]
    else:
        picked = list(spawns)
    if len(picked) > MAX_PANELS:
        raise ValueError('%d agents (%s): pick up to %d with --agents' % (len(picked), ', '.join(names), MAX_PANELS))
    return picked


def topic_of(ns, layer):
    return '/%s/%s' % (ns, layer.topic)


def leave_seconds(n_subs):
    return n_subs * goal_relay.LEAVE_GAP_S + goal_relay.LEAVE_PAUSE_S


# ---------------------------------------------------------------------------- drawing (numpy, cv2)

def bgr(c):
    return (int(255 * min(max(c.b, 0.0), 1.0)), int(255 * min(max(c.g, 0.0), 1.0)), int(255 * min(max(c.r, 0.0), 1.0)))


def _fx(u, v):
    k = 1 << SHIFT
    return (int(round(min(max(u, -1e5), 1e5) * k)), int(round(min(max(v, -1e5), 1e5) * k)))


class Painter:
    """Draws a layer's primitives with alpha: opaque ones straight onto the image, the rest on one
    copy per alpha step, blended back once per step (faint first)."""

    def __init__(self, img):
        self.img, self.ops = img, {}

    def add(self, alpha, fn):
        if alpha < 0.03:
            return
        a = 1.0 if alpha >= 0.95 else max(0.1, round(alpha * 10.0) / 10.0)
        self.ops.setdefault(a, []).append(fn)

    def flush(self):
        for a in sorted(self.ops):
            fns = self.ops[a]
            if a >= 1.0:
                for fn in fns:
                    fn(self.img)
            else:
                ov = self.img.copy()
                for fn in fns:
                    fn(ov)
                cv2.addWeighted(ov, a, self.img, 1.0 - a, 0.0, dst=self.img)
        self.ops = {}


def _local_rect(full, cx, cy, hx, hy):
    """Corners (px, cv2 fixed point) of the rectangle +/- hx, hy (m) round (cx, cy) in the marker's
    own axes, through full (marker local -> px)."""
    return np.array([_fx(*full.apply(cx + x, cy + y)) for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))],
                    np.int32)


def draw_marker(p, m, to_px, res, fade):
    """One marker onto Painter p; to_px: Aff marker frame -> px. Returns 'drawn' or 'skipped'."""
    t = m.type
    if t in SKIPPED_TYPES or not 0 <= t <= TRIANGLES:
        return 'skipped'
    base_a = m.color.a * fade
    base = bgr(m.color)
    sx, sy = m.scale.x / res, m.scale.y / res
    per_vertex = len(m.colors) == len(m.points) and len(m.colors) > 0
    full = to_px @ pose_aff(m.pose)[0]          # marker local (x, y) -> px

    def vcol(i):
        if per_vertex:
            c = m.colors[i]
            return bgr(c), c.a * fade
        return base, base_a

    if t in (CUBE, SPHERE, CYLINDER, TEXT) or (t == ARROW and len(m.points) < 2):
        u, v = full.apply(0.0, 0.0)
        if t == TEXT:
            p.add(base_a, lambda img: draw_text(img, m.text, u, v, base))
        elif t == CUBE:
            poly = _local_rect(full, 0.0, 0.0, m.scale.x / 2.0, m.scale.y / 2.0)
            p.add(base_a, lambda img: cv2.fillPoly(img, [poly], base, cv2.LINE_AA, SHIFT))
        elif t in (SPHERE, CYLINDER):
            du, dv = full.apply(1.0, 0.0)
            angle = math.degrees(math.atan2(dv - v, du - u))
            axes = (int(min(max(1.0, sx / 2.0), 1e5) * (1 << SHIFT)), int(min(max(1.0, sy / 2.0), 1e5) * (1 << SHIFT)))
            p.add(base_a, lambda img: cv2.ellipse(img, _fx(u, v), axes, angle, 0, 360, base, -1, cv2.LINE_AA, SHIFT))
        else:   # ARROW from its pose: length scale.x along the pose's x, shaft width scale.y
            end = full.apply(m.scale.x, 0.0)
            th = min(max(1, int(round(sy))), MAX_LINE_PX)
            p.add(base_a, lambda img: cv2.arrowedLine(img, _fx(u, v), _fx(*end), base, th, cv2.LINE_AA, SHIFT, 0.3))
        return 'drawn'

    pts = marker_points(m, to_px)
    if not pts:
        return 'drawn'
    th = min(max(1, int(round(sx))), MAX_LINE_PX)
    if t == ARROW:   # start, end; shaft diameter scale.x
        p.add(base_a, lambda img: cv2.arrowedLine(img, _fx(*pts[0]), _fx(*pts[1]), base, th, cv2.LINE_AA, SHIFT,
                                                  0.25))
    elif t == LINE_STRIP:
        if per_vertex:
            for i in range(len(pts) - 1):
                c, a = vcol(i)
                p.add(a, lambda img, i=i, c=c: cv2.line(img, _fx(*pts[i]), _fx(*pts[i + 1]), c, th, cv2.LINE_AA,
                                                        SHIFT))
        elif len(pts) > 1:
            poly = np.array([_fx(*q) for q in pts], np.int32)
            p.add(base_a, lambda img: cv2.polylines(img, [poly], False, base, th, cv2.LINE_AA, SHIFT))
    elif t == LINE_LIST:
        for i in range(0, len(pts) - 1, 2):
            c, a = vcol(i)
            p.add(a, lambda img, i=i, c=c: cv2.line(img, _fx(*pts[i]), _fx(*pts[i + 1]), c, th, cv2.LINE_AA, SHIFT))
    elif t == SPHERE_LIST:
        r = int(min(max(2.0, sx / 2.0), 1e5) * (1 << SHIFT))
        for i, q in enumerate(pts):
            c, a = vcol(i)
            p.add(a, lambda img, q=q, c=c: cv2.circle(img, _fx(*q), r, c, -1, cv2.LINE_AA, SHIFT))
    elif t == CUBE_LIST:
        for i, q in enumerate(m.points):
            c, a = vcol(i)
            poly = _local_rect(full, q.x, q.y, m.scale.x / 2.0, m.scale.y / 2.0)
            p.add(a, lambda img, poly=poly, c=c: cv2.fillPoly(img, [poly], c, cv2.LINE_AA, SHIFT))
    elif t == POINTS:   # screen-facing squares, as RViz draws them seen from above
        hw, hh = min(max(1.0, sx / 2.0), 1e5), min(max(1.0, (sy or sx) / 2.0), 1e5)
        for i, q in enumerate(pts):
            c, a = vcol(i)
            p.add(a, lambda img, q=q, c=c: cv2.rectangle(img, _fx(q[0] - hw, q[1] - hh), _fx(q[0] + hw, q[1] + hh),
                                                         c, -1, cv2.LINE_8, SHIFT))
    return 'drawn'


def draw_text(img, text, u, v, color, scale=0.4):
    """Small centred text with a contrasting halo (goal_selector's ids are black)."""
    lines = (text or '').split('\n')
    halo = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
    (_, th), _ = cv2.getTextSize('Ag', cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = v - (len(lines) - 1) * (th + 4) / 2.0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        org = (int(u - tw / 2.0), int(y + th / 2.0))
        cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, scale, halo, 3, cv2.LINE_AA)
        cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += th + 4


def grid_labels(msg):
    """OccupancyGrid -> uint8 labels: 0 unknown, 1 free, 2 occupied."""
    info = msg.info
    raw = msg.data              # rclpy: array('b'); a list in tests
    data = (np.frombuffer(raw.tobytes(), np.int8) if hasattr(raw, 'tobytes') else np.asarray(raw, np.int8)
            ).reshape(info.height, info.width)
    labels = np.zeros(data.shape, np.uint8)
    labels[(data >= 0) & (data < OCC_THRESHOLD)] = 1
    labels[data >= OCC_THRESHOLD] = 2
    return labels


class AgentPanel:
    """One agent's received state (callbacks only store here) and its panel's drawing."""

    def __init__(self, spawn, cls, color, layers):
        self.name, self.cls, self.color, self.spawn = spawn.name, cls, color, spawn
        self.layers = layers
        self.msgs = {}                    # grid and pose layers: the newest message
        self.rates = dict((k, Rate()) for k in LAYERS)
        self.labels = {}                  # grid layer -> (msg, labels): converted once per message
        self.boxes = {}                   # grid layer -> (msg, its known cells' world box corners)
        self.checked = {}                 # grid and pose layers -> (msg, usable), checked once per message
        self.invalid = {}                 # grid and pose layers -> messages that could not be drawn
        self.errors, self.last_error = 0, ''
        self.pose_raw, self.pose_new = None, 0
        self.stats = {}

    def store(self, key):
        def cb(msg):
            self.msgs[key] = msg
            self.rates[key].tick(time.monotonic())
        return cb

    def pose_sink(self, raw):
        """Poller handler, per message: only kept, not deserialized."""
        self.pose_raw = raw
        self.pose_new += 1

    def take_pose(self, now, deserialize):
        """After a poll: count what came (the header's true rate), deserialize only the newest."""
        if not self.pose_new:
            return
        self.rates['pose'].tick(now, self.pose_new)
        self.pose_new = 0
        try:
            self.msgs['pose'] = deserialize(self.pose_raw)
        except Exception:  # noqa: BLE001  a bad sample must not end the session
            self.invalid['pose'] = self.invalid.get('pose', 0) + 1

    def usable(self, key):
        """The layer's newest message if it can be drawn (RViz: a grid whose data does not fit its
        size is dropped, a NaN pose rejected), else None; each bad message counted once."""
        msg = self.msgs.get(key)
        if msg is None:
            return None
        c = self.checked.get(key)
        if c is None or c[0] is not msg:
            ok = pose_ok(msg.pose) if LAYERS[key].kind == 'pose' else grid_ok(msg)
            c = self.checked[key] = (msg, ok)
            if not ok:
                self.invalid[key] = self.invalid.get(key, 0) + 1
        return msg if c[1] else None

    def error(self, e):
        """A drawing that failed anyway: counted and shown, the session goes on."""
        self.errors += 1
        self.last_error = '%s: %s' % (type(e).__name__, e)

    def invalid_count(self, store):
        return sum(self.invalid.values()) + sum(store.invalid.get(topic_of(self.name, LAYERS[k]), 0)
                                                for k in self.layers if LAYERS[k].kind == 'markers')

    def store_markers(self, key, store):
        topic = topic_of(self.name, LAYERS[key])

        def cb(msg):
            now = time.monotonic()
            store.apply(topic, msg.markers, now)
            self.rates[key].tick(now)
        return cb

    def pose_world(self, frames):
        """(x, y, yaw) of the agent's DLIO pose in the world, else (none yet, not finite, another
        frame) its spawn pose."""
        msg = self.usable('pose')
        to_world = resolve_frame(frames, msg.header.frame_id) if msg is not None else None
        if to_world is None:
            return self.spawn.x, self.spawn.y, self.spawn.yaw
        q = msg.pose
        x, y = to_world.apply(q.position.x, q.position.y)
        return x, y, to_world.yaw() + yaw_of(q.orientation)

    def world_points(self, frames):
        """What the world view keeps in sight: the DLIO pose and the grids' known cells (a planner
        window can be mostly unknown)."""
        out = []
        msg = self.usable('pose')
        if msg is not None and resolve_frame(frames, msg.header.frame_id) is not None:
            out.append(self.pose_world(frames)[:2])
        for key in ('occ', 'map'):
            msg = self.usable(key)
            to_world = resolve_frame(frames, msg.header.frame_id) if msg is not None else None
            if to_world is not None and msg.info.width and msg.info.height:
                out += self.known_box(key, msg, to_world)
        return out

    def known_box(self, key, msg, to_world):
        """Corners of the world box round the grid's known cells, once per message. The extremes of a
        turned grid lie at each row's first and last known cell, so only those are transformed."""
        cached = self.boxes.get(key)
        if cached is not None and cached[0] is msg:
            return cached[1]
        known = self.grid_labels(key, msg) > 0
        rows = np.nonzero(known.any(axis=1))[0]
        box = []
        if len(rows):
            sub = known[rows]
            first = sub.argmax(axis=1)
            last = sub.shape[1] - sub[:, ::-1].argmax(axis=1)          # one past the last known cell
            i = np.concatenate([first, first, last, last]).astype(np.float64)
            j = np.concatenate([rows, rows + 1, rows, rows + 1]).astype(np.float64)
            m = cell_aff(msg.info, to_world)
            x, y = m.a * i + m.b * j + m.c, m.d * i + m.e * j + m.f
            box = [(x.min(), y.min()), (x.max(), y.max())]
        self.boxes[key] = (msg, box)
        return box

    def grid_labels(self, key, msg):
        cached = self.labels.get(key)
        if cached is None or cached[0] is not msg:
            cached = (msg, grid_labels(msg))
            self.labels[key] = cached
        return cached[1]

    def draw(self, view, shown, store, frames, now):
        img = np.full((view.h, view.w, 3), BG, np.uint8)
        va = view.aff()
        stats = {'drawn': 0, 'skipped': 0, 'frames': {}}
        sp = grid_spacing(view.res)
        xmin, ymin, xmax, ymax = view.extent()
        for i in range(int(math.floor(xmin / sp)), int(math.ceil(xmax / sp)) + 1):
            u, _ = va.apply(i * sp, 0.0)
            cv2.line(img, _fx(u, 0), _fx(u, view.h), GRIDLINE, 1, cv2.LINE_8, SHIFT)
        for j in range(int(math.floor(ymin / sp)), int(math.ceil(ymax / sp)) + 1):
            _, v = va.apply(0.0, j * sp)
            cv2.line(img, _fx(0, v), _fx(view.w, v), GRIDLINE, 1, cv2.LINE_8, SHIFT)
        for key in DRAW_ORDER:
            if key not in self.layers or key not in shown:
                continue
            kind = LAYERS[key].kind
            if kind == 'grid':
                self._draw_grid(img, key, view, frames, stats)
            elif kind == 'markers':
                p = Painter(img)
                markers, fade = store.markers(topic_of(self.name, LAYERS[key]), now)
                for m in markers:
                    to_world = resolve_frame(frames, m.header.frame_id)
                    if to_world is None:
                        f = m.header.frame_id or "''"
                        stats['frames'][f] = stats['frames'].get(f, 0) + 1
                        continue
                    stats[draw_marker(p, m, va @ to_world, view.res, fade)] += 1
                p.flush()
            elif kind == 'pose' and self.usable('pose') is not None:
                if resolve_frame(frames, self.msgs['pose'].header.frame_id) is None:
                    f = self.msgs['pose'].header.frame_id or "''"
                    stats['frames'][f] = stats['frames'].get(f, 0) + 1
                    continue
                x, y, yaw = self.pose_world(frames)
                u, v = va.apply(x, y)
                r = max(4.0, 0.3 / view.res)
                ln = max(12.0, 0.8 / view.res)
                cv2.circle(img, _fx(u, v), int(r * (1 << SHIFT)), self.color, -1, cv2.LINE_AA, SHIFT)
                cv2.circle(img, _fx(u, v), int(r * (1 << SHIFT)), (0, 0, 0), 1, cv2.LINE_AA, SHIFT)
                cv2.line(img, _fx(u, v), _fx(u + ln * math.cos(yaw), v - ln * math.sin(yaw)), (0, 0, 0), 2,
                         cv2.LINE_AA, SHIFT)
        self.stats = stats
        return img

    def _draw_grid(self, img, key, view, frames, stats):
        msg = self.usable(key)
        if msg is None or msg.info.width == 0 or msg.info.height == 0:
            return
        to_world = resolve_frame(frames, msg.header.frame_id)
        if to_world is None:
            f = msg.header.frame_id or "''"
            stats['frames'][f] = stats['frames'].get(f, 0) + 1
            return
        m = grid_aff(msg.info, to_world, view)
        out = cv2.warpAffine(self.grid_labels(key, msg), np.array(m.rows(), np.float64), (view.w, view.h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        free, occ = (OCC_FREE, OCC_OCC) if key == 'occ' else (MAP_FREE, MAP_OCC)
        img[out == 1] = free
        img[out == 2] = occ
        if key == 'occ':
            corners = np.array([_fx(u, v) for u, v in grid_outline(m, msg.info.width, msg.info.height)], np.int32)
            cv2.polylines(img, [corners], True, OCC_OUTLINE, 1, cv2.LINE_AA, SHIFT)


def put(img, text, org, color, scale=FONT_SCALE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def wrap_items(items, width, scale=FONT_SCALE, gap=18):
    """[(text, color)] -> rows of [(text, color, x)] that fit in width px."""
    rows, row, x = [], [], 0
    for text, color in items:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if row and x + tw > width:
            rows.append(row)
            row, x = [], 0
        row.append((text, color, x))
        x += tw + gap
    if row:
        rows.append(row)
    return rows


def draw_header(a, w, shown, follow, zoom, follow_m, view, now, invalid=0):
    head = np.full((HEADER_ROWS * ROW_H + 8, w, 3), HEAD_BG, np.uint8)
    cv2.rectangle(head, (0, 0), (8, head.shape[0]), a.color, -1)
    mode = ('follow %.0f m across (zoom x%.2g)' % (follow_m / zoom, zoom) if follow else 'world view')
    put(head, '%s (%s)' % (a.name, a.cls.upper()), (16, ROW_H), (255, 255, 255), 0.6, 2)
    put(head, '%s, %g m grid' % (mode, grid_spacing(view.res)), (150, ROW_H), TEXT_ON)
    items = []
    for i, (k, lay) in enumerate(LAYERS.items()):
        sub = k in a.layers
        r = a.rates[k]
        age = r.age(now)
        color = TEXT_OFF if not sub or k not in shown else TEXT_STALE if age is None or age > STALE_S else TEXT_ON
        items.append((layer_text(i + 1, lay, sub, k in shown, r.hz(now), age), color))
    rows = wrap_items(items, w - 24)[:HEADER_ROWS - 2]
    for j, row in enumerate(rows):
        for text, color, x in row:
            put(head, text, (16 + x, ROW_H * (j + 2)), color)
    s = a.stats
    frames = ', '.join('%s x%d' % kv for kv in sorted(s.get('frames', {}).items()))
    errors = ', draw errors %d (%s)' % (a.errors, a.last_error[:60]) if a.errors else ''
    put(head, 'markers drawn %d, skipped (mesh/triangles) %d, invalid (NaN/inf, bad grid) %d, unsupported frame %s%s'
        % (s.get('drawn', 0), s.get('skipped', 0), invalid, frames or 'none', errors), (16, ROW_H * HEADER_ROWS),
        TEXT_STALE if frames or invalid or errors else (190, 190, 190))
    return head


LEGEND = (('occ: free', OCC_FREE), ('occupied', OCC_OCC), ('window', OCC_OUTLINE),
          ('original HGP path', (0, 128, 255)), ('HGP path', (0, 0, 255)), ('committed traj (speed)', (255, 200, 0)),
          ('frontier: active', (255, 204, 0)), ('dormant', (153, 153, 153)), ('robot -> goal', (0, 255, 255)),
          ('map: free', MAP_FREE), ('occupied', MAP_OCC))
POSE_LEGEND = 'disc: DLIO pose (its panel\'s colour)'
KEYS_HELP = '1-7 layers   f follow   +/- zoom (follow)   q/Esc quit'


def legend_rows(width, pose_color):
    items = [('  ' + t, c) for t, c in LEGEND + ((POSE_LEGEND, pose_color),)] + [(KEYS_HELP, TEXT_ON)]
    return wrap_items(items, width - 20)


def draw_legend(width, pose_color=(200, 200, 200)):
    """pose_color: one panel's colour, or grey for several (each header bar shows its own)."""
    rows = legend_rows(width, pose_color)
    img = np.full((len(rows) * ROW_H + 8, width, 3), LEGEND_BG, np.uint8)
    for j, row in enumerate(rows):
        for text, color, x in row:
            y = ROW_H * (j + 1)
            if text.startswith('  '):
                cv2.rectangle(img, (10 + x, y - 10), (20 + x, y), color, -1)
                cv2.rectangle(img, (10 + x, y - 10), (20 + x, y), (130, 130, 130), 1)
                put(img, text.strip(), (26 + x, y), TEXT_ON)
            else:
                put(img, text, (10 + x, y), (170, 170, 170))
    return img


class Viewer:
    """Panels, toggles and views; render() is the whole window."""

    def __init__(self, panels, frames, store, bounds, size, follow, follow_m, shown, grow=False):
        self.panels, self.frames, self.store, self.bounds, self.grow = panels, frames, store, bounds, grow
        self.fit = None                   # grow: the box round what the agents have seen so far
        self.width, self.height = size
        self.follow, self.follow_m, self.zoom = follow, follow_m, 1.0
        self.shown = set(shown)
        self.legend = draw_legend(self.width, panels[0].color if len(panels) == 1 else (200, 200, 200))
        self.rows, self.cols, self.pw, self.ph = panel_size(len(panels), self.width, self.height,
                                                            self.legend.shape[0])

    def key(self, k):
        """True to quit."""
        if k in (ord('q'), 27):
            return True
        if ord('1') <= k < ord('1') + len(LAYERS):
            key = list(LAYERS)[k - ord('1')]
            self.shown.symmetric_difference_update({key})
        elif k == ord('f'):
            self.follow = not self.follow
        elif k in (ord('+'), ord('=')) and self.follow:
            self.zoom = min(self.zoom * 1.25, 20.0)
        elif k in (ord('-'), ord('_')) and self.follow:
            self.zoom = max(self.zoom / 1.25, 0.1)
        return False

    def view_of(self, a):
        if self.follow:
            x, y, _ = a.pose_world(self.frames)
            return follow_view(x, y, self.follow_m, self.zoom, self.pw, self.ph)
        return world_view(self.bounds, self.pw, self.ph)

    def render(self, now):
        if self.grow:
            seen = []
            for a in self.panels:
                try:
                    seen += [q for q in a.world_points(self.frames)
                             if _finite(*q) and near_spawns(q, [b.spawn for b in self.panels])]
                except Exception as e:  # noqa: BLE001  one agent's bad grid must not end the session
                    a.error(e)
            if seen:                       # until then the spawn poses' box
                self.fit = grow_bounds(self.fit, seen + [(a.spawn.x, a.spawn.y) for a in self.panels])
                self.bounds = self.fit
        tiles = []
        for a in self.panels:
            view = self.view_of(a)
            try:
                body = a.draw(view, self.shown, self.store, self.frames, now)
            except Exception as e:  # noqa: BLE001  as RViz: an error on the display, the rest runs on
                a.error(e)
                body = np.full((view.h, view.w, 3), BG, np.uint8)
            head = draw_header(a, self.pw, self.shown, self.follow, self.zoom, self.follow_m, view, now,
                               a.invalid_count(self.store))
            tile = np.vstack([head, body])
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (20, 20, 20), 2)
            tiles.append(tile)
        while len(tiles) < self.rows * self.cols:
            tiles.append(np.full_like(tiles[0], 15))
        grid = np.vstack([np.hstack(tiles[r * self.cols:(r + 1) * self.cols]) for r in range(self.rows)])
        return np.vstack([self.legend[:, :grid.shape[1]], grid])

    def summary(self, now):
        """One line per agent for the log: every layer's message count, rate and age, and the counts."""
        out = []
        for a in self.panels:
            parts = ['%s %d msgs %.1f Hz %s' % (k, a.rates[k].count, a.rates[k].hz(now), fmt_age(a.rates[k].age(now)))
                     for k in a.layers]
            s, v = a.stats, self.view_of(a)
            out.append('%s: %s; markers drawn %d, skipped %d, invalid %d, unsupported frames %s, draw errors %d%s; '
                       'view centre (%.2f, %.2f) m, %.4f m/px, %dx%d px under a %d px header' % (
                           a.name, ', '.join(parts), s.get('drawn', 0), s.get('skipped', 0),
                           a.invalid_count(self.store), s.get('frames') or '{}', a.errors,
                           ' (%s)' % a.last_error if a.errors else '', v.cx, v.cy, v.res, v.w, v.h,
                           HEADER_ROWS * ROW_H + 8))
        return out


# ---------------------------------------------------------------------------- the window (a child process)

def window_closed(shown):
    """(closed, shown) of WIN. GTK (the image's OpenCV 4.5.4) answers WND_PROP_VISIBLE -1 always, but
    WND_PROP_AUTOSIZE -1 once the window is destroyed (0 before, even unshown); Qt answers VISIBLE 0
    once it is closed, after having been 1."""
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_AUTOSIZE) < 0:
        return True, shown
    vis = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE)
    shown = shown or vis >= 1
    return shown and 0 <= vis < 1, shown


def window_main(argv):
    """sim_viz.py --window FD W H: Display's child. No ROS: it shows the frames the node puts in the
    shared memory FD and sends back keys; it ends when the window is closed or its stdin is."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the pane's Ctrl-C is for the node, which closes this
    fd, w, h = (int(v) for v in argv[:3])
    img = np.ndarray((h, w, 3), np.uint8, mmap.mmap(fd, w * h * 3))
    cv2.setNumThreads(1)
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.imshow(WIN, img)
        cv2.resizeWindow(WIN, w, h)
        cv2.waitKey(1)
    except cv2.error as e:
        print('sim_viz: no window on DISPLAY=%r: %s' % (os.environ.get('DISPLAY'), str(e).strip()), file=sys.stderr)
        return 2
    out, shown = sys.stdout.fileno(), False
    os.set_blocking(0, False)
    try:
        os.write(out, b'R')
        while True:
            try:
                got = os.read(0, 64)
                if not got:
                    return 0                       # the node closed it
            except BlockingIOError:
                got = b''
            if b'F' in got:
                cv2.imshow(WIN, img)               # copies: the node may draw the next one now
                os.write(out, b'A')
            k = cv2.waitKey(WINDOW_POLL_MS)
            if k != -1:
                os.write(out, b'K' + bytes([k & 0xFF]))
            closed, shown = window_closed(shown)
            if closed:
                return 0
    except OSError:                                # the node is gone
        return 0


class Display:
    """The window, in a child process with no ROS (window_main). GDK _exit()s a process that loses its
    X connection (gdk_x_io_error), which here would undeclare every subscription at once; so only the
    child holds the display, and the node leaves gently when it goes. Frames go through shared memory
    (a memfd), one in flight: 'F' shows it, the child answers 'A' once it has copied it and 'K' + byte
    for a key. Closing its stdin closes the window."""

    def __init__(self, shape):
        h, w = shape[:2]
        fd = os.memfd_create('sim_viz_frame')
        try:
            os.ftruncate(fd, h * w * 3)
            self.frame = np.ndarray((h, w, 3), np.uint8, mmap.mmap(fd, h * w * 3))
            self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), WINDOW_FLAG, str(fd), str(w),
                                          str(h)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, pass_fds=(fd,))
        finally:
            os.close(fd)                           # the mapping and the child keep it
        self.out = self.proc.stdout.fileno()
        os.set_blocking(self.out, False)
        self.pending, self.ready, self.busy, self.gone = b'', False, True, False

    def events(self):
        """Keys pressed since the last call, or None once the window is gone."""
        try:
            got = os.read(self.out, 4096)
            if not got:
                self.gone = True
        except BlockingIOError:
            got = b''
        except OSError:
            self.gone, got = True, b''
        buf, keys, i = self.pending + got, [], 0
        while i < len(buf):
            c = buf[i:i + 1]
            if c == b'K':
                if i + 1 == len(buf):
                    break                          # its key byte comes with the next read
                keys.append(buf[i + 1])
                i += 2
                continue
            if c in (b'R', b'A'):
                self.ready = self.ready or c == b'R'
                self.busy = False
            i += 1
        self.pending = buf[i:]
        return None if self.gone else keys

    def wait_ready(self, timeout, stopped):
        """None once the window is up, else why not."""
        deadline = time.monotonic() + timeout
        while not self.ready:
            if self.events() is None:
                return 'the window process ended (exit %s) before its window opened' % self.reap()
            if stopped():
                return 'stopped before the window opened'
            if time.monotonic() > deadline:
                return 'no window after %.0f s' % timeout
            select.select([self.out], [], [], 0.1)
        return None

    def show(self, img):
        self.frame[...] = img
        try:
            self.proc.stdin.write(b'F')
            self.proc.stdin.flush()
            self.busy = True
        except OSError:
            self.gone = True

    def how(self):
        rc = self.reap(1.0)                        # its stdout closed: it is exiting
        return 'closed' if rc == 0 else 'its process exited %d: X connection lost?' % rc

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass

    def reap(self, timeout=2.0):
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait()


# ---------------------------------------------------------------------------- main

def build_parser():
    ap = argparse.ArgumentParser(prog='sim_viz.py', description=__doc__.split('\n\n')[0])
    ap.add_argument('--spawn', default='/rover_sim/run/spawn.yaml', help='the run\'s spawn.yaml (sim.py writes it)')
    ap.add_argument('--agents', default='', help='comma-separated agents to show (default: all, up to %d)' % MAX_PANELS)
    ap.add_argument('--layers', default='default',
                    help='comma-separated layers to subscribe: %s; default (%s), all' % (', '.join(LAYERS),
                                                                                         ','.join(DEFAULT_LAYERS)))
    ap.add_argument('--follow', action='store_true', help='start in the follow view (f toggles)')
    ap.add_argument('--follow-m', type=float, default=15.0, help='follow view: metres across (default 15)')
    ap.add_argument('--bounds', type=float, nargs=4, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                    help='world view, m (default: spawn.yaml walls, else the spawn poses +/- %g m)' % SPAWN_MARGIN_M)
    ap.add_argument('--size', type=int, nargs=2, metavar=('W', 'H'), default=(1600, 1000), help='window pixels')
    ap.add_argument('--hz', type=float, default=5.0, help='redraw rate (default 5)')
    ap.add_argument('--snapshot', metavar='PNG', help='no window: write one frame to PNG and exit')
    ap.add_argument('--snapshot-after', type=float, default=8.0, metavar='S',
                    help='seconds of listening before the snapshot (default 8: frontiers come every 2 s)')
    return ap


def setup(args):
    """Everything before rclpy: (spawn data, spawns, the panels' agents, layers, bounds)."""
    import yaml
    with open(args.spawn) as f:
        data = yaml.safe_load(f)
    spawns = sim_tf.spawn_poses(data, args.spawn)
    agents = pick_agents(spawns, [n for n in (t.strip() for t in args.agents.split(',')) if n])
    layers = parse_layers(args.layers)
    if not (args.hz > 0 and args.follow_m > 0 and args.snapshot_after >= 0 and min(args.size) >= 200):
        raise ValueError('--hz and --follow-m must be > 0, --snapshot-after >= 0, --size at least 200 x 200')
    return data, spawns, agents, layers, view_bounds(data, spawns, args.bounds)


def say(text):
    """Print, best effort: a closed stdout must not cut the leave short."""
    try:
        print(text, flush=True)
    except Exception:  # noqa: BLE001
        pass


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if argv[1:2] == [WINDOW_FLAG]:
        return window_main(argv[2:])
    try:
        from rclpy.utilities import remove_ros_args
        rest = remove_ros_args(argv)[1:]
    except ImportError:                    # no ROS here: only the refusals can run
        rest = argv[1:]
    args = build_parser().parse_args(rest)
    try:                                   # all of it before rclpy opens a session
        data, spawns, agents, layers, bounds = setup(args)
    except (OSError, ValueError, sim_tf.SpawnError) as e:
        print('sim_viz: %s' % e, file=sys.stderr)
        return 2
    uri = os.environ.get('ZENOH_SESSION_CONFIG_URI')
    if uri != drive_pattern.SIM_SESSION_URI:
        print('sim_viz: not in a rover_sim container (ZENOH_SESSION_CONFIG_URI=%r, the sim\'s is %s): refusing to '
              'join another graph' % (uri, drive_pattern.SIM_SESSION_URI), file=sys.stderr)
        return 2
    if cv2 is None:
        print('sim_viz: needs numpy and cv2 (the gazebo image has them)', file=sys.stderr)
        return 2
    cv2.setNumThreads(1)                   # its cores are shared with nothing heavy, but keep it small

    state = {'stop': None, 'leaving': False}

    def on_signal(signum, _frame):
        if state['leaving']:
            say('sim_viz: signal %d while leaving the graph: still leaving, one subscription at a time' % signum)
        elif state['stop'] is None:
            state['stop'] = signum
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    snap = bool(args.snapshot)
    store, frames = MarkerStore(), frame_table(spawns)
    classes = dict((a.get('name'), a.get('class', '?')) for a in data['agents'])
    panels = [AgentPanel(p, classes.get(p.name, '?'), PANEL_COLORS[i % len(PANEL_COLORS)], layers)
              for i, p in enumerate(agents)]
    # Only the spawn poses' view grows: --bounds and the walls are the world.
    viewer = Viewer(panels, frames, store, bounds, args.size, args.follow, args.follow_m, layers,
                    grow=not args.bounds and not data.get('walls'))
    display = None
    if not snap:                           # the window first: a display that fails costs no join
        display = Display(viewer.render(time.monotonic()).shape)
        why = display.wait_ready(WINDOW_READY_S, lambda: state['stop'] is not None)
        if why:
            display.close()
            display.reap()
            print('sim_viz: %s; not joining the graph' % why, file=sys.stderr)
            return 2

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.serialization import deserialize_message
    from rclpy.signals import SignalHandlerOptions
    from visualization_msgs.msg import MarkerArray

    entities, nodes, node = [], [], None
    try:
        # No rclpy signal handler: it would shut the context down under the paced leave.
        rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
        name = NODE_NAME + ('_snapshot' if snap else '')
        node = Node(name, namespace=NODE_NAMESPACE, enable_rosout=False, start_parameter_services=False)
        nodes.append(node)
        # The 100 Hz pose is read every POLL_S, not by a callback per message (Poller: a tenth of the CPU).
        poller = goal_relay.Poller(Node, name + '_poll', NODE_NAMESPACE)
        nodes.insert(0, poller.node)
        # Markers reliable with room for a burst, so a DELETE or DELETEALL is not lost; grids best effort,
        # newest only; the polled pose best effort with room for a poll's worth.
        reliable = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latest = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        polled = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=POSE_DEPTH,
                            reliability=ReliabilityPolicy.BEST_EFFORT)
        types = {'grid': OccupancyGrid, 'markers': MarkerArray}
        for a in panels:
            for k in layers:
                lay = LAYERS[k]
                if lay.kind == 'pose':
                    poller.add(PoseStamped, topic_of(a.name, lay), polled, a.pose_sink, raw=True, latest=False)
                    continue
                cb = a.store_markers(k, store) if lay.kind == 'markers' else a.store(k)
                sub = node.create_subscription(types[lay.kind], topic_of(a.name, lay), cb,
                                               reliable if lay.kind == 'markers' else latest)
                entities.append((node, 'subscription', sub))
        entities[:0] = poller.entities()
        say('sim_viz: %d subscription(s) for %s (%s); leaving takes %.1f s (q, Esc, closing the window, '
            'Ctrl-C or SIGTERM; never SIGKILL)' % (len(entities), ', '.join(a.name for a in panels),
                                                   ', '.join(layers), leave_seconds(len(entities))))
        next_poll = [0.0]

        def pump(deadline):
            """Spin until deadline or a stop, reading the polled pose every POLL_S."""
            while state['stop'] is None:
                now = time.monotonic()
                if now >= next_poll[0]:
                    poller.poll()
                    for a in panels:
                        a.take_pose(now, lambda raw: deserialize_message(raw, PoseStamped))
                    next_poll[0] = now + POLL_S
                if now >= deadline:
                    return
                rclpy.spin_once(node, timeout_sec=min(deadline, next_poll[0]) - now)

        if snap:
            pump(time.monotonic() + args.snapshot_after)
            if state['stop'] is not None:      # stopped before the frame: no PNG, and not a success
                say('sim_viz: signal %d before the snapshot: no frame written' % state['stop'])
                return 128 + state['stop']
            now = time.monotonic()
            img = viewer.render(now)
            d = os.path.dirname(os.path.abspath(args.snapshot))
            os.makedirs(d, exist_ok=True)
            if not cv2.imwrite(args.snapshot, img):
                raise OSError('could not write %s' % args.snapshot)
            for line in viewer.summary(now):
                say('sim_viz: ' + line)
            say('sim_viz: wrote %s (%dx%d) after %.1f s' % (args.snapshot, img.shape[1], img.shape[0],
                                                            args.snapshot_after))
            return 0
        period, next_draw = 1.0 / args.hz, 0.0
        while state['stop'] is None:
            keys = display.events()
            if keys is None:
                say('sim_viz: the window is gone (%s)' % display.how())
                break
            if any(viewer.key(k) for k in keys):
                break
            if keys:
                next_draw = 0.0            # show a toggle at once
            now = time.monotonic()
            if now >= next_draw and not display.busy:
                display.show(viewer.render(now))
                next_draw = now + period
            # Waiting for the child's answer, or for the next frame: either way back within 20 ms.
            pump(now + 0.02 if display.busy else min(next_draw, now + 0.02))
        return 0
    finally:
        state['leaving'] = True
        if display is not None:
            display.close()                # the window goes now; the node leaves after
        if not snap:
            try:
                for line in viewer.summary(time.monotonic()):
                    say('sim_viz: ' + line)
            except Exception:  # noqa: BLE001  leaving: nothing may keep the graph entities
                pass

        def spin():
            try:
                rclpy.spin_once(node, timeout_sec=0)
            except Exception:  # noqa: BLE001  a callback that fails again must not cut the leave short
                pass
        if nodes:
            t0 = time.monotonic()
            goal_relay.leave_gently(entities, nodes, spin)
            say('sim_viz: left the graph: %d subscription(s) undeclared one every %.1f s (%.1f s)'
                % (len(entities), goal_relay.LEAVE_GAP_S, time.monotonic() - t0))
        rclpy.try_shutdown()
        if display is not None:
            display.reap()


if __name__ == '__main__':
    sys.exit(main())
