#!/usr/bin/env python3
"""Grade a running demo (swap or explore) from the sim's ground truth and the run's pane logs.

  scripts/demo_eval.py [auto|swap|explore] [--duration 300] [--out f.json] [--run-dir D] [--log-dir D]
  scripts/demo_eval.py grade --recording f.json [--out f.json]     (grade a saved recording again)
  demo_eval.py record [--duration 300] [--maps]                     (in sim-mighty: the probe alone)

On the host (python3.8 + PyYAML + docker CLI): it reads the run's session.json and spawn.yaml
(scripts/sim.py's run dir, $ROVER_SIM_RUN or /tmp/rover_sim-<uid>/run), notes how long every pane
log of the run is (~/.local/state/rover_sim/logs/latest/<stack>/<pane>.log), records --duration
seconds with one rclpy probe in sim-mighty (`demo_eval.py record`, through docker exec; sim
containers only), then grades what was recorded together with what the pane logs gained meanwhile.
It writes the report and the recording as JSON (default <log dir>/demo_eval/) and prints one
PASS/FAIL line per criterion. auto takes the session's demo (swap when it gives goals).

The probe (sim-mighty: it has the planner's messages) subscribes per agent ground_truth/odom_wall
and dlio/odom_node/odom (serialized, not deserialised), sim/rtf, mode, term_goal, goal_reached and
mighty_node's traj_received markers, plus /trajs, and with --maps (explore) persistent_map/grid. It
has no rosout publisher and no parameter services and leaves the graph one subscription at a time
(goal_relay.leave_gently): rmw_zenoh 0.1.9 stalls delivery for everyone when a session undeclares
many entities at once. It is still one more session on the graph: run one probe at a time.

Criteria (Verification PR 5):
  legs         swap: every agent reached >= --min-legs legs and none timed out (goal_relay.py's LEG
               lines in the window; a leg reached by the estimate is checked against ground truth)
  frontiers    explore: every agent's goal_selector committed >= 1 frontier goal (its log lines
               "Exploration: -> frontier", else the term_goal commitments the probe saw)
  coverage     explore: the union of every agent's persistent_map/grid known cells, in the world
               frame, grew by >= --coverage-growth m^2 over the window
  separation   no two chassis came within --contact m of each other (oriented footprints from each
               agent's spawned URDF, <run>/urdf/<ns>.urdf, collision geometry; ground truth chassis
               poses at 20 Hz); the worst approach is reported
  contact      no chassis came within --contact m of the world's boxes (the collision boxes of
               src/rover_sim/worlds/<world>.world, model, link and collision poses composed)
  trajs        no 'Dropping traj' line in a pane log, and every agent accepted every other agent's
               /trajs: mighty_node runs at --log-level error, so its throttled 'Dropping traj' WARN
               never reaches the logs; the probe counts instead each sender's /trajs and each
               receiver's traj_received markers (published only for a trajectory that passed the
               frame-alignment check, mighty_node.cpp trajCallback)
  errors       no ERROR/FATAL line in the pane logs other than mighty_node's 'Map hold:' state lines
               (logged at ERROR because of that log level; counted and listed apart)
  rtf          no sim/rtf run below --rtf-min longer than 5 s and no new rtf_violations.log line
  odometry     each agent's DLIO estimate stayed within --odom-max m (xy) of ground truth; the xy
               ATE RMSE and the distance driven in the window are reported with it
Exit 0 every criterion PASS, 1 a FAIL, 2 a setup error (no run, no probe, no recording).

The pure parts (geometry, the URDF footprint, the world's boxes, the log scan, the grading)
import without rclpy and need no numpy; tests/test_demo_eval.py covers them.
"""

import argparse
import bisect
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import goal_relay  # noqa: E402  LEG lines, frames, the paced leave

REPO = Path(HERE).parent
WORLDS_DIR = REPO / 'src' / 'rover_sim' / 'worlds'
# scripts/sim.py's (tests/test_demo_eval.py checks they agree).
RUN_DIR_DEFAULT = '/tmp/rover_sim-%d/run' % os.getuid()
RUN_DIR_IN = '/rover_sim/run'
LOG_ROOT = Path.home() / '.local' / 'state' / 'rover_sim' / 'logs'
SIM_CONTAINERS = ('sim-gazebo', 'sim-drive', 'sim-dlio-jr', 'sim-dlio-rr', 'sim-mapping', 'sim-elevation',
                  'sim-mighty', 'sim-viz')
SIM_PROJECT = 'rover-sim-stack'
PROBE_CONTAINER = 'sim-mighty'
PROBE_SETUP = 'source /opt/ros/humble/setup.bash && source /home/swarm/code/mighty_ws/install/setup.bash'
MARK = '@@DEMO_EVAL@@ '
RTF_LOG = 'rtf_violations.log'

TRUTH_DT, EST_DT, COVERAGE_DT = 0.05, 0.1, 5.0     # s between recorded samples
POLL_S = 0.05                                     # the polled subscriptions are read this often
BIN_S = 1.0                                       # /trajs and traj_received counts per bin
ROBOT_TOP_M = 1.0                                 # boxes reaching below this can touch a chassis


class EvalError(Exception):
    pass


# ---------------------------------------------------------------------------- geometry

def rpy_matrix(r, p, y):
    """URDF's fixed-axis roll, pitch, yaw: Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr))


def mmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def mvec(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def compose3(parent, child):
    """(R, t) of child expressed in parent's parent."""
    (ra, ta), (rb, tb) = parent, child
    v = mvec(ra, tb)
    return mmul(ra, rb), (ta[0] + v[0], ta[1] + v[1], ta[2] + v[2])


def rot2(yaw, x, y):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


def rect_poly(x, y, yaw, fp):
    """World corners (counter-clockwise) of footprint fp = (xmin, xmax, ymin, ymax) at (x, y, yaw)."""
    xmin, xmax, ymin, ymax = fp
    out = []
    for bx, by in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)):
        dx, dy = rot2(yaw, bx, by)
        out.append((x + dx, y + dy))
    return out


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def seg_point(p, a, b):
    ax, ay = b[0] - a[0], b[1] - a[1]
    n = ax * ax + ay * ay
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / n))
    return math.hypot(p[0] - a[0] - t * ax, p[1] - a[1] - t * ay)


def segs_cross(a, b, c, d):
    d1, d2, d3, d4 = _cross(c, d, a), _cross(c, d, b), _cross(a, b, c), _cross(a, b, d)
    return ((d1 > 0) != (d2 > 0) or d1 == 0 or d2 == 0) and ((d3 > 0) != (d4 > 0) or d3 == 0 or d4 == 0) and \
        max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0])) and \
        max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))


def inside(p, poly):
    """p in the convex counter-clockwise polygon (its boundary included)."""
    return all(_cross(poly[i], poly[(i + 1) % len(poly)], p) >= 0 for i in range(len(poly)))


def poly_distance(a, b):
    """Distance between two convex polygons, 0 when they touch or overlap."""
    ea = [(a[i], a[(i + 1) % len(a)]) for i in range(len(a))]
    eb = [(b[i], b[(i + 1) % len(b)]) for i in range(len(b))]
    if inside(a[0], b) or inside(b[0], a) or any(segs_cross(p, q, r, s) for p, q in ea for r, s in eb):
        return 0.0
    return min(min(seg_point(p, r, s) for p in a for r, s in eb), min(seg_point(p, r, s) for p in b for r, s in ea))


def circumradius(fp):
    return max(math.hypot(x, y) for x in fp[:2] for y in fp[2:])


# ---------------------------------------------------------------------------- footprints (URDF)

def _origin(el):
    o = el.find('origin')
    xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
    rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
    return rpy_matrix(*rpy), tuple(xyz)


def _geometry_points(g):
    """Points whose xy hull bounds a URDF collision geometry, in its own frame; None for a mesh."""
    box, cyl, sph = g.find('box'), g.find('cylinder'), g.find('sphere')
    if box is not None:
        sx, sy, sz = (float(v) / 2 for v in box.get('size').split())
        return [(i * sx, j * sy, k * sz) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    if cyl is not None:
        r, h = float(cyl.get('radius')), float(cyl.get('length')) / 2
        ring = [(r * math.cos(2 * math.pi * i / 64), r * math.sin(2 * math.pi * i / 64)) for i in range(64)]
        return [(x, y, z) for x, y in ring for z in (-h, h)]
    if sph is not None:
        return [(0.0, 0.0, 0.0)], float(sph.get('radius'))
    return None


def urdf_footprint(text):
    """(xmin, xmax, ymin, ymax) of every collision geometry of the robot, in its root link's frame
    (the chassis: the floor point under the turning centre, which ground truth reports)."""
    robot = ET.fromstring(text)
    links = {l.get('name'): l for l in robot.findall('link')}
    joints = collections.defaultdict(list)
    children = set()
    for j in robot.findall('joint'):
        joints[j.find('parent').get('link')].append((j.find('child').get('link'), _origin(j)))
        children.add(j.find('child').get('link'))
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise EvalError('URDF has %d root links (%s), want 1' % (len(roots), ', '.join(roots)))
    ident = (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0))
    poses, todo, xs, ys = {roots[0]: ident}, [roots[0]], [], []
    while todo:
        name = todo.pop()
        for child, o in joints.get(name, ()):
            poses[child] = compose3(poses[name], o)
            todo.append(child)
        for c in links[name].findall('collision'):
            pts = _geometry_points(c.find('geometry'))
            if pts is None:
                raise EvalError('link %s: a collision mesh has no footprint here' % name)
            pose = compose3(poses[name], _origin(c))
            if isinstance(pts, tuple):                               # a sphere: centre +- radius
                (centre,), r = pts
                cx, cy, _ = compose3(pose, (ident[0], centre))[1]
                xs += [cx - r, cx + r]
                ys += [cy - r, cy + r]
                continue
            for p in pts:
                v = mvec(pose[0], p)
                xs.append(pose[1][0] + v[0])
                ys.append(pose[1][1] + v[1])
    if not xs:
        raise EvalError('URDF has no collision geometry')
    return round(min(xs), 4), round(max(xs), 4), round(min(ys), 4), round(max(ys), 4)


# ---------------------------------------------------------------------------- the world's boxes (SDF)

Box = collections.namedtuple('Box', 'name cx cy cz yaw sx sy sz')


def _sdf_pose(el, where):
    p = el.find('pose')
    if p is None or not (p.text or '').strip():
        return 0.0, 0.0, 0.0, 0.0
    if p.get('relative_to') or p.get('frame'):
        raise EvalError('%s: a pose relative to another frame is not handled' % where)
    v = [float(x) for x in p.text.split()]
    if len(v) != 6:
        raise EvalError('%s: pose %r is not x y z roll pitch yaw' % (where, p.text))
    if abs(v[3]) > 1e-9 or abs(v[4]) > 1e-9:
        raise EvalError('%s: roll/pitch in its pose is not handled' % where)
    return v[0], v[1], v[2], v[5]


def _compose2(a, b):
    dx, dy = rot2(a[3], b[0], b[1])
    return a[0] + dx, a[1] + dy, a[2] + b[2], a[3] + b[3]


def world_boxes(text, top=ROBOT_TOP_M):
    """[Box] of a Gazebo world's box collisions, in the world frame (poses composed with yaw
    only, as the rover_sim worlds use), that reach below `top` (a chassis can touch them)."""
    root = ET.fromstring(text)
    world = root.find('world')
    if world is None:
        raise EvalError('no <world>')
    out = []

    def walk(model, parent, path):
        if model.find('include') is not None:
            raise EvalError('%s: <include> is not handled (geometry from a model database)' % path)
        mp = _compose2(parent, _sdf_pose(model, path))
        for link in model.findall('link'):
            lp = _compose2(mp, _sdf_pose(link, path + '/' + link.get('name', '')))
            for c in link.findall('collision'):
                where = '%s/%s/%s' % (path, link.get('name', ''), c.get('name', ''))
                g = c.find('geometry')
                size = g.find('box/size') if g is not None else None
                if size is None:
                    if g is not None and g.find('plane') is not None:
                        continue                                  # the ground
                    raise EvalError('%s: only box and plane collisions are handled' % where)
                cx, cy, cz, yaw = _compose2(lp, _sdf_pose(c, where))
                sx, sy, sz = (float(v) for v in size.text.split())
                if cz - sz / 2 < top:
                    out.append(Box(where, round(cx, 6), round(cy, 6), round(cz, 6),
                                   round(math.remainder(yaw, 2 * math.pi), 9), sx, sy, sz))
        for sub in model.findall('model'):
            walk(sub, mp, path + '/' + sub.get('name', ''))

    if world.find('include') is not None:
        raise EvalError('<include> is not handled (geometry from a model database)')
    for m in world.findall('model'):
        walk(m, (0.0, 0.0, 0.0, 0.0), m.get('name', ''))
    return out


def box_poly(b):
    return rect_poly(b.cx, b.cy, b.yaw, (-b.sx / 2, b.sx / 2, -b.sy / 2, b.sy / 2))


def point_box(b, x, y):
    """Distance from (x, y) to box b's footprint, 0 inside."""
    lx, ly = rot2(-b.yaw, x - b.cx, y - b.cy)
    return math.hypot(max(abs(lx) - b.sx / 2, 0.0), max(abs(ly) - b.sy / 2, 0.0))


# ---------------------------------------------------------------------------- serialized messages

def _align(pos, n):
    return (pos + n - 1) & ~(n - 1)


def _string(buf, pos):
    """(text, end) of the CDR string at payload offset pos (after the 4-byte encapsulation)."""
    pos = _align(pos, 4)
    (n,) = struct.unpack_from('<I', buf, 4 + pos)
    if n < 1 or len(buf) < 8 + pos + n:
        raise ValueError('bad CDR string at %d' % pos)
    return bytes(buf[8 + pos:7 + pos + n]).decode('utf-8', 'replace'), pos + 4 + n


def _check(buf):
    if len(buf) < 16 or bytes(buf[0:2]) != b'\x00\x01':
        raise ValueError('not a little-endian CDR buffer')


def stamp_of(buf):
    _check(buf)
    sec, nsec = struct.unpack_from('<iI', buf, 4)
    return sec + nsec * 1e-9


def marker_ns(buf):
    """visualization_msgs/Marker's ns (after its std_msgs/Header)."""
    _check(buf)
    _, pos = _string(buf, 8)
    return _string(buf, pos)[0]


def dyntraj_id(buf):
    """dynus_interfaces/DynTraj's id: after its Header and float32[] bbox."""
    _check(buf)
    _, pos = _string(buf, 8)
    pos = _align(pos, 4)
    (n,) = struct.unpack_from('<I', buf, 4 + pos)
    pos = _align(pos + 4 + 4 * n, 4)
    return struct.unpack_from('<i', buf, 4 + pos)[0]


# ---------------------------------------------------------------------------- pane logs

CSI_RE = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[()][0-9A-Za-z]')
ROS_RE = re.compile(r'\[(DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\] \[(\d+(?:\.\d+)?)\] \[([^\]]*)\]: ?(.*)$')
ZENOH_RE = re.compile(r'\d{4}-\d\d-\d\dT[\d:.]+Z\s+(ERROR)\s+(\S+)\s*(.*)$')
FRONTIER_RE = re.compile(r'Exploration: -> frontier \d+')
# mighty_node runs at --log-level error, so it logs its map-age hold's state lines at ERROR on purpose.
EXPECTED_ERRORS = (('mighty_node', re.compile(r'^Map hold: ')),)
NUM_RE = re.compile(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?')


def log_sizes(log_dir):
    """{relative path: [bytes, inode]} of every pane log under log_dir. The inode follows a log that
    a restarted stack renamed (sim.py rotate_pane_logs) while the new one takes its path."""
    out = {}
    for p in sorted(Path(log_dir).rglob('*.log')):
        try:
            st = p.stat()
        except OSError:
            continue
        out[str(p.relative_to(log_dir))] = [st.st_size, st.st_ino]
    return out


def _size_ino(v):
    """(bytes, inode or None) of a log_sizes() value; a recording from before the inode is bytes."""
    return (v, None) if isinstance(v, int) else (v[0], v[1])


def read_new(path, start, end):
    try:
        with open(path, 'rb') as f:
            f.seek(start)
            data = f.read(max(0, end - start))
    except OSError:
        return []
    text = CSI_RE.sub('', data.decode('utf-8', 'replace'))
    return [l.rstrip() for part in text.split('\n') for l in part.split('\r') if l.strip()]


def logger_agent(logger):
    m = re.match(r'^([A-Z]{2}\d{2})\.', logger)
    return m.group(1) if m else None


def expected_error(logger, msg, allowed=()):
    """The name of the expected kind an ERROR line is (EXPECTED_ERRORS, --allow-error), or None."""
    for node, rx in EXPECTED_ERRORS:
        if logger.endswith(node) and rx.search(msg):
            return '%s: %s' % (node, rx.pattern.strip('^ '))
    for pattern in allowed:
        if re.search(pattern, '%s: %s' % (logger, msg)):
            return 'allowed: ' + pattern
    return None


def scan_logs(log_dir, start, end, allowed=()):
    """What the pane logs gained between two log_sizes() snapshots. A log is followed by its inode:
    one renamed meanwhile is read from where it was, the new one at its path from its start (a
    `sim.py start goals`), and one that shrank from its start."""
    out = {'files': 0, 'lines': 0, 'dropping_traj': collections.Counter(), 'errors': [], 'expected_errors':
           collections.Counter(), 'frontiers': collections.Counter(), 'legs': []}
    by_ino = dict((ino, size) for size, ino in map(_size_ino, start.values()) if ino is not None)
    for rel, v in sorted(end.items()):
        size, ino = _size_ino(v)
        begin = by_ino.get(ino, 0) if ino is not None else _size_ino(start.get(rel, 0))[0]
        lines = read_new(os.path.join(str(log_dir), rel), begin if begin <= size else 0, size)
        if not lines:
            continue
        out['files'] += 1
        out['lines'] += len(lines)
        for line in lines:
            leg = goal_relay.parse_leg_line(line)
            if leg is not None:
                out['legs'].append(leg)
                continue
            m = ROS_RE.search(line)
            if m:
                level, logger, msg = m.group(1), m.group(3), m.group(4)
                agent = logger_agent(logger)
                if 'Dropping traj' in msg:
                    out['dropping_traj'][agent or rel] += 1
                if logger.endswith('.goal_selector') and FRONTIER_RE.search(msg):
                    out['frontiers'][agent] += 1
                if level in ('ERROR', 'FATAL'):
                    kind = expected_error(logger, msg, allowed)
                    if kind:
                        out['expected_errors'][kind] += 1
                    else:
                        out['errors'].append((rel, logger, msg))
                continue
            if 'Dropping traj' in line:
                out['dropping_traj'][rel] += 1
            z = ZENOH_RE.search(line)
            if z:
                kind = expected_error('zenoh ' + z.group(2), z.group(3), allowed)
                if kind:
                    out['expected_errors'][kind] += 1
                else:
                    out['errors'].append((rel, 'zenoh ' + z.group(2), z.group(3)))
    return out


def error_kinds(errors, top=10):
    """The most frequent error messages, numbers masked: [(count, 'logger: message')]."""
    c = collections.Counter('%s: %s' % (logger, NUM_RE.sub('#', msg)[:100]) for _, logger, msg in errors)
    return [[n, k] for k, n in c.most_common(top)]


# ---------------------------------------------------------------------------- grading

def nearest_sample(times, rows, t, max_gap):
    i = bisect.bisect_left(times, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(times) and abs(times[j] - t) <= max_gap and (best is None or
                                                                       abs(times[j] - t) < abs(times[best] - t)):
            best = j
    return rows[best] if best is not None else None


def common_ticks(truth, dt):
    spans = [(rows[0][0], rows[-1][0]) for rows in truth.values() if rows]
    if not spans:
        return []
    t0, t1 = min(s[0] for s in spans), max(s[1] for s in spans)
    return [t0 + k * dt for k in range(int((t1 - t0) / dt) + 1)]


def events(ticks, dt):
    """Contiguous runs of tick times -> [[start, end]]."""
    out = []
    for t in ticks:
        if out and t - out[-1][1] <= 1.5 * dt:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


def separation(truth, fps, contact, dt=TRUTH_DT, max_gap=0.15):
    """Pairwise chassis distances over the recording (oriented footprints, ground truth)."""
    names = sorted(n for n in truth if truth[n] and n in fps)
    times = {n: [r[0] for r in truth[n]] for n in names}
    pairs = {'%s-%s' % p: {'min_m': None, 't': None, 'contact_ticks': []} for p in itertools.combinations(names, 2)}
    worst = None
    radius = {n: circumradius(fps[n]) for n in names}
    for t in common_ticks({n: truth[n] for n in names}, dt):
        at = {n: nearest_sample(times[n], truth[n], t, max_gap) for n in names}
        for a, b in itertools.combinations(names, 2):
            ra, rb = at[a], at[b]
            if ra is None or rb is None:
                continue
            p = pairs['%s-%s' % (a, b)]
            lb = math.hypot(ra[1] - rb[1], ra[2] - rb[2]) - radius[a] - radius[b]
            if lb > contact and p['min_m'] is not None and lb >= p['min_m']:
                continue
            d = poly_distance(rect_poly(ra[1], ra[2], ra[3], fps[a]), rect_poly(rb[1], rb[2], rb[3], fps[b]))
            if p['min_m'] is None or d < p['min_m']:
                p['min_m'], p['t'] = d, t
            if d <= contact:
                p['contact_ticks'].append(t)
            if worst is None or d < worst['min_m']:
                worst = {'pair': '%s-%s' % (a, b), 'min_m': d, 't': t, a: list(ra[1:4]), b: list(rb[1:4])}
    for p in pairs.values():
        p['contacts'] = events(p.pop('contact_ticks'), dt)
        p['min_m'] = None if p['min_m'] is None else round(p['min_m'], 3)
    if worst:
        worst['min_m'] = round(worst['min_m'], 3)
    return {'pairs': pairs, 'worst': worst}


def box_contact(truth, fps, boxes, contact, dt=TRUTH_DT):
    """Per agent: the smallest chassis-to-box distance and every contact with a world box."""
    out = {}
    aabb = []
    for b in boxes:
        r = math.hypot(b.sx, b.sy) / 2
        aabb.append((b, b.cx - r, b.cx + r, b.cy - r, b.cy + r))
    for n in sorted(truth):
        if n not in fps or not truth[n]:
            continue
        rad = circumradius(fps[n])
        best, hits, last = None, collections.defaultdict(list), -1e18
        for t, x, y, yaw in truth[n]:
            if t - last < dt * 0.999:
                continue
            last = t
            limit = max(contact, best[0] if best else 1e18) + rad
            poly = None
            for b, x0, x1, y0, y1 in aabb:
                if x < x0 - limit or x > x1 + limit or y < y0 - limit or y > y1 + limit:
                    continue
                if point_box(b, x, y) - rad > max(contact, best[0] if best else 1e18):
                    continue
                poly = poly or rect_poly(x, y, yaw, fps[n])
                d = poly_distance(poly, box_poly(b))
                if best is None or d < best[0]:
                    best = (d, t, b.name)
                if d <= contact:
                    hits[b.name].append(t)
        out[n] = {'min_m': None if best is None else round(best[0], 3), 't': best[1] if best else None,
                  'box': best[2] if best else None,
                  'contacts': [{'box': k, 'spans': events(v, dt)} for k, v in sorted(hits.items())]}
    return out


def leg_summary(legs, agents, t0, t1, min_legs, truth=None, goal_radius=0.5, slack=0.5):
    """Swap legs per agent that ended in the window, with ground truth's word on the reached ones."""
    times = {n: [r[0] for r in rows] for n, rows in (truth or {}).items()}
    out = {}
    for n in agents:
        mine = [g for g in legs if g.get('agent') == n and g.get('kind') == 'swap' and
                t0 <= float(g.get('t_end', -1)) <= t1]
        res = collections.Counter(g.get('result') for g in mine)
        far = []
        for g in mine:
            if g.get('result') != 'reached' or n not in times:
                continue
            row = nearest_sample(times[n], truth[n], float(g['t_end']), 0.2)
            if row is not None:
                d = math.hypot(row[1] - g['target_world'][0], row[2] - g['target_world'][1])
                if d > goal_radius + slack:
                    far.append({'leg': g.get('leg'), 'truth_m': round(d, 3)})
        durs = [float(g['duration_s']) for g in mine if g.get('result') == 'reached']
        out[n] = {'reached': res.get('reached', 0), 'timeout': res.get('timeout', 0),
                  'preempted': res.get('preempted', 0), 'durations_s': [round(d, 1) for d in durs],
                  'mean_s': round(sum(durs) / len(durs), 1) if durs else None, 'reached_but_far': far}
    ok = bool(agents) and all(v['reached'] - len(v['reached_but_far']) >= min_legs and v['timeout'] == 0
                              for v in out.values())
    return ok, out


def traj_summary(sent, accepted, agents, min_sent=10, min_ratio=0.5):
    """sent {sender id: n}, accepted {receiver: {sender id: n}} -> (ok, per ordered pair). A pair
    is judged once its sender published min_sent trajectories; with none judged, not ok."""
    ids = {n: int(n[2:]) for n in agents}
    pairs, ok, judged = {}, True, 0
    for ego in agents:
        for other in agents:
            if other == ego:
                continue
            s = int(sent.get(str(ids[other]), 0))
            a = int((accepted.get(ego) or {}).get(str(ids[other]), 0))
            good = s < min_sent or (a > 0 and a >= min_ratio * s)
            judged += s >= min_sent
            ok = ok and good
            pairs['%s<-%s' % (ego, other)] = {'sent': s, 'accepted': a, 'ok': good}
    return ok and (judged > 0 or len(agents) < 2), pairs


def rtf_summary(samples, rtf_min, hold_s=5.0):
    """The bridges' own rule (livox_sim_bridge.RtfMonitor): below rtf_min for more than hold_s
    since the first sample below it. sim/rtf is NaN while the bridge has under 1 s of on-time IMU
    to span; such a sample neither starts nor ends a run below and is left out of min and mean."""
    out, ok = {}, True
    for n, rows in sorted(samples.items()):
        vals = [v for _, v in rows if math.isfinite(v)]
        longest, run = 0.0, None
        for t, v in rows:
            if not math.isfinite(v):
                continue
            if v < rtf_min:
                run = t if run is None else run
                longest = max(longest, t - run)
            else:
                run = None
        out[n] = {'min': round(min(vals), 4) if vals else None,
                  'mean': round(sum(vals) / len(vals), 5) if vals else None, 'n': len(vals),
                  'unknown': len(rows) - len(vals), 'longest_below_s': round(longest, 1)}
        ok = ok and bool(vals) and longest <= hold_s
    return ok and bool(samples), out


def odometry_summary(est, truth, spawns, limit):
    """Per agent: the largest xy distance between DLIO's estimate (<ns>/map -> world) and truth,
    which grades, and the xy ATE RMSE over the window with the truth path driven meanwhile, as
    odom_eval reports them (base_link is right above the chassis root, so no lever arm in xy)."""
    out, ok = {}, True
    for n in sorted(est):
        if n not in spawns or not truth.get(n):
            continue
        times = [r[0] for r in truth[n]]
        worst, cnt, sq = None, 0, 0.0
        for t, x, y in est[n]:
            row = nearest_sample(times, truth[n], t, 0.06)
            if row is None:
                continue
            wx, wy = goal_relay.to_world(spawns[n], x, y)
            d = math.hypot(wx - row[1], wy - row[2])
            cnt += 1
            sq += d * d
            if worst is None or d > worst[0]:
                worst = (d, t)
        path = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(truth[n], truth[n][1:]))
        out[n] = {'max_m': None if worst is None else round(worst[0], 3), 't': worst[1] if worst else None,
                  'rmse_m': round(math.sqrt(sq / cnt), 4) if cnt else None, 'path_m': round(path, 1),
                  'samples': cnt}
        ok = ok and worst is not None and worst[0] <= limit
    return ok and bool(out), out


def coverage_summary(cov, growth):
    if len(cov) < 2:
        return False, {'samples': len(cov)}
    first, last = cov[0], cov[-1]
    out = {'first_m2': first[1], 'last_m2': last[1], 'growth_m2': round(last[1] - first[1], 2),
           'per_agent_last_m2': last[2], 'samples': len(cov),
           'timeline': [[round(t - first[0], 1), a] for t, a, _ in cov]}
    return last[1] - first[1] >= growth, out


def goal_commits(term_goal):
    """{ns: number of distinct goal_selector commitments (term_goal stamps)}."""
    return {n: len({r[1] for r in rows}) for n, rows in term_goal.items()}


def grade(rec, logs, boxes, opts):
    """The report of a recording, the pane-log scan and the world's boxes."""
    mode, agents = rec['mode'], sorted(rec['agents'])
    t0, t1 = rec['t0'], rec['t1']
    truth = {n: [tuple(r) for r in rows] for n, rows in rec['truth'].items()}
    spawns = {p.name: p for p in goal_relay.sim_tf.spawn_poses(rec['spawn'])}
    fps = {n: tuple(v) for n, v in rec['footprints'].items()}
    crit = []

    def add(name, ok, summary, detail):
        crit.append({'name': name, 'pass': bool(ok), 'summary': summary, 'detail': detail})

    if mode == 'swap':
        ok, per = leg_summary(logs['legs'], agents, t0, t1, opts.min_legs, truth, opts.goal_radius)
        add('legs', ok, 'every agent reached >= %d legs, no timeout: %s' % (opts.min_legs, ', '.join(
            '%s %d reached / %d timeout%s' % (n, v['reached'], v['timeout'],
                                              ' (mean %.1f s)' % v['mean_s'] if v['mean_s'] else '')
            for n, v in sorted(per.items()))), per)
    else:
        commits = goal_commits(rec.get('term_goal') or {})
        fr = {n: logs['frontiers'].get(n, 0) for n in agents}
        counts = {n: fr[n] or commits.get(n, 0) for n in agents}
        add('frontiers', all(v >= 1 for v in counts.values()) and bool(agents),
            'every goal_selector committed a frontier goal: %s' % ', '.join(
                '%s %d' % (n, counts[n]) for n in agents),
            {'frontier_log_lines': fr, 'term_goal_commitments': commits})
        ok, cov = coverage_summary(rec.get('coverage') or [], opts.coverage_growth)
        add('coverage', ok, 'known area grew by >= %.1f m^2: %s' % (
            opts.coverage_growth, '%.1f -> %.1f m^2' % (cov['first_m2'], cov['last_m2']) if 'first_m2' in cov
            else 'no persistent_map/grid samples (%d)' % cov['samples']), cov)

    sep = separation(truth, fps, opts.contact)
    w = sep['worst']
    hit = {k: v['contacts'] for k, v in sep['pairs'].items() if v['contacts']}
    alone = sum(bool(truth.get(n)) for n in agents) < 2     # one agent (--agents JR81): no pair to check
    add('separation', alone or (w is not None and w['min_m'] > opts.contact and not hit),
        'chassis never within %.2f m of another: closest %s' % (
            opts.contact, '%.3f m (%s at %.1f s)' % (w['min_m'], w['pair'], w['t'] - t0) if w else
            'n/a (one agent)' if alone else 'n/a (no truth)'), sep)
    bc = box_contact(truth, fps, boxes, opts.contact)
    add('contact', bool(bc) and all(not v['contacts'] for v in bc.values()),
        'no chassis within %.2f m of a world box: %s' % (opts.contact, ', '.join(
            '%s %s' % (n, '%.3f m' % v['min_m'] if v['min_m'] is not None else 'n/a') for n, v in sorted(bc.items()))),
        bc)
    tok, pairs = traj_summary(rec.get('trajs') or {}, rec.get('traj_received') or {}, agents)
    drops = sum(logs['dropping_traj'].values())
    add('trajs', tok and drops == 0,
        "no 'Dropping traj' in the logs (%d) and every agent took the others' /trajs (accepted/sent): %s" % (
            drops, ', '.join('%s %d/%d' % (k, v['accepted'], v['sent']) for k, v in sorted(pairs.items()))
            or 'one agent'), {'log_lines': dict(logs['dropping_traj']), 'pairs': pairs})
    add('errors', not logs['errors'], 'no unexpected ERROR line in %d new log lines of %d panes: %d (%s)' % (
        logs['lines'], logs['files'], len(logs['errors']), ', '.join(
            '%s x%d' % (k, n) for k, n in sorted(logs['expected_errors'].items())) or 'no map hold'),
        {'unexpected': error_kinds(logs['errors']), 'expected': dict(logs['expected_errors'])})
    rok, rtf = rtf_summary({n: [tuple(r) for r in rows] for n, rows in (rec.get('rtf') or {}).items()},
                           opts.rtf_min)
    viol = logs.get('rtf_violations', [])
    add('rtf', rok and not viol, 'sim/rtf never below %.2f for over 5 s, %d new violation line(s): %s' % (
        opts.rtf_min, len(viol), ', '.join('%s min %s mean %s' % (n, v['min'], v['mean']) for n, v in rtf.items())),
        {'agents': rtf, 'violations': viol})
    ook, odo = odometry_summary({n: [tuple(r) for r in rows] for n, rows in (rec.get('est') or {}).items()},
                                truth, spawns, opts.odom_max)
    add('odometry', ook, 'DLIO within %.2f m of ground truth: %s' % (opts.odom_max, ', '.join(
        '%s max %s m (RMSE %s m, %s m driven)' % (n, v['max_m'], v['rmse_m'], v['path_m'])
        for n, v in odo.items())), odo)
    return {'mode': mode, 'session': rec.get('session', {}).get('name'), 'agents': agents,
            'window': {'t0': t0, 't1': t1, 'duration_s': round(t1 - t0, 1)},
            'pass': all(c['pass'] for c in crit), 'criteria': crit, 'notes': rec.get('notes', [])}


def report_lines(report):
    out = ['%s  %-10s  %s' % ('PASS' if c['pass'] else 'FAIL', c['name'], c['summary']) for c in report['criteria']]
    out.append('%s: %s demo of %s, %.0f s, %d criteria' % ('PASS' if report['pass'] else 'FAIL', report['mode'],
                                                           report['session'], report['window']['duration_s'],
                                                           len(report['criteria'])))
    return out


# ---------------------------------------------------------------------------- the probe (rclpy)

class Recorder:
    """The probe's subscriptions; everything it keeps is plain data for the recording."""

    def __init__(self, node, spawns, agents, maps, poller=None):
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import OccupancyGrid, Odometry
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Empty, Float64, String
        from visualization_msgs.msg import Marker

        self.node, self.spawns, self.agents, self.poller = node, spawns, agents, poller
        self.entities, self.notes = [], []
        self.truth = {n: [] for n in agents}
        self.est = {n: [] for n in agents}
        self.rtf = {n: [] for n in agents}
        self.mode = {n: [] for n in agents}
        self.term_goal = {n: [] for n in agents}
        self.reached = {n: [] for n in agents}
        self.accepted = {n: collections.Counter() for n in agents}
        self.sent = collections.Counter()
        self.grids, self.coverage, self.last_cov = {}, [], 0.0
        latest = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        # every message between two polls (POLL_S): /trajs and the markers come at the replan rates
        burst = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=400, reliability=ReliabilityPolicy.BEST_EFFORT)
        every = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        for n in agents:
            self._poll(Odometry, '/%s/ground_truth/odom_wall' % n, latest, self._odom(self.truth[n], TRUTH_DT, True))
            self._poll(Odometry, '/%s/dlio/odom_node/odom' % n, latest, self._odom(self.est[n], EST_DT, False))
            self._sub(Float64, '/%s/sim/rtf' % n, lambda m, n=n: self.rtf[n].append((round(time.time(), 3), m.data)),
                      latest)
            self._sub(String, '/%s/mode' % n, self._mode(n), latest)
            self._sub(PoseStamped, '/%s/term_goal' % n, self._goal(n), every)
            self._sub(Empty, '/%s/goal_reached' % n, lambda m, n=n: self.reached[n].append(round(time.time(), 3)),
                      every)
            self._poll(Marker, '/%s/traj_received' % n, burst, self._marker(n), latest=False)
            if maps:
                self._sub(OccupancyGrid, '/%s/persistent_map/grid' % n, self._grid(n), latest)
        try:
            from dynus_interfaces.msg import DynTraj
        except ImportError:
            self.notes.append('no dynus_interfaces here: /trajs not counted (run the probe in sim-mighty)')
        else:
            self._poll(DynTraj, '/trajs', burst, self._traj, latest=False)
        if maps and importlib.util.find_spec('numpy') is None:
            self.notes.append('no numpy here: coverage not computed')
            maps = False
        self.maps = maps

    def _sub(self, msg_type, topic, cb, qos, raw=False):
        self.entities.append((self.node, 'subscription', self.node.create_subscription(msg_type, topic, cb, qos,
                                                                                       raw=raw)))

    def _poll(self, msg_type, topic, qos, handler, latest=True):
        """Serialized and read every POLL_S (goal_relay.Poller), or a callback per message without one."""
        if self.poller is not None:
            self.poller.add(msg_type, topic, qos, handler, raw=True, latest=latest)
        else:
            self._sub(msg_type, topic, handler, qos, True)

    def all_entities(self):
        return (self.poller.entities() if self.poller else []) + self.entities

    @staticmethod
    def _odom(rows, dt, with_yaw):
        import livox_sim_bridge

        def f(buf):
            try:
                sec, nsec, _, _, v = livox_sim_bridge.odom_parse(buf)
            except ValueError:
                return
            t = sec + nsec * 1e-9
            if rows and t - rows[-1][0] < dt * 0.8:     # polled every POLL_S: keep about one per dt
                return
            row = (round(t, 3), round(v[0], 4), round(v[1], 4))
            rows.append(row + (round(goal_relay.quat_yaw(*v[3:7]), 4),) if with_yaw else row)
        return f

    def _mode(self, n):
        def f(m):
            if not self.mode[n] or self.mode[n][-1][1] != m.data:
                self.mode[n].append((round(time.time(), 3), m.data))
        return f

    def _goal(self, n):
        def f(m):
            stamp = m.header.stamp.sec * 1000000000 + m.header.stamp.nanosec
            self.term_goal[n].append((round(time.time(), 3), stamp, round(m.pose.position.x, 3),
                                      round(m.pose.position.y, 3)))
        return f

    def _marker(self, n):
        def f(buf):
            try:
                ns = marker_ns(buf)
            except (ValueError, struct.error):
                return
            if ns.startswith('other_agent_traj_'):
                self.accepted[n][ns[len('other_agent_traj_'):]] += 1
        return f

    def _traj(self, buf):
        try:
            self.sent[str(dyntraj_id(buf))] += 1
        except (ValueError, struct.error):
            pass

    def _grid(self, n):
        def f(m):
            self.grids[n] = (m.info, m.data)
        return f

    def sample_coverage(self, now):
        """The union of every agent's latest persistent map known cells, in the world frame."""
        import numpy as np
        if not self.maps or now - self.last_cov < COVERAGE_DT or not self.grids:
            return
        self.last_cov = now
        keys, per = [], {}
        res = min(float(info.resolution) for info, _ in self.grids.values())
        for n, (info, data) in self.grids.items():
            d = np.array(data, np.int8) if isinstance(data, list) else np.frombuffer(memoryview(data), dtype=np.int8)
            idx = np.nonzero(d >= 0)[0]
            per[n] = round(float(idx.size) * float(info.resolution) ** 2, 2)
            if not idx.size:
                continue
            w, r = int(info.width), float(info.resolution)
            gx, gy = (idx % w + 0.5) * r, (idx // w + 0.5) * r
            o = info.origin
            oyaw = goal_relay.quat_yaw(o.orientation.x, o.orientation.y, o.orientation.z, o.orientation.w)
            mx = o.position.x + math.cos(oyaw) * gx - math.sin(oyaw) * gy
            my = o.position.y + math.sin(oyaw) * gx + math.cos(oyaw) * gy
            p = self.spawns[n]
            wx = p.x + math.cos(p.yaw) * mx - math.sin(p.yaw) * my
            wy = p.y + math.sin(p.yaw) * mx + math.cos(p.yaw) * my
            keys.append((np.floor(wx / res).astype(np.int64) + (1 << 20)) * (1 << 21)
                        + np.floor(wy / res).astype(np.int64) + (1 << 20))
        union = int(np.unique(np.concatenate(keys)).size) if keys else 0
        self.coverage.append((round(now, 3), round(union * res * res, 2), per))

    def result(self):
        return {'truth': self.truth, 'est': self.est, 'rtf': self.rtf, 'mode': self.mode,
                'term_goal': self.term_goal, 'goal_reached': self.reached,
                'traj_received': {n: dict(c) for n, c in self.accepted.items()}, 'trajs': dict(self.sent),
                'coverage': self.coverage, 'notes': self.notes}


def record(args):
    """In a sim container: subscribe, record for --duration s (or until SIGINT/SIGTERM), print the
    recording on one MARK line, then leave the graph gently."""
    goal_relay.drive_pattern.require_sim()
    try:
        poses = goal_relay.sim_tf.load_spawn(args.spawn)
    except (OSError, goal_relay.sim_tf.SpawnError) as e:
        raise EvalError(str(e))
    spawns = {p.name: p for p in poses}
    agents = [n for n in args.agents.split(',') if n] or sorted(spawns)
    unknown = [n for n in agents if n not in spawns]
    if unknown:
        raise EvalError('no sim agent %s in %s' % (', '.join(unknown), args.spawn))

    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    stop = []

    def on_signal(*_):
        stop.append(1)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node('demo_eval', namespace='/sim', enable_rosout=False, start_parameter_services=False)
    poller = goal_relay.Poller(Node, 'demo_eval_poll', '/sim')
    rec = Recorder(node, spawns, agents, args.maps, poller)
    t0 = time.time()
    end, next_poll = t0 + args.duration, t0
    sys.stderr.write('demo_eval: recording %s for %.0f s (%d subscriptions)\n' % (', '.join(agents), args.duration,
                                                                                  len(rec.all_entities())))
    while not stop and time.time() < end:
        rclpy.spin_once(node, timeout_sec=max(0.0, min(POLL_S, next_poll - time.time())))
        now = time.time()
        if now >= next_poll:
            poller.poll()
            next_poll = now + POLL_S
        rec.sample_coverage(now)
    t1 = time.time()
    out = dict(rec.result(), version=1, t0=round(t0, 3), t1=round(t1, 3), agents=agents,
               stopped_early=bool(stop))
    try:
        print(MARK + json.dumps(out, separators=(',', ':')), flush=True)
    except OSError:                  # the docker exec client is gone: still leave the graph gently
        pass
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    goal_relay.leave_gently(rec.all_entities(), [poller.node, node], lambda: rclpy.spin_once(node, timeout_sec=0))
    rclpy.try_shutdown()
    return 0


# ---------------------------------------------------------------------------- the host side

def sim_container(name):
    if name not in SIM_CONTAINERS:
        raise EvalError('refusing container %r: only the sim containers (%s)' % (name, ', '.join(SIM_CONTAINERS)))
    p = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}} {{index .Config.Labels '
                        '"com.docker.compose.project"}}', name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       universal_newlines=True)
    state, _, project = p.stdout.strip().partition(' ')
    if p.returncode != 0 or state != 'true':
        raise EvalError('%s is not running (scripts/sim.py up <session>)' % name)
    if project != SIM_PROJECT:
        raise EvalError('refusing %s: it belongs to compose project %r, not %s' % (name, project, SIM_PROJECT))
    return name


def load_run(run_dir):
    run_dir = Path(run_dir)
    try:
        session = json.loads((run_dir / 'session.json').read_text())
        import yaml
        spawn = yaml.safe_load((run_dir / 'spawn.yaml').read_text())
    except (OSError, ValueError) as e:
        raise EvalError('no running session in %s (%s): scripts/sim.py up <session>' % (run_dir, e))
    return session, spawn


def pick_mode(mode, session):
    if mode != 'auto':
        return mode
    if session.get('demo') in ('swap', 'explore'):
        return session['demo']
    return 'swap' if any(a.get('goals') for a in session.get('agents') or []) else 'explore'


def footprints(run_dir, agents):
    out = {}
    for n in agents:
        try:
            out[n] = urdf_footprint((Path(run_dir) / 'urdf' / ('%s.urdf' % n)).read_text())
        except OSError as e:
            raise EvalError('%s: %s (the spawned URDF gives the footprint)' % (n, e))
    return out


def world_file(world):
    return WORLDS_DIR / ('%s.world' % world)


def read_rtf_log(path, start, end):
    return read_new(str(path), start, end) if path else []


def isolation_problems():
    """sim.py check_isolation(): the probe opens a ROS session, and no sim tool does that while the
    sim's isolation from the host and the fleet is broken (status, arm, start and viz fail closed)."""
    spec = importlib.util.spec_from_file_location('sim', os.path.join(HERE, 'sim.py'))
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    return sim.check_isolation()


def probe_command(duration, agents, maps):
    budget = int(math.ceil(duration + 25 + goal_relay.LEAVE_GAP_S * (9 * len(agents) + 1)))
    inner = ('%s && exec timeout -k 5 %d python3 /rover_sim/scripts/demo_eval.py record --duration %g '
             '--spawn %s/spawn.yaml --agents %s%s' % (PROBE_SETUP, budget, duration, RUN_DIR_IN, ','.join(agents),
                                                     ' --maps' if maps else ''))
    return inner, budget


def run(args):
    run_dir = Path(args.run_dir or os.environ.get('ROVER_SIM_RUN') or RUN_DIR_DEFAULT)
    session, spawn = load_run(run_dir)
    mode = pick_mode(args.mode, session)
    agents = [n for n in args.agents.split(',') if n] or sorted(a['name'] for a in spawn['agents'])
    log_dir = Path(args.log_dir) if args.log_dir else (LOG_ROOT / 'latest').resolve()
    if not log_dir.is_dir():
        raise EvalError('no pane-log dir %s' % log_dir)
    fps = footprints(run_dir, agents)
    sim_container(args.container)
    problems = isolation_problems()
    if problems:
        raise EvalError('the sim is not isolated from this host: no ROS session is opened:\n  - '
                        + '\n  - '.join(problems))
    rtf_path = run_dir / RTF_LOG
    start, rtf0 = log_sizes(log_dir), rtf_path.stat().st_size if rtf_path.exists() else 0
    inner, budget = probe_command(args.duration, agents, mode == 'explore')
    print('demo_eval: %s demo of %s, %s; recording %.0f s in %s' % (mode, session.get('name'), ', '.join(agents),
                                                                   args.duration, args.container), flush=True)
    try:
        p = subprocess.run(['docker', 'exec', args.container, 'bash', '-c', inner], stdout=subprocess.PIPE,
                           universal_newlines=True, timeout=budget + 15)
    except subprocess.TimeoutExpired:
        raise EvalError('the probe did not return within %d s' % (budget + 15))
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith(MARK)), None)
    if line is None:
        raise EvalError('the probe gave no recording (exit %d): %s' % (p.returncode, p.stdout[-500:]))
    rec = json.loads(line[len(MARK):])
    end, rtf1 = log_sizes(log_dir), rtf_path.stat().st_size if rtf_path.exists() else 0
    rec.update(mode=mode, session=session, spawn=spawn, footprints=fps, world=spawn.get('world') or session['world'],
               host={'log_dir': str(log_dir), 'log_start': start, 'log_end': end, 'run_dir': str(run_dir),
                     'rtf_log': [rtf0, rtf1]})
    return rec


def grade_recording(rec, args):
    host = rec['host']
    logs = scan_logs(host['log_dir'], host['log_start'], host['log_end'], args.allow_error or ())
    logs['rtf_violations'] = read_rtf_log(Path(host['run_dir']) / RTF_LOG, *host['rtf_log'])
    wf = world_file(rec['world'])
    try:
        text = wf.read_text()
    except OSError as e:
        raise EvalError('world %s: %s' % (rec['world'], e))
    boxes = world_boxes(text)
    report = grade(rec, logs, boxes, args)
    report['world'] = {'file': str(wf.relative_to(REPO)), 'sha256': hashlib.sha256(text.encode()).hexdigest()[:16],
                       'boxes': len(boxes)}
    report['footprints'] = rec['footprints']
    report['log_dir'] = host['log_dir']
    return report


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')
    tmp.replace(path)


def build_parser():
    doc_head, doc_rest = __doc__.split('\n', 1)
    ap = argparse.ArgumentParser(description=doc_head, epilog=doc_rest,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', nargs='?', default='auto', choices=('auto', 'swap', 'explore', 'grade', 'record'))
    ap.add_argument('--duration', type=float, default=300.0, help='s to record (default 300)')
    ap.add_argument('--agents', default='', help='comma-separated; default every agent of the run')
    ap.add_argument('--run-dir', default=None, help='default $ROVER_SIM_RUN or %s' % RUN_DIR_DEFAULT)
    ap.add_argument('--log-dir', default=None, help='default %s/latest' % LOG_ROOT)
    ap.add_argument('--container', default=PROBE_CONTAINER)
    ap.add_argument('--out', default=None, help='report JSON (default <log dir>/demo_eval/<mode>_<time>.json)')
    ap.add_argument('--recording', default=None, help='grade: a recording this tool saved')
    ap.add_argument('--min-legs', type=int, default=2, help='swap: reached legs per agent (default 2)')
    ap.add_argument('--goal-radius', type=float, default=0.5, help="m, mighty_node's (for the truth check)")
    ap.add_argument('--contact', type=float, default=0.02, help='m: closer is contact (default 0.02)')
    ap.add_argument('--coverage-growth', type=float, default=2.0, help='explore: m^2 (default 2)')
    ap.add_argument('--rtf-min', type=float, default=0.95)
    ap.add_argument('--odom-max', type=float, default=0.5, help='m, DLIO vs truth xy (default 0.5)')
    ap.add_argument('--allow-error', action='append', metavar='REGEX',
                    help="an ERROR line ('<logger>: <message>') this run expects (repeatable)")
    ap.add_argument('--spawn', default=RUN_DIR_IN + '/spawn.yaml', help='record: the run\'s spawn.yaml')
    ap.add_argument('--maps', action='store_true', help='record: also persistent_map/grid (coverage)')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.what == 'record':
            return record(args)
        if args.what == 'grade':
            if not args.recording:
                raise EvalError('grade needs --recording')
            rec = json.loads(Path(args.recording).read_text())
            name = Path(args.recording).name
            out = Path(args.out) if args.out else Path(args.recording).with_name(
                name.replace('_recording', '_report') if '_recording' in name else name + '.report.json')
        else:
            args.mode = args.what
            rec = run(args)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            base = Path(rec['host']['log_dir']) / 'demo_eval' / ('%s_%s' % (rec['mode'], stamp))
            write_json(str(base) + '_recording.json', rec)
            out = Path(args.out) if args.out else Path(str(base) + '_report.json')
        report = grade_recording(rec, args)
    except (EvalError, OSError, ValueError, KeyError) as e:
        print('demo_eval: %s' % e, file=sys.stderr)
        return 2
    except goal_relay.drive_pattern.PlanError as e:
        print('demo_eval: %s' % e, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('demo_eval: interrupted; a probe already started finishes its --duration in %s and leaves the '
              'graph by itself' % args.container, file=sys.stderr)
        return 130
    write_json(out, report)
    for line in report_lines(report):
        print(line)
    print('report: %s' % out)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
