#!/usr/bin/env python3
"""Grade one agent's DLIO odometry against the sim's ground truth.

  odom_eval.py --ns JR81 [--spawn /rover_sim/run/spawn.yaml] [--duration s] [--out f.json]
               [--target 0.10] [--save-raw f.json] [--drive <drive_pattern.py args>]
  odom_eval.py --offline raw.json [--out f.json]        (re-grade a --save-raw recording)

Records /<ns>/dlio/odom_node/odom (frame <ns>/odom, child <ns>/base_link) and
/<ns>/ground_truth/odom_wall (frame world, child <ns>/chassis) until --duration s, Ctrl-C,
or, with --drive, until that drive ends plus --tail s. --drive takes everything after it as
drive_pattern.py arguments (--ns and --spawn are shared) and drives from this same node, one
zenoh session; it waits for DLIO to run first (its odom carries the seed pose, unchanged,
until the IMU calibration ends), as the agent has to be still for that. The agent must be in
AUTO already (scripts/sim.py arm <ns>, or a session with auto_arm).

What it compares, in the world frame:
  DLIO       T_world_map * T_map_odom * T_odom_base_link(DLIO). T_world_map is the spawn pose
             (spawn.yaml; sim_tf.py publishes the same), T_map_odom map_odom_tf.py's
             identity; /tf is read once to check both, and a non-identity map->odom seen
             there is used, as consumers would.
  truth      T_world_chassis(odom_wall) * (0, 0, lidar_height): base_link is the lidar origin,
             level and right above the chassis root.
Grading starts at the first DLIO sample that DLIO computed (dlio_live_ns): the seed pose it
holds through its IMU calibration equals the truth by construction. Each sample is matched to
truth interpolated at its own stamp (both are wall-mapped by the same ClockMapper) and graded
at offset 0, as consumers see it. The stamp offset that minimises the error is reported as a
diagnostic only: truth(t + offset) against DLIO(t), so a negative offset means DLIO's pose
stamped t is where the agent was |offset| earlier.

Reports ATE RMSE/max (3D, xy, z), yaw and rotation error, end-point drift (m and % of the
truth path length), the SE(2)-aligned xy ATE (what is left once a constant offset is
removed), sample counts and rates, and prints one verdict line: PASS when the 3D ATE RMSE is
under --target (the plan's 10 cm). Exit 0 PASS, 1 FAIL, 2 no data or a setup error, 3 when
the --drive pattern did not complete (the run is still graded).

The math (SE(3) compose, interpolation, ATE, the offset search) is pure Python and imports
without rclpy; tests/test_odom_eval.py covers it.
"""

import argparse
import bisect
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import drive_pattern  # noqa: E402  pure at import: spawn.yaml parsing and the DriveRunner

NS_PER_S = 1000000000
IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------- SE(3)
# A pose is ((x, y, z), (qx, qy, qz, qw)).

def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def qconj(q):
    return (-q[0], -q[1], -q[2], q[3])


def qnormalize(q):
    n = math.sqrt(sum(c * c for c in q))
    if not n > 0:
        raise ValueError('zero quaternion')
    return tuple(c / n for c in q)


def qrot(q, v):
    """v rotated by unit quaternion q."""
    x, y, z, w = q
    # v + 2w (u x v) + 2 u x (u x v), u = (x, y, z)
    cx, cy, cz = y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]
    return (v[0] + 2.0 * (w * cx + y * cz - z * cy),
            v[1] + 2.0 * (w * cy + z * cx - x * cz),
            v[2] + 2.0 * (w * cz + x * cy - y * cx))


def compose(a, b):
    """a * b: b expressed in a's parent frame."""
    r = qrot(a[1], b[0])
    return ((a[0][0] + r[0], a[0][1] + r[1], a[0][2] + r[2]), qnormalize(qmul(a[1], b[1])))


def inverse(a):
    qi = qconj(a[1])
    t = qrot(qi, a[0])
    return ((-t[0], -t[1], -t[2]), qi)


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_angle(q):
    """Rotation angle of q, in [0, pi]."""
    return 2.0 * math.atan2(math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2), abs(q[3]))


def slerp(q0, q1, f):
    d = sum(a * b for a, b in zip(q0, q1))
    if d < 0.0:
        q1, d = tuple(-c for c in q1), -d
    if d > 0.9995:                                   # near-parallel: nlerp is exact enough
        return qnormalize(tuple(a + f * (b - a) for a, b in zip(q0, q1)))
    th = math.acos(min(1.0, d))
    s0, s1 = math.sin((1.0 - f) * th) / math.sin(th), math.sin(f * th) / math.sin(th)
    return tuple(s0 * a + s1 * b for a, b in zip(q0, q1))


def wrap(a):
    return math.remainder(a, 2.0 * math.pi)


def is_identity(p, tol=1e-6):
    return all(abs(c) <= tol for c in p[0]) and quat_angle(p[1]) <= tol


# ---------------------------------------------------------------------------- trajectories

class Track:
    """Poses sorted by stamp (float seconds), with interpolation."""

    def __init__(self, samples):
        rows = sorted(samples, key=lambda r: r[0])
        self.t = [r[0] for r in rows]
        self.p = [r[1] for r in rows]

    def __len__(self):
        return len(self.t)

    def bracket(self, t, max_gap):
        """(i, f) with t = t[i] + f (t[i+1] - t[i]), or None outside the track or a gap."""
        n = len(self.t)
        if n == 0 or t < self.t[0] or t > self.t[-1]:
            return None
        i = bisect.bisect_right(self.t, t) - 1
        if i >= n - 1:
            return (n - 1, 0.0)
        span = self.t[i + 1] - self.t[i]
        if span > max_gap:
            return None
        return (i, 0.0 if span <= 0 else (t - self.t[i]) / span)

    def position_at(self, t, max_gap):
        b = self.bracket(t, max_gap)
        if b is None:
            return None
        i, f = b
        a = self.p[i][0]
        if f == 0.0:
            return a
        c = self.p[i + 1][0]
        return (a[0] + f * (c[0] - a[0]), a[1] + f * (c[1] - a[1]), a[2] + f * (c[2] - a[2]))

    def pose_at(self, t, max_gap):
        b = self.bracket(t, max_gap)
        if b is None:
            return None
        i, f = b
        if f == 0.0:
            return self.p[i]
        return (self.position_at(t, max_gap), slerp(self.p[i][1], self.p[i + 1][1], f))


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None


def path_length(points, min_step=0.01):
    """xy length of a polyline, counting a step only once it reaches min_step (so
    jitter at rest does not add up)."""
    total, last = 0.0, None
    for p in points:
        if last is None:
            last = p
            continue
        d = math.hypot(p[0] - last[0], p[1] - last[1])
        if d >= min_step:
            total += d
            last = p
    return total


def se2_align(est, ref):
    """(dx, dy, dyaw, rmse) of the rigid planar motion that best maps est xy onto ref xy."""
    n = len(est)
    if n < 2:
        return None
    ex, ey = sum(p[0] for p in est) / n, sum(p[1] for p in est) / n
    rx, ry = sum(p[0] for p in ref) / n, sum(p[1] for p in ref) / n
    sxx = sxy = 0.0
    for a, b in zip(est, ref):
        ax, ay, bx, by = a[0] - ex, a[1] - ey, b[0] - rx, b[1] - ry
        sxx += ax * bx + ay * by
        sxy += ax * by - ay * bx
    th = math.atan2(sxy, sxx)
    c, s = math.cos(th), math.sin(th)
    dx, dy = rx - (c * ex - s * ey), ry - (s * ex + c * ey)
    err = [math.hypot(c * a[0] - s * a[1] + dx - b[0], s * a[0] + c * a[1] + dy - b[1]) for a, b in zip(est, ref)]
    return dx, dy, th, rms(err)


def ate_at(dlio, truth, offset, max_gap):
    """3D position RMSE of DLIO samples [(t, (x, y, z))] against truth.position_at(t + offset)."""
    sq, n = 0.0, 0
    for t, p in dlio:
        q = truth.position_at(t + offset, max_gap)
        if q is not None:
            sq += (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
            n += 1
    return (math.sqrt(sq / n), n) if n else (None, 0)


def best_offset(dlio, truth, max_gap, search=0.3, coarse=0.01, fine=0.001, max_samples=1000):
    """(offset s, rmse) minimising ate_at over [-search, search]: a coarse grid, then a fine
    one around its best, on at most max_samples evenly spread samples."""
    if not dlio:
        return None, None
    sub = dlio[::max(1, len(dlio) // max_samples)]

    def grid(lo, hi, step):
        best = (None, None)
        k = 0
        while lo + k * step <= hi + 1e-12:
            d = lo + k * step
            e, n = ate_at(sub, truth, d, max_gap)
            if e is not None and n >= len(sub) // 2 and (best[1] is None or e < best[1]):
                best = (d, e)
            k += 1
        return best

    d, _ = grid(-search, search, coarse)
    if d is None:
        return None, None
    d, _ = grid(max(-search, d - coarse), min(search, d + coarse), fine)
    return round(d, 6), ate_at(dlio, truth, d, max_gap)[0]


# ---------------------------------------------------------------------------- evaluation

def to_pose(row):
    """[stamp_ns, x, y, z, qx, qy, qz, qw, ...] -> (t s, pose)."""
    return row[0] / NS_PER_S, ((row[1], row[2], row[3]), qnormalize((row[4], row[5], row[6], row[7])))


def rate(stamps):
    if len(stamps) < 2 or stamps[-1] <= stamps[0]:
        return None
    return (len(stamps) - 1) / (stamps[-1] - stamps[0])


def stat(xs):
    xs = [abs(x) for x in xs]
    return {'rmse': rms(xs), 'max': max(xs) if xs else None}


def evaluate(raw, target=0.10, max_gap=0.1, offset_search=0.3):
    """The report for a recording (the --save-raw format): {ns, world_map, map_odom,
    lidar_height, truth: [[stamp_ns, x, y, z, qx, qy, qz, qw]], dlio: [[stamp_ns, x, y, z,
    qx, qy, qz, qw, recv_ns]], ...}."""
    ns = raw.get('ns', '')
    h = float(raw['lidar_height'])
    world_map = (tuple(raw['world_map'][0]), qnormalize(tuple(raw['world_map'][1])))
    map_odom = (tuple(raw['map_odom'][0]), qnormalize(tuple(raw['map_odom'][1]))) if raw.get('map_odom') else IDENTITY
    world_odom = compose(world_map, map_odom)
    lever = ((0.0, 0.0, h), IDENTITY[1])

    counts = {'dlio_received': len(raw.get('dlio', [])), 'dlio_zero_stamp': 0, 'dlio_pre_seed': 0,
              'dlio_pre_live': 0, 'dlio_bad_quaternion': 0, 'dlio_unmatched': 0, 'dlio_matched': 0,
              'dlio_duplicate_stamps': 0, 'truth_received': len(raw.get('truth', [])), 'truth_dropped': 0}
    # None: DLIO never ran while recording. A recording made before the key existed is graded
    # from the seed pose on.
    gated, live_ns = 'dlio_live_ns' in raw, raw.get('dlio_live_ns')
    tr = []
    for r in raw.get('truth', []):
        try:
            if r[0] > 0:
                t, p = to_pose(r)
                tr.append((t, compose(p, lever)))
                continue
        except ValueError:
            pass
        counts['truth_dropped'] += 1
    truth = Track(tr)

    dl, last = [], None
    for r in sorted(raw.get('dlio', []), key=lambda r: r[0]):
        if r[0] <= 0:                     # published before DLIO had an IMU sample
            counts['dlio_zero_stamp'] += 1
            continue
        try:
            t, p = to_pose(r)
        except ValueError:                # a zero quaternion
            counts['dlio_bad_quaternion'] += 1
            continue
        if is_identity(p, 0.0):           # before the seed pose arrived
            counts['dlio_pre_seed'] += 1
            continue
        if gated and (live_ns is None or r[0] < live_ns):     # the seed pose, held through the IMU calibration
            counts['dlio_pre_live'] += 1
            continue
        if r[0] == last:
            counts['dlio_duplicate_stamps'] += 1
        last = r[0]
        dl.append((t, compose(world_odom, p)))
    dl_stamps = [t for t, _ in dl]

    report = {'ns': ns, 'target_m': target, 'lidar_height_m': h, 'samples': counts,
              'frames': raw.get('frames', {}), 'transforms': raw.get('transforms', {}),
              'rates_hz': {'dlio_stamps': rate(dl_stamps),
                           'dlio_unique_stamps': rate(sorted(set(dl_stamps))),
                           'dlio_receipt': rate(sorted(r[8] / NS_PER_S for r in raw.get('dlio', []) if len(r) > 8)),
                           'truth_stamps': rate(truth.t)}}

    pos_e, z_e, xy_e, yaw_e, rot_e, matched_est, matched_ref, ts = [], [], [], [], [], [], [], []
    for t, p in dl:
        q = truth.pose_at(t, max_gap)
        if q is None:
            counts['dlio_unmatched'] += 1
            continue
        dx, dy, dz = (p[0][k] - q[0][k] for k in range(3))
        xy_e.append(math.hypot(dx, dy))
        z_e.append(dz)
        pos_e.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        yaw_e.append(wrap(quat_yaw(p[1]) - quat_yaw(q[1])))
        rot_e.append(quat_angle(qmul(qconj(q[1]), p[1])))
        matched_est.append(p[0])
        matched_ref.append(q[0])
        ts.append(t)
    counts['dlio_matched'] = len(ts)
    if not ts:
        report['verdict'] = 'NO DATA'
        report['line'] = ('odom_eval %s: NO DATA: %d DLIO samples (%d before DLIO ran), %d truth samples, none '
                          'matched' % (ns, counts['dlio_received'], counts['dlio_pre_live'], counts['truth_received']))
        return report

    i0, i1 = bisect.bisect_left(truth.t, ts[0]), bisect.bisect_right(truth.t, ts[-1])
    length = path_length([p[0] for p in truth.p[i0:i1]])
    ate = stat(pos_e)
    report.update({
        'span_s': ts[-1] - ts[0],
        'path_length_m': length,
        'ate_m': {'rmse': ate['rmse'], 'max': ate['max'], 'xy_rmse': stat(xy_e)['rmse'], 'xy_max': stat(xy_e)['max'],
                  'z_rmse': stat(z_e)['rmse'], 'z_max': stat(z_e)['max'],
                  'z_mean': sum(z_e) / len(z_e)},
        'yaw_deg': {'rmse': math.degrees(rms(yaw_e)), 'max': math.degrees(max(abs(e) for e in yaw_e)),
                    'mean': math.degrees(sum(yaw_e) / len(yaw_e))},
        'rotation_deg': {'rmse': math.degrees(rms(rot_e)), 'max': math.degrees(max(rot_e))},
        'start': {'xy_m': xy_e[0], 'z_m': z_e[0], 'yaw_deg': math.degrees(yaw_e[0])},
        'end': {'xy_m': xy_e[-1], 'z_m': z_e[-1], 'yaw_deg': math.degrees(yaw_e[-1]),
                'drift_pct': 100.0 * xy_e[-1] / length if length > 0 else None},
    })
    al = se2_align(matched_est, matched_ref)
    if al:
        report['se2_aligned'] = {'xy_rmse_m': al[3], 'dx_m': al[0], 'dy_m': al[1], 'dyaw_deg': math.degrees(al[2])}
    d = e = None
    if length >= 1.0:                     # at rest every offset fits equally well
        d, e = best_offset([(t, p[0]) for t, p in dl], truth, max_gap, offset_search)
    report['best_offset'] = {'offset_s': d, 'ate_rmse_m': e, 'searched_s': offset_search,
                             'note': 'truth(t + offset) vs DLIO(t); < 0: DLIO lags its stamps'
                                     + ('' if length >= 1.0 else '; not searched: the agent barely moved')}
    ok = ate['rmse'] < target
    report['verdict'] = 'PASS' if ok else 'FAIL'
    report['line'] = (
        'odom_eval %s: %s ATE %.3f m %s %.2f m target (xy %.3f, z %.3f; max %.3f) over %.1f m in %.0f s; '
        'end drift %.3f m (%s); yaw rmse %.2f deg (max %.2f); %d samples, DLIO %.1f Hz; best offset %s'
        % (ns, report['verdict'], ate['rmse'], '<' if ok else '>=', target, report['ate_m']['xy_rmse'],
           report['ate_m']['z_rmse'], ate['max'], length, report['span_s'], xy_e[-1],
           '%.2f %%' % report['end']['drift_pct'] if report['end']['drift_pct'] is not None else 'no path',
           report['yaw_deg']['rmse'], report['yaw_deg']['max'], len(ts),
           report['rates_hz']['dlio_stamps'] or 0.0,
           'n/a' if d is None else '%+.0f ms (ATE %.3f)' % (d * 1000.0, e)))
    return report


# ---------------------------------------------------------------------------- spawn

def spawn_pose(path, ns):
    """(T_world_map, lidar_height) of ns from spawn.yaml (sim.py spawn_table)."""
    pose, _, h = drive_pattern.load_spawn_entry(path, ns)
    return ((pose.x, pose.y, 0.0), yaw_quat(pose.yaw)), h


def pose_list(p):
    return [list(p[0]), list(p[1])]


def pose_delta(a, b):
    """(translation m, rotation deg) between two poses."""
    d = compose(inverse(a), b)
    return math.sqrt(sum(c * c for c in d[0])), math.degrees(quat_angle(d[1]))


# ---------------------------------------------------------------------------- ROS

class LiveGate:
    """When DLIO starts running, seen from its odom: from node start it publishes at 100 Hz,
    stamped 0 until the first IMU sample, then an all-zero pose until the seed pose arrives,
    then the seed with zero twist through the IMU calibration. Its state only changes once it
    runs. (A seed at the origin with identity rotation would look like no seed.)"""

    def __init__(self):
        self.seed = None
        self.since = None                # monotonic time it was first seen running
        self.since_ns = None             # the stamp of that first sample DLIO computed

    def feed(self, row, twist, now):
        """row [stamp_ns, x, y, z, qx, qy, qz, qw], twist the 6 twist components."""
        if self.since is not None or row[0] <= 0:
            return
        pose = tuple(row[1:8])
        if pose == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0):
            return
        if self.seed is None:
            self.seed = pose
        if any(abs(v) > 1e-9 for v in twist) or max(abs(a - b) for a, b in zip(pose, self.seed)) > 1e-7:
            self.since, self.since_ns = now, row[0]


class Recorder:
    """DLIO odom, ground truth and, once, the two /tf transforms the composition uses."""

    TF_WAIT = 10.0

    def __init__(self, node, ns, dlio_topic, truth_topic):
        from nav_msgs.msg import Odometry
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from tf2_msgs.msg import TFMessage

        self.node, self.ns = node, ns
        self.truth, self.dlio = [], []
        self.frames = {}
        self.tf = {}                                 # (parent, child) -> pose
        self.live = LiveGate()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=500)
        node.create_subscription(Odometry, dlio_topic, self._on_dlio, qos)
        node.create_subscription(Odometry, truth_topic, self._on_truth, qos)
        self.want = {('world', ns + '/map'), (ns + '/map', ns + '/odom')}
        self.tf_sub = node.create_subscription(TFMessage, '/tf', self._on_tf, QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=100))
        self.t0 = time.monotonic()

    @staticmethod
    def row(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        return [m.header.stamp.sec * NS_PER_S + m.header.stamp.nanosec, p.x, p.y, p.z, q.x, q.y, q.z, q.w,
                time.time_ns()]

    def _on_truth(self, m):
        self.truth.append(self.row(m)[:8])
        self.frames.setdefault('truth', [m.header.frame_id, m.child_frame_id])

    def _on_dlio(self, m):
        r = self.row(m)
        self.dlio.append(r)
        self.frames.setdefault('dlio', [m.header.frame_id, m.child_frame_id])
        if self.live.since is None:
            tw = m.twist.twist
            self.live.feed(r, (tw.linear.x, tw.linear.y, tw.linear.z, tw.angular.x, tw.angular.y, tw.angular.z),
                           time.monotonic())

    def _on_tf(self, m):
        for t in m.transforms:
            key = (t.header.frame_id, t.child_frame_id)
            if key in self.want and key not in self.tf:
                tr, q = t.transform.translation, t.transform.rotation
                self.tf[key] = ((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))
        self.poll_tf()

    def poll_tf(self):
        if self.tf_sub is not None and (self.want <= set(self.tf) or time.monotonic() - self.t0 > self.TF_WAIT):
            self.node.destroy_subscription(self.tf_sub)     # /tf is busy; one look is enough
            self.tf_sub = None


def check_frames(frames, ns):
    want = {'dlio': [ns + '/odom', ns + '/base_link'], 'truth': ['world', ns + '/chassis']}
    return {k: {'seen': frames.get(k), 'expected': v, 'ok': frames.get(k) == v} for k, v in want.items()}


def split_argv(argv):
    if '--drive' in argv:
        i = argv.index('--drive')
        return argv[:i], argv[i + 1:]
    return argv, None


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ns', help='agent namespace, e.g. JR81')
    ap.add_argument('--spawn', default='/rover_sim/run/spawn.yaml')
    ap.add_argument('--duration', type=float, default=None, help='s to record (with --drive: at most)')
    ap.add_argument('--out', default=None, help='write the report here (JSON)')
    ap.add_argument('--save-raw', default=None, help='write the recording here (JSON, for --offline)')
    ap.add_argument('--offline', default=None, help='grade a --save-raw recording; no ROS')
    ap.add_argument('--target', type=float, default=0.10, help='ATE RMSE pass mark, m')
    ap.add_argument('--lidar-height', type=float, default=None, help="m; default spawn.yaml's")
    ap.add_argument('--max-gap', type=float, default=0.1, help='s between truth samples to interpolate across')
    ap.add_argument('--offset-search', type=float, default=0.3, help='+- s for the best-offset diagnostic')
    ap.add_argument('--tail', type=float, default=3.0, help='s to keep recording after the drive ends')
    ap.add_argument('--dlio-wait', type=float, default=90.0, help='s to wait for DLIO before driving')
    ap.add_argument('--settle', type=float, default=2.0, help='s DLIO must have run before driving')
    ap.add_argument('--dlio-topic', default=None, help='default /<ns>/dlio/odom_node/odom')
    ap.add_argument('--truth-topic', default=None, help='default /<ns>/ground_truth/odom_wall')
    return ap


def write_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=1)
        f.write('\n')
    os.replace(tmp, path)


def finish(report, args, drive=None):
    if drive is not None:
        report['drive'] = drive
        if drive.get('code') != 0:
            report['line'] += ('; DRIVE DID NOT COMPLETE (%s)' % {-1: 'never started', 2: 'setup error',
                                                                  3: 'timeout'}.get(drive.get('code'), drive.get('code')))
    print(report['line'], flush=True)
    if args.out:
        write_json(args.out, report)
    if report['verdict'] == 'NO DATA':
        return 2
    if drive is not None and drive.get('code') != 0:
        return 3
    return 0 if report['verdict'] == 'PASS' else 1


def record(args, drive_argv):
    ns = args.ns
    world_map, h = spawn_pose(args.spawn, ns)
    if args.lidar_height is not None:
        h = args.lidar_height
    dargs = None
    if drive_argv is not None:
        dap = argparse.ArgumentParser(prog='odom_eval.py ... --drive')
        drive_pattern.add_drive_args(dap, with_ns=False)
        dargs = dap.parse_args(drive_pattern.join_negative_values(drive_argv))
        drive_pattern.require_sim()            # before rclpy opens a session (DriveRunner checks again)
        drive_pattern.need_size(dargs.pattern, dargs.size)

    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = Node('odom_eval_%s' % ns)
    log = node.get_logger()
    rec = Recorder(node, ns, args.dlio_topic or '/%s/dlio/odom_node/odom' % ns,
                   args.truth_topic or '/%s/ground_truth/odom_wall' % ns)
    runner, drive, code = None, None, None
    t0 = time.monotonic()
    end = t0 + args.duration if args.duration else None
    try:
        if dargs is not None:
            runner = drive_pattern.DriveRunner(node, dargs, ns, args.spawn)
            runner.armed = False
            log.info('waiting for DLIO to run (calibrated, state changing) before driving')
        stop_at = None
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            rec.poll_tf()
            if end is not None and now >= end:
                break
            if runner is not None and not runner.armed:
                if rec.live.since is not None and now - rec.live.since >= args.settle:
                    log.info('DLIO running for %.1f s: driving' % (now - rec.live.since))
                    runner.arm()
                elif now - t0 > args.dlio_wait:
                    log.error('DLIO odom did not start changing within %.0f s: not driving' % args.dlio_wait)
                    code = 2
                    break
            if runner is not None and runner.finished is not None:
                stop_at = stop_at or now + args.tail
                if now >= stop_at:
                    break
    except drive_pattern.PlanError as e:
        log.error(str(e))
        code = 2
    except KeyboardInterrupt:
        if runner is not None and runner.armed:
            runner.stop_now()
    finally:
        if runner is not None:
            drive = dict(runner.stats, code=runner.finished if runner.finished is not None else -1)
        node.destroy_node()
        rclpy.try_shutdown()

    tf = {k: v for k, v in rec.tf.items()}
    map_odom = tf.get((ns + '/map', ns + '/odom'))
    world_map_tf = tf.get(('world', ns + '/map'))
    transforms = {
        'world_map': {'source': 'spawn.yaml', 'pose': pose_list(world_map),
                      'tf': pose_list(world_map_tf) if world_map_tf else None,
                      'tf_vs_spawn_m_deg': list(pose_delta(world_map, world_map_tf)) if world_map_tf else None},
        'map_odom': {'source': '/tf' if map_odom else 'assumed identity (not seen on /tf)',
                     'pose': pose_list(map_odom or IDENTITY),
                     'identity': is_identity(map_odom) if map_odom else None}}
    if world_map_tf and max(pose_delta(world_map, world_map_tf)) > 1e-3:
        print('odom_eval: warning: /tf world->%s/map differs from spawn.yaml by %.4f m, %.3f deg'
              % ((ns,) + pose_delta(world_map, world_map_tf)), file=sys.stderr)
    if map_odom and not is_identity(map_odom):
        print('odom_eval: warning: /tf %s/map->%s/odom is not identity; using it' % (ns, ns), file=sys.stderr)
    raw = {'ns': ns, 'lidar_height': h, 'world_map': pose_list(world_map),
           'map_odom': pose_list(map_odom) if map_odom else None, 'dlio_live_ns': rec.live.since_ns,
           'frames': check_frames(rec.frames, ns), 'transforms': transforms,
           'recorded_s': time.monotonic() - t0, 'truth': rec.truth, 'dlio': rec.dlio}
    for k, v in raw['frames'].items():
        if v['seen'] is not None and not v['ok']:
            print('odom_eval: warning: %s frames %s, expected %s' % (k, v['seen'], v['expected']), file=sys.stderr)
    return raw, drive, code


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    own, drive_argv = split_argv(argv)
    args = build_parser().parse_args(own)
    code = None
    if args.offline:
        with open(args.offline) as f:
            raw = json.load(f)
    else:
        if not args.ns:
            build_parser().error('--ns is required (or --offline)')
        try:
            raw, drive, code = record(args, drive_argv)
        except drive_pattern.PlanError as e:      # spawn.yaml, arguments: nothing recorded
            print('odom_eval: %s' % e, file=sys.stderr)
            return 2
        if drive is not None:
            raw['drive'] = drive
        if args.save_raw:
            write_json(args.save_raw, raw)
    report = evaluate(raw, args.target, args.max_gap, args.offset_search)
    rc = finish(report, args, raw.get('drive'))
    return 2 if code == 2 else rc


if __name__ == '__main__':
    sys.exit(main())
