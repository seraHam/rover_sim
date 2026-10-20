#!/usr/bin/env python3
"""odometry: truth (NOT PARITY): one agent's DLIO outputs, made from Gazebo's ground truth.

  truth_odom.py <ns> [--spawn /rover_sim/run/spawn.yaml] [--dlio-cfg DIR] [--check]

scripts/sim.py runs it in the agent's DLIO window (sim-dlio-<class>) in place of DLIO's pane,
for an agent whose session gives it `odometry: truth`: DLIO's estimate is replaced by the truth,
and nothing else changes. Its purpose is a control: what is left in the maps and the planner with
it is not DLIO's drift. It publishes what DLIO publishes, under the same names, frames and stamps
(direct_lidar_inertial_odometry src/dlio/odom.cc publishPose, publishToROS, publishCloud):
  dlio/odom_node/odom      nav_msgs/Odometry <ns>/odom -> <ns>/base_link, one per ground-truth
                           sample (100 Hz); twist linear in odom axes, angular in base_link axes
  dlio/odom_node/pose      geometry_msgs/PoseStamped in <ns>/odom, with each odom
  /tf                      <ns>/odom -> <ns>/base_link with each odom; <ns>/base_link -> <ns>/imu
                           (cfg baselink2imu) with each cloud
  dlio/odom_node/pointcloud/deskewed  each lidar scan with DLIO's crop box removed, in <ns>/odom,
                           full resolution (DLIO's map/dense/filtered is false), DLIO's 32-byte
                           point layout, the scan's stamp
  dlio/session             std_msgs/String {"id", "ns", "start"} (dlio_ws dlio_session.py): from
                           the start, then 1 Hz, with an id of its own, so every mapper resets as
                           on a DLIO restart, and the paused arm's session gate passes
from
  ground_truth/odom_wall   the chassis pose in world, wall-stamped (livox_sim_bridge.py)
  livox/lidar              the scan in <ns>/lidar, wall-stamped by the same clock mapping
  <run>/spawn.yaml         the spawn pose (<ns>/map = <ns>/odom = it, at z 0, as sim_tf.py and
                           map_odom_tf.py publish) and the lidar height: base_link is the lidar
                           origin, right above the chassis root (odom_eval.py's truth)
  DLIO's cfg in the image  baselink2lidar, baselink2imu, the crop box half-extent, the IMU
                           calibration time and the frame names, from the files DLIO itself loads
                           (cfg/dlio.yaml, then cfg/params.yaml; the later wins), so each class
                           gets its own reference robot's values (JR: JR02, RR: RR08)
A scan is placed at the ground-truth pose interpolated at its stamp: Gazebo casts a scan at one
instant, so the deskewed cloud is T_odom_base(truth) * baselink2lidar * p, DLIO's own formula with
perfect odometry. The timeline follows DLIO's: nothing but the session while the world is paused
(no ground truth flows then; the mapping panes' odom -> base_link gate degrades as with DLIO),
odometry from the first ground-truth sample, clouds once the IMU calibration time has passed
after it (DLIO holds its clouds back that long).

--check validates the spawn entry and the cfg, prints what it read and exits 0 (the pane's gate
in scripts/sim.py, before the node marker). It refuses an agent whose spawn.yaml entry is not
odometry: truth (two odometry publishers on one name), and runs only on the sim's zenoh session
(drive_pattern.require_sim). Exit 0, 2 on a setup error.

Frames, the cfg, interpolation, the session payload and the gates are pure Python (no rclpy,
no numpy: the host runs their tests); only the cloud is numpy. tests/test_truth_odom.py.
"""

import argparse
import bisect
import json
import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import drive_pattern  # noqa: E402  spawn.yaml (spawn_entry, spawn_odometry), require_sim
import goal_relay  # noqa: E402  the paced leave
import odom_eval  # noqa: E402  SE(3): compose, inverse, qrot, slerp, yaw_quat

ODOMETRY_MODES = drive_pattern.ODOMETRY_MODES
DLIO_CFG_DIR = '/home/swarm/code/dlio_ws/install/share/direct_lidar_inertial_odometry/cfg'   # sim.py DLIO_CFG_IN
DLIO_CFG_FILES = ('dlio.yaml', 'params.yaml')     # dlio.launch.py's parameters, in its order
SPAWN_DEFAULT = '/rover_sim/run/spawn.yaml'
SESSION_TAG = 'truth'                # in the session id: <time_ns>-truth-<ns>
GT_KEEP_S = 2.0                      # ground truth kept for the scans' interpolation
GT_RESTART_S = 1.0                   # a stamp this much older than the newest: the bridge restarted
SCAN_WAIT_S = 0.15                   # a scan newer than the newest ground truth waits this long for it
SESSION_PERIOD_S = 1.0               # dlio_session.py's
REPORT_S = 30.0
LEAVE_GAP_S, LEAVE_PAUSE_S = goal_relay.LEAVE_GAP_S, goal_relay.LEAVE_PAUSE_S
IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)


class SetupError(ValueError):
    pass


# ---------------------------------------------------------------------------- DLIO's cfg

class DlioCfg:
    """What the stand-in mirrors of DLIO's parameters. Poses are ((x, y, z), (qx, qy, qz, qw))."""

    def __init__(self, base_lidar, base_imu, crop, hold_s, frames, sources):
        self.base_lidar, self.base_imu, self.crop, self.hold_s = base_lidar, base_imu, crop, hold_s
        self.frames, self.sources = frames, sources       # frames: odom, baselink, imu (DLIO's names)


def flatten(d, prefix=''):
    """A ROS 2 parameter block, nested or slash-spelled, as {'a/b/c': value}."""
    out = {}
    for k, v in d.items():
        key = '%s/%s' % (prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def cfg_params(texts):
    """The parameters DLIO's odom node gets from [(name, yaml text)] in its load order: each
    file's /** block (and any block for a dlio_odom_node), the later file winning per key."""
    import yaml
    params = {}
    for name, text in texts:
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise SetupError('%s: %s' % (name, e))
        if not isinstance(data, dict):
            raise SetupError('%s: not a ROS 2 parameter file' % name)
        for node, block in data.items():
            if str(node) != '/**' and not str(node).endswith('dlio_odom_node'):
                continue
            ros = block.get('ros__parameters') if isinstance(block, dict) else None
            if not isinstance(ros, dict):
                raise SetupError('%s: %s has no ros__parameters' % (name, node))
            params.update(flatten(ros))
    return params


def rot_to_quat(R):
    """Unit quaternion (x, y, z, w) of a row-major 3x3 rotation [[...], [...], [...]]."""
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = ((R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, 0.25 * s)
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        q = (0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s)
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        q = ((R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s)
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        q = ((R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s, (R[1][0] - R[0][1]) / s)
    return odom_eval.qnormalize(q)


def quat_to_rot(q):
    """Row-major 3x3 rotation of a unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def rpy_deg(q):
    """(roll, pitch, yaw) in degrees, for the --check line."""
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))


def _extrinsic(params, which, source):
    R, t = params.get('extrinsics/%s/R' % which), params.get('extrinsics/%s/t' % which)
    if not (isinstance(R, list) and len(R) == 9 and isinstance(t, list) and len(t) == 3):
        raise SetupError('%s: extrinsics/%s needs R (9 values) and t (3)' % (source, which))
    try:
        R = [[float(v) for v in R[r * 3:r * 3 + 3]] for r in range(3)]
        t = tuple(float(v) for v in t)
    except (TypeError, ValueError):
        raise SetupError('%s: extrinsics/%s: not numbers' % (source, which))
    # A rotation: orthonormal and proper, to the 7 digits the files carry.
    for i in range(3):
        for j in range(3):
            dot = sum(R[i][k] * R[j][k] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-5:
                raise SetupError('%s: extrinsics/%s/R is not a rotation' % (source, which))
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    if abs(det - 1.0) > 1e-5:
        raise SetupError('%s: extrinsics/%s/R is not a proper rotation (det %.6f)' % (source, which, det))
    return t, rot_to_quat(R)


def load_dlio_cfg(texts):
    """DlioCfg from [(name, yaml text)] in DLIO's load order (DLIO_CFG_FILES)."""
    params = cfg_params(texts)
    source = ' + '.join(n for n, _ in texts)
    frames = {}
    for k in ('odom', 'baselink', 'imu'):
        v = params.get('frames/' + k)
        if not isinstance(v, str) or not v:
            raise SetupError('%s: frames/%s missing' % (source, k))
        frames[k] = v
    crop = params.get('odom/preprocessing/cropBoxFilter/size')
    calib, calib_s = params.get('imu/calibration'), params.get('odom/imu/calibration/time')
    if not isinstance(crop, (int, float)) or isinstance(crop, bool) or not crop >= 0:
        raise SetupError('%s: odom/preprocessing/cropBoxFilter/size missing or negative' % source)
    if not isinstance(calib, bool):
        raise SetupError('%s: imu/calibration missing' % source)
    if calib and (not isinstance(calib_s, (int, float)) or isinstance(calib_s, bool) or not calib_s >= 0):
        raise SetupError('%s: odom/imu/calibration/time missing' % source)
    return DlioCfg(_extrinsic(params, 'baselink2lidar', source), _extrinsic(params, 'baselink2imu', source),
                   float(crop), float(calib_s) if calib else 0.0, frames, source)


def read_dlio_cfg(directory):
    texts = []
    for f in DLIO_CFG_FILES:
        path = os.path.join(directory, f)
        try:
            with open(path) as fh:
                texts.append((path, fh.read()))
        except OSError as e:
            raise SetupError('DLIO cfg: %s' % e)
    return load_dlio_cfg(texts)


# ---------------------------------------------------------------------------- spawn

def check_spawn(data, ns, source=SPAWN_DEFAULT):
    """(T_world_map, lidar_height) of ns, which spawn.yaml must mark odometry: truth."""
    try:
        pose, _, h = drive_pattern.spawn_entry(data, ns, source)
        mode = drive_pattern.spawn_odometry(data, ns, source)
    except drive_pattern.PlanError as e:
        raise SetupError(str(e))
    if mode != 'truth':
        raise SetupError('%s: %s has odometry: %s, so its DLIO publishes these names: refusing to run '
                         'beside it (scripts/sim.py up --odometry truth:%s)' % (source, ns, mode, ns))
    return ((pose.x, pose.y, 0.0), odom_eval.yaw_quat(pose.yaw)), h


# ---------------------------------------------------------------------------- frames

class Frames:
    """The chassis pose in world -> base_link in <ns>/odom, and the lidar in <ns>/odom."""

    def __init__(self, world_map, lidar_height, cfg):
        self.map_world = odom_eval.inverse(world_map)       # <ns>/odom = <ns>/map
        self.lever = (0.0, 0.0, lidar_height)
        self.chassis_base = (self.lever, IDENTITY_Q)
        self.cfg = cfg

    def base_in_odom(self, world_chassis):
        return odom_eval.compose(self.map_world, odom_eval.compose(world_chassis, self.chassis_base))

    def lidar_in_odom(self, odom_base):
        return odom_eval.compose(odom_base, self.cfg.base_lidar)

    def twist(self, odom_base, v_chassis, w_chassis):
        """DLIO's twist: linear in odom axes, angular in base_link axes, from odom_wall's (in chassis
        axes, livox_sim_bridge.odom_to_body). base_link shares the chassis' orientation, lever up."""
        w = w_chassis
        lever_v = (w[1] * self.lever[2] - w[2] * self.lever[1], w[2] * self.lever[0] - w[0] * self.lever[2],
                   w[0] * self.lever[1] - w[1] * self.lever[0])
        v_base = tuple(a + b for a, b in zip(v_chassis, lever_v))
        return odom_eval.qrot(odom_base[1], v_base), tuple(w)


# ---------------------------------------------------------------------------- ground truth

class GtBuffer:
    """Ground-truth chassis poses (wall s, pose), interpolated at a scan's stamp."""

    def __init__(self, keep_s=GT_KEEP_S, restart_s=GT_RESTART_S):
        self.t, self.poses, self.keep, self.restart = [], [], keep_s, restart_s
        self.resets = 0

    def add(self, t, pose):
        """False for a stamp not after the newest. One more than restart_s older starts over (a
        restarted bridge maps sim time to wall again)."""
        if self.t and t <= self.t[-1]:
            if t >= self.t[-1] - self.restart:
                return False
            self.t, self.poses = [], []
            self.resets += 1
        self.t.append(t)
        self.poses.append(pose)
        while self.t[0] < t - self.keep:
            self.t.pop(0)
            self.poses.pop(0)
        return True

    def newest(self):
        return (self.t[-1], self.poses[-1]) if self.t else None

    def at(self, t):
        """The pose at t, interpolated (linear, slerp); None outside the buffer."""
        if not self.t or t < self.t[0] or t > self.t[-1]:
            return None
        i = bisect.bisect_left(self.t, t)
        if self.t[i] == t:
            return self.poses[i]
        t0, t1 = self.t[i - 1], self.t[i]
        f = (t - t0) / (t1 - t0)
        (p0, q0), (p1, q1) = self.poses[i - 1], self.poses[i]
        return tuple(a + f * (b - a) for a, b in zip(p0, p1)), odom_eval.slerp(q0, q1, f)


def scan_pose(buf, t, waited_s, wait_s=SCAN_WAIT_S):
    """(what, pose) for a scan stamped t that has waited waited_s: 'pose' interpolated; 'wait' for
    ground truth that has not come yet; 'held' the newest pose, once it has waited wait_s (the scan
    is a little newer than the newest truth); 'drop' older than the buffer, or no truth at all."""
    newest = buf.newest()
    if newest is not None and t <= newest[0]:
        pose = buf.at(t)
        return ('pose', pose) if pose is not None else ('drop', None)
    if waited_s < wait_s:
        return 'wait', None
    return ('held', newest[1]) if newest is not None else ('drop', None)


class CloudGate:
    """DLIO publishes no cloud during its IMU calibration (cfg odom/imu/calibration/time), which
    starts with its first IMU sample: here, the first ground-truth stamp."""

    def __init__(self, hold_s):
        self.hold_s, self.first = hold_s, None

    def saw_truth(self, t):
        if self.first is None:
            self.first = t

    def open(self, t):
        return self.first is not None and t - self.first >= self.hold_s


# ---------------------------------------------------------------------------- names

def node_name(ns):
    """/<ns>/truth_odom_<ns>: tools/reference/jr02/allowed_deviations.yaml odometry.truth.sim_only."""
    return 'truth_odom_' + ns


def session_payload(ns, now_ns):
    """(json, id) as dlio_ws dlio_session.py publishes them; the id is one word, its first token
    the start in ns, as sim.py dlio_session_id and the mappers read it."""
    sid = '%d-%s-%s' % (now_ns, SESSION_TAG, ns)
    return json.dumps({'id': sid, 'ns': ns, 'start': now_ns / 1e9}, sort_keys=True), sid


def summary(ns, cfg, lidar_height):
    """The --check line: what the stand-in took from where."""
    (lt, lq), (it, iq) = cfg.base_lidar, cfg.base_imu
    return ('truth_odom %s: baselink2lidar rpy (%.4f, %.4f, %.4f) deg t (%g, %g, %g); baselink2imu rpy '
            '(%.4f, %.4f, %.4f) deg t (%g, %g, %g); crop %.2f m; IMU calibration %.1f s; frames %s/%s %s/%s '
            '%s/%s; lidar height %.2f m (from %s)'
            % ((ns,) + rpy_deg(lq) + lt + rpy_deg(iq) + it
               + (cfg.crop, cfg.hold_s, ns, cfg.frames['odom'], ns, cfg.frames['baselink'], ns, cfg.frames['imu'],
                  lidar_height, cfg.sources)))


# ---------------------------------------------------------------------------- the cloud (numpy)

def livox_dtype(np):
    """livox_sim_bridge.py's 26-byte point: x y z intensity f32, tag line u8, timestamp f64."""
    return np.dtype({'names': ['x', 'y', 'z', 'intensity', 'tag', 'line', 'timestamp'],
                     'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'],
                     'offsets': [0, 4, 8, 12, 16, 17, 18], 'itemsize': 26})


# pcl::toROSMsg of dlio::Point: x y z (PCL_ADD_POINT4D, 16 B), intensity, then the union of t (u32) /
# time (f32) / timestamp (f64) aligned to 8, EIGEN_ALIGN16: 32 B. (name, offset, PointField datatype)
OUT_FIELDS = (('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('intensity', 16, 7), ('t', 24, 6), ('time', 24, 7),
              ('timestamp', 24, 8))
OUT_STEP = 32


def out_dtype(np):
    return np.dtype({'names': ['x', 'y', 'z', 'intensity', 'timestamp'],
                     'formats': ['<f4', '<f4', '<f4', '<f4', '<f8'], 'offsets': [0, 4, 8, 16, 24],
                     'itemsize': OUT_STEP})


def crop_mask(np, xyz, size):
    """True for the points DLIO keeps: finite, and outside the +-size cube around the lidar."""
    finite = np.isfinite(xyz).all(axis=1)
    inside = (np.abs(xyz) <= size).all(axis=1)
    return finite & ~inside


def cloud_out(np, data, n, odom_lidar, crop):
    """The deskewed cloud's bytes and point count from n raw livox points in the lidar frame."""
    raw = np.frombuffer(data, dtype=livox_dtype(np), count=n)
    xyz = np.stack([raw['x'], raw['y'], raw['z']], axis=1).astype(np.float64)
    keep = crop_mask(np, xyz, crop)
    R, t = np.array(quat_to_rot(odom_lidar[1])), np.array(odom_lidar[0])
    pts = xyz[keep] @ R.T + t
    out = np.zeros(len(pts), dtype=out_dtype(np))
    out['x'], out['y'], out['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    out['intensity'], out['timestamp'] = raw['intensity'][keep], raw['timestamp'][keep]
    return out.tobytes(), len(out)


# ---------------------------------------------------------------------------- ROS

def stamp_s(st):
    return st.sec + st.nanosec * 1e-9


def run(ns, frames, cfg):
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    odom_f, base_f, imu_f = ('%s/%s' % (ns, cfg.frames[k]) for k in ('odom', 'baselink', 'imu'))
    in_step = livox_dtype(np).itemsize
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: stop.append(s))
    node = Node(node_name(ns), namespace='/' + ns, enable_rosout=False, start_parameter_services=False)
    log = node.get_logger()
    # DLIO's: every publisher depth 1, reliable (odom.cc:57-67); the tf broadcaster's depth 100.
    rel1 = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE)
    pubs = [
        ('session', node.create_publisher(String, 'dlio/session', QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE))),
        ('odom', node.create_publisher(Odometry, 'dlio/odom_node/odom', rel1)),
        ('pose', node.create_publisher(PoseStamped, 'dlio/odom_node/pose', rel1)),
        ('cloud', node.create_publisher(PointCloud2, 'dlio/odom_node/pointcloud/deskewed', rel1)),
        ('tf', node.create_publisher(TFMessage, '/tf', QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=100, reliability=ReliabilityPolicy.RELIABLE))),
    ]
    pub = dict(pubs)
    gt, gate, scans = GtBuffer(), CloudGate(cfg.hold_s), []
    stats = dict.fromkeys(('gt', 'scans', 'clouds', 'held', 'dropped', 'calibrating', 'pts', 'bad'), 0)
    payload, sid = session_payload(ns, time.time_ns())
    out_fields = [PointField(name=nm, offset=off, datatype=dt, count=1) for nm, off, dt in OUT_FIELDS]
    log.warning('odometry: truth (NOT PARITY): ground truth stands in for DLIO; session %s' % sid)
    print(summary(ns, cfg, frames.lever[2]), flush=True)

    def tf(stamp, parent, child, pose):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, parent, child
        tr, rot = t.transform.translation, t.transform.rotation
        tr.x, tr.y, tr.z = pose[0]
        rot.x, rot.y, rot.z, rot.w = pose[1]
        return t

    def on_gt(m):
        stats['gt'] += 1
        p, o = m.pose.pose.position, m.pose.pose.orientation
        world_chassis = ((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        ts = stamp_s(m.header.stamp)
        if not gt.add(ts, world_chassis):
            return
        gate.saw_truth(ts)
        odom_base = frames.base_in_odom(world_chassis)
        lv, av = m.twist.twist.linear, m.twist.twist.angular
        v, w = frames.twist(odom_base, (lv.x, lv.y, lv.z), (av.x, av.y, av.z))
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = m.header.stamp, odom_f, base_f
        pp, po = od.pose.pose.position, od.pose.pose.orientation
        pp.x, pp.y, pp.z = odom_base[0]
        po.x, po.y, po.z, po.w = odom_base[1]
        od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.linear.z = v
        od.twist.twist.angular.x, od.twist.twist.angular.y, od.twist.twist.angular.z = w
        pub['odom'].publish(od)
        ps = PoseStamped()
        ps.header, ps.pose = od.header, od.pose.pose
        pub['pose'].publish(ps)
        pub['tf'].publish(TFMessage(transforms=[tf(m.header.stamp, odom_f, base_f, odom_base)]))

    def on_scan(m):
        stats['scans'] += 1
        scans.append((stamp_s(m.header.stamp), m, time.monotonic()))

    def flush_scans():
        while scans:
            ts, m, arrived = scans[0]
            what, world_chassis = scan_pose(gt, ts, time.monotonic() - arrived)
            if what == 'wait':
                return
            scans.pop(0)
            if what == 'drop':
                stats['dropped'] += 1
                continue
            if not gate.open(ts):
                stats['calibrating'] += 1
                continue
            if m.point_step != in_step or len(m.data) < m.width * m.height * in_step:
                stats['bad'] += 1
                if stats['bad'] == 1:
                    log.warning('livox/lidar: point_step %d, expected livox_sim_bridge.py\'s %d: dropping such scans'
                                % (m.point_step, in_step))
                continue
            stats['held'] += what == 'held'
            odom_base = frames.base_in_odom(world_chassis)
            data, n = cloud_out(np, bytes(m.data), m.width * m.height, frames.lidar_in_odom(odom_base), cfg.crop)
            c = PointCloud2()
            c.header.stamp, c.header.frame_id = m.header.stamp, odom_f
            c.height, c.width, c.fields = 1, n, out_fields
            c.is_bigendian, c.point_step, c.row_step, c.is_dense = False, OUT_STEP, OUT_STEP * n, True
            c.data = data
            pub['cloud'].publish(c)
            pub['tf'].publish(TFMessage(transforms=[tf(m.header.stamp, base_f, imu_f, cfg.base_imu)]))
            stats['clouds'] += 1
            stats['pts'] += n

    subs = [node.create_subscription(Odometry, 'ground_truth/odom_wall', on_gt, QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=50, reliability=ReliabilityPolicy.RELIABLE)),
            node.create_subscription(PointCloud2, 'livox/lidar', on_scan, QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE))]
    last_session, last_report = -SESSION_PERIOD_S, time.monotonic()
    try:
        while not stop:
            now = time.monotonic()
            if now - last_session >= SESSION_PERIOD_S:
                last_session = now
                pub['session'].publish(String(data=payload))
            rclpy.spin_once(node, timeout_sec=0.005)
            flush_scans()
            if now - last_report >= REPORT_S:
                last_report = now
                print('[truth_odom] %s: ground truth %d (%d restarts), scans %d in, %d clouds (%d held, %d dropped, '
                      '%d in calibration, %d bad), %.0f pts/cloud'
                      % (ns, stats['gt'], gt.resets, stats['scans'], stats['clouds'], stats['held'], stats['dropped'],
                         stats['calibrating'], stats['bad'], stats['pts'] / max(1, stats['clouds'])), flush=True)
    finally:
        goal_relay.leave_gently([(node, 'subscription', s) for s in subs]
                                + [(node, 'publisher', p) for _, p in pubs],
                                [node], lambda: rclpy.spin_once(node, timeout_sec=0), LEAVE_GAP_S, LEAVE_PAUSE_S)
        rclpy.try_shutdown()
        print('[truth_odom] %s: left the graph' % ns, flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('ns')
    ap.add_argument('--spawn', default=SPAWN_DEFAULT, help='spawn.yaml (default %(default)s)')
    ap.add_argument('--dlio-cfg', default=DLIO_CFG_DIR, help="the image's DLIO cfg dir (default %(default)s)")
    ap.add_argument('--check', action='store_true', help='check the spawn entry and the cfg, print them, exit 0')
    args = ap.parse_args(argv)
    try:
        world_map, h = check_spawn(drive_pattern.load_spawn(args.spawn), args.ns, args.spawn)
        cfg = read_dlio_cfg(args.dlio_cfg)
        if args.check:
            print(summary(args.ns, cfg, h), flush=True)
            return 0
        drive_pattern.require_sim()
    except (SetupError, drive_pattern.PlanError) as e:
        print('truth_odom.py: %s' % e, file=sys.stderr, flush=True)
        return 2
    return run(args.ns, Frames(world_map, h, cfg), cfg)


if __name__ == '__main__':
    sys.exit(main())
