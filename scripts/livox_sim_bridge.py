#!/usr/bin/env python3
"""Gazebo MID-360 + IMU -> what livox_ros_driver2 publishes on the robot, on wall time.

One per agent, in sim-gazebo:

  livox_sim_bridge.py --ns JR81 [--imu-units g|mps2] [--min-range 0.1] [--max-range 40.0]
                      [--slew 0.05] [--run-dir /rover_sim/run] [--input pc2|custom]

Node /<ns>/livox_sim_bridge_<ns>. In (sim-time stamps, from the patched plugins):
mid360_PointCloud2 (or the mid360 CustomMsg with --input custom), sim/imu_raw (m/s^2),
ground_truth/odom. Out, as the robot's driver (lddc.cpp) has them:

  livox/lidar   PointCloud2 PointXYZRTLT: x y z intensity f32 @0 4 8 12, tag u8 @16,
                line u8 @17, timestamp f64 @18 (absolute ns, > 1e14 so DLIO takes its
                Livox path), point_step 26, height 1, is_dense, RELIABLE depth 256,
                frame <ns>/lidar
  livox/imu     frame livox_frame, accel in g (the SDK's unit), orientation and
                covariances left at their defaults as the driver leaves them
  ground_truth/odom_wall   ground truth with the same wall stamps, as nav_msgs/Odometry
                defines it: child_frame_id <ns>/chassis (the model's root, on the floor
                under the turning centre; base_link is lidar_height above it) and the
                twist in that frame. The plugin's ground_truth/odom has the twist in
                world axes and child_frame_id 'chassis'.
  sim/time_offset, sim/rtf (1 Hz)

Why a bridge: the stack runs on wall clock as on the robot, Gazebo on sim time. Every
input stamp is mapped through one offset (ClockMapper) shared by lidar, IMU and ground
truth, so DLIO sees them exactly as consistent as they are in sim time. The offset
follows the lower envelope of (arrival - sim stamp) with a slew limit, so stamps never
jump and lag wall time by tens of ms. RTF < 0.95 for > 5 s invalidates a run (the stack's
staleness gates assume real time): it is logged and appended to <run-dir>/rtf_violations.log.

The pure parts (ClockMapper, RtfMonitor, the CDR codec, the numpy packing) import
without rclpy; tests/test_livox_sim_bridge.py covers them.
"""

import argparse
import array
import collections
import datetime
import math
import os
import re
import struct
import sys
import time

try:
    import numpy as np
except ImportError:  # host-side unit tests skip the packing tests
    np = None

NS_PER_S = 1000000000
G = 9.80665                      # livox_ros_driver2 passes the SDK's accel through, in g
LIVOX_FRAME_IMU = 'livox_frame'  # hardcoded in lddc.cpp InitImuMsg
MID360_LINES = 4                 # pub_handler.cpp: line = point index % kLineNumberMid360
DRIVER_QUEUE = 256               # lddc.cpp: kMinEthPacketQueueSize * 8, single topic
WALL_EPOCH_NS = 10 ** 17         # a sim stamp past this (3 years of sim) is a wall stamp

# sensor_msgs/PointField datatypes
INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32, FLOAT64 = range(1, 9)
_NP_TYPES = {INT8: 'i1', UINT8: 'u1', INT16: '<i2', UINT16: '<u2', INT32: '<i4',
             UINT32: '<u4', FLOAT32: '<f4', FLOAT64: '<f8'}

# (name, offset, datatype) of lddc.cpp InitPointcloud2MsgHeader
LIVOX_FIELDS = (('x', 0, FLOAT32), ('y', 4, FLOAT32), ('z', 8, FLOAT32),
                ('intensity', 12, FLOAT32), ('tag', 16, UINT8), ('line', 17, UINT8),
                ('timestamp', 18, FLOAT64))
LIVOX_POINT_STEP = 26


# ---------------------------------------------------------------------------- clock

class ClockMapper:
    """sim stamp -> wall stamp, one per agent, shared by all its streams.

    Each message gives d = wall arrival - sim stamp = offset + transport latency, and
    latency is positive, so the target offset is the minimum of d over a sliding window
    (bucketed, O(1) per sample). Gazebo's real-time cap never runs sim ahead of wall, so
    d has no low outliers to reject. The applied offset moves toward the target by at
    most `slew` s per wall second: stamps never jump, and at RTF r < 1 the offset grows
    at 1 - r, which keeps stamps advancing at the wall rate while r >= 1 - slew.

    Re-initialised (offset := d) on the first sample; when d stays more than `jump_s`
    above the offset for `confirm_s` without falling (pause/unpause, a stall the world's
    real_time_lock gave up on, world reset, a sustained RTF < 1 - slew: an RTF violation by
    then); and when the target falls more than `ahead_s` below the offset (sim stamps
    jumping ahead). A single late message, e.g. after a stall of this process, does not
    move the offset. Per stream, published stamps are strictly increasing: a stamp not
    after the previous one is nudged to previous + 1 us and counted.

    jump_s sits below real_time_lock's max_lag (0.5 s in worlds/*.world): a stall the lock
    will not catch up is re-anchored here within confirm_s, instead of the slew taking
    ~lag / slew s to absorb it while every stamp is that stale. A stall it does catch up (at up to
    1.5x) shows as d falling back: while d falls by more than settle_s within a
    confirmation window, the window restarts, so the offset is not moved up only to leave
    stamps in the future as sim time catches up.
    """

    def __init__(self, window_s=2.0, slew=0.05, jump_s=0.4, bucket_s=0.1, confirm_s=0.2,
                 settle_s=0.02, ahead_s=1.0, rtf_window_s=5.0, stall_s=1.0, nudge_ns=1000):
        if window_s <= 0 or bucket_s <= 0 or slew < 0 or jump_s <= 0 or ahead_s <= 0:
            raise ValueError('window, bucket, jump and ahead must be > 0 and slew >= 0')
        self.bucket_ns = int(bucket_s * NS_PER_S)
        self.nbuckets = max(1, int(round(window_s / bucket_s)))
        self.slew = float(slew)
        self.jump_ns = int(jump_s * NS_PER_S)
        self.confirm_ns = int(confirm_s * NS_PER_S)
        self.settle_ns = int(settle_s * NS_PER_S)
        self.ahead_ns = int(ahead_s * NS_PER_S)
        self.rtf_window_ns = int(rtf_window_s * NS_PER_S)
        self.stall_ns = int(stall_s * NS_PER_S)
        self.nudge_ns = int(nudge_ns)
        self.offset_ns = None            # applied: published = sim + offset
        self.target_ns = None
        self._buckets = collections.deque()  # [bucket index, min d]
        self._last_wall = None
        self._prev_d = {}                # stream -> last d (labels a jump as sudden)
        self._high = None                # [since wall, min d, samples, kind, d at since] while d >> offset
        self._last_out = {}              # stream -> last published stamp
        self.nudges = collections.Counter()
        self.events = collections.Counter()   # init / jump / drift
        self._rtf = collections.deque()  # (wall, sim) of the RTF stream
        self._rtf_seen = False

    def _reset(self, d, wall_ns):
        self.offset_ns = self.target_ns = d
        self._buckets.clear()
        self._buckets.append([wall_ns // self.bucket_ns, d])
        self._last_wall = wall_ns
        self._high = None

    def observe(self, stream, sim_ns, wall_ns):
        """Feed one message's (sim stamp, arrival). Returns None, or the event that
        re-initialised the offset: 'init', 'jump' or 'drift'."""
        d = wall_ns - sim_ns
        prev = self._prev_d.get(stream)
        self._prev_d[stream] = d
        if self.offset_ns is None:
            self._reset(d, wall_ns)
            self.events['init'] += 1
            return 'init'
        if d - self.offset_ns > self.jump_ns:
            h = self._high
            if h is None:
                sudden = prev is None or d - prev > self.jump_ns
                self._high = h = [wall_ns, d, 0, 'jump' if sudden else 'drift', d]
            elif h[4] - d > self.settle_ns:
                h[0], h[1], h[2], h[4] = wall_ns, d, 0, d     # still falling: sim catching up
            h[1] = min(h[1], d)
            h[2] += 1
            if wall_ns - h[0] >= self.confirm_ns and h[2] >= 3:
                kind = h[3]
                self._reset(h[1], wall_ns)
                self.events[kind] += 1
                return kind
            return None
        self._high = None

        b = wall_ns // self.bucket_ns
        if self._buckets and self._buckets[-1][0] == b:
            if d < self._buckets[-1][1]:
                self._buckets[-1][1] = d
        else:
            self._buckets.append([b, d])
        while self._buckets[0][0] <= b - self.nbuckets:
            self._buckets.popleft()
        self.target_ns = min(m for _, m in self._buckets)

        # (d never enters the window more than jump_s above the offset: see above.)
        if self.offset_ns - self.target_ns > self.ahead_ns:
            self._reset(self.target_ns, wall_ns)
            self.events['drift'] += 1
            return 'drift'
        dt = wall_ns - self._last_wall
        if dt > 0:
            self._last_wall = wall_ns
            step = int(self.slew * dt)
            err = self.target_ns - self.offset_ns
            self.offset_ns += max(-step, min(step, err))
        return None

    def stamp(self, stream, sim_ns):
        """Published stamp (ns) for a sim stamp of `stream`, after observe()."""
        out = sim_ns + self.offset_ns
        last = self._last_out.get(stream)
        if last is not None and out <= last:
            out = last + self.nudge_ns
            self.nudges[stream] += 1
        self._last_out[stream] = out
        return out

    def map(self, stream, sim_ns, wall_ns):
        ev = self.observe(stream, sim_ns, wall_ns)
        return self.stamp(stream, sim_ns), ev

    def observe_rtf(self, sim_ns, wall_ns):
        """The RTF stream (the IMU: the densest and most regular). A pause or hang stays
        in the window, as the lost real time it is; a world reset (sim going backwards)
        restarts it."""
        if self._rtf and sim_ns <= self._rtf[-1][1]:
            self._rtf.clear()
        self._rtf.append((wall_ns, sim_ns))
        self._rtf_seen = True
        while self._rtf and self._rtf[0][0] < wall_ns - self.rtf_window_ns:
            self._rtf.popleft()

    def rtf(self, now_ns, min_span_s=1.0):
        """d(sim)/d(wall) over the RTF window. 0.0 once samples stop for stall_s after
        having flowed (a stalled or paused sim), None while unknown."""
        if not self._rtf_seen:
            return None
        if not self._rtf or now_ns - self._rtf[-1][0] > self.stall_ns:
            return 0.0
        w0, s0 = self._rtf[0]
        w1, s1 = self._rtf[-1]
        if w1 - w0 < min_span_s * NS_PER_S:
            return None
        return (s1 - s0) / float(w1 - w0)


class RtfMonitor:
    """RTF < threshold for longer than hold_s is a violation (one event per episode)."""

    def __init__(self, threshold=0.95, hold_s=5.0):
        self.threshold = threshold
        self.hold_s = hold_s
        self.below_since = None
        self.violating = False
        self.worst = None

    def update(self, rtf, now_s):
        """Returns 'violation' when an episode starts, 'recovered' when it ends, else
        None. An unknown RTF (None) neither starts nor ends an episode."""
        if rtf is None or (isinstance(rtf, float) and math.isnan(rtf)):
            return None
        if rtf < self.threshold:
            if self.below_since is None:
                self.below_since = now_s
                self.worst = rtf
            self.worst = min(self.worst, rtf)
            if not self.violating and now_s - self.below_since > self.hold_s:
                self.violating = True
                return 'violation'
            return None
        self.below_since = None
        if self.violating:
            self.violating = False
            return 'recovered'
        return None


def violation_line(ns, rtf, when=None):
    when = when or datetime.datetime.now().astimezone()
    return '%s %s rtf=%.3f\n' % (when.isoformat(timespec='milliseconds'), ns, rtf)


def append_violation(run_dir, line):
    """Append to <run_dir>/rtf_violations.log; False (no raise) if the run dir is
    missing or unwritable, so a bridge outside a sim.py run still works."""
    if not run_dir or not os.path.isdir(run_dir):
        return False
    try:
        with open(os.path.join(run_dir, 'rtf_violations.log'), 'a') as f:
            f.write(line)
        return True
    except OSError:
        return False


def stamp_ns(stamp):
    return stamp.sec * NS_PER_S + stamp.nanosec


def set_stamp(stamp, ns):
    stamp.sec, stamp.nanosec = divmod(int(ns), NS_PER_S)


# ---------------------------------------------------------------------------- CDR

# IMU and ground truth (300 msg/s) go through as serialized CDR: building rclpy message
# objects for them cost more CPU than everything else in the bridge. Layouts are XCDR1
# little endian, aligned from the payload start after the 4-byte encapsulation header,
# as rmw_zenoh_cpp (Fast-CDR) writes them; tests check them against rclpy's serializer.
CDR_LE = b'\x00\x01\x00\x00'
_HDR = struct.Struct('<iII')           # header.stamp sec, nanosec; frame_id length
_V3 = struct.Struct('<3d')
_IMU_OUT = struct.Struct('<4siII12s4d9d3d9d3d9d')
_ZERO9 = (0.0,) * 9


def _align(pos, n):
    return (pos + n - 1) & ~(n - 1)


def _check_cdr(buf):
    if len(buf) < 16 or buf[0:2] != CDR_LE[0:2]:
        raise ValueError('not a little-endian CDR buffer')


def cdr_stamp_ns(buf):
    """header.stamp of a serialized message whose first field is a std_msgs/Header."""
    _check_cdr(buf)
    sec, nsec, _ = _HDR.unpack_from(buf, 4)
    return sec * NS_PER_S + nsec


def cdr_restamp(buf, ns):
    """The same serialized message with header.stamp set to ns."""
    out = bytearray(buf)
    sec, nsec = divmod(int(ns), NS_PER_S)
    struct.pack_into('<iI', out, 4, sec, nsec)
    return bytes(out)


def imu_parse(buf):
    """(stamp ns, angular_velocity, linear_acceleration) of a serialized sensor_msgs/Imu."""
    _check_cdr(buf)
    sec, nsec, slen = _HDR.unpack_from(buf, 4)
    p = 4 + _align(12 + slen, 8)       # orientation, after the frame_id string
    if len(buf) < p + 296:
        raise ValueError('Imu needs %d bytes, got %d' % (p + 296, len(buf)))
    # orientation 4d, its covariance 9d, then angular_velocity 3d at +104, its covariance,
    # linear_acceleration 3d at +200
    return sec * NS_PER_S + nsec, _V3.unpack_from(buf, p + 104), _V3.unpack_from(buf, p + 200)


def imu_build(stamp, gyro, accel):
    """Serialized sensor_msgs/Imu as lddc.cpp InitImuMsg fills it: frame livox_frame,
    gyro and accel set, orientation (0, 0, 0, 1) and covariances zero as constructed."""
    sec, nsec = divmod(int(stamp), NS_PER_S)
    return _IMU_OUT.pack(CDR_LE, sec, nsec, len(LIVOX_FRAME_IMU) + 1,
                         LIVOX_FRAME_IMU.encode(), 0.0, 0.0, 0.0, 1.0, *_ZERO9,
                         *gyro, *_ZERO9, *accel, *_ZERO9)


def _cdr_string(buf, pos):
    """(text, end) of the CDR string whose length word is at payload offset pos."""
    pos = _align(pos, 4)
    (n,) = struct.unpack_from('<I', buf, 4 + pos)
    if n < 1 or len(buf) < 4 + pos + 4 + n:
        raise ValueError('bad CDR string at %d' % pos)
    return bytes(buf[4 + pos + 4:4 + pos + 3 + n]).decode('utf-8', 'replace'), pos + 4 + n


_ODOM_TAIL = struct.Struct('<85d')     # pose 7, its covariance 36, twist 6, its covariance 36


def odom_parse(buf):
    """(sec, nanosec, frame_id, child_frame_id, 85 doubles: pose, pose covariance, twist,
    twist covariance) of a serialized nav_msgs/Odometry."""
    _check_cdr(buf)
    sec, nsec = struct.unpack_from('<iI', buf, 4)
    frame, pos = _cdr_string(buf, 8)
    child, pos = _cdr_string(buf, pos)
    pos = _align(pos, 8)
    if len(buf) < 4 + pos + _ODOM_TAIL.size:
        raise ValueError('Odometry needs %d bytes, got %d' % (4 + pos + _ODOM_TAIL.size, len(buf)))
    return sec, nsec, frame, child, _ODOM_TAIL.unpack_from(buf, 4 + pos)


def _cdr_put_string(out, text):
    raw = text.encode() + b'\0'
    out += b'\0' * (_align(len(out) - 4, 4) - (len(out) - 4))
    out += struct.pack('<I', len(raw)) + raw


def odom_build(stamp, frame, child, values):
    sec, nsec = divmod(int(stamp), NS_PER_S)
    out = bytearray(CDR_LE + struct.pack('<iI', sec, nsec))
    _cdr_put_string(out, frame)
    _cdr_put_string(out, child)
    out += b'\0' * (_align(len(out) - 4, 8) - (len(out) - 4))
    return bytes(out + _ODOM_TAIL.pack(*values))


def rotate_inverse(q, v):
    """v (world axes) in the axes of the frame with orientation q = (x, y, z, w)."""
    x, y, z, w = q
    # R(q)^T v = v + 2 w (v x u) + 2 u x (u x v) with u = (x, y, z): the conjugate rotation
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] - w * tx + (y * tz - z * ty), v[1] - w * ty + (z * tx - x * tz),
            v[2] - w * tz + (x * ty - y * tx))


def odom_to_body(buf, stamp, child):
    """gazebo_ros_p3d's Odometry -> the nav_msgs convention: restamped, child_frame_id
    replaced (p3d writes the bare link name), and the twist rotated from world axes, which
    p3d reports, into the child frame's. The pose and both covariances (p3d's are diagonal
    and isotropic) are unchanged."""
    _, _, frame, _, v = odom_parse(buf)
    q = v[3:7]
    twist = rotate_inverse(q, v[43:46]) + rotate_inverse(q, v[46:49])
    return odom_build(stamp, frame, child, v[:43] + twist + v[49:])


# ---------------------------------------------------------------------------- points

def _require_numpy():
    if np is None:
        raise RuntimeError('livox_sim_bridge needs numpy')


_LIVOX_DT = []


def livox_dtype():
    _require_numpy()
    if not _LIVOX_DT:
        _LIVOX_DT.append(np.dtype({'names': [f[0] for f in LIVOX_FIELDS],
                                   'formats': [_NP_TYPES[f[2]] for f in LIVOX_FIELDS],
                                   'offsets': [f[1] for f in LIVOX_FIELDS],
                                   'itemsize': LIVOX_POINT_STEP}))
    return _LIVOX_DT[0]


# Input field names, by role: the plugin's PointCloud2 comes from
# sensor_msgs::convertPointCloudToPointCloud2 (x, y, z + one float field per channel).
_INTENSITY_NAMES = ('intensity', 'reflectivity')
_LINE_NAMES = ('line', 'ring')


def cloud_view(data, fields, point_step, n):
    """Structured view of n points of a PointCloud2 buffer. fields: [(name, offset,
    datatype)]. Returns {'x','y','z'[, 'intensity'][, 'line']: arrays}."""
    _require_numpy()
    by_name = {f[0]: f for f in fields}
    want = []
    for role, names in (('x', ('x',)), ('y', ('y',)), ('z', ('z',)),
                        ('intensity', _INTENSITY_NAMES), ('line', _LINE_NAMES)):
        f = next((by_name[nm] for nm in names if nm in by_name), None)
        if f is None:
            if role in ('x', 'y', 'z'):
                raise ValueError('cloud has no %r field (fields: %s)' % (role, sorted(by_name)))
            continue
        if f[2] not in _NP_TYPES:
            raise ValueError('field %r has unknown datatype %d' % (f[0], f[2]))
        want.append((role, f[1], _NP_TYPES[f[2]]))
    if len(data) < n * point_step:
        raise ValueError('cloud data is %d bytes, %d points x %d need %d'
                         % (len(data), n, point_step, n * point_step))
    dt = np.dtype({'names': [w[0] for w in want], 'formats': [w[2] for w in want],
                   'offsets': [w[1] for w in want], 'itemsize': point_step})
    arr = np.frombuffer(data, dtype=dt, count=n)
    return {name: arr[name] for name in dt.names}


# livox_ros_driver2/CustomMsg in CDR (little endian, XCDR1 alignment from the payload
# start after the 4-byte encapsulation header). CustomPoint is 19 bytes, and the next
# element's uint32 is 4-aligned, so elements are 20 apart.
_CUSTOM_POINT = (('offset_time', 0, '<u4'), ('x', 4, '<f4'), ('y', 8, '<f4'), ('z', 12, '<f4'),
                 ('reflectivity', 16, 'u1'), ('tag', 17, 'u1'), ('line', 18, 'u1'))
CUSTOM_POINT_STRIDE = 20


def parse_custom_msg(buf):
    """(sec, nanosec, frame_id, points) from a serialized CustomMsg, without building
    20k Python CustomPoint objects (what a typed rclpy subscription would do)."""
    _require_numpy()
    _check_cdr(buf)
    base = 4
    sec, nanosec, slen = struct.unpack_from('<iII', buf, base)
    pos = 12
    frame_id = bytes(buf[base + pos:base + pos + max(0, slen - 1)]).decode('utf-8', 'replace')
    pos = _align(pos + slen, 8) + 8          # timebase uint64
    pos = _align(pos, 4) + 4 + 1 + 3         # point_num, lidar_id, rsvd[3]
    pos = _align(pos, 4)
    (n,) = struct.unpack_from('<I', buf, base + pos)
    pos += 4
    need = base + pos + (n - 1) * CUSTOM_POINT_STRIDE + 19 if n else base + pos
    if len(buf) < need:
        raise ValueError('CustomMsg with %d points needs %d bytes, got %d' % (n, need, len(buf)))
    dt = np.dtype({'names': [p[0] for p in _CUSTOM_POINT], 'formats': [p[2] for p in _CUSTOM_POINT],
                   'offsets': [p[1] for p in _CUSTOM_POINT], 'itemsize': 19})
    pts = np.ndarray((n,), dtype=dt, buffer=buf, offset=base + pos, strides=(CUSTOM_POINT_STRIDE,))
    return sec, nanosec, frame_id, {'x': pts['x'], 'y': pts['y'], 'z': pts['z'],
                                    'intensity': pts['reflectivity'], 'line': pts['line']}


class PointFilter:
    """Keeps finite returns with min_range < r < max_range. The plugin clamps a no-hit
    ray to exactly its <range><max> and a too-near one to <min>; both are dropped (with
    a tolerance for float32 rounding). If the plugin's max is below --max-range, its
    clamp shows up as a pile of points at one range: those are dropped too and
    `clamp_seen` records the range so the caller can warn once."""

    def __init__(self, min_range=0.1, max_range=40.0):
        self.min2 = (min_range * (1 + 1e-4)) ** 2
        self.max2 = (max_range * (1 - 1e-4)) ** 2
        self.clamp_seen = None

    def mask(self, x, y, z):
        r2 = x * x + y * y + z * z           # NaN/inf in any coordinate propagate here
        with np.errstate(invalid='ignore'):
            keep = np.isfinite(r2) & (r2 > self.min2) & (r2 < self.max2)
        if keep.any():
            top = r2[keep].max()
            at_top = keep & (r2 >= top * (1 - 2e-5))
            n_top = int(np.count_nonzero(at_top))
            if n_top >= max(10, keep.size // 100):
                keep &= ~at_top
                self.clamp_seen = float(math.sqrt(top))
        return keep


def pack_livox(cols, keep, stamp_ns):
    """PointXYZRTLT bytes of the kept points. Every point gets the scan's stamp (the sim
    scan is instantaneous). line = point index % 4, as the driver numbers it, unless the
    input has its own line ids."""
    dt = livox_dtype()
    idx = np.flatnonzero(keep)
    out = np.zeros(idx.size, dtype=dt)
    out['x'] = cols['x'][idx]
    out['y'] = cols['y'][idx]
    out['z'] = cols['z'][idx]
    if 'intensity' in cols:
        out['intensity'] = cols['intensity'][idx]
    if 'line' in cols:
        out['line'] = cols['line'][idx]
    else:
        out['line'] = idx & (MID360_LINES - 1)
    out['timestamp'] = float(stamp_ns)
    return idx.size, out.tobytes()


# ---------------------------------------------------------------------------- node

NS_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ns', required=True, help='agent namespace, e.g. JR81')
    ap.add_argument('--imu-units', choices=('g', 'mps2'), default='g',
                    help='livox/imu linear_acceleration unit (the robot driver: g)')
    ap.add_argument('--min-range', type=float, default=0.1)
    ap.add_argument('--max-range', type=float, default=40.0,
                    help="no-hit cut, the MID-360 xacro's <range><max>")
    ap.add_argument('--slew', type=float, default=0.05, help='max offset change, s per s')
    ap.add_argument('--window', type=float, default=2.0, help='offset target window, s')
    ap.add_argument('--input', choices=('pc2', 'custom'), default='pc2',
                    help='mid360_PointCloud2 (default) or the mid360 CustomMsg')
    ap.add_argument('--run-dir', default='/rover_sim/run')
    a = ap.parse_args(argv)
    if not NS_RE.match(a.ns):
        ap.error('--ns %r is not a namespace name' % a.ns)
    if not 0 <= a.min_range < a.max_range:
        ap.error('need 0 <= --min-range < --max-range')
    return a


def main(argv=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.qos_event import PublisherEventCallbacks, SubscriptionEventCallbacks
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, PointCloud2, PointField
    from std_msgs.msg import Float64

    argv = sys.argv if argv is None else argv
    args = parse_args(rclpy.utilities.remove_ros_args(argv)[1:])
    _require_numpy()

    def qos(reliable, depth):
        return QoSProfile(reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=depth,
                          durability=DurabilityPolicy.VOLATILE)

    class Bridge(Node):
        def __init__(self):
            super().__init__('livox_sim_bridge_' + args.ns, namespace='/' + args.ns)
            self.ns = args.ns
            self.clock_map = ClockMapper(window_s=args.window, slew=args.slew)
            self.rtf_mon = RtfMonitor()
            self.filter = PointFilter(args.min_range, args.max_range)
            self.accel_scale = 1.0 / G if args.imu_units == 'g' else 1.0
            self.lidar_frame = args.ns + '/lidar'
            self.chassis_frame = args.ns + '/chassis'
            self.counts = collections.Counter()
            self.warned = set()
            self.run_dir = args.run_dir

            # Driver QoS (rclcpp default + depth 256); inputs best effort, which matches
            # the plugins whether they publish best effort (livox) or reliable. No default
            # QoS-event handlers: each is one more waitable the executor rebuilds its wait
            # set with on every one of ~400 wakeups/s.
            def pub(t, topic, depth):
                return self.create_publisher(t, topic, qos(True, depth),
                                             event_callbacks=PublisherEventCallbacks(use_default_callbacks=False))
            self.pub_lidar = pub(PointCloud2, 'livox/lidar', DRIVER_QUEUE)
            self.pub_imu = pub(Imu, 'livox/imu', DRIVER_QUEUE)
            self.pub_odom = pub(Odometry, 'ground_truth/odom_wall', 10)
            self.pub_offset = pub(Float64, 'sim/time_offset', 10)
            self.pub_rtf = pub(Float64, 'sim/rtf', 10)

            self.fields_out = [PointField(name=n, offset=o, datatype=t, count=1)
                               for n, o, t in LIVOX_FIELDS]
            self._in_fields = None

            def sub(t, topic, cb, depth, raw=False):
                self.create_subscription(t, topic, cb, qos(False, depth), raw=raw,
                                         event_callbacks=SubscriptionEventCallbacks(use_default_callbacks=False))
            if args.input == 'pc2':
                sub(PointCloud2, 'mid360_PointCloud2', self.on_pc2, 2)
                src = 'mid360_PointCloud2'
            else:
                from livox_ros_driver2.msg import CustomMsg
                sub(CustomMsg, 'mid360', self.on_custom, 2, raw=True)
                src = 'mid360 (CustomMsg, raw CDR)'
            sub(Imu, 'sim/imu_raw', self.on_imu, 100, raw=True)
            sub(Odometry, 'ground_truth/odom', self.on_odom, 50, raw=True)
            self.create_timer(1.0, self.on_second)
            self.create_timer(30.0, self.report)
            self.get_logger().info(
                '%s + sim/imu_raw + ground_truth/odom -> livox/lidar, livox/imu (%s), '
                'ground_truth/odom_wall; range (%.2f, %.1f) m, slew %.3f s/s'
                % (src, args.imu_units, args.min_range, args.max_range, args.slew))

        def warn_once(self, key, msg):
            if key not in self.warned:
                self.warned.add(key)
                self.get_logger().warning(msg)

        def map_stamp(self, stream, sim_ns, wall_ns):
            if sim_ns > WALL_EPOCH_NS:
                # One wall-stamped stream drags the shared offset to ~0 for all of them.
                self.warn_once('epoch-' + stream, '%s stamps look like wall time (%.3f s), not sim '
                               'time: are the sensor plugins the sim-stamp patched ones?'
                               % (stream, sim_ns / 1e9))
            out, ev = self.clock_map.map(stream, sim_ns, wall_ns)
            if ev == 'init':
                self.get_logger().info('time offset initialised from %s: wall - sim = %.6f s'
                                       % (stream, self.clock_map.offset_ns / 1e9))
            elif ev is not None:
                self.get_logger().warning(
                    'time offset re-initialised (%s, from %s): wall - sim = %.6f s. %s'
                    % (ev, stream, self.clock_map.offset_ns / 1e9,
                       'Sim paused/reset?' if ev == 'jump' else
                       'Sim slower than wall for long (RTF < 1 - slew): stamps jumped.'))
            return out

        def publish_cloud(self, sim_ns, wall_ns, cols, frame_in):
            if frame_in and frame_in != self.lidar_frame:
                self.warn_once('frame', 'input lidar frame %r, publishing as %r'
                               % (frame_in, self.lidar_frame))
            keep = self.filter.mask(cols['x'], cols['y'], cols['z'])
            if self.filter.clamp_seen is not None:
                self.warn_once('clamp', 'no-hit returns clamped at %.3f m (< --max-range %.1f): '
                               'dropping them; pass --max-range to match the xacro'
                               % (self.filter.clamp_seen, args.max_range))
            out_ns = self.map_stamp('lidar', sim_ns, wall_ns)
            n, data = pack_livox(cols, keep, out_ns)
            self.counts['lidar_in'] += 1
            self.counts['points_in'] += keep.size
            if n == 0:
                # The driver never publishes an empty cloud, and DLIO reads points[0].
                self.counts['lidar_empty'] += 1
                return
            msg = PointCloud2()
            set_stamp(msg.header.stamp, out_ns)
            msg.header.frame_id = self.lidar_frame
            msg.height = 1
            msg.width = n
            msg.fields = self.fields_out
            msg.is_bigendian = False
            msg.point_step = LIVOX_POINT_STEP
            msg.row_step = n * LIVOX_POINT_STEP
            # array.array skips the generated setter's per-byte Python check of bytes.
            msg.data = array.array('B', data)
            msg.is_dense = True
            self.pub_lidar.publish(msg)
            self.counts['lidar_out'] += 1
            self.counts['points_out'] += n

        def on_pc2(self, msg):
            wall_ns = time.time_ns()
            if msg.is_bigendian:
                self.warn_once('endian', 'big-endian cloud ignored')
                return
            n = msg.width * msg.height
            if msg.height > 1 and msg.row_step != msg.width * msg.point_step:
                self.warn_once('rowpad', 'cloud rows are padded: ignored')
                return
            sig = (msg.point_step, tuple((f.name, f.offset, f.datatype) for f in msg.fields))
            if sig != self._in_fields:
                self._in_fields = sig
                self.get_logger().info('input cloud: point_step %d, fields %s'
                                       % (sig[0], ', '.join('%s@%d' % f[:2] for f in sig[1])))
            try:
                cols = cloud_view(msg.data, sig[1], msg.point_step, n)
            except ValueError as e:
                self.warn_once('cloud', 'bad input cloud: %s' % e)
                return
            self.publish_cloud(stamp_ns(msg.header.stamp), wall_ns, cols, msg.header.frame_id)

        def on_custom(self, raw):
            wall_ns = time.time_ns()
            try:
                sec, nsec, frame, cols = parse_custom_msg(raw)
            except (ValueError, struct.error) as e:
                self.warn_once('custom', 'bad CustomMsg: %s' % e)
                return
            if not cols['line'].any():
                del cols['line']                 # the plugin leaves line 0: number as the driver
            self.publish_cloud(sec * NS_PER_S + nsec, wall_ns, cols, frame)

        def on_imu(self, raw):
            wall_ns = time.time_ns()
            try:
                sim_ns, gyro, a = imu_parse(raw)
            except (ValueError, struct.error) as e:
                self.warn_once('imu', 'bad Imu: %s' % e)
                return
            self.clock_map.observe_rtf(sim_ns, wall_ns)
            s = self.accel_scale
            self.pub_imu.publish(imu_build(self.map_stamp('imu', sim_ns, wall_ns), gyro,
                                           (a[0] * s, a[1] * s, a[2] * s)))
            self.counts['imu'] += 1

        def on_odom(self, raw):
            wall_ns = time.time_ns()
            try:
                sim_ns = cdr_stamp_ns(raw)
                out = self.map_stamp('odom', sim_ns, wall_ns)
                self.pub_odom.publish(odom_to_body(raw, out, self.chassis_frame))
            except (ValueError, struct.error) as e:
                self.warn_once('odom', 'bad Odometry: %s' % e)
                return
            self.counts['odom'] += 1

        def on_second(self):
            now_ns = time.time_ns()
            if self.clock_map.offset_ns is not None:
                self.pub_offset.publish(Float64(data=self.clock_map.offset_ns / 1e9))
            rtf = self.clock_map.rtf(now_ns)
            self.pub_rtf.publish(Float64(data=float('nan') if rtf is None else rtf))
            ev = self.rtf_mon.update(rtf, now_ns / 1e9)
            if ev == 'violation':
                m = self.rtf_mon
                logged = append_violation(self.run_dir, violation_line(self.ns, m.worst))
                self.get_logger().error(
                    'RTF %.3f < %.2f for more than %.0f s%s: this run is not real time and its '
                    'results are invalid%s'
                    % (m.worst, m.threshold, m.hold_s,
                       ' (no IMU from the sim: stalled or paused)' if rtf == 0.0 else '',
                       '' if logged else ' (no run dir %s to log to)' % self.run_dir))
            elif ev == 'recovered':
                self.get_logger().warning('RTF back to %.3f (worst %.3f)' % (rtf, self.rtf_mon.worst))

        def report(self):
            c, cm = self.counts, self.clock_map
            if cm.offset_ns is None:
                self.get_logger().info('waiting for sensor data')
                return
            rtf = cm.rtf(time.time_ns())
            self.get_logger().info(
                'lidar %d in / %d out (%d empty), points kept %.1f%%, imu %d, odom %d; '
                'offset %.6f s (target %+.1f ms), rtf %s, nudges %s, reinits %s'
                % (c['lidar_in'], c['lidar_out'], c['lidar_empty'],
                   100.0 * c['points_out'] / max(1, c['points_in']), c['imu'], c['odom'],
                   cm.offset_ns / 1e9, (cm.target_ns - cm.offset_ns) / 1e6,
                   'n/a' if rtf is None else '%.3f' % rtf, dict(cm.nudges) or 0,
                   dict(cm.events)))

    rclpy.init(args=argv)
    node = Bridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()            # SIGINT may already have shut the context down


if __name__ == '__main__':
    main()
