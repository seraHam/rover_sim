"""Unit tests for scripts/livox_sim_bridge.py's pure logic (no rclpy).

The clock tests run anywhere; the packing tests need numpy (skipped without it, as on
the host). With rclpy and livox_ros_driver2 importable (the gazebo image, sourced), the
CustomMsg parser is also checked against rclpy's own serializer.

    python3 -m unittest discover -s tests -v
"""
import importlib.util
import math
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('livox_sim_bridge', str(REPO / 'scripts' / 'livox_sim_bridge.py'))
bridge = importlib.util.module_from_spec(_spec)
sys.modules['livox_sim_bridge'] = bridge
_spec.loader.exec_module(bridge)

np = bridge.np
NS = bridge.NS_PER_S
MS = 1000000
WALL0 = 1760000000 * NS          # an epoch-like wall time


def feed(cm, streams, duration_s, rtf=1.0, latency_ms=(1.0, 1.0), sim0=0, wall0=WALL0,
         seed=1, rtf_stream='imu'):
    """Simulate arrivals: each stream (name, hz) publishes at sim times k/hz, which
    arrive at wall0 + (sim - sim0) / rtf + latency. Returns [(stream, sim, wall, out, ev)]
    in arrival order."""
    rnd = random.Random(seed)
    events = []
    for name, hz in streams:
        n = int(duration_s * rtf * hz)
        last = 0
        for k in range(1, n + 1):
            sim = sim0 + k * NS // hz
            lat = rnd.uniform(*latency_ms) * MS
            # One publisher's messages arrive in order (transport is FIFO per stream).
            last = max(last + 1000, wall0 + int((sim - sim0) / rtf + lat))
            events.append((last, name, sim))
    events.sort()
    out = []
    for wall, name, sim in events:
        if name == rtf_stream:
            cm.observe_rtf(sim, wall)
        stamp, ev = cm.map(name, sim, wall)
        out.append((name, sim, wall, stamp, ev))
    return out


STREAMS = (('imu', 200), ('lidar', 10), ('odom', 100))


def feed_stall(cm, stall_s, catch_up, at_s=5.0, until_s=30.0, max_lag_s=0.5):
    """IMU at 200 Hz through a gzserver stall of stall_s at at_s, followed as the worlds'
    real_time_lock follows it: a lag up to max_lag_s is made up at up to catch_up x real
    time (rate 1 + lag / 0.5 s, at most 1.5), a larger one is accepted (re-anchored).
    Returns [(wall arrival, published stamp, event)]."""
    wall, sim, anchor, stalled, rows = 0.0, 0.0, (0.0, 0.0), False, []
    while wall < until_s:
        if not stalled and wall >= at_s:
            wall, stalled = wall + stall_s, True
        lag = (wall - anchor[0]) - (sim - anchor[1])
        if lag > max_lag_s:
            anchor, lag = (wall, sim), 0.0
        wall += 0.001 / min(1 + min(max(lag / 0.5, 0.0), 0.5), catch_up)    # one 1 ms step
        sim += 0.001
        if round(sim * 1000) % 5 == 0:
            arr = WALL0 + int((wall + 0.002) * NS)
            out, ev = cm.map('imu', int(round(sim * 1000)) * MS, arr)
            rows.append((arr, out, ev))
    return rows


def strictly_increasing(rows, stream):
    s = [r[3] for r in rows if r[0] == stream]
    return all(b > a for a, b in zip(s, s[1:]))


class ClockMapperTest(unittest.TestCase):
    def test_first_sample_initialises(self):
        cm = bridge.ClockMapper()
        out, ev = cm.map('imu', 5 * NS, WALL0)
        self.assertEqual(ev, 'init')
        self.assertEqual(out, WALL0)
        self.assertEqual(cm.offset_ns, WALL0 - 5 * NS)

    def test_rtf1_offset_is_min_latency_and_stable(self):
        cm = bridge.ClockMapper()
        rows = feed(cm, STREAMS, 20.0, latency_ms=(1.0, 8.0))
        # After the window, the offset sits within ~0.1 ms of wall0 + min latency and
        # moves less than 1 ms over the last 10 s.
        late = [r[3] - r[1] for r in rows if r[2] > WALL0 + 10 * NS]
        self.assertLess(max(late) - min(late), 1 * MS)
        self.assertAlmostEqual((cm.offset_ns - WALL0) / MS, 1.0, delta=0.2)
        for s, _ in STREAMS:
            self.assertTrue(strictly_increasing(rows, s), s)
        self.assertEqual(sum(cm.nudges.values()), 0)
        self.assertEqual(dict(cm.events), {'init': 1})
        self.assertAlmostEqual(cm.rtf(rows[-1][2]), 1.0, delta=0.005)

    def test_rtf097_slews_monotonic(self):
        cm = bridge.ClockMapper()
        rows = feed(cm, STREAMS, 30.0, rtf=0.97, latency_ms=(1.0, 5.0))
        for s, _ in STREAMS:
            self.assertTrue(strictly_increasing(rows, s), s)
        self.assertEqual(dict(cm.events), {'init': 1})
        # Stamps advance at the wall rate and lag arrival by the window's share of the
        # drift (2 s x 3%) plus latency, well under the stack's staleness gates.
        imu = [r for r in rows if r[0] == 'imu' and r[2] > WALL0 + 10 * NS]
        rate = (imu[-1][3] - imu[0][3]) / float(imu[-1][2] - imu[0][2])
        self.assertAlmostEqual(rate, 1.0, delta=0.005)
        lag = [(r[2] - r[3]) / MS for r in imu]
        self.assertLess(max(lag), 80.0)
        self.assertGreater(min(lag), 0.0)
        self.assertAlmostEqual(cm.rtf(rows[-1][2]), 0.97, delta=0.005)

    def test_slew_limit(self):
        cm = bridge.ClockMapper(slew=0.05)
        cm.map('imu', 0, WALL0 + 500 * MS)
        # The first sample was 0.5 s late: the offset comes down at no more than 50 ms
        # per s, and the stamps keep increasing meanwhile.
        prev, last, prev_wall = cm.offset_ns, None, WALL0 + 500 * MS
        for k in range(1, 401):
            sim = 500 * MS + k * 5 * MS
            wall = WALL0 + sim + 1 * MS
            out, _ = cm.map('imu', sim, wall)
            self.assertLessEqual(prev - cm.offset_ns, 0.05 * (wall - prev_wall) + 1)
            self.assertTrue(last is None or out > last)
            prev, last, prev_wall = cm.offset_ns, out, wall
        self.assertAlmostEqual((WALL0 + 500 * MS - cm.offset_ns) / MS, 100.0, delta=2.0)  # ~2 s x 50 ms/s
        self.assertEqual(sum(cm.nudges.values()), 0)

    def test_pause_reinitialises_forward(self):
        cm = bridge.ClockMapper()
        rows = feed(cm, STREAMS, 5.0)
        # Paused 3 s: sim resumes where it stopped, wall moved on.
        rows += feed(cm, STREAMS, 5.0, sim0=5 * NS, wall0=WALL0 + 8 * NS)
        self.assertEqual(cm.events['jump'], 1)
        self.assertAlmostEqual((cm.offset_ns - (WALL0 + 3 * NS)) / MS, 1.0, delta=0.2)
        for s, _ in STREAMS:
            self.assertTrue(strictly_increasing(rows, s), s)
        self.assertEqual(sum(cm.nudges.values()), 0)

    def test_world_reset_keeps_stamps_increasing(self):
        cm = bridge.ClockMapper()
        rows = feed(cm, STREAMS, 5.0, sim0=100 * NS)
        rows += feed(cm, STREAMS, 5.0, sim0=0, wall0=WALL0 + 5 * NS)
        self.assertEqual(cm.events['jump'], 1)
        for s, _ in STREAMS:
            self.assertTrue(strictly_increasing(rows, s), s)
        # Only the confirm period (0.2 s) is nudged.
        self.assertLess(cm.nudges['imu'], 50)

    def test_stall_not_caught_up_reanchors(self):
        # real_time_lock accepts a lag over 0.5 s for good: the offset follows within the
        # confirm period, not after ~0.7 s / slew of stamps that stale.
        cm = bridge.ClockMapper()
        rows = feed_stall(cm, 0.7, catch_up=1.5)
        self.assertEqual(dict(cm.events), {'init': 1, 'jump': 1})
        resumed = next(a for a, _, _ in rows if a > WALL0 + int(5.6 * NS))
        stale = [(a - o) / NS for a, o, _ in rows if a > resumed + int(0.25 * NS)]
        self.assertLess(max(stale), 0.01)
        self.assertGreaterEqual(min(stale), 0.0)
        out = [o for _, o, _ in rows]
        self.assertTrue(all(b > a for a, b in zip(out, out[1:])))
        self.assertEqual(sum(cm.nudges.values()), 0)

    def test_stall_caught_up_is_not_a_jump(self):
        # A lag the lock makes up (here 0.48 s, at 1.5x or a CPU-bound 1.2x) is not
        # re-anchored: moving the offset up would leave stamps in the future as sim time
        # catches up. The stamps are as late as the data meanwhile.
        for catch_up in (1.5, 1.2):
            cm = bridge.ClockMapper()
            rows = feed_stall(cm, 0.48, catch_up=catch_up)
            self.assertEqual(dict(cm.events), {'init': 1}, catch_up)
            err = [(a - o) / NS for a, o, _ in rows]
            self.assertGreater(min(err), -0.01, catch_up)                # never in the future
            self.assertLess(max(err[-1000:]), 0.01, catch_up)           # caught up afterwards
            self.assertEqual(sum(cm.nudges.values()), 0)

    def test_one_late_message_is_ignored(self):
        cm = bridge.ClockMapper()
        feed(cm, (('imu', 200),), 5.0)
        before = cm.offset_ns
        sim = 5 * NS + NS // 200
        out, ev = cm.map('lidar', sim, WALL0 + sim + 1500 * MS)  # 1.5 s late
        self.assertIsNone(ev)
        self.assertEqual(cm.offset_ns, before)
        self.assertEqual(out, sim + before)
        out, ev = cm.map('imu', sim + 5 * MS, WALL0 + sim + 6 * MS)
        self.assertIsNone(ev)
        self.assertEqual(dict(cm.events), {'init': 1})

    def test_sim_jumping_ahead_reinitialises(self):
        cm = bridge.ClockMapper()
        feed(cm, (('imu', 200),), 3.0)
        sim = 3 * NS + 2 * NS
        _, ev = cm.map('imu', sim, WALL0 + 3 * NS + 6 * MS)
        self.assertEqual(ev, 'drift')

    def test_rtf09_bounded_and_monotonic(self):
        cm = bridge.ClockMapper()
        rows = feed(cm, STREAMS, 60.0, rtf=0.9)
        self.assertGreaterEqual(cm.events['drift'], 1)
        for s, _ in STREAMS:
            self.assertTrue(strictly_increasing(rows, s), s)
        lag = [(r[2] - r[3]) / NS for r in rows]
        self.assertLess(max(lag), 1.5)
        self.assertAlmostEqual(cm.rtf(rows[-1][2]), 0.9, delta=0.005)

    def test_nudge(self):
        cm = bridge.ClockMapper()
        a, _ = cm.map('imu', NS, WALL0)
        b, _ = cm.map('imu', NS, WALL0 + 1)
        self.assertEqual(b, a + 1000)
        self.assertEqual(cm.nudges['imu'], 1)
        c, _ = cm.map('lidar', NS, WALL0 + 2)   # other streams are independent
        self.assertEqual(c, a)

    def test_rtf_unknown_and_stall(self):
        cm = bridge.ClockMapper()
        self.assertIsNone(cm.rtf(WALL0))
        cm.observe_rtf(0, WALL0)
        cm.observe_rtf(NS // 2, WALL0 + NS // 2)
        self.assertIsNone(cm.rtf(WALL0 + NS // 2))          # < 1 s of data
        cm.observe_rtf(2 * NS, WALL0 + 2 * NS)
        self.assertAlmostEqual(cm.rtf(WALL0 + 2 * NS), 1.0)
        self.assertEqual(cm.rtf(WALL0 + 4 * NS), 0.0)        # stalled

    def test_rtf_across_pause_and_reset(self):
        cm = bridge.ClockMapper()
        rows = feed(cm, STREAMS, 5.0)
        # A 2.5 s pause is lost real time: the 5 s window 2 s after it holds 2.5 s of sim.
        rows += feed(cm, STREAMS, 2.0, sim0=5 * NS, wall0=WALL0 + 7500 * MS)
        self.assertAlmostEqual(cm.rtf(rows[-1][2]), 0.5, delta=0.01)
        # A world reset restarts the window instead of reading negative.
        rows += feed(cm, STREAMS, 0.5, sim0=0, wall0=rows[-1][2])
        self.assertIsNone(cm.rtf(rows[-1][2]))
        rows += feed(cm, STREAMS, 1.5, sim0=rows[-1][1], wall0=rows[-1][2])
        self.assertAlmostEqual(cm.rtf(rows[-1][2]), 1.0, delta=0.01)


class RtfMonitorTest(unittest.TestCase):
    def test_episode(self):
        m = bridge.RtfMonitor(threshold=0.95, hold_s=5.0)
        evs = [m.update(r, float(t)) for t, r in enumerate([1.0, 0.9, 0.9, None, 0.92, 0.9, 0.9,
                                                             0.85, 0.9, 0.99, 0.9])]
        self.assertEqual(evs.count('violation'), 1)
        self.assertEqual(evs[7], 'violation')                # below since t=1, > 5 s at t=7
        self.assertEqual(evs[9], 'recovered')
        self.assertIsNone(m.update(float('nan'), 11.0))

    def test_worst_of_episode(self):
        m = bridge.RtfMonitor(threshold=0.95, hold_s=5.0)
        for t, r in enumerate([0.9, 0.9, 0.85, 0.9, 0.9, 0.9, 0.9]):
            m.update(r, float(t))
        self.assertTrue(m.violating)
        self.assertAlmostEqual(m.worst, 0.85)
        self.assertEqual(m.update(1.0, 7.0), 'recovered')
        self.assertAlmostEqual(m.worst, 0.85)        # still readable for the recovery log

    def test_short_dip_is_not_a_violation(self):
        m = bridge.RtfMonitor()
        evs = [m.update(r, float(t)) for t, r in enumerate([0.5] * 5 + [1.0] + [0.5] * 5)]
        self.assertNotIn('violation', evs)

    def test_violation_log(self):
        import datetime
        when = datetime.datetime(2026, 10, 18, 12, 0, 1, 250000, tzinfo=datetime.timezone.utc)
        line = bridge.violation_line('JR81', 0.9123, when)
        self.assertEqual(line, '2026-10-18T12:00:01.250+00:00 JR81 rtf=0.912\n')
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(bridge.append_violation(d, line))
            self.assertTrue(bridge.append_violation(d, line))
            with open(os.path.join(d, 'rtf_violations.log')) as f:
                self.assertEqual(f.read(), line * 2)
            self.assertFalse(bridge.append_violation(os.path.join(d, 'missing'), line))
        self.assertFalse(bridge.append_violation('', line))


class ArgsTest(unittest.TestCase):
    def test_defaults(self):
        a = bridge.parse_args(['--ns', 'JR81'])
        self.assertEqual((a.ns, a.imu_units, a.min_range, a.max_range, a.slew, a.run_dir, a.input),
                         ('JR81', 'g', 0.1, 40.0, 0.05, '/rover_sim/run', 'pc2'))

    def test_bad(self):
        import contextlib
        import io
        for argv in (['--ns', 'JR81/x'], ['--ns', 'JR81', '--min-range', '50'], []):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bridge.parse_args(argv)


def imu_cdr(frame, stamp=(5, 6), orient=(0.1, 0.2, 0.3, 0.9), gyro=(1.0, -2.0, 3.0),
            accel=(-4.8, 0.0, 8.55), cov=0.5):
    """sensor_msgs/Imu as Fast-CDR (XCDR1 LE) writes it."""
    body = bytearray(struct.pack('<iI', *stamp))
    fb = frame.encode() + b'\x00'
    body += struct.pack('<I', len(fb)) + fb
    while len(body) % 8:
        body.append(0)
    c = [cov] * 9
    body += struct.pack('<4d9d3d9d3d9d', *orient, *c, *gyro, *c, *accel, *c)
    return bytes(b'\x00\x01\x00\x00' + body)


def odom_cdr(frame, child, pose, twist, stamp=(5, 6), cov=0.25):
    """nav_msgs/Odometry as Fast-CDR (XCDR1 LE) writes it."""
    body = bytearray(struct.pack('<iI', *stamp))
    for text in (frame, child):
        while len(body) % 4:
            body.append(0)
        raw = text.encode() + b'\x00'
        body += struct.pack('<I', len(raw)) + raw
    while len(body) % 8:
        body.append(0)
    c = [cov] * 36
    body += struct.pack('<7d36d6d36d', *pose, *c, *twist, *c)
    return bytes(b'\x00\x01\x00\x00' + body)


def rclpy_serializer():
    """(serialize, deserialize) when rclpy and an RMW are usable, else None."""
    try:
        from rclpy.serialization import deserialize_message, serialize_message
        from std_msgs.msg import Header
        serialize_message(Header())
    except Exception:
        return None
    return serialize_message, deserialize_message


class CdrTest(unittest.TestCase):
    def test_imu_parse_every_frame_length(self):
        for frame in ('', 'a', 'JR81/lidar', 'JR81/lidar_x', 'livox_frame', 'x' * 17):
            ns, gyro, accel = bridge.imu_parse(imu_cdr(frame))
            self.assertEqual((ns, gyro, accel), (5 * NS + 6, (1.0, -2.0, 3.0), (-4.8, 0.0, 8.55)))
        with self.assertRaises(ValueError):
            bridge.imu_parse(imu_cdr('JR81/lidar')[:-8])
        with self.assertRaises(ValueError):
            bridge.imu_parse(b'\x01\x00' + imu_cdr('a')[2:])          # big endian

    def test_imu_build_is_the_driver_message(self):
        buf = bridge.imu_build(1760000000 * NS + 42, (0.1, 0.2, 0.3), (-0.49, 0.0, 0.87))
        ref = imu_cdr('livox_frame', stamp=(1760000000, 42), orient=(0.0, 0.0, 0.0, 1.0),
                      gyro=(0.1, 0.2, 0.3), accel=(-0.49, 0.0, 0.87), cov=0.0)
        self.assertEqual(len(buf), 324)
        self.assertEqual(buf, ref)

    def test_restamp(self):
        buf = imu_cdr('world', stamp=(3, 4))
        out = bridge.cdr_restamp(buf, 1760000000 * NS + 7)
        self.assertEqual(bridge.cdr_stamp_ns(out), 1760000000 * NS + 7)
        self.assertEqual(out[12:], buf[12:])
        self.assertEqual(bridge.cdr_stamp_ns(buf), 3 * NS + 4)

    def test_odom_codec(self):
        pose = (1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        for frame, child in (('world', 'chassis'), ('', 'a'), ('world', 'JR81/chassis'), ('abc', 'defgh')):
            buf = odom_cdr(frame, child, pose, (0.5, 0.0, 0.0, 0.0, 0.0, 0.25))
            sec, nsec, f, c, v = bridge.odom_parse(buf)
            self.assertEqual((sec, nsec, f, c), (5, 6, frame, child))
            self.assertEqual(v[:7], pose)
            self.assertEqual(v[43:49], (0.5, 0.0, 0.0, 0.0, 0.0, 0.25))
            self.assertEqual(bridge.odom_build(5 * NS + 6, f, c, v), buf)
        with self.assertRaises(ValueError):
            bridge.odom_parse(odom_cdr('world', 'chassis', pose, (0,) * 6)[:-8])

    def test_odom_twist_into_the_body(self):
        h = math.sqrt(0.5)
        # Heading +90 deg, moving along world +y at 1 m/s and turning at +0.5 rad/s:
        # forward 1 m/s in the body.
        q = (0.0, 0.0, h, h)
        buf = odom_cdr('world', 'chassis', (3.0, 4.0, 0.0) + q, (0.0, 1.0, 0.0, 0.0, 0.0, 0.5))
        out = bridge.odom_to_body(buf, 1760000000 * NS + 7, 'JR81/chassis')
        sec, nsec, f, c, v = bridge.odom_parse(out)
        self.assertEqual((sec, nsec, f, c), (1760000000, 7, 'world', 'JR81/chassis'))
        self.assertEqual(v[:7], (3.0, 4.0, 0.0) + q)
        for got, want in zip(v[43:49], (1.0, 0.0, 0.0, 0.0, 0.0, 0.5)):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(v[7:43], (0.25,) * 36)
        self.assertEqual(v[49:], (0.25,) * 36)
        # Pitched nose-down 30 deg (about +y), heading 0: world (cos30, 0, -sin30) is body +x.
        p = math.radians(30)
        q = (0.0, math.sin(p / 2), 0.0, math.cos(p / 2))
        buf = odom_cdr('world', 'chassis', (0.0, 0.0, 0.0) + q,
                       (math.cos(p), 0.0, -math.sin(p), 0.0, 1.0, 0.0))
        v = bridge.odom_parse(bridge.odom_to_body(buf, 0, 'c'))[4]
        for got, want in zip(v[43:49], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)):
            self.assertAlmostEqual(got, want, places=12)

    def test_against_rclpy(self):
        ser = rclpy_serializer()
        if ser is None:
            self.skipTest('rclpy serializer unavailable (needs ROS sourced and an RMW)')
        serialize, deserialize = ser
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        for frame in ('', 'JR81/lidar', 'abcdefgh'):
            m = Imu()
            m.header.stamp.sec, m.header.stamp.nanosec = 12, 34
            m.header.frame_id = frame
            m.orientation.w = 0.5
            m.angular_velocity.x, m.angular_velocity.z = 0.25, -1.5
            m.linear_acceleration.x, m.linear_acceleration.z = -4.8, 8.6
            m.linear_acceleration_covariance[4] = 0.01
            raw = serialize(m)
            self.assertEqual(bridge.imu_parse(raw), (12 * NS + 34, (0.25, 0.0, -1.5), (-4.8, 0.0, 8.6)))
        out = deserialize(bridge.imu_build(1760000000 * NS + 5, (0.1, 0.2, 0.3), (1.0, 2.0, 3.0)), Imu)
        ref = Imu()                       # lddc.cpp: only stamp, frame, gyro, accel are set
        ref.header.stamp.sec, ref.header.stamp.nanosec = 1760000000, 5
        ref.header.frame_id = 'livox_frame'
        ref.angular_velocity.x, ref.angular_velocity.y, ref.angular_velocity.z = 0.1, 0.2, 0.3
        ref.linear_acceleration.x, ref.linear_acceleration.y, ref.linear_acceleration.z = 1.0, 2.0, 3.0
        self.assertEqual(out, ref)
        self.assertEqual(serialize(ref), bridge.imu_build(1760000000 * NS + 5, (0.1, 0.2, 0.3), (1.0, 2.0, 3.0)))
        o = Odometry()
        o.header.stamp.sec = 9
        o.header.frame_id, o.child_frame_id = 'world', 'chassis'
        o.pose.pose.position.x, o.twist.covariance[35] = 1.25, 0.5
        back = deserialize(bridge.cdr_restamp(serialize(o), 1760000000 * NS + 99), Odometry)
        self.assertEqual((back.header.stamp.sec, back.header.stamp.nanosec), (1760000000, 99))
        back.header.stamp = o.header.stamp
        self.assertEqual(back, o)
        for child in ('chassis', 'JR81/chassis', 'abcdefg'):
            o.child_frame_id = child
            o.pose.pose.orientation.z, o.pose.pose.orientation.w = 1.0, 0.0      # heading 180 deg
            o.twist.twist.linear.x, o.twist.twist.angular.z = -0.75, 0.25
            raw = serialize(o)
            self.assertEqual(bridge.odom_parse(raw)[2:4], ('world', child))
            back = deserialize(bridge.odom_to_body(raw, 1760000000 * NS + 1, 'RR83/chassis'), Odometry)
            self.assertEqual(back.child_frame_id, 'RR83/chassis')
            self.assertAlmostEqual(back.twist.twist.linear.x, 0.75)
            self.assertAlmostEqual(back.twist.twist.angular.z, 0.25)
            self.assertEqual(back.pose, o.pose)
            self.assertEqual(list(back.twist.covariance), list(o.twist.covariance))


@unittest.skipIf(np is None, 'numpy not installed')
class PackingTest(unittest.TestCase):
    def cloud(self, pts, extra=()):
        """A plugin-like PointCloud2 buffer: x, y, z f32 (+ extra f32 channels)."""
        names = ['x', 'y', 'z'] + list(extra)
        arr = np.zeros((len(pts), len(names)), dtype='<f4')
        arr[:, :] = pts
        fields = [(n, 4 * i, bridge.FLOAT32) for i, n in enumerate(names)]
        return arr.tobytes(), fields, 4 * len(names)

    def test_livox_layout(self):
        dt = bridge.livox_dtype()
        self.assertEqual(dt.itemsize, 26)
        self.assertEqual([(n, dt.fields[n][1]) for n in dt.names],
                         [('x', 0), ('y', 4), ('z', 8), ('intensity', 12), ('tag', 16),
                          ('line', 17), ('timestamp', 18)])
        self.assertEqual(dt['timestamp'], np.dtype('<f8'))

    def test_filter_and_pack(self):
        nan, inf = float('nan'), float('inf')
        pts = [[1, 2, 3], [nan, 0, 0], [0, inf, 0], [0.05, 0, 0], [0.1, 0, 0],   # 0.1 = min clamp
               [40, 0, 0], [0, 0, 39.9999], [10, -5, 0.5], [0, 0, -0.36], [-3, 0, 0]]
        data, fields, step = self.cloud(pts)
        cols = bridge.cloud_view(data, fields, step, len(pts))
        keep = bridge.PointFilter(0.1, 40.0).mask(cols['x'], cols['y'], cols['z'])
        self.assertEqual(np.flatnonzero(keep).tolist(), [0, 7, 8, 9])
        stamp = 1760000000123456789
        n, buf = bridge.pack_livox(cols, keep, stamp)
        self.assertEqual((n, len(buf)), (4, 4 * 26))
        rows = [struct.unpack_from('<4f2Bd', buf, 26 * i) for i in range(n)]
        self.assertEqual([r[:3] for r in rows], [(1, 2, 3), (10, -5, 0.5),
                                                 (0, 0, np.float32(-0.36)), (-3, 0, 0)])
        self.assertEqual([r[3:5] for r in rows], [(0, 0)] * 4)          # intensity, tag
        self.assertEqual([r[5] for r in rows], [0, 3, 0, 1])           # input index % 4
        self.assertTrue(all(r[6] == float(stamp) and r[6] > 1e14 for r in rows))

    def test_intensity_and_line_channels(self):
        # x y z f32, intensity f32, ring u8, 3 bytes padding: point_step 20
        dt = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring'],
                       'formats': ['<f4'] * 4 + ['u1'], 'offsets': [0, 4, 8, 12, 16], 'itemsize': 20})
        arr = np.zeros(3, dtype=dt)
        arr['x'] = [1, 2, 3]
        arr['intensity'] = [7, 9, 11]
        arr['ring'] = [2, 1, 3]
        fields = [('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('intensity', 12, 7), ('ring', 16, 2)]
        cols = bridge.cloud_view(arr.tobytes(), fields, 20, 3)
        n, out = bridge.pack_livox(cols, np.array([True, False, True]), 2 * 10 ** 17)
        r = [struct.unpack_from('<4f2Bd', out, 26 * i) for i in range(n)]
        self.assertEqual([(x[0], x[3], x[5]) for x in r], [(1, 7, 2), (3, 11, 3)])

    def test_clamp_below_max_range_detected(self):
        rnd = np.random.default_rng(0)
        d = rnd.normal(size=(2000, 3)).astype('<f4')
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = rnd.uniform(0.5, 15.0, size=2000).astype('<f4')
        r[:300] = 20.0                                                 # xacro max 20, no hit
        pts = d * r[:, None]
        f = bridge.PointFilter(0.1, 40.0)
        keep = f.mask(pts[:, 0], pts[:, 1], pts[:, 2])
        self.assertEqual(int(keep.sum()), 1700)
        self.assertFalse(keep[:300].any())
        self.assertAlmostEqual(f.clamp_seen, 20.0, places=3)
        # A real scene has no such pile: nothing extra dropped, nothing recorded.
        f2 = bridge.PointFilter(0.1, 40.0)
        self.assertEqual(int(f2.mask(pts[300:, 0], pts[300:, 1], pts[300:, 2]).sum()), 1700)
        self.assertIsNone(f2.clamp_seen)

    def test_cloud_view_errors(self):
        data, fields, step = self.cloud([[1, 2, 3]])
        with self.assertRaises(ValueError):
            bridge.cloud_view(data, fields[:2], step, 1)                # no z
        with self.assertRaises(ValueError):
            bridge.cloud_view(data, fields, step, 2)                    # short buffer

    def custom_cdr(self, frame, pts, sec=12, nanosec=345):
        """CustomMsg CDR as Fast-CDR (XCDR1) lays it out."""
        b = bytearray(b'\x00\x01\x00\x00')
        body = bytearray()

        def align(n):
            while len(body) % n:
                body.append(0)
        body += struct.pack('<iI', sec, nanosec)
        fb = frame.encode() + b'\x00'
        body += struct.pack('<I', len(fb)) + fb
        align(8)
        body += struct.pack('<Q', 999)
        align(4)
        body += struct.pack('<I', len(pts)) + b'\x01' + b'\x00\x00\x00'
        align(4)
        body += struct.pack('<I', len(pts))
        for i, (x, y, z) in enumerate(pts):
            align(4)
            body += struct.pack('<I3f3B', i * 10, x, y, z, i % 256, 0, i % 4)
        return bytes(b + body)

    def test_parse_custom_msg(self):
        pts = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (-1.0, 0.5, 0.25)]
        for frame in ('', 'a', 'JR81/lidar', 'RR83/lidar_xyz'):       # every alignment
            sec, nsec, fr, cols = bridge.parse_custom_msg(self.custom_cdr(frame, pts))
            self.assertEqual((sec, nsec, fr), (12, 345, frame))
            self.assertEqual(np.stack([cols['x'], cols['y'], cols['z']], 1).tolist(), [list(p) for p in pts])
            self.assertEqual(cols['line'].tolist(), [0, 1, 2])
            self.assertEqual(cols['intensity'].tolist(), [0, 1, 2])
        _, _, _, cols = bridge.parse_custom_msg(self.custom_cdr('x', []))
        self.assertEqual(cols['x'].size, 0)
        with self.assertRaises(ValueError):
            bridge.parse_custom_msg(self.custom_cdr('JR81/lidar', pts)[:-5])

    def test_parse_custom_msg_matches_rclpy(self):
        ser = rclpy_serializer()
        try:
            from livox_ros_driver2.msg import CustomMsg, CustomPoint
        except ImportError:
            ser = None
        if ser is None:
            self.skipTest('rclpy serializer / livox_ros_driver2 unavailable')
        serialize_message = ser[0]
        m = CustomMsg()
        m.header.stamp.sec, m.header.stamp.nanosec = 77, 123456789
        m.header.frame_id = 'JR81/lidar'
        m.timebase = 5
        for i in range(5):
            m.points.append(CustomPoint(offset_time=i, x=float(i), y=-float(i), z=0.5 * i,
                                        reflectivity=i, tag=0, line=i % 4))
        m.point_num = 5
        buf = serialize_message(m)
        sec, nsec, fr, cols = bridge.parse_custom_msg(buf)
        self.assertEqual((sec, nsec, fr), (77, 123456789, 'JR81/lidar'))
        self.assertEqual(cols['x'].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(cols['z'].tolist(), [0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(cols['line'].tolist(), [0, 1, 2, 3, 0])

    def test_pack_speed(self):
        # 20k points, about what one MID-360 scan holds: well under a ms per scan here.
        import time
        rnd = np.random.default_rng(1)
        pts = rnd.uniform(-20, 20, size=(20000, 3)).astype('<f4')
        data, fields, step = pts.tobytes(), [('x', 0, 7), ('y', 4, 7), ('z', 8, 7)], 12
        f = bridge.PointFilter()
        t0 = time.perf_counter()
        for _ in range(20):
            cols = bridge.cloud_view(data, fields, step, 20000)
            bridge.pack_livox(cols, f.mask(cols['x'], cols['y'], cols['z']), 2 * 10 ** 17)
        per_scan = (time.perf_counter() - t0) / 20
        self.assertLess(per_scan, 0.02)


if __name__ == '__main__':
    unittest.main()
