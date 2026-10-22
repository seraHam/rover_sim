#!/usr/bin/env python3
"""The rover_sim world in Isaac Sim 5.0, in gzserver's place (session `simulator: isaac`).

  /isaac-sim/python.sh isaac_sim.py --world <w>.world --spawn <run>/spawn.yaml --urdf-dir <run>/urdf
                       --classes classes.yaml --package-dir <rover_sim share> --state-dir <run>/isaac
                       --mid360-csv mid360.csv --calib-dir <run>/sensor_calib [--lidar-samples N]
                       [--physics-hz 200] [--urdf-wait 300] [--camera 1280x720|off]

Per agent, what its Gazebo model publishes (urdf/*.xacro), on sim-time stamps, for
livox_sim_bridge.py:
  <ns>/mid360_PointCloud2   x y z intensity f32, frame <ns>/lidar, 10 Hz: the MID-360 pattern
                            (mid360_profile.py) on an RTX lidar, lidar_samples rays a scan
  <ns>/sim/imu_raw          at the lidar origin in its axes, m/s^2 with the gravity reaction,
                            200 Hz, with mid360.xacro's noise
  <ns>/ground_truth/odom    p3d's: the chassis in world, child_frame_id chassis, the twist in
                            world axes, 100 Hz
and it takes <ns>/cmd_vel through the firmware's command handling (config/classes.yaml).
Isaac only, per agent, rendered with the lidar at 10 Hz (--camera WxH, or off): the robot's
RealSense D455 as rover_sensors' d455.launch.yaml runs it, colour and depth aligned to it, where
state_publisher.launch.py mounts <ns>_d455_link (<calib-dir>/<ns>_calib.npy; sim_tf.py --d455 has
the camera's own frames), on the bridge's wall stamps (its sim/time_offset; none before the first):
  <ns>/d455/color/image_raw                     rgb8, 90 deg across
  <ns>/d455/aligned_depth_to_color/image_raw    16UC1 mm, 0 outside the depth imagers' field of
                                                view and range (D455_DEPTH_*); no stereo noise
and their camera_info, frame <ns>_d455_color_optical_frame.

Unlike Gazebo:
- the drive is a push: every physics step a horizontal force at the chassis' centre of mass
  and a yaw torque take the chassis origin (on the floor under the turning centre) half the
  way to the command, within what the wheels' max_wheel_torque can push. Pushed at its COM,
  it does not pitch under acceleration as a robot driven at the floor does; the wheels are
  frictionless and do not turn; PhysX does the rest (gravity, contacts);
- the IMU is the lidar origin's motion (its velocity differenced over a physics step, plus
  the gravity reaction, in the lidar's axes), not a physics sensor;
- the world is its boxes (demo_eval.world_boxes) on a ground plane.

The world starts paused. Once every agent is in it writes <state-dir>/spawned, which the
spawn pane prints; it runs from when <state-dir>/unpause appears (sim.py up), and writes
<state-dir>/running. Sim time is held to the wall clock as real_time_lock.cpp holds Gazebo's:
never ahead, catching up at most 1.5x, a lag past 0.5 s given up.

The pure parts (Drive, ImuNoise, Pacer, the URDF, motion and CDR helpers) import without
Isaac; tests/test_isaac_sim.py covers them.
"""

import argparse
import math
import os
import random
import re
import struct
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import livox_sim_bridge as bridge
import mid360_profile
import sim_tf

G0 = 9.80665
IMU_HZ = 200
ODOM_HZ = 100
SPAWN_Z = 0.005                  # spawn_agent.launch.py's
# mid360.xacro: stddev per sample and bias mean; Gazebo draws each bias once, bias_stddev bias/4
GYRO_NOISE, GYRO_BIAS = 2e-3, 5e-4
ACCEL_NOISE, ACCEL_BIAS = 2e-2, 1e-2
DRIVE_GAIN = 0.5                 # of the velocity error the drive closes per physics step
WHEEL_JOINT_RE = re.compile(r'_wheel_joint$')
CAMERA_APERTURE_MM = 20.955         # USD's default horizontal aperture
D455_COLOR_HFOV_DEG = 90.0          # the colour imager's
D455_DEPTH_FOV_DEG = (87.0, 58.0)   # the depth imagers'
D455_DEPTH_RANGE_M = (0.52, 20.0)   # min-z at 848x480, and Intel's stated maximum
# USD cameras look along -z with +y up, optical frames along +z with +y down
USD_FROM_OPTICAL = (0.0, 1.0, 0.0, 0.0)
ROS_LIB = '/isaac-sim/exts/isaacsim.ros2.bridge/%s/rclpy'
# The boxes' colours, in turn, so neighbours tell apart on camera (the worlds give none Isaac reads)
BOX_COLORS = ((0.55, 0.42, 0.30), (0.30, 0.42, 0.55), (0.45, 0.52, 0.35), (0.60, 0.55, 0.45), (0.50, 0.35, 0.40))


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


class Drive:
    """gazebo_ros_diff_drive's command handling: v and w clamped, v slewed at `accel` up and
    down from its own last value (w is not), both zero once no command came for `timeout` s."""

    def __init__(self, v_max, omega_max, accel, timeout):
        self.v_max, self.omega_max, self.accel, self.timeout = v_max, omega_max, accel, timeout
        self.v_cmd = self.w_cmd = self.v = self.w = 0.0
        self.last = None

    @classmethod
    def from_class(cls, cfg):
        return cls(cfg['v_max'], cfg['omega_max'], cfg['linear_acceleration'], cfg['command_timeout'])

    def command(self, v, w, t):
        self.v_cmd = clamp(v, -self.v_max, self.v_max)
        self.w_cmd = clamp(w, -self.omega_max, self.omega_max)
        self.last = t

    def update(self, t, dt):
        if self.last is None or t - self.last > self.timeout:
            self.v_cmd = self.w_cmd = 0.0
        step = self.accel * dt
        self.v += clamp(self.v_cmd - self.v, -step, step)
        self.w = self.w_cmd
        return self.v, self.w


class ImuNoise:
    """Gazebo's GaussianNoiseModel per axis: each bias drawn once from N(bias, bias/4) with a
    random sign, then every sample gets it plus N(0, stddev)."""

    def __init__(self, stddev, bias, rng, axes=3):
        self.stddev, self.rng = stddev, rng
        self.bias = []
        for _ in range(axes):
            b = rng.gauss(bias, bias / 4)
            self.bias.append(-b if rng.random() < 0.5 else b)

    def apply(self, values):
        return [v + b + self.rng.gauss(0.0, self.stddev) for v, b in zip(values, self.bias)]


class Pacer:
    """real_time_lock.cpp's hold on sim time: a step ending at sim time s runs no earlier than
    the wall time s maps to, and no faster than max_rate x real time while catching up; a lag
    past max_lag is given up (the mapping moves, `lost` adds it up)."""

    def __init__(self, max_rate=1.5, max_lag=0.5):
        self.max_rate, self.max_lag = max_rate, max_lag
        self.offset = self.wall = self.sim = None
        self.lost = 0.0

    def start(self, wall, sim):
        self.offset, self.wall, self.sim = wall - sim, wall, sim

    def delay(self, wall, sim):
        """Seconds to wait before the step that ends at sim time `sim`."""
        due = sim + self.offset
        if wall - due > self.max_lag:
            self.lost += wall - due
            self.offset, due = wall - sim, wall
        at = max(due, self.wall + (sim - self.sim) / self.max_rate, wall)
        self.wall, self.sim = at, sim
        return at - wall


# ---------------------------------------------------------------------------- URDF

def resolve_packages(text, packages):
    """package://<pkg>/ -> <dir>/ for each package in `packages` ({name: dir}); any other
    package is an error (the importer would leave the mesh out without a word)."""
    def sub(m):
        if m.group(1) not in packages:
            raise ValueError('package://%s/: no directory for it' % m.group(1))
        return packages[m.group(1)].rstrip('/') + '/'
    return re.sub(r'package://([^/"\s]+)/', sub, text)


def rename_robot(text, name):
    """The <robot name> the importer names the prim after: two agents of a class collide."""
    out, n = re.subn(r'<robot(\s[^>]*?)?\sname="[^"]*"', lambda m: '<robot%s name="%s"' % (m.group(1) or '', name),
                     text, count=1)
    if n != 1:
        raise ValueError('no <robot name="..."> in the URDF')
    return out


def _floats(el, attr, n=3):
    v = el.get(attr) if el is not None else None
    return tuple(float(x) for x in v.split()) if v else (0.0,) * n


def link_com(text, link):
    """A URDF link's <inertial><origin xyz>, (0, 0, 0) when it has none."""
    for el in ET.fromstring(text).findall('link'):
        if el.get('name') == link:
            return _floats(el.find('inertial/origin'), 'xyz')
    raise ValueError('no link %r in the URDF' % link)


def body_inertia(text, root):
    """(mass, centre of mass, yaw inertia about root's z axis) of the whole URDF in root's
    frame, every link rigidly placed."""
    mass = izz = 0.0
    com = [0.0, 0.0, 0.0]
    for el in ET.fromstring(text).findall('link'):
        i = el.find('inertial')
        if i is None:
            continue
        m = float(i.find('mass').get('value'))
        t, q = link_transform(text, root, el.get('name'))
        c = tuple(a + b for a, b in zip(t, rotate(q, _floats(i.find('origin'), 'xyz'))))
        mass += m
        com = [a + m * b for a, b in zip(com, c)]
        izz += float(i.find('inertia').get('izz', 0.0)) + m * (c[0] ** 2 + c[1] ** 2)
    return mass, tuple(c / mass for c in com), izz


def wheel_count(text):
    return sum(1 for j in ET.fromstring(text).findall('joint') if WHEEL_JOINT_RE.search(j.get('name', '')))


def link_transform(text, parent, child):
    """(t, q) of `child` in `parent`'s frame, through the joints' <origin>s (rpy fixed-axis XYZ)."""
    joints = {j.find('child').get('link'): j for j in ET.fromstring(text).findall('joint')}
    t, q, link = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), child
    while link != parent:
        j = joints.get(link)
        if j is None:
            raise ValueError('no joint chain from %s to %s' % (parent, child))
        o = j.find('origin')
        jt, jq = _floats(o, 'xyz'), quat_rpy(*_floats(o, 'rpy'))
        t, q = tuple(a + b for a, b in zip(jt, rotate(jq, t))), quat_mul(jq, q)
        link = j.find('parent').get('link')
    return t, q


# ---------------------------------------------------------------------------- motion

def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def rotate(q, v):
    """v by the unit quaternion q = (w, x, y, z)."""
    w, u = q[0], q[1:]
    t = tuple(2 * c for c in cross(u, v))
    c = cross(u, t)
    return tuple(v[i] + w * t[i] + c[i] for i in range(3))


def rotate_inv(q, v):
    return rotate((q[0], -q[1], -q[2], -q[3]), v)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def quat_rpy(r, p, y):
    """URDF rpy: about the fixed x, then y, then z."""
    def axis(i, a):
        q = [math.cos(a / 2), 0.0, 0.0, 0.0]
        q[i] = math.sin(a / 2)
        return tuple(q)
    return quat_mul(axis(3, y), quat_mul(axis(2, p), axis(1, r)))


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def drive_push(q, v0, wz, v, w, mass, izz, f_max, dt, gain=DRIVE_GAIN):
    """(force, yaw torque), world axes: what takes the chassis origin's horizontal velocity v0
    and yaw rate wz `gain` of the way to v along its heading and w in one step of dt, the
    force at most f_max (the wheels' traction) and the torque at most f_max times the
    wheelbase radius sqrt(izz / mass)."""
    yaw = yaw_of(q)
    fx = gain * mass * (v * math.cos(yaw) - v0[0]) / dt
    fy = gain * mass * (v * math.sin(yaw) - v0[1]) / dt
    n = math.hypot(fx, fy)
    if n > f_max:
        fx, fy = fx * f_max / n, fy * f_max / n
    t_max = f_max * math.sqrt(izz / mass)
    return (fx, fy, 0.0), clamp(gain * izz * (w - wz) / dt, -t_max, t_max)


def point_velocity(v, ang, r):
    """Velocity of the point at r from the one moving at v, on a body turning at ang."""
    return tuple(a + b for a, b in zip(v, cross(ang, r)))


def imu_sample(q_body, ang, v_now, v_before, dt, q_mount):
    """(specific force, rate, orientation) of an IMU mounted at q_mount on a body at q_body
    turning at ang (world), its point's velocity v_before then v_now dt later: what an
    accelerometer (gravity reaction included) and a gyro read, in the IMU's axes."""
    q = quat_mul(q_body, q_mount)
    f = tuple((a - b) / dt for a, b in zip(v_now, v_before))
    return rotate_inv(q, (f[0], f[1], f[2] + G0)), rotate_inv(q, ang), q


# ---------------------------------------------------------------------------- CDR

_IMU_TAIL = struct.Struct('<4d9d3d9d3d9d')


def imu_cdr(stamp_ns, frame, q, gyro, accel):
    """Serialized sensor_msgs/Imu (q = (w, x, y, z)); covariances zero."""
    sec, nsec = divmod(int(stamp_ns), bridge.NS_PER_S)
    out = bytearray(bridge.CDR_LE + struct.pack('<iI', sec, nsec))
    bridge._cdr_put_string(out, frame)
    out += b'\0' * (bridge._align(len(out) - 4, 8) - (len(out) - 4))
    zero = (0.0,) * 9
    return bytes(out + _IMU_TAIL.pack(q[1], q[2], q[3], q[0], *zero, *gyro, *zero, *accel, *zero))


def odom_cdr(stamp_ns, pos, q, lin, ang):
    """gazebo_ros_p3d's Odometry: frame world, child chassis, the twist in world axes."""
    return bridge.odom_build(stamp_ns, 'world', 'chassis', tuple(pos) + (q[1], q[2], q[3], q[0])
                             + (0.0,) * 36 + tuple(lin) + tuple(ang) + (0.0,) * 36)


def camera_size(text):
    """'640x480' -> (640, 480); 'off' -> None."""
    if text == 'off':
        return None
    m = re.fullmatch(r'([1-9][0-9]*)x([1-9][0-9]*)', text)
    if not m:
        raise argparse.ArgumentTypeError('%r: expected WxH (640x480) or off' % text)
    return int(m.group(1)), int(m.group(2))


def camera_k(width, height, hfov_deg=D455_COLOR_HFOV_DEG):
    """(fx, fy, cx, cy) of a pinhole camera with square pixels and that horizontal field of view."""
    f = width / 2 / math.tan(math.radians(hfov_deg) / 2)
    return f, f, width / 2, height / 2


def read_calib(path):
    """A sensor_calib <ns>_calib.npy: base_link -> <ns>_d455_link as x y z roll pitch yaw."""
    import numpy as np
    v = [float(x) for x in np.load(path).reshape(-1)]
    if len(v) != 6:
        raise ValueError('%s: %d values, expected x y z roll pitch yaw' % (path, len(v)))
    return v


def d455_camera_pose(calib):
    """(t, q) in base_link of the USD camera that sees what <ns>_d455_color_optical_frame does,
    <ns>_d455_link at calib as state_publisher.launch.py puts it."""
    link_q = quat_rpy(*calib[3:])
    x, y, z, w = sim_tf.D455_OPTICAL_Q
    t = tuple(a + b for a, b in zip(calib[:3], rotate(link_q, sim_tf.D455_COLOR_T)))
    return t, quat_mul(link_q, quat_mul((w, x, y, z), USD_FROM_OPTICAL))


def depth_mask(width, height, k, fov_deg=D455_DEPTH_FOV_DEG):
    """(height, width) bool: the pixels whose rays the depth imagers see."""
    import numpy as np
    fx, fy, cx, cy = k
    u = np.abs(np.arange(width) + 0.5 - cx) / fx <= math.tan(math.radians(fov_deg[0]) / 2)
    v = np.abs(np.arange(height) + 0.5 - cy) / fy <= math.tan(math.radians(fov_deg[1]) / 2)
    return v[:, None] & u[None, :]


def depth_mm(z, mask, depth_range=D455_DEPTH_RANGE_M):
    """distance_to_image_plane (m, inf where nothing) -> 16UC1 mm, 0 where the D455 has no depth."""
    import numpy as np
    ok = mask & (z >= depth_range[0]) & (z <= depth_range[1])
    return np.where(ok, np.rint(np.where(ok, z, 0.0) * 1000.0), 0).astype(np.uint16)


def stamp_ns(t):
    return int(round(t * 1e9))


def wait_for(paths, timeout, poll=0.5):
    t0 = time.time()
    while True:
        missing = [p for p in paths if not os.path.isfile(p)]
        if not missing:
            return
        if time.time() - t0 > timeout:
            raise SystemExit('[isaac_sim] still missing after %.0f s: %s' % (timeout, ', '.join(missing)))
        time.sleep(poll)


def write_state(state_dir, name, text):
    tmp = os.path.join(state_dir, '.%s.tmp' % name)
    with open(tmp, 'w') as f:
        f.write(text + '\n')
    os.replace(tmp, os.path.join(state_dir, name))


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('--world', '--spawn', '--urdf-dir', '--classes', '--package-dir', '--state-dir', '--mid360-csv',
              '--calib-dir'):
        ap.add_argument(k, required=True)
    ap.add_argument('--lidar-samples', type=int, default=mid360_profile.FRAME_POINTS)
    ap.add_argument('--physics-hz', type=int, default=IMU_HZ)
    ap.add_argument('--urdf-wait', type=float, default=300.0)
    ap.add_argument('--seed', type=int, default=None, help='IMU noise seed (default: random, as Gazebo)')
    ap.add_argument('--camera', type=camera_size, default=(1280, 720),
                    help="the D455's WxH, or off (default 1280x720, d455_image_transport.wifi.yaml's colour)")
    args = ap.parse_args(argv)
    for hz in (IMU_HZ, ODOM_HZ, mid360_profile.SCAN_RATE_HZ):
        if args.physics_hz % hz:
            ap.error('--physics-hz %d: needs to be a multiple of %d' % (args.physics_hz, hz))
    return args


# ---------------------------------------------------------------------------- Isaac

class Agent:
    """One robot: its URDF geometry, drive, IMU noise and last state."""

    def __init__(self, name, cls, cfg, text, rng):
        self.name, self.cls, self.cfg = name, cls, cfg
        self.com = link_com(text, 'chassis')
        self.mass, self.body_com, self.izz = body_inertia(text, 'chassis')
        self.f_max = wheel_count(text) * cfg['max_wheel_torque'] / cfg['wheel_radius']
        self.lidar_t, self.lidar_q = link_transform(text, 'chassis', 'lidar')
        self.drive = Drive.from_class(cfg)
        self.gyro = ImuNoise(GYRO_NOISE, GYRO_BIAS, rng)
        self.accel = ImuNoise(ACCEL_NOISE, ACCEL_BIAS, rng)
        self.cmds = []
        self.v_imu = None
        self.last_scan = None
        self.rgb = self.depth = None
        self.time_offset = None         # the bridge's wall - sim, s


class IsaacWorld:

    def __init__(self, args, poses, classes_of, classes, boxes, states):
        import carb
        import numpy as np
        import omni.kit.commands
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid, GroundPlane
        from isaacsim.core.prims import Articulation, RigidPrim, SingleXFormPrim
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
        enable_extension('isaacsim.asset.importer.urdf')
        enable_extension('omni.physx.fabric')
        from isaacsim.asset.importer.urdf import _urdf
        settings = carb.settings.get_settings()
        # PhysX on this thread (its worker threads cost more to sync than a few robots take to
        # solve), writing poses to Fabric, which RTX renders from, not back to USD every step
        settings.set_int('/persistent/physics/numThreads', 0)
        for k in ('/physics/updateToUsd', '/physics/updateVelocitiesToUsd', '/physics/updateParticlesToUsd'):
            settings.set_bool(k, False)
        settings.set_bool('/physics/collisionApproximateCylinders', False)   # round wheels, not polygons
        self.np, self.Vt = np, Vt
        self.states = states
        self.dt = 1.0 / args.physics_hz
        self.world = World(physics_dt=self.dt, rendering_dt=1.0 / mid360_profile.SCAN_RATE_HZ,
                           stage_units_in_meters=1.0)
        physics = self.world.get_physics_context()
        physics.enable_gpu_dynamics(False)
        physics.set_broadphase_type('MBP')
        stage = omni.usd.get_context().get_stage()
        world_prim = UsdGeom.Xform.Define(stage, '/World').GetPrim()
        GroundPlane('/World/ground', size=400.0, color=np.array([0.28, 0.28, 0.27]))
        UsdLux.DomeLight.Define(stage, '/World/sky').CreateIntensityAttr(250.0)
        sun = UsdLux.DistantLight.Define(stage, '/World/sun')
        sun.CreateIntensityAttr(1200.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(40.0, 0.0, 30.0))
        for i, b in enumerate(boxes):
            FixedCuboid('/World/boxes/b%03d' % i, position=np.array([b.cx, b.cy, b.cz]),
                        orientation=np.array(yaw_quat(b.yaw)), scale=np.array([b.sx, b.sy, b.sz]),
                        color=np.array(BOX_COLORS[i % len(BOX_COLORS)]))

        slip = UsdShade.Material.Define(stage, '/World/Looks/frictionless')
        m = UsdPhysics.MaterialAPI.Apply(slip.GetPrim())
        m.CreateStaticFrictionAttr(0.0)
        m.CreateDynamicFrictionAttr(0.0)
        m.CreateRestitutionAttr(0.0)
        px = PhysxSchema.PhysxMaterialAPI.Apply(slip.GetPrim())
        px.CreateFrictionCombineModeAttr('min')
        px.CreateRestitutionCombineModeAttr('min')

        _, cfg = omni.kit.commands.execute('URDFCreateImportConfig')
        cfg.merge_fixed_joints = False
        cfg.fix_base = False
        cfg.import_inertia_tensor = True
        cfg.make_default_prim = False
        cfg.create_physics_scene = False
        cfg.self_collision = False
        cfg.distance_scale = 1.0
        cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_NONE

        os.makedirs(os.path.join(args.state_dir, 'urdf'), exist_ok=True)
        profile = os.path.join(args.state_dir, 'mid360.usda')
        with open(profile, 'w') as f:
            f.write(mid360_profile.render_usda(states[:1]))
        rng = random.Random(args.seed)
        self.agents, by_class = [], {}
        for p in poses:
            cls = classes_of[p.name]
            with open(os.path.join(args.urdf_dir, p.name + '.urdf')) as f:
                text = f.read()
            a = Agent(p.name, cls, classes[cls], text, rng)
            urdf = os.path.join(args.state_dir, 'urdf', p.name + '.urdf')
            with open(urdf, 'w') as f:
                f.write(rename_robot(resolve_packages(text, {'rover_sim': args.package_dir}), p.name))
            stage.SetDefaultPrim(world_prim)   # the importer puts the robot under it, and moves it
            _, a.root = omni.kit.commands.execute('URDFParseAndImportFile', urdf_path=urdf, import_config=cfg)
            SingleXFormPrim(a.root).set_world_pose(position=np.array([p.x, p.y, SPAWN_Z]),
                                                   orientation=np.array(yaw_quat(p.yaw)))
            links = {q.GetName(): q for q in Usd.PrimRange(stage.GetPrimAtPath(a.root))
                     if q.HasAPI(UsdPhysics.RigidBodyAPI)}
            for q in Usd.PrimRange(stage.GetPrimAtPath(a.root)):
                if q.IsInstanceable():                  # the importer's colliders are instance proxies
                    q.SetInstanceable(False)
            colliders = [q for q in Usd.PrimRange(stage.GetPrimAtPath(a.root)) if q.HasAPI(UsdPhysics.CollisionAPI)]
            for q in Usd.PrimRange(stage.GetPrimAtPath(a.root)):
                if q.HasAPI(UsdPhysics.ArticulationRootAPI):    # never asleep: the IMU reads every step
                    art = PhysxSchema.PhysxArticulationAPI.Apply(q)
                    art.CreateSleepThresholdAttr(0.0)
                    art.CreateStabilizationThresholdAttr(0.0)
            for q in colliders:
                UsdShade.MaterialBindingAPI.Apply(q).Bind(slip, UsdShade.Tokens.weakerThanDescendants, 'physics')
            print('[isaac_sim] %s: %d colliders made frictionless' % (p.name, len(colliders)), flush=True)
            lidar = stage.DefinePrim(links['lidar'].GetPath().AppendChild('MID360'), 'OmniLidar')
            lidar.GetReferences().AddReference(profile)
            rp = rep.create.render_product(str(lidar.GetPath()), [1, 1], name=p.name + '_mid360',
                                           render_vars=['GenericModelOutput', 'RtxSensorMetadata'])
            a.annotator = rep.AnnotatorRegistry.get_annotator('IsaacExtractRTXSensorPointCloudNoAccumulator')
            a.annotator.attach([rp.path])
            if args.camera:
                w, h = args.camera
                t, q = d455_camera_pose(read_calib(os.path.join(args.calib_dir, p.name + '_calib.npy')))
                cam = UsdGeom.Camera.Define(stage, links['base_link'].GetPath().AppendChild('d455_color'))
                xf = UsdGeom.Xformable(cam)
                xf.AddTranslateOp().Set(Gf.Vec3d(*t))
                xf.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
                cam.CreateHorizontalApertureAttr(CAMERA_APERTURE_MM)
                cam.CreateVerticalApertureAttr(CAMERA_APERTURE_MM * h / w)
                cam.CreateFocalLengthAttr(CAMERA_APERTURE_MM / 2 / math.tan(math.radians(D455_COLOR_HFOV_DEG) / 2))
                cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
                rpc = rep.create.render_product(str(cam.GetPath()), (w, h), name=p.name + '_d455')
                a.rgb = rep.AnnotatorRegistry.get_annotator('rgb')
                a.rgb.attach([rpc.path])
                a.depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
                a.depth.attach([rpc.path])
                a.k = camera_k(w, h)
                a.depth_mask = depth_mask(w, h, a.k)
                print('[isaac_sim] %s: D455 at (%.3f, %.3f, %.3f) in base_link, %dx%d'
                      % ((p.name,) + t + (w, h)), flush=True)
            a.az = lidar.GetAttribute('omni:sensor:Core:emitterState:s001:azimuthDeg')
            a.el = lidar.GetAttribute('omni:sensor:Core:emitterState:s001:elevationDeg')
            self.agents.append(a)
            by_class.setdefault(cls, []).append(a)
        self.groups = []
        for cls, agents in by_class.items():
            view = Articulation([a.root for a in agents], name='agents_' + cls)
            chassis = RigidPrim([a.root + '/chassis' for a in agents], name='chassis_' + cls)
            self.world.scene.add(view)
            self.world.scene.add(chassis)
            self.groups.append((view, chassis, agents))
        self.world.reset()
        for view, chassis, agents in self.groups:
            want = [a.root + '/chassis' for a in agents]
            if list(view.prim_paths) != want or list(chassis.prim_paths) != want:
                raise RuntimeError('views out of order: %s, %s, want %s' % (view.prim_paths, chassis.prim_paths, want))
        self.world.pause()
        self.state = 0
        self.k = 0

    def sense_and_drive(self, t):
        """Each agent's state at t (a.truth, a.imu) and its drive's push over the next step."""
        np = self.np
        for view, chassis, agents in self.groups:
            pos, quat = view.get_world_poses()
            vel = view.get_velocities()
            forces = np.zeros((len(agents), 3), dtype=np.float32)
            torques = np.zeros((len(agents), 3), dtype=np.float32)
            at = np.zeros((len(agents), 3), dtype=np.float32)
            for i, a in enumerate(agents):
                q = tuple(float(x) for x in quat[i])
                lin, ang = tuple(float(x) for x in vel[i, :3]), tuple(float(x) for x in vel[i, 3:])
                r_com = rotate(q, a.com)
                v0 = point_velocity(lin, ang, tuple(-x for x in r_com))
                a.truth = (tuple(float(x) for x in pos[i]), q, v0, ang)
                v_imu = point_velocity(v0, ang, rotate(q, a.lidar_t))
                a.imu = imu_sample(q, ang, v_imu, a.v_imu if a.v_imu else v_imu, self.dt, a.lidar_q)
                a.v_imu = v_imu
                for v, w in a.cmds:
                    a.drive.command(v, w, t)
                a.cmds = []
                v, w = a.drive.update(t, self.dt)
                forces[i], torques[i, 2] = drive_push(q, v0, ang[2], v, w, a.mass, a.izz, a.f_max, self.dt)
                at[i] = tuple(float(x) for x in pos[i]) + np.array(rotate(q, a.body_com))
            chassis.apply_forces_and_torques_at_pos(forces=forces, torques=torques, positions=at, is_global=True)

    def step(self):
        self.world.step(render=False)
        self.k += 1

    def render(self):
        self.state = (self.state + 1) % len(self.states)
        az = self.Vt.FloatArray([-x for x, _ in self.states[self.state]])
        el = self.Vt.FloatArray([e for _, e in self.states[self.state]])
        for a in self.agents:
            a.az.Set(az)
            a.el.Set(el)
        self.world.render()

    def image(self, a):
        """The D455's last frame: colour (h, w, 3) uint8 and distance to the image plane (h, w)
        float32 m, copies; or None."""
        if a.rgb is None:
            return None
        rgb, z = a.rgb.get_data(), a.depth.get_data()
        if any(d is None or getattr(d, 'size', 0) == 0 for d in (rgb, z)):
            return None
        np = self.np
        rgb = np.ascontiguousarray(np.asarray(rgb)[:, :, :3])
        return rgb, np.array(z, dtype=np.float32).reshape(rgb.shape[:2])

    def scan(self, a):
        d = a.annotator.get_data()
        if not d or 'data' not in d or len(d['data']) == 0:
            return None
        p = self.np.asarray(d['data'], dtype=self.np.float32).reshape(-1, 3)
        key = (len(p), p[0].tobytes())
        if key == a.last_scan:
            return None
        a.last_scan = key
        return p


class Ros:
    """rclpy (Isaac's bundled Humble) on the session's rmw: one node, publishers per agent
    (IMU and ground truth as serialized CDR), cmd_vel taken on a spin thread and applied at
    the next physics step. The publishers are best effort, as the bridge subscribes: a
    reliable one blocks in rmw_zenoh while delivery stalls (a session leaving the graph), and
    with it the physics loop (2.6 s once, live). D455 frames go out from a thread of their own,
    the latest per agent, so packing megabytes a frame stays off the physics loop."""

    def __init__(self, agents):
        sys.path.insert(0, ROS_LIB % os.environ.get('ROS_DISTRO', 'humble'))
        import threading
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
        from std_msgs.msg import Float64
        rclpy.init()
        self.PointCloud2, self.PointField, self.Image, self.CameraInfo = PointCloud2, PointField, Image, CameraInfo
        self.node = rclpy.create_node('isaac_sim')
        self.pub, self.cam = {}, {}
        for a in agents:
            self.pub[a.name] = tuple(self.node.create_publisher(t, '/%s/%s' % (a.name, topic), qos_profile_sensor_data)
                                     for t, topic in ((PointCloud2, 'mid360_PointCloud2'), (Imu, 'sim/imu_raw'),
                                                      (Odometry, 'ground_truth/odom')))
            self.cam[a.name] = tuple(self.node.create_publisher(t, '/%s/d455/%s' % (a.name, topic),
                                                                qos_profile_sensor_data)
                                     for t, topic in ((Image, 'color/image_raw'), (CameraInfo, 'color/camera_info'),
                                                      (Image, 'aligned_depth_to_color/image_raw'),
                                                      (CameraInfo, 'aligned_depth_to_color/camera_info')))
            self.node.create_subscription(Twist, '/%s/cmd_vel' % a.name,
                                          lambda m, a=a: a.cmds.append((m.linear.x, m.angular.z)), 10)
            self.node.create_subscription(Float64, '/%s/sim/time_offset' % a.name,
                                          lambda m, a=a: setattr(a, 'time_offset', m.data), qos_profile_sensor_data)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        self.frames, self.frames_ready = {}, threading.Condition()
        threading.Thread(target=self._publish_frames, daemon=True).start()

    def cloud(self, a, t, xyz):
        import array
        import numpy as np
        m = self.PointCloud2()
        m.header.stamp.sec, m.header.stamp.nanosec = divmod(stamp_ns(t), bridge.NS_PER_S)
        m.header.frame_id = '%s/lidar' % a.name
        pts = np.zeros((len(xyz), 4), dtype=np.float32)
        pts[:, :3] = xyz
        m.height, m.width = 1, len(xyz)
        m.fields = [self.PointField(name=n, offset=4 * i, datatype=self.PointField.FLOAT32, count=1)
                    for i, n in enumerate(('x', 'y', 'z', 'intensity'))]
        m.is_bigendian, m.point_step, m.row_step, m.is_dense = False, 16, 16 * len(xyz), True
        m.data = array.array('B', pts.tobytes())
        self.pub[a.name][0].publish(m)

    def image(self, a, t, rgb, z):
        """Queues the D455 frame rendered at sim time t, on the bridge's wall stamp."""
        if a.time_offset is None:
            return
        with self.frames_ready:
            self.frames[a.name] = (a, stamp_ns(t) + int(round(a.time_offset * 1e9)), rgb, z)
            self.frames_ready.notify()

    def _publish_frames(self):
        while True:
            with self.frames_ready:
                while not self.frames:
                    self.frames_ready.wait()
                frames, self.frames = list(self.frames.values()), {}
            for a, stamp, rgb, z in frames:
                self._publish_frame(a, stamp, rgb, z)

    def _publish_frame(self, a, stamp, rgb, z):
        import array
        h, w = rgb.shape[:2]
        fx, fy, cx, cy = a.k
        pubs = self.cam[a.name]
        for (img, info), enc, data in ((pubs[:2], 'rgb8', rgb), (pubs[2:], '16UC1', depth_mm(z, a.depth_mask))):
            m = self.Image()
            m.header.stamp.sec, m.header.stamp.nanosec = divmod(stamp, bridge.NS_PER_S)
            m.header.frame_id = '%s_d455_color_optical_frame' % a.name
            m.height, m.width, m.encoding, m.is_bigendian, m.step = h, w, enc, 0, data.strides[0]
            m.data = array.array('B', data.tobytes())
            img.publish(m)
            i = self.CameraInfo()
            i.header = m.header
            i.height, i.width, i.distortion_model = h, w, 'plumb_bob'
            i.d = [0.0] * 5
            i.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            i.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            i.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            info.publish(i)

    def imu(self, a, t):
        accel, gyro, q = a.imu
        self.pub[a.name][1].publish(imu_cdr(stamp_ns(t), '%s/lidar' % a.name, q, a.gyro.apply(gyro),
                                            a.accel.apply(accel)))

    def odom(self, a, t):
        self.pub[a.name][2].publish(odom_cdr(stamp_ns(t), *a.truth))


def main(argv=None):
    args = parse_args(argv)
    import yaml
    import demo_eval
    import sim_tf
    with open(args.spawn) as f:
        spawn = yaml.safe_load(f)
    poses = sim_tf.spawn_poses(spawn, args.spawn)
    classes_of = {a['name']: a['class'] for a in spawn['agents']}
    with open(args.classes) as f:
        classes = yaml.safe_load(f)
    with open(args.world) as f:
        boxes = demo_eval.world_boxes(f.read(), top=math.inf)
    states = mid360_profile.emitter_states(mid360_profile.read_pattern(args.mid360_csv), args.lidar_samples)
    os.makedirs(args.state_dir, exist_ok=True)
    wait_for([os.path.join(args.urdf_dir, p.name + '.urdf') for p in poses], args.urdf_wait)
    print('[isaac_sim] %s: %d boxes, %s; MID-360 %d rays a scan, %d scans in the pattern, physics %d Hz'
          % (args.world, len(boxes), ', '.join('%s (%s)' % (p.name, classes_of[p.name]) for p in poses),
             len(states[0]), len(states), args.physics_hz), flush=True)

    print('[isaac_sim] starting Isaac Sim: ~25 s, or ~3 min the first time while it compiles shaders '
          '(it prints nothing meanwhile; its caches are compose volumes)', flush=True)
    t0 = time.time()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    try:
        print('[isaac_sim] Isaac Sim up after %.0f s; building the world' % (time.time() - t0), flush=True)
        sim = IsaacWorld(args, poses, classes_of, classes, boxes, states)
        ros = Ros(sim.agents)
        write_state(args.state_dir, 'spawned', ' '.join(p.name for p in poses))
        print('[isaac_sim] spawned %s into the paused world; waiting for %s/unpause'
              % (' '.join(p.name for p in poses), args.state_dir), flush=True)
        while not os.path.exists(os.path.join(args.state_dir, 'unpause')):
            time.sleep(0.2)
        run(args, sim, ros, app)
    except Exception:
        traceback.print_exc()   # app.close() exits the process before Python would print it
        sys.stdout.flush()
        raise
    finally:
        app.close()


def run(args, sim, ros, app):
    imu_every, odom_every = args.physics_hz // IMU_HZ, args.physics_hz // ODOM_HZ
    scan_every = args.physics_hz // mid360_profile.SCAN_RATE_HZ
    sim.world.play()
    pacer = Pacer()
    pacer.start(time.time(), 0.0)
    write_state(args.state_dir, 'running', '%.3f' % time.time())
    print('[isaac_sim] running', flush=True)
    report = time.time() + 10.0
    while app.is_running():
        t = sim.k * sim.dt
        sim.sense_and_drive(t)
        if sim.k % imu_every == 0:
            for a in sim.agents:
                ros.imu(a, t)
        if sim.k % odom_every == 0:
            for a in sim.agents:
                ros.odom(a, t)
        time.sleep(pacer.delay(time.time(), t + sim.dt))
        sim.step()
        if sim.k % scan_every == 0:
            sim.render()
            for a in sim.agents:
                xyz = sim.scan(a)
                if xyz is not None:
                    ros.cloud(a, sim.k * sim.dt, xyz)
                frame = sim.image(a)
                if frame is not None:
                    ros.image(a, sim.k * sim.dt, *frame)
        if time.time() > report:
            print('[isaac_sim] sim %.1f s, %.2f s lost to lag' % (sim.k * sim.dt, pacer.lost), flush=True)
            report = time.time() + 10.0


if __name__ == '__main__':
    sys.exit(main())
