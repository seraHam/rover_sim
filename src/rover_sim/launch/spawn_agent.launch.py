# Spawns one JR02-parity agent into a running world.launch.py: xacro to a file, then
# spawn_entity -file at floor height, as entity <namespace>. There is no
# robot_state_publisher: the model publishes no TF (rover_description's rsp in sim-drive owns
# <ns>/base_link -> <ns>/lidar, from the same lidar mount file).
#
#   ros2 launch rover_sim spawn_agent.launch.py namespace:=JR81 class:=jr x:=-2.0 y:=0.0 yaw:=0.0
#
# yaw is in radians. lidar_mount_file defaults to the run dir's generated
# /rover_sim/run/sensor_calib/<ns>_lidar_mount.yaml; lidar_height to the class's
# (config/classes.yaml). lidar_samples is the most rays a 10 Hz scan casts, spread over the
# MID-360's 20000-point frame: every k-th point with k = ceil(20000 / lidar_samples), which
# casts ceil(20000 / k) (logged below). The robot's full 20000 is far from real time (4
# agents in empty: RTF 0.63); config/worlds.yaml has what 4 agents can afford per world
# (sim.py passes it), and the default fits both shipped worlds.
import math
import os
import re

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Dropped from just above the floor, so it never starts interpenetrating it.
SPAWN_Z = 0.005
FRAME_POINTS = 20000      # mid360.xacro: pattern points in one 100 ms MID-360 frame


def number(name: str, value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise RuntimeError('%s:=%s is not a number' % (name, value))
    if not math.isfinite(v):
        raise RuntimeError('%s:=%s is not finite' % (name, value))
    return v


def spawn(context):
    arg = {k: LaunchConfiguration(k).perform(context) for k in (
        'namespace', 'class', 'x', 'y', 'yaw', 'lidar_mount_file', 'lidar_height', 'lidar_samples',
        'urdf_file')}
    ns, cls = arg['namespace'].strip('/'), arg['class']
    # The plugins' nodes take it as their namespace, and rclcpp aborts gzserver (every
    # agent) on an invalid one.
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', ns):
        raise RuntimeError('namespace:=%s: expected a ROS name such as JR81' % ns)
    share = get_package_share_directory('rover_sim')
    with open(os.path.join(share, 'config', 'classes.yaml')) as f:
        classes = yaml.safe_load(f)
    if cls not in classes:
        raise RuntimeError('class:=%s: expected one of %s' % (cls, ', '.join(classes)))
    x, y, yaw = (number(k, arg[k]) for k in ('x', 'y', 'yaw'))
    if abs(yaw) > 2 * math.pi + 1e-6:
        raise RuntimeError('yaw:=%s: radians expected (|yaw| <= 2 pi)' % arg['yaw'])
    height = number('lidar_height', arg['lidar_height'] or str(classes[cls]['lidar_height']))
    samples = arg['lidar_samples']
    if not samples.isdigit() or not 1 <= int(samples) <= 20000:
        raise RuntimeError('lidar_samples:=%s: expected 1 to 20000 (one MID-360 frame)' % samples)
    mount = arg['lidar_mount_file'] or '/rover_sim/run/sensor_calib/%s_lidar_mount.yaml' % ns
    if not os.path.isfile(mount):
        raise RuntimeError('lidar_mount_file:=%s does not exist (sim.py generates it per agent)' % mount)
    urdf_file = arg['urdf_file'] or '/tmp/rover_sim_%s.urdf' % ns

    doc = xacro.process_file(os.path.join(share, 'urdf', cls + '.urdf.xacro'), mappings={
        'namespace': ns, 'lidar_mount_file': mount, 'lidar_height': repr(height),
        'lidar_samples': samples})
    os.makedirs(os.path.dirname(os.path.abspath(urdf_file)), exist_ok=True)
    with open(urdf_file, 'w') as f:
        f.write(doc.toprettyxml(indent='  '))

    spawner = Node(
        package='gazebo_ros', executable='spawn_entity.py', name='spawn_entity', namespace=ns,
        output='screen',
        # spawn_entity.py waits on the service's reply without a timeout of its own, and a
        # SIGTERM'd rclpy process can hang in its shutdown: -k SIGKILLs it, so the pane fails.
        prefix='timeout -k 5 120',
        parameters=[{'use_sim_time': False}],
        arguments=['-file', urdf_file, '-entity', ns, '-timeout', '60',
                   '-x', repr(x), '-y', repr(y), '-z', repr(SPAWN_Z), '-Y', repr(yaw)])

    def check(event, _context):
        # ros2 launch exits 0 whatever its processes do; an exception here makes it exit 1,
        # so sim.py's spawn pane stops before unpausing a world that is missing an agent.
        if event.returncode != 0:
            raise RuntimeError('%s: spawn_entity failed (exit %d)' % (ns, event.returncode))

    # What mid360.xacro's downsample makes of it.
    every = -(-FRAME_POINTS // int(samples))
    rays = -(-FRAME_POINTS // every)
    return [
        LogInfo(msg='%s (%s): lidar %.3f m up, mount %s, lidar_samples %s: %d rays per scan (every %d of %d), '
                'at x %.3f y %.3f yaw %.4f rad; %s'
                % (ns, cls, height, mount, samples, rays, every, FRAME_POINTS, x, y, yaw, urdf_file)),
        spawner,
        RegisterEventHandler(OnProcessExit(target_action=spawner, on_exit=check)),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('namespace', description='Agent name, e.g. JR81: entity name and ROS namespace'),
        DeclareLaunchArgument('class', description='jr (Pioneer 3-DX) or rr (Pioneer 3-AT)'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='Radians'),
        DeclareLaunchArgument('lidar_mount_file', default_value='',
                              description='rover_description-format <ns>_lidar_mount.yaml'),
        DeclareLaunchArgument('lidar_height', default_value='',
                              description='m above the floor; default per class (jr 0.36, rr 0.40)'),
        DeclareLaunchArgument('lidar_samples', default_value='500',
                              description='at most this many MID-360 rays per scan (the robot has 20000; see config/worlds.yaml)'),
        DeclareLaunchArgument('urdf_file', default_value='', description='default /tmp/rover_sim_<ns>.urdf'),
        OpaqueFunction(function=spawn),
    ])
