# Spawns one robot into a running Gazebo: robot_state_publisher (TF prefixed with
# <namespace>/) and spawn_entity, as mighty's onboard_mighty.launch.py does.
# ros2 launch rover_sim spawn_robot.launch.py namespace:=RR01 model:=rr x:=4.0 y:=4.0 yaw:=-135
from math import radians

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# red rover (Pioneer 3-AT), red rover junior (Pioneer 3-DX)
MODELS = {'rr': 'p3at', 'jr': 'p3dx'}


def spawn(context):
    namespace = LaunchConfiguration('namespace').perform(context)
    model = LaunchConfiguration('model').perform(context) or 'rr'
    if model not in MODELS:
        raise RuntimeError(f'model:={model} unknown; one of {tuple(MODELS)}')
    urdf = [FindPackageShare('rover_sim').perform(context), f'/urdf/{MODELS[model]}.urdf.xacro']
    x, y, z = (LaunchConfiguration(k).perform(context) for k in ('x', 'y', 'z'))
    yaw = str(radians(float(LaunchConfiguration('yaw').perform(context))))
    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=namespace,
            output='screen',
            parameters=[{
                'robot_description': ParameterValue(
                    Command(['xacro ', *urdf, ' namespace:=', namespace]), value_type=str),
                'use_sim_time': False,
                'frame_prefix': namespace + '/',
            }],
            arguments=['--ros-args', '--log-level', 'error'],
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_entity',
            namespace=namespace,
            parameters=[{'use_sim_time': False}],
            arguments=['-topic', 'robot_description', '-entity', namespace,
                       '-x', x, '-y', y, '-z', z, '-Y', yaw, '--ros-args', '--log-level', 'error'],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('namespace', description='Robot name, e.g. RR01'),
        DeclareLaunchArgument('model', default_value='rr', description=f'One of {tuple(MODELS)}'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='Degrees'),
        OpaqueFunction(function=spawn),
    ])
