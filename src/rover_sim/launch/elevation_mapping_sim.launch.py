"""elevation_mapping_cupy as the rovers run it (elevation_mapping_dlio.launch.py's
config), fed the sim's lidar instead of DLIO's cloud. params_file:=<yaml> is
loaded on top of the rovers' config (use a `/**:` key)."""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _ros_parameters(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    # core_param.yaml is keyed by absolute node name, which wouldn't apply under a namespace.
    return next(v['ros__parameters'] for v in data.values() if 'ros__parameters' in v)


def _setup(context):
    ns = LaunchConfiguration('quad').perform(context)
    params_file = LaunchConfiguration('params_file').perform(context)
    config = os.path.join(get_package_share_directory('elevation_mapping_cupy'), 'config')
    params = _ros_parameters(os.path.join(config, 'core', 'core_param.yaml'))
    params.update(_ros_parameters(os.path.join(config, 'setups', 'dlio', 'base.yaml')))
    params.update({
        'extra_plugin_config_file': os.path.join(config, 'core', 'traversability_plugin_config.yaml'),
        'subscribers': {'lidar': {'topic_name': f'/{ns}/mid360_PointCloud2', 'data_type': 'pointcloud'}},
        'map_frame': 'map',
        'corrected_map_frame': 'map',
        'base_frame': f'{ns}/base_link',
        'sensor_frame': f'{ns}/{ns}_livox',
    })
    return [Node(package='elevation_mapping_cupy', executable='elevation_mapping_node.py',
                 name='elevation_mapping_node', namespace=ns, output='screen',
                 parameters=[params] + ([params_file] if params_file else []))]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('quad', description='Robot namespace, e.g. RR01'),
        DeclareLaunchArgument('params_file', default_value='',
                              description="Overrides of the rovers' config"),
        OpaqueFunction(function=_setup),
    ])
