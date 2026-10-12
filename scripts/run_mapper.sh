#!/bin/bash
# Starts robot $1's mapper: MAPPER=acl (default) or elevation (the rovers'
# elevation_mapping_cupy and config, with config/elevation_sim.yaml on top).
ns=$1
case ${MAPPER:-acl} in
  acl)
    exec ros2 launch global_mapper_ros global_mapper_node.launch.py use_gazebo:=true \
      use_obstacle_tracker:=false param_file:=/tmp/rover_sim/mapper.yaml quad:=$ns ;;
  elevation)
    "$(dirname "$0")/elevation_start.py" $ns &
    exec ros2 launch rover_sim elevation_mapping_sim.launch.py quad:=$ns \
      params_file:=/home/swarm/rover_sim/config/elevation_sim.yaml ;;
  *)
    echo "unknown mapper: $MAPPER (expected acl or elevation)"; exit 1 ;;
esac
