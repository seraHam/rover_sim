#!/bin/bash
# Writes the mapper and mighty configs the launch/ sessions use to /tmp/rover_sim/.
# With PRIOR_MAP set to a nav2 map yaml, the mapper starts from that map on a
# fixed grid covering it, and mighty plans with hard obstacles over a 40 m horizon.
# PRIOR_WINDOW (m) sets mighty's minimum planning window; it is centered on the
# robot, so it must be twice the farthest goal offset to hold the goal.
set -e
out=/tmp/rover_sim
mkdir -p $out
mapper=/home/swarm/code/rover_sim_ws/install/global_mapper_ros/share/global_mapper_ros/cfg/sim_ground_robot.yaml
mighty=/home/swarm/code/rover_sim_ws/install/mighty/share/mighty/config/mighty_ground_robot.yaml

if [ -z "$PRIOR_MAP" ]; then
  cp $mapper $out/mapper.yaml
  cp $mighty $out/mighty.yaml
  exit 0
fi

image=$(awk '/^image:/{print $2}' "$PRIOR_MAP")
[[ $image = /* ]] || image=$(dirname "$PRIOR_MAP")/$image
read -r w h < <(grep -av '^#' "$image" | head -c 64 | tr -s ' \t\r\n' '\n' | sed -n '2,3p' | paste -sd' ')
read -r cx cy dx dy < <(awk -v w="$w" -v h="$h" '
  /^resolution:/ {r = $2}
  /^origin:/ {gsub(/[][,]/, " "); ox = $2; oy = $3}
  END {printf "%.3f %.3f %.3f %.3f\n", ox + w*r/2, oy + h*r/2, w*r, h*r}' "$PRIOR_MAP")

sed -e "s|^\( *origin:\) \[[^,]*,[^,]*,\([^]]*\)\]|\1 [$cx, $cy,\2]|" \
    -e "s|^\( *world_dimensions:\) \[[^,]*,[^,]*,\([^]]*\)\]|\1 [$dx, $dy,\2]|" \
    -e "s|^\( *\)lock_origin: false|\1lock_origin: true\n\1prior_map.yaml: $PRIOR_MAP|" \
    $mapper > $out/mapper.yaml
# A goal outside the window is clamped to its edge, which can land on a wall.
window=()
[ -n "$PRIOR_WINDOW" ] && window=(-e "s/^\( *min_wd[xy]:\) [0-9.]*/\1 $(printf %.1f "$PRIOR_WINDOW")/")
sed -e 's/^\( *use_soft_cost_obstacles:\) true/\1 false/' \
    -e 's/^\( *horizon:\) [0-9.]*/\1 40.0/' \
    "${window[@]}" \
    $mighty > $out/mighty.yaml
