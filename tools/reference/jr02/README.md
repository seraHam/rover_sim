# JR02 reference

What rover JR02 actually ran on 2026-10-18: the parameters, launch params, graph edges and
TF tree that `scripts/parity_check.py` compares a sim agent against. Each file is either a
verbatim copy of a robot capture or derived from captures. Derived files say so in their
header, and the tables below show which is which.

```bash
scripts/parity_check.py JR81                      # every node; NOT RUNNING ones don't fail
scripts/parity_check.py RR83 --nodes mighty_node,mpc --require-running
scripts/parity_check.py JR81 --mapping fallback   # also allow the fallback's documented keys
scripts/parity_check.py JR81 --show-ref mpc       # the reference, renamed to JR81
scripts/parity_check.py JR81 --selftest           # the checker on these files alone
```

The checker replaces `JR02` with the agent's name before comparing. Every deliberate
difference is listed in `allowed_deviations.yaml` with its reason, and is reported as
ALLOWED.

## What JR02 was running

Sources: the survey (`jr02_vs_rover_sim_survey.md`), the 2026-10-18 captures, and
`processes/` (image IDs and full process command lines).

| Stack (container) | Code | Config |
|---|---|---|
| hw-mighty | mighty binary built from `5094cdb` (`main`); mpc `193c36b`; goal_selector_msgs `4a03aed` | mighty `db3018a` `config/`, mounted read-only over the installed share. It is byte-identical to `3cf7320`, the sim's pin |
| mapping | acl-mapping `f44e045`, map_fusion `8ea4ca7` (image `rover-mapping:local`, built from rover `b226206`) | mapping_ws `c1233f7` `config/mapping`, mounted at `/mapping_config` |
| dlio | DLIO `73c2489`; image `dlio:JR02-04c7239` | `cfg_overlay/JR02` baked in, byte-identical to dlio_ws `e676331`; scripts `59f6ee2`, mounted |
| drive | rover `b39c898` (tree `0dc816f`), acl_ctrl_safety `5cb2be0` | rover_control `teleop_logitech.yaml`; `sensor_calib/JR02_*` |
| all | `rmw_zenoh_cpp` `0.1.9-1jammy.20260907.200640`, domain 22 | `/etc/rover/rover.env` (`rover.env` here) |

**JR02 state at capture time.**
- **The MID-360 lidar link was down.**
  - `enp86s0` read `operstate down`, `carrier 0` at 2026-10-18 19:23 UTC.
  - It had been down since boot (2026-10-15 18:07 EDT), and no Livox driver was running.
  - Effect on the captures: lidar-derived topics have their subscribers but no live publisher. For example, `livox/lidar` shows DLIO subscribed with 0 publishers.
  - DLIO was seeded but idle. The elevation mapper and map_fusion published nothing, and the geofence held AUTO at zero.
- **Unaffected:** parameters, launch params and endpoint sets. Nodes create their endpoints at start-up, whether or not data flows.
- **Not captured:** live data (rates, field contents, TF stamps).
- **Containers:** up since 2026-10-15 22:07 UTC. dlio and drive were recreated at 19:20 and 19:05 UTC that day, which left stale `/tmp/launch_params_*` files from the earlier runs. Only the files of the running processes are kept here.

## Files

**This repo is public.** Files that come from private repos (GitLab `mit-acl`, and github
`blakete/dlio_ws`) are therefore not stored here. `manifest.yaml`'s `external:` section
names each by repo, commit and path, together with `yaml_sha256`, a hash of the robot's
copy (parsed, so launch's rewritten copy of a params file still matches). `parity_check.py`
reads them at check time from the pinned checkouts `scripts/build_images.sh` keeps under
`.cache/`, or clones the repo into `.cache/reference/` (with the GitLab/GitHub key the
image builds use), and refuses a file whose hash differs. Captures that described the
rest of the fleet or the robot's hardware are left out or trimmed, as marked below.

### Captured on the robot: verbatim copies (or fetched, where marked)

| File | What | Captured |
|---|---|---|
| `params/mighty_node.yaml` | `ros2 param dump /JR02/mighty_node`: the full declared set (218 parameters once flattened) | survey, 2026-10-18 ~17:14 UTC (ros2 CLI on JR02) |
| `params/goal_selector.yaml` | `ros2 param dump /JR02/goal_selector` | survey, same time |
| `launch_params/mighty_node.yaml` | `hw-mighty:/tmp/launch_params_87incphn`. The single merged file the launch passes to mighty_node (pid 6158), from `mighty.yaml` ← `hw_mighty_ground_robot.yaml` ← launch args | survey; re-read 2026-10-18 19:24 UTC, identical |
| `launch_params/goal_selector.yaml` | `hw-mighty:/tmp/launch_params_m35qw1qv` (goal_selector, pid 5806) | survey; re-read 19:24 UTC, identical |
| `launch_params/mpc.yaml` | `hw-mighty:/tmp/launch_params_pdqrnef9` (mpc, pid 6156): `cmd_vel_topic: cmd_vel_auto`, applied after `config/mpc/mpc.yaml` | 2026-10-18 19:24 UTC |
| `launch_params/global_mapper_ros.yaml` | `mapping:/tmp/launch_params_cd7zaa25` (global_mapper_node, pid 6097), applied after `/mapping_config/global_mapper_junior_dlio.yaml` | 19:24 UTC |
| `launch_params/map_fusion_node.yaml` | `mapping:/tmp/launch_params_ntv38ooe` (map_fusion_node, pid 5971), after `map_fusion_junior.yaml` | 19:24 UTC |
| `launch_params/persistent_map_node.yaml` | `mapping:/tmp/launch_params_iyxq73_k` (persistent_map_node, pid 5691), after `persistent_map.yaml` | 19:24 UTC |
| `launch_params/dlio_odom_node.*.yaml` | `dlio:/tmp/launch_params_gedp22vx` (`initial_pose_topic: world`) and `_q7ftxy6g` (`output/twoDOnly: false`), applied after `cfg/dlio.yaml` and `cfg/params.yaml` (dlio_odom_node, pid 5756) | 19:24 UTC |
| `launch_params/safety_node.yaml` (**fetched**) | `drive:/tmp/launch_params_5pz6vhj3` (safety_node, pid 6026): launch's copy of the deployed `teleop_logitech.yaml`, with all robots' blocks; joy_node and teleop_twist_joy get the same file. Same content as rover `b39c898` (what JR02 runs) `rover_control/config/teleop_logitech.yaml`, which is what is fetched | 19:24 UTC |
| `launch_params/robot_state_publisher.yaml` (**trimmed**) | `drive:/tmp/launch_params_z_ufdiqh`. Its `robot_description`, the deployed URDF (private rover_description), is replaced by a placeholder: the checker compares the sim's URDF against `tf/tf_tree.yaml`, which holds its joints | 19:24 UTC |
| `config/mighty/{mighty,hw_mighty_ground_robot,hw_goal_selector}.yaml` | mounted mighty config, byte-identical to `db3018a` and `3cf7320`. `robot_type:=red_rover` loads `mighty.yaml` and then `hw_mighty_ground_robot.yaml`. The unused `hw_mighty.yaml` (quadrotor) and `mighty_ground_robot.yaml` are left out | survey |
| `config/mpc/mpc.yaml` (**fetched**) | `hw-mighty:.../share/mpc/config/mpc.yaml`, identical to mpc `193c36b` `config/mpc.yaml` | 19:24 UTC |
| `config/mapping/*.yaml` (**fetched**) | `mapping:/mapping_config/{global_mapper_junior_dlio,map_fusion_junior,persistent_map}.yaml`, byte-identical to mapping_ws `c1233f7` `config/mapping/`. Their mtimes predate the container start | 19:24 UTC |
| `config/dlio/{dlio,params}.yaml` (**fetched**) | `dlio:/home/swarm/code/dlio_ws/install/share/direct_lidar_inertial_odometry/cfg/*.yaml`, byte-identical to dlio_ws `e676331` `cfg_overlay/JR02` | 19:24 UTC |
| `rover.env` (**trimmed**) | `/etc/rover/rover.env`: identity, RMW, `DLIO_SEED_Z=0.36`, `MAPPER_*`. Its change-history comments, the router config path and the serial device line are left out. It sets no `ZENOH_SESSION_CONFIG_URI`; the containers get that from their compose files | 19:23 UTC |
| `graph/node_info.txt` | `ros2 node info` of mighty_node, goal_selector, convert_odom_to_state, mpc | survey, ~17:15 UTC |
| `processes/docker_top_*.txt` | `docker top <c> -eo pid,lstart,args` for drive, dlio, mapping, hw-mighty: the full command lines. In the drive one (**trimmed**) robot_state_publisher's URDF argument and the firmware serial device are left out | 19:24 UTC |
| `processes/images.txt` (**trimmed**) | `docker inspect` image IDs, creation times and image labels of the 4 stack containers compared here; the sensors and geofence containers and compose's labels are left out | 19:24 UTC |

**Not kept here** (they describe the rest of the fleet): the survey's `ros2 node list` +
`ros2 topic list -t` of the whole graph JR02 saw, and its two `ros2 topic info -v` captures
(19 planner topics, 19 control/mapping chain topics), which list other robots' and C2's
endpoints. `graph/edges.yaml` keeps what the checker needs from them: JR02's own endpoints.

**How it was captured.** The 2026-10-18 19:23–19:24 UTC captures are read-only, using
`ssh -o BatchMode=yes jr02.local` plus:
- `cat /etc/rover/rover.env` and `cat /sys/class/net/enp86s0/{operstate,carrier}`;
- `docker exec <c> cat <file>`, and `docker exec mapping sh -c 'ls /tmp/launch_params_*; cat ...'` (the same for dlio, drive and hw-mighty);
- `docker top <c> -eo pid,lstart,args`, which ties each `/tmp/launch_params_*` file to the process holding it;
- `docker inspect`.

No ros2 CLI was run on the robot for these. The survey's earlier ros2 CLI calls did run there, and one of them left a stray `ros2-daemon` in the mapping container, which is why the checker always passes `--no-daemon`.

### Derived: not a single capture

| File | Derived from |
|---|---|
| `graph/edges.yaml` | The survey's graph captures (`node_info.txt` and the three not kept here), normalised. `nodes:` holds the complete pub/sub sets of the 4 hw-mighty nodes (from `node_info.txt`). `topics:` holds, per topic, the endpoints of the JR02 stack only: namespace `/JR02`, plus its root-level `map_odom_tf_broadcaster` and `dlio_map_node`. Other robots, C2 and base-station RViz are dropped. `/tf`'s two `map_odom_tf_broadcaster` (two robots', same root name) collapse into one. Types come from the topic list and `node_info.txt` |
| `tf/tf_tree.yaml` | **JR: static joints** come from the deployed URDF (robot_state_publisher's launch params; the URDF itself is not kept here). They were checked identical to the derivation from rover `9b0b4ba` `red_rover_sensors_ns_frame_ids.urdf.xml`, `sensor_calib/JR02_lidar_mount.yaml` (pitch 0.5117309 rad = 29.32°) and `JR02_calib.npy`, which applies `state_publisher.launch.py`'s logic. **RR: static joints** come from the same derivation with RR08's files (pitch 0.3590419 rad = 20.5716°). That part is not a capture. **Dynamic edges** come from the process list: `<ns>/map→<ns>/odom` is `map_odom_tf.py` (periodic `/tf`), and `<ns>/odom→<ns>/base_link` is DLIO. `world→<ns>/map` is sim only. No `tf2_echo` was run on the robot |
| `manifest.yaml` | Node table: robot name, sim container, and which files make up each node's reference |
| `class_rr/` (**fetched**, all four files) | **Not from a robot**; no RR robot was surveyed. Used when the agent is RR-class. Contents: mapping_ws `c1233f7` `global_mapper_red_rover.yaml` and `map_fusion_red_rover.yaml`, and dlio_ws `e676331` `cfg_overlay/RR08/{dlio,params}.yaml`. The RR mapping panes use the same launch args as JR (`mapping_host_tmux.sh`), so JR02's launch params are reused with the name substituted |

## How the checker reads these

- **`kind: dump` nodes (mighty_node, goal_selector).** Every key must match. A key the sim declares and the robot did not is drift too.
- **`kind: files` nodes.** These nodes (mpc, mapping, DLIO, safety_node, robot_state_publisher) have no dump. The reference is what their param files and launch params set, merged the way rcl merges `--params-file` arguments. Keys the sim node adds (its defaults) are not compared.
  - rclcpp silently drops file params that a node never declares. Such keys show up as a NOTE, not a DIFF. One known case is `radius_drone` in `global_mapper_junior_dlio.yaml`.
  - The two nodes with both a dump and a launch-param file show how common this is. Every key they share has the same value. But mighty_node's file carries 61 keys it never declares (for example the old `exploration.*`, which now lives in goal_selector), and goal_selector's file carries 216.
  - The `launch_params/` files of mighty_node and goal_selector contain `!!python/tuple` tags, which is how launch_ros writes tuples. rcl ignores them, and the checker reads them as lists.
- **robot_state_publisher's `robot_description`.** It is compared joint by joint against `tf/tf_tree.yaml`.
- **Graph checks.**
  - `node info` sets for the 4 hw-mighty nodes.
  - `topic info -v` endpoints and their reliability and durability, for every reference topic that has a running endpoint node.
  - Only the agent's own endpoints count, since other agents share `/trajs` and the other root topics.
- **Not covered:** the elevation mapper. It runs on JR02's Orin (JX02), whose launch params were not captured for this reference. The node appears in the graph checks only.

## Checked against real nodes (2026-10-18)

These are the robot images run as `JR81` in throwaway containers with `--network none` and a private router, using the robot's launch commands:

| Stack | Result |
|---|---|
| mighty-hw `3cf7320` (the sim's pin) | mighty_node, goal_selector, mpc and convert_odom_to_state match JR02's `5094cdb`/`db3018a` params and edges. The only difference is the allowed `num_agents` |
| rover-mapping `b226206` + mapping_ws `c1233f7` config | Only allowed differences (`save_dir`, `corrected_path_topic`), plus the `radius_drone` note |
| rover-drive `0dc816f` | 2 DIFFs, both expected until later phases. There is no `JR81_lidar_mount.yaml`, so the lidar joint is stripped (Phase 2 generates the file), and there is no `/JR81/safety_node` `ang_vel_max: 4.0` block (Phase 3's `teleop_sim.yaml` adds it) |

## Updating

1. Re-capture with the same read-only commands. Never run ros2 CLI on the robot. If you ever must, use `--no-daemon`.
2. Map each `/tmp/launch_params_*` file to its node by the `--params-file` argument in `docker top`.
3. Keep the robot's file names in this README, and record the new date and the deployed refs.
4. Regenerate `graph/edges.yaml` and `tf/tf_tree.yaml` with the rules above.
5. Run `scripts/parity_check.py JR81 --selftest`.
6. If a sim deviation becomes intended, add it to `allowed_deviations.yaml`, with the reason, in the same change.
