# rover_sim

A multi-rover simulation of the ACL fleet's ground-robot autonomy stack, at parity with what
rover **JR02** runs. Each simulated agent runs the robot's own images and command lines:
- DLIO;
- the drive and safety stack;
- the mapping stack (global_mapper, map_fusion, persistent_map, elevation_mapping_cupy);
- the mighty planner, with mpc and goal_selector.

They run against a Gazebo physics model of its chassis (Pioneer 3-DX for JR, Pioneer 3-AT for
RR) with a simulated Livox MID-360 and IMU. What the sim does differently from the robot, and
why, is listed under [Deviations from the robot](#deviations-from-the-robot).

No ROS install is needed on the host. Everything runs in docker, and `scripts/sim.py` drives it.

## Quickstart

**What you need**
- Linux with Docker Engine and its compose and buildx plugins, your user in the `docker` group,
  and `git`, `tmux` and Python 3.8+ with PyYAML (`sudo apt install git tmux python3-yaml`).
- 12 or more CPU cores (with fewer, the containers run unpinned), and about 30 GB free under
  docker's root (45 GB with the GPU image).
- Optional: an NVIDIA GPU with the NVIDIA Container Toolkit, for the `mapping: gpu` sessions.
  `smoke` and `odom_eval` run without one.
- An ssh key on your GitHub and GitLab accounts, loaded in an ssh agent (`ssh-add`) or given as
  `SSH_KEY=<key>`. The image builds forward it to clone private repos, and never bake it in.
- Read access to the private repos the stacks are built from:
  - GitHub: `blakete/dlio_ws`;
  - mit-acl GitLab: `ugv/redrover/mapping_ws`, `ugv/redrover/rover`, `ugv/redrover/map_fusion`,
    `ugv/acl_ctrl_safety`, `ugv/ugv_control/mpc` and `lab/acl-mapping`.

  The pins decide this list: `scripts/bootstrap.py --check` reads it from them and names each
  repo you cannot read.

**Run it**

```bash
git clone https://github.com/seraHam/rover_sim.git && cd rover_sim
scripts/bootstrap.py --run
```

`scripts/bootstrap.py` checks the host and your access to every repo first, so a missing grant
fails in seconds, not part-way through the build. It then builds the images (without a GPU, all
but the elevation image) and starts `smoke`: JR81 and RR83 swap places in a walled arena. The
first build takes roughly an hour on a 20-core machine; after a failure, run it again and it
resumes. Without `--run` it stops after a dry run of `smoke`; `--check` only checks. Then:

```bash
scripts/sim.py status          # containers, panes, sim time, DLIO, modes
scripts/sim.py viz             # the live 2D view (needs a display)
scripts/demo_eval.py swap      # grade the swap from ground truth (records 300 s)
scripts/sim.py down            # stop it; make down also removes the containers
```

## Architecture

- **One container per robot stack** (`docker/compose.yaml`, project `rover-sim-stack`). Each
  container idles; a session `docker exec`s every node into it from a host tmux session per
  stack (`rsim-<stack>`), the way the robot's `*_host_tmux.sh` units do. Ctrl-C, Up, Enter in a
  pane restarts that one node.

  | container | image | runs |
  |---|---|---|
  | `sim-drive` | rover `rover_bringup` (`DRIVE_SHA`) | the zenoh router, teleop + safety_node, rover_description |
  | `sim-gazebo` | this repo's `gazebo` target | gzserver (and gzclient), spawns, the sensor bridges, world → `<ns>/map`, RViz |
  | `sim-dlio-jr`, `sim-dlio-rr` | dlio_ws (`DLIO_SHA`), one per class | DLIO, its seed pose, map → odom |
  | `sim-mapping` | mapping_ws (`MAPPING_SHA`) | global_mapper, map_fusion, persistent_map |
  | `sim-elevation` | this repo's `elevation` target (GPU) | elevation_mapping_cupy, as the robot's Orin runs it |
  | `sim-mighty` | mighty `Dockerfile.hw` (`MIGHTY_SHA`) | mighty_node, convert_odom_to_state, mpc, goal_selector, frame_align, goal_relay.py |
  | `sim-viz` | `gazebo` (profile `viz`) | sim_viz.py, the 2D viewer |

- **Transport:** rmw_zenoh, the fleet's exact build (`docker/stacks.env` pins it; every image
  is checked with `dpkg-query`). One router runs in `sim-drive` on `tcp/sim-drive:7447`, and
  every node is a peer of it (`docker/zenoh/`). There is no zenoh gateway.
- **Isolation from the fleet.** The containers sit on an internal docker network (`rover-sim`)
  with IPv6 off, no ports and no host networking. The host runs the fleet's C2 router, and the
  sim must never reach it. `sim.py` re-checks this on every `up` and `status` and refuses to
  start nodes when broken. Every tool that publishes refuses to run outside the sim's zenoh
  session (`ZENOH_SESSION_CONFIG_URI=/rover_sim/zenoh/session.json5`).
- **CPU budget** (hosts with 12 or more CPUs):
  - CPUs 0-1 for the host;
  - 2-5 for gzserver;
  - the rest shared by the other containers.

  Host processes are not pinned: see Open issues.
- **Bring-up in waves into a paused world.** `up` starts the stacks in this order:
  1. the router and Gazebo (the spawn);
  2. drive, frame_align and DLIO;
  3. mapping;
  4. elevation;
  5. the planner;
  6. RViz and the goals.

  Each wave waits for its panes' node-start markers and for the containers to go idle. Panes
  start 0.25 s apart. Planner sessions are armed while still paused. A **graph check** then
  verifies each agent's data path (lidar → DLIO → mappers → map_fusion → planner → mpc →
  safety), and only then does `up` unpause, once. A pane that fails, a planner TF gate that
  times out, or a missing graph edge stops `up` with the world still paused and every agent
  DISARMED. `<logdir>/bringup.log` has every step.

### Frames and topics

- `world` → `<ns>/map` is the agent's spawn pose (`sim_tf.py`). `<ns>/map` → `<ns>/odom` is
  identity (DLIO's `map_odom_tf.py`), and `<ns>/odom` → `<ns>/base_link` is DLIO.
  rover_description publishes `base_link` → `<ns>/lidar`.
- `scripts/livox_sim_bridge.py` turns the plugins' sim-stamped streams into the robot's
  `/<ns>/livox/{lidar,imu}` on wall-clock stamps, and publishes `/<ns>/ground_truth/odom_wall`
  plus `sim/rtf` and `sim/time_offset`.
- Agents are named class prefix + two digits (`JR81`, `RR83`). The digits are mighty's agent
  id, unique across classes. Each class is calibrated from one real robot (JR from JR02, RR from
  RR08): the Gazebo lidar mount, rover_description's base_link → lidar, and the DLIO image's
  cfg overlay all come from that rover.

## Images

`make images` (`scripts/build_images.sh`, which `bootstrap.py` runs) builds each stack from its
own repo's Dockerfile at the `docker/stacks.env` pins, plus this repo's `gazebo` and `elevation`
targets. It reuses an image whose inputs are unchanged. Useful options:
- `make images STACKS="mighty mapping dlio drive gazebo"` builds those only (here: all but
  elevation);
- `--src UNIT=DIR` builds a local checkout into the stack's tag;
- `--reuse UNIT=IMAGE` retags an existing image;
- `--list` shows the state.

Images are labelled with the commit and inputs they were built from, and `up` / `status` warn
about one built from uncommitted changes.

## Usage

```bash
scripts/sim.py up smoke                      # or: make session S=smoke
scripts/sim.py status [--maps]               # containers, isolation, router, panes, sim time, DLIO, modes
scripts/sim.py viz                           # the live 2D view (sim_viz.py) on $DISPLAY
scripts/sim.py attach mighty                 # the rsim-mighty tmux session
scripts/sim.py arm [<ns>...] / disarm [<ns>...]
scripts/sim.py start goals|rviz              # restart that one stack on the running session
scripts/sim.py down [--containers]
scripts/sim.py up <session> --dry-run        # every pane and command; touches nothing
```

**Sessions** live in `config/sessions/<name>.yaml`, or you can pass a path:

| session | world | agents | mapping | demo |
|---|---|---|---|---|
| `smoke` | arena | JR81, RR83 | fallback | swap (crossing paths) |
| `multi_agent_swap` | arena | JR81, JR82, RR83, RR84 | gpu | swap (pinwheel) |
| `office_explore` | ACL_office_simple | 4 | gpu | explore (goal_selector's frontiers) |
| `odom_eval` | arena | 4 | fallback | no planner: `odom_eval.py --drive` |

**Session keys.** The `scripts/sim.py` docstring has the full list. The main ones:
- `world`;
- `gui` (gzclient);
- `mapping: gpu | fallback`;
- `odometry: dlio | truth`, also settable per agent;
- `rviz`;
- `planner`;
- `demo: none | swap | explore`;
- `auto_arm`;
- `agents`, each `{name, class, pose: [x, y, yaw_deg], goals}`.

Per-run overrides:
- `ROVER_SIM_AGENTS=JR81:jr,RR85:rr` replaces the agents;
- `up --rviz` adds RViz;
- `up --odometry truth[:NS,...]` puts those agents on ground truth.

**Goals.**
- RViz's 2D Goal Pose publishes on `/sim/goal` (world frame). `scripts/goal_relay.py` sends it
  to the nearest agent's `term_goal_rviz`, in that agent's `<ns>/map`, or to the agent named
  on `/sim/goal_agent`.
- The swap demo's legs go the same way. Each agent first gets a 1.2 m drive off its spot (see
  Deviations).
- Your own goal sender: publish a `geometry_msgs/PoseStamped` on `/sim/goal` from a container
  on the sim's network.

### Tools

- `scripts/demo_eval.py [swap|explore] --duration S` grades a running demo from ground truth and
  the pane logs. It checks legs or frontiers and coverage, separation, box contact, /trajs per
  pair, unexpected ERROR lines, RTF, and DLIO against truth.
- `scripts/odom_eval.py --ns JR81 [--drive ...]`: DLIO's ATE against ground truth, optionally
  while `drive_pattern.py` drives a pattern.
- `scripts/parity_check.py <ns>` diffs an agent's parameters and graph edges against JR02's
  captured reference (`tools/reference/jr02/`). What the sim changes on purpose is in
  `tools/reference/jr02/allowed_deviations.yaml`.
- `scripts/check_pins.py [--robot SSH_TARGET] [--orin SSH_TARGET]` checks the pins against each
  other and, read-only over ssh, against what a robot's containers run.
- `scripts/truth_odom.py`: ground truth published under DLIO's names (`odometry: truth`).

Every tool that opens a ROS session joins the graph with as few entities as it can, and leaves
one entity at a time, 0.5 s apart. With rmw_zenoh 0.1.9 many undeclarations at once stall
delivery graph-wide. Run probes before or after a window that is being graded, not during one.

## GPU vs fallback mapping

- `mapping: gpu` is parity: global_mapper, elevation_mapping_cupy, map_fusion and
  persistent_map, with the robot's files and command lines.
- `mapping: fallback` (no GPU) is **NOT PARITY**:
  - there is no elevation mapper and no map_fusion;
  - global_mapper runs the robot's file plus a 3-key overlay (`config/mapping/`);
  - `scripts/grid_relay.py` publishes its grids under map_fusion's names.

  `up`, its dry run, `status` and `parity_check` all label it.

## Odometry: DLIO vs truth

`odometry: truth` (**NOT PARITY**, a control) replaces an agent's DLIO with `truth_odom.py`.
- It publishes Gazebo's ground truth under DLIO's names, frames and stamps.
- It reads the extrinsics, crop box and calibration time from the class image's own DLIO cfg.
- It applies per agent: `up --odometry truth:JR81` runs JR81 on truth next to the others on DLIO.

What stays wrong in the maps or the planner with it on is not DLIO's drift. demo_eval and
parity_check report truth agents apart, and `odom_eval` refuses them.

## Pins

- `docker/stacks.env` is the one pin file: each stack's repo, commit and image tag, the base
  image digest, and the rmw_zenoh build.
- `rover_sim.sim.repos` pins the gazebo image's sim-only sources: the Gazebo plugins (mit-acl
  copies with the sim-stamp fixes), DecompROS2, and the Livox SDK and driver.
- To bump a pin:
  1. Change the `*_SHA` and the short sha in its `*_IMAGE` together (`build_images.sh` refuses
     a tag that does not carry the pin).
  2. Run `make images`.
  3. Run `scripts/check_pins.py`.
- Re-verify against the robot (`check_pins.py --robot ...`) before relying on a pin. The robots
  move fast.

## Tests

`make test` (`python3 -m unittest discover -s tests`) runs on the host with Python 3.8 and
PyYAML. The numpy, cv2 and rclpy-dependent tests skip there and run in the gazebo image:

```bash
docker run --rm --network none -v "$PWD":/src:ro -w /src rover-sim/gazebo:local \
  bash -c 'source /opt/ros/humble/setup.bash && python3 -m unittest discover -s tests'
```

## Deviations from the robot

The sim changes these on purpose. `allowed_deviations.yaml` lists the parameter-level ones.

**Physics and sensors**
- **Clock:** sim time is capped at real time and catches up after overruns (at most 1.5×, never
  ahead of the wall clock). A lag over 0.5 s is accepted and re-anchored.
- **MID-360 density** is budgeted per world (`src/rover_sim/config/worlds.yaml`): 2858 rays in
  the arena, 1429 in the office: about a tenth of the robot's ~20k points per scan, which
  would not be real time with 4 agents.
- **Drive model:** the Gazebo diff drive applies the firmware's command clamp, forward-speed
  slew and 0.5 s timeout. It does not model the per-wheel PI loop or the ARM latch. The RR uses
  an effective skid-steer track (0.578 m) so that in-place yaw matches the command.
- **Calibration** reaches sim-drive's rover_description share through symlinks, not a mount.
- **Worlds:** swap and smoke run in a walled arena (DLIO drifts metres on a bare floor). The
  office has sim-only pilasters, shelves and a corridor ceiling, where DLIO slid along bare
  corridors.

**Bring-up and tooling**
- Stacks start in waves into a paused world, and each pane prints one extra start-marker line.
- Panes start 0.25 s apart, and a graph check of each agent's data path runs before the unpause.
- The DLIO pane waits until its seed pose has reached its container (`wait_for_seed.py`).
- TF gates:
  - `up` stops on a planner TF-gate timeout (the robot proceeds degraded);
  - a gate goes on after its logged verdict even if its zenoh session then fails to close.
- Planner sessions arm before the unpause (on the robot an operator arms).
- Every sim tool leaves the graph 0.5 s per entity; `status` takes about 22 s with 4 agents.
- cupy's kernel cache is a named docker volume (the Orin keeps it on disk).
- `dlio_map_node` is renamed per agent; its `/save_pcd`, `/map` and keyframe subscription stay
  at the root, shared.
- `persistent_map save_dir:=none`: maps stay in memory and nothing carries over between runs.

**Planner and demos**
- `num_agents:=<highest id>` is the only planner parameter that differs from JR02.
- **First drive:** goal_relay drives each agent 1.2 m straight ahead in safety_node's JOY mode,
  then back to AUTO, before any goal. It stands in for the operator's drive off the spot,
  without which a cold-started rover plans nothing (Open issues 2).
- **Swap goals** sit 20° past the opposite start, not on it: mighty refuses a terminal goal on
  an occupied cell.
- The goals stack (goal_relay.py) runs for every planner session, in the last paused wave.
- **Not parity, labelled:** `mapping: fallback` and `odometry: truth`.

## Open issues

1. **mighty: an occupied terminal goal within `horizon` can never be planned to**, and
   remembered robots occupy goals (robot-side, multi-robot). Its fallback subgoal is the goal
   itself, so the replan fails silently and goal_selector gives up. This causes most swap and
   explore leg timeouts.
2. **mighty cannot leave a spot whose surroundings it never saw** (robot-side, cold start). The
   sim works around it with the first drive.
3. **rmw_zenoh 0.1.9 stalls the graph:**
   - about 0.3 s per undeclaration on a 4-agent planner graph;
   - declarations lost at bring-up, which the graph check now catches;
   - with RViz up and more than 2 agents, a probe joining mid-run stalled delivery for tens of
     seconds.

   Run graded tools without RViz, and close or restart RViz only with the world paused.
4. **Real time on a shared host.** Host processes (a browser, an antivirus scanner, other
   builds) are not pinned and can land on gzserver's CPUs. The 4-agent office explore then
   averages an RTF of 0.97, with dips. Run graded windows on a quiet host.
5. **mpc runs at 23–29 Hz instead of 30 Hz** with 4 agents: the shared CPUs are 88–95 % busy.
   Each robot has its own computers.
6. **Planner debug logs grow** in sim-mighty's `/tmp`, about 240 MB/h for 4 agents.
7. **'Dropping traj' is invisible at `--log-level error`**, on the robot too. demo_eval counts
   the received trajectories instead.
