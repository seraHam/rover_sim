# rover_sim

Standalone Docker build of the mighty ground-robot simulation stack: mighty +
mpc + global_mapper_ros (acl-mapping) + Gazebo/RViz/Livox-sim, pulled from
their upstream repos (`rover_sim.repos`, and `docker/Dockerfile`'s `MIGHTY_GIT_*`
args for mighty's `feature/rover_sim` branch) rather than requiring a local
checkout of any of them. No native ROS install needed on the host.

Your own stack runs alongside it: the container is `rover-sim`, on the
host network, so its ROS graph is shared with the host.

## First-time setup

### 1. Install Docker

**Linux**: install Docker Engine + the Compose plugin, then add yourself to
the `docker` group so you don't need `sudo` for every command:
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```
Log out and back in (or `newgrp docker`) for the group change to take effect.
Verify with `docker compose version`.

**macOS**: install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
(includes Compose). See "Other prerequisites" below for the networking/GUI
caveats before you build.

### 2. Set up SSH keys for GitHub and GitLab

This repo clones from both GitHub (`mighty`, most dependencies) and GitLab
(`mpc`, `acl-mapping`), so you need a key added to both, not just one.

```bash
# Generate a key if you don't already have one (skip if `ls ~/.ssh/id_ed25519.pub` exists)
ssh-keygen -t ed25519 -C "you@example.com"

# Make sure ssh-agent is running and has the key loaded
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Print the public key to paste in
cat ~/.ssh/id_ed25519.pub
```

Add that public key in both places:
- GitHub: **Settings → SSH and GPG keys → New SSH key** (https://github.com/settings/keys)
- GitLab: **Preferences → SSH Keys** (https://gitlab.com/-/user_settings/ssh_keys)

Verify each connection works:
```bash
ssh -T git@github.com   # expect: "Hi <username>! You've successfully authenticated..."
ssh -T git@gitlab.com   # expect: "Welcome to GitLab, @<username>!"
```

`make build` forwards whatever key `ssh-agent` is holding via BuildKit
(`--ssh default`) — the key is never baked into the image.

### 3. Other prerequisites

- Linux + X11: `make up` already runs `xhost +local:` for you, so the
  container's Gazebo/RViz/rqt windows can open on your display.
- **macOS**: `network_mode: host` (`docker/compose.yaml`) is Linux-only —
  Docker Desktop for Mac runs containers in a Linux VM with no real "host"
  network to share. Docker Desktop has a beta "host networking" toggle
  (Settings → Resources → Network) that may just work; if not, the fallback
  is bridge networking with FastDDS discovery configured across it — not yet
  built.
  GUI display also needs XQuartz on Mac instead of native X11, not yet
  wired up either.

### 4. Build the image

From the repo root:
```bash
cd rover_sim         # wherever you cloned it
make build           # 1 parallel job (default), about 1.5 h the first time
make build JOBS=4    # more jobs: faster, but uses more RAM
```
Rerunning a killed build resumes from the last finished step, as long as `JOBS`
is the same; a different `JOBS` recompiles everything.

## Usage

Start the container, from the repo root (the `make` commands all run there):
```bash
cd rover_sim # wherever you cloned it
make up      # bring the container up
```
Open a shell in it (one per terminal you need):
```bash
make shell   # docker exec -it rover-sim bash
```
Inside the container, run the 4-rover swap demo:
```bash
run_multi    # tmuxp load /home/swarm/rover_sim/launch/multi_agent_swap_demo.yaml
```
Stop the run, still inside the container:
```bash
stop         # kills everything the sim started, then the current (or most recent) tmux session
```
Stop the container, from the host:
```bash
make down    # stop the container (and everything running in it)
```
After changing `docker/Dockerfile`, `rover_sim.repos` or the mighty pin, rebuild
with `make build` (see "Build the image" above).

These files are bind-mounted over mighty's installed copies, so an edit
takes effect in a running container with no rebuild or mighty push:
`config/mighty_ground_robot.yaml`, `rviz/mighty_sim_ground_robot.rviz`,
`urdf/p3at.urdf.xacro` (camera commented out), `urdf/pioneer3at_body.urdf.xacro`.
An editor that saves by replacing the file needs `make down && make up` to
show the change. RViz's "Save Config" writes into `rviz/mighty_sim_ground_robot.rviz`
here. `docker/overrides/` does the same for two files from other pinned repos:
mpc's `mpc_sim.yaml` (sim tuning) and the Livox sim's `mid360.xacro` (lidar at
10 Hz instead of 1000). `docker/Dockerfile`, `rover_sim.repos` and mighty's C++
source need `make build`.

## Demo session (`launch/`)

`multi_agent_swap_demo.yaml` (`run_multi`): 4 ground robots (`RR01`-`RR04`),
empty world, swapping to opposite corners via `goal_monitor_node.py` (no manual
RViz goal-clicking). Windows: `sim`, `core`, `mapping`, `state`, `goals`.
`env:=ACL_office_simple` in `base_mighty.launch.py` loads the office world
instead of `empty`. Run it inside the container:
```bash
run_multi    # tmuxp load /home/swarm/rover_sim/launch/multi_agent_swap_demo.yaml
```

To stop a run, `stop` inside the container (or `make down` from the host).
`tmux kill-session` alone can leave `ros2 launch` children and the
backgrounded static TF publishers running, which then publish duplicate TF
into the next run.

## Connecting your own planner or goal sender

Per robot:
- **Goals in**: publish a `geometry_msgs/PoseStamped` (frame `map`) on
  `/RRxx/term_goal`; mighty plans and drives to it. In the demo,
  `goal_monitor_node.py` (the `goals` window) does this, so leave that window
  out of your own session to send goals yourself.
- **State out**: `/RRxx/state` (`dynus_interfaces/State`) and TF
  `map -> RRxx/base_link`.
- **Map out**: `/RRxx/occ_2d_topic` (`nav_msgs/OccupancyGrid`).

Your stack can run on the host or inside the container (`make shell`); the
container shares the host network. Inside the container everything is already on
`ROS_DOMAIN_ID=20` with FastDDS; on the host, set `ROS_DOMAIN_ID=20` and
`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`. The container always uses FastDDS:
under Zenoh, Gazebo's `/plug/set_entity_state` server can stall, so fake_sim's
pose updates stop reaching Gazebo and the lidar scans from a frozen model
(smeared map).

## Known issues

- **macOS support is unbuilt** — see "First-time setup" above.
- **`mpc_node`'s `tracking_frame`**: `onboard_mighty.launch.py` tries to pass
  it as an absolute frame (`'/' + map_frame_id`) for sim, but the pinned
  `mpc_node.py` unconditionally strips a leading `/` before checking, so it
  always gets namespace-prefixed to `{ns}/map` regardless — which nothing
  else publishes. A consumer needs a static identity transform
  (`map -> {ns}/map`) per robot to work around it (the demo session
  publishes one per robot) rather than patching the private `mpc` repo.
