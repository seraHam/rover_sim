#!/usr/bin/env python3
"""rover_sim session runner (host side) for the per-stack sim.

docker/compose.stack.yaml is a set of idle containers, one per robot stack. This
script brings them up and runs every node as a `docker exec` from a host tmux session
per stack (rsim-<stack>), the way the robot's *_host_tmux.sh units do, so Ctrl-C / Up /
Enter in a pane restarts that one node.

  scripts/sim.py up <session> [--stacks a,b] [--partial] [--no-arm] [--rviz] [--odometry MODE[:NS,...]] [--dry-run]
  scripts/sim.py down [--containers]
  scripts/sim.py attach [<stack>]
  scripts/sim.py status [--maps]
  scripts/sim.py arm [<ns>...] [--now] [--timeout S]     (safety_node AUTO)
  scripts/sim.py disarm [<ns>...]
  scripts/sim.py start rviz|goals        (restart that one stack on the running session)
  scripts/sim.py viz [--agents a,b] [--layers l,m] [--follow] [--stop] [--snapshot PNG [--snapshot-after S]]
                                         (the 2D viewer, sim_viz.py; --snapshot: one frame, no window)
  scripts/sim.py loadtest [--agents N] [--variant V|all ...]
  scripts/sim.py compose <docker compose args...>
  scripts/sim.py cpus gz|rest            (inside sim-gazebo: its panes' CPU split)

<session> is config/sessions/<name>.yaml, or a path to one:

  name: smoke
  world: arena                  # src/rover_sim/worlds/<world>.world; not empty: DLIO drifts
                                # metres on a bare floor (src/rover_sim/config/worlds.yaml)
  gui: false                    # gzclient on $DISPLAY
  mapping: fallback             # gpu (parity) | fallback (no GPU, "not parity")
  odometry: dlio                # dlio (parity) | truth (ground truth as DLIO, "not parity"); per agent too
  rviz: false                   # RViz on $DISPLAY (<run>/rviz/session.rviz)
  planner: true                 # mighty_hw.sh's panes per agent (default true)
  demo: swap                    # none | swap (start <-> goals, repeatedly) | explore
  auto_arm: true                # true: put each safety_node in AUTO once its DLIO tracks
  agents:                       # pose: world frame, [x m, y m, yaw deg]; goals [x, y] likewise
    - {name: JR81, class: jr, pose: [-2.0, 0.0, 0.0], goals: [[2.0, 0.0]]}
    - {name: RR83, class: rr, pose: [2.0, 0.0, 180.0], goals: [[-2.0, 0.0]], odometry: truth}

Agent names are a class prefix (JR, RR) plus two digits. The digits must be unique
across classes: mighty takes its agent id from them. Each class is calibrated from one
real robot, the one its DLIO image is built for (docker/stacks.env DLIO_<CLASS>_ROBOT).
ROVER_SIM_AGENTS=JR81:jr,RR82:rr,... replaces the session's agents for one `up`
(apply_agents_env). Each may carry its own pose (JR85:jr:1.0:2.0:90). The others take the
session's places in order, with their goals; with more agents than the session places, in
the arena (and empty) every one goes on a 4 m circle round the centre, facing it, its swap goal
on the far side, past the opposite start (circle_goal); elsewhere the extra ones need a pose.

Stacks are a registry (STACKS below). Each names its containers and builds its panes
from the session: `zenoh` (the router in sim-drive), `gazebo` (world, spawns, sensor
bridges, world-><ns>/map), `drive` (teleop + safety_node, rover_description per agent),
`frame_align` (ground-truth alignment for mighty), `dlio` (DLIO, its seed pose and
map->odom per agent; truth_odom.py and map->odom for odometry: truth), `mapping` (global_mapper, map_fusion, persistent_map per agent), for
`mapping: gpu` `elevation` (the Orin's elevation mapper per agent), `mighty` (mighty_hw.sh's
mighty_node, convert_odom_to_state, mpc and goal_selector per agent), with rviz `rviz`, and
for every planner session `goals` (scripts/goal_relay.py: each agent's first drive, the session's
goals, and RViz's SetGoal on /sim/goal, to each agent's term_goal_rviz in its own <ns>/map).
`mapping: fallback` is NOT PARITY: no GPU, so no elevation and no map_fusion; global_mapper
runs a sim config and grid_relay.py puts its grids on the planner's names.
`odometry: truth` is NOT PARITY too, and a control: scripts/truth_odom.py publishes Gazebo's
ground truth under DLIO's names (odom, pose, deskewed cloud, TF, dlio/session) from the image's
own DLIO cfg, so what the maps and the planner still do wrong is not DLIO's drift. It is per
agent (an agent's odometry: wins over the session's), and `up --odometry truth[:NS,...]` sets it
for one run (apply_odometry; session.json and spawn.yaml record each agent's). parity_check.py,
demo_eval.py, odom_eval.py and sim_viz.py read it there.

`up` starts them in waves into a paused world (Stack.wave): first the router and Gazebo,
whose spawn pane spawns every agent while nothing else is on the graph (wait_spawned; it
fails if the pane does); then drive, frame_align and DLIO; then mapping; then elevation;
then the planner (its wave counts as started only once each mpc has built its NLP:
Pane.ready); then RViz and the goals.
Each wave waits until its panes have started their nodes and the containers are idle
(settle_wave) before the next one starts; a pane that ends before its node starts stops `up`
there, with the world still paused, and so does a planner pane whose TF gate timed out
(wait_for_tf.py exits 0 then; <ns>/map -> <ns>/odom is on /tf while paused, so a timeout
means it is missing). Then `up` unpauses physics once, over gazebo transport
(unpause_physics). A start-up burst in the first seconds of physics costs real time, and
session joins on a loaded graph stall zenoh delivery for everyone; both made DLIO diverge in
the Phase 3 and 4 runs, so no stack joins once physics runs. That holds for RViz and for
goal_relay.py too: the relay waits for every agent to be in AUTO before it sends a goal, so
its first goal goes out the moment arming ends, before goal_selector's exploration has sent
the agent anywhere. RViz's <ns>/base_link -> <ns>/lidar comes latched on /tf_static (gated by
wait_zenoh_ready.py like every pane), everything else it draws on periodic /tf. DLIO
calibrates its IMU only after the unpause: there is no IMU while paused.

safety_node starts DISARMED, holding the agent still (zero cmd_vel) while DLIO calibrates
its IMU, and stays so, as on the robot, until `arm` calls its /<ns>/auto_mode once DLIO
registers scans from the seed pose. With auto_arm, `up` does that itself. From then on
/<ns>/cmd_vel_auto drives the agent, and safety_node zeroes the command when cmd_vel_auto
stops for 1 s. `disarm` calls /<ns>/disarm. Every call is logged to <pane-log dir>/arm.log.
The planner sessions arm (auto_arm: true), and they do it before the unpause, without the
DLIO gate: nothing commands motion until DLIO tracks (mpc publishes cmd_vel_auto at 30 Hz from
its start, zero until mighty_node sends it a path; mighty_node, goal_selector and
goal_relay.py act on DLIO's estimate), and the arm query's leave, which stalls the planner's
big graph for a moment, then meets a world with no sensor data. goal_relay.py then drives each
agent 1.2 m straight ahead in JOY mode, as the operator does before handing a rover to
autonomy, and back to AUTO: from a cold start mighty plans no path over floor the rover never
saw (its tilted MID-360 never sees the floor just in front of it). odom_eval runs no planner
(planner: false): its drives publish cmd_vel_auto themselves, and mpc's zeros would be a
second publisher there; it arms after the unpause, once DLIO registers scans.

Per-run generated files go to $ROVER_SIM_RUN (default /tmp/rover_sim-<uid>/run), which
is mounted at /rover_sim/run in every container. It must be yours and writable by no one
else: the containers run as root and write, chown and execute files there. Pane logs go
to ~/.local/state/rover_sim/logs/<timestamp>/<stack>/<pane>.log.

`status` exits 2 when a sensor bridge logged an RTF violation (real-time factor below
0.95 for over 5 s) during the current run: such a run is invalid. It exits 3 when the
isolation check fails, and then opens no ROS session to read the bridges' values. With
rmw_zenoh 0.1.9, a ROS session that leaves the loaded 4-agent mapping graph all at once stalls
zenoh delivery for everyone for 1-3 s (Phase 4: parity_check's, a probe's, status's own before
it left one subscription at a time, STATUS_LEAVE_GAP_S): run such tools before or after a
window that is being graded. `status --maps` also reads the planner's grids, the multi-MB
persistent map among them. The same holds for any node that leaves: a restarted
mighty_node undeclares the num_agents - 1 frame_align subscriptions it made (83 with ids up
to 84), and closing RViz all of its displays' subscriptions. `start rviz|goals` joins the
running graph (and its old process leaves it): do that outside a graded window too.

`viz` opens scripts/sim_viz.py's live 2D view of the agents (occ_2d_topic, both HGP paths, the
committed trajectory, the frontiers, DLIO's pose; world frame or following an agent) on $DISPLAY,
from a container of its own (sim-viz: compose profile viz, never started by `up`; Gazebo's image,
X like sim-gazebo, cores off gzserver's and their SMT siblings) in tmux session rsim-viz. It joins
the running graph as `start` does, and leaves it one subscription at a time: `viz --stop` and `down`
stop it first and wait for that, before anything else goes down (a docker stop of sim-viz waits too:
compose.stack.yaml). `viz --snapshot PNG` writes one frame of it to PNG on this host, no window.

Host needs: python3 >= 3.8 with PyYAML, tmux, the docker CLI with compose v2.
"""

import argparse
import base64
import copy
import dataclasses
import inspect
import ipaddress
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - host setup problem
    sys.exit('sim.py needs PyYAML (sudo apt install python3-yaml)')


# ---------------------------------------------------------------------------- paths

REPO = Path(__file__).resolve().parent.parent
DOCKER_DIR = REPO / 'docker'
COMPOSE_FILE = DOCKER_DIR / 'compose.stack.yaml'
LOCAL_COMPOSE_FILE = DOCKER_DIR / 'compose.local.yaml'
STACKS_ENV = DOCKER_DIR / 'stacks.env'
ZENOH_DIR = DOCKER_DIR / 'zenoh'
SESSIONS_DIR = REPO / 'config' / 'sessions'

PROJECT = 'rover-sim-stack'          # compose project; every compose call is scoped to it
NETWORK = 'rover-sim'                # internal docker network (compose.stack.yaml)
GPU_PROFILE = 'gpu'                  # compose profile of sim-elevation
TMUX_PREFIX = 'rsim-'
ROUTER_CONTAINER = 'sim-drive'
ROUTER_HOST = 'sim-drive'            # DNS name on the rover-sim network
ROUTER_PORT = 7447
FLEET_PORTS = (7447,)                # host listeners the sim must never reach (acl-c2-zenoh)

# Per user: this host is shared, and a run dir in a common /tmp path could be made a
# symlink by someone else before docker mounts it (see check_run_dir).
RUN_DIR_DEFAULT = '/tmp/rover_sim-%d/run' % os.getuid()
RUN_DIR_IN = '/rover_sim/run'        # the run dir inside every container
SESSION_URI_IN = '/rover_sim/zenoh/session.json5'
SCRIPTS_IN = '/rover_sim/scripts'    # this repo's scripts/ inside every container
WORLDS_DIR = REPO / 'src' / 'rover_sim' / 'worlds'
WORLD_BUDGETS = REPO / 'src' / 'rover_sim' / 'config' / 'worlds.yaml'   # MID-360 rays per world
CLASSES_YAML = REPO / 'src' / 'rover_sim' / 'config' / 'classes.yaml'
# rover_sim's own mapping overlays (the fallback's), mounted with config/ at /rover_sim/config.
SIM_MAPPING_DIR = REPO / 'config' / 'mapping'
SIM_MAPPING_IN = '/rover_sim/config/mapping'

# Per-run files (in the run dir; contracts with the sim's own nodes).
SESSION_FILE = 'session.json'        # Session.to_json: status, arm, start, goal_relay.py
SPAWN_FILE = 'spawn.yaml'            # agents and spawn poses: sim_tf.py, the spawn pane, goal_relay.py
CALIB_DIR = 'sensor_calib'           # <ns>_lidar_mount.yaml, <ns>_calib.npy
URDF_DIR = 'urdf'                    # <ns>.urdf, as spawned (xacro output)
FRAME_ALIGN_FILE = 'frame_align.yaml'
CLASSES_FILE = 'classes.yaml'        # the checkout's classes.yaml: drive_pattern.py's limits
TELEOP_FILE = 'teleop_sim.yaml'      # rover_control's teleop file + the sim agents' blocks
DLIO_LAUNCH_FILE = 'dlio_sim.launch.py'   # the robot's DLIO launch with a per-agent map node name
MAPPING_RUN_DIR = 'mapping'          # the fallback's param files: robot file + rover_sim overlay
RVIZ_RUN_DIR = 'rviz'                # session.rviz, from rviz/templates/ (rviz_config)
RVIZ_FILE = 'session.rviz'
ARMED_SESSIONS_FILE = 'armed_sessions.json'   # {ns: DLIO session id} of each agent's last arm (arm_z_check)
ARM_LOG = 'arm.log'                  # in the pane-log dir: every auto_mode / disarm call
BRINGUP_LOG = 'bringup.log'          # in the pane-log dir: bring_up's waves, load samples, unpause
RTF_LOG = 'rtf_violations.log'       # appended by livox_sim_bridge.py; any line = run invalid

LOG_ROOT = Path.home() / '.local' / 'state' / 'rover_sim' / 'logs'
LOG_KEEP = 20                        # newest pane-log dirs kept

# A detached session from a non-tty needs an explicit size; attaching resizes it.
TMUX_SIZE = ('220', '56')
TMUX_HISTORY = 50000                 # scrollback lines per pane (pipe-pane logs keep it all)
TMUX_PLACEHOLDER = '_starting'
PANE_SLUG_OPTION = '@rsim_slug'      # tmux pane option: which pane is which (titles are display only)
PANE_TF_GATE_OPTION = '@rsim_tf_gate'  # 'strict' on a Pane.strict_tf_gate pane: `status` reads its gate
MAX_PANES_PER_WINDOW = 8             # beyond this tmux runs out of room to split


class SimError(Exception):
    """A user-facing failure: printed without a traceback."""


class SessionError(SimError):
    pass


def info(msg: str) -> None:
    print('[sim] ' + msg, flush=True)


def warn(msg: str) -> None:
    print('[sim] WARNING: ' + msg, file=sys.stderr, flush=True)


def sh(args: Sequence[str], check: bool = True, capture: bool = True,
       env: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    args = [str(a) for a in args]
    try:
        proc = subprocess.run(args, stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.PIPE if capture else None,
                              universal_newlines=True, env=env, timeout=timeout)
    except FileNotFoundError:
        raise SimError('%s not found on PATH' % args[0])
    except subprocess.TimeoutExpired:
        raise SimError('timed out after %ss: %s' % (timeout, shlex_join(args)))
    if check and proc.returncode != 0:
        detail = (proc.stderr or '').strip() if capture else ''
        raise SimError('command failed (exit %d): %s%s' % (
            proc.returncode, shlex_join(args), '\n' + detail if detail else ''))
    return proc


def shlex_join(args: Iterable[str]) -> str:
    return ' '.join(shlex.quote(str(a)) for a in args)


def parallel(fn: Callable[[Any], Any], items: Sequence[Any]) -> List[Any]:
    """fn over items on threads (docker calls are latency-bound); results in order."""
    out = [None] * len(items)  # type: List[Any]
    errs = []  # type: List[BaseException]

    def work(i: int, item: Any) -> None:
        try:
            out[i] = fn(item)
        except BaseException as e:  # re-raised on the caller's thread
            errs.append(e)

    threads = [threading.Thread(target=work, args=(i, it)) for i, it in enumerate(items)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise errs[0]
    return out


# ---------------------------------------------------------------------------- sessions

AGENT_NAME_RE = re.compile(r'^(?P<prefix>[A-Z]{2})(?P<num>[0-9]{2})$')


@dataclass(frozen=True)
class RobotClass:
    prefix: str                        # agent name prefix
    chassis: str
    reference_key: str                 # docker/stacks.env key: the real robot it is calibrated from
    lidar_height: float                # m above the floor: base_link = lidar origin = DLIO's seed z


# The reference robot is the one the class's DLIO image bakes the cfg_overlay of, so the
# Gazebo lidar pose, rover_description's base_link->lidar and DLIO's extrinsics all come
# from the same rover. Heights are the robots' DLIO_SEED_Z.
ROBOT_CLASSES = OrderedDict([
    ('jr', RobotClass('JR', 'Pioneer 3-DX, diff drive', 'DLIO_JR_ROBOT', 0.36)),
    ('rr', RobotClass('RR', 'Pioneer 3-AT, skid steer', 'DLIO_RR_ROBOT', 0.40)),
])
CLASS_PREFIXES = OrderedDict((k, c.prefix) for k, c in ROBOT_CLASSES.items())  # class -> name prefix
MAPPING_MODES = ('gpu', 'fallback')
# Where an agent's odometry comes from: dlio, the robot's DLIO; truth (NOT PARITY), Gazebo's
# ground truth published under DLIO's names by scripts/truth_odom.py, to tell what DLIO's drift
# does from what the mapping and the planner do. Per agent, with the session's as the default.
ODOMETRY_MODES = ('dlio', 'truth')
# What the goals stack does with the planner (goal_relay.py --mode): swap drives each agent
# between its start and its goals, over and over; explore leaves the goals to goal_selector's
# exploration (hw_goal_selector.yaml has it on), as on the robot; none sends nothing.
DEMOS = ('none', 'swap', 'explore')
IDENT_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
SESSION_KEYS = ('name', 'world', 'gui', 'mapping', 'odometry', 'rviz', 'auto_arm', 'planner', 'demo', 'agents')
AGENT_KEYS = ('name', 'class', 'pose', 'goals', 'odometry')


@dataclass(frozen=True)
class Agent:
    name: str
    cls: str
    x: float
    y: float
    yaw_deg: float
    goals: Tuple[Tuple[float, ...], ...] = ()
    odometry: str = 'dlio'             # ODOMETRY_MODES: this agent's, the session's default unless it sets one

    @property
    def id(self) -> int:
        """mighty's agent id: the last two digits of the namespace (mighty_node.cpp:57-63)."""
        return int(self.name[2:])

    @property
    def prefix(self) -> str:
        return self.name[:2]

    @property
    def yaw(self) -> float:
        """Radians in [-pi, pi] (exact for a yaw_deg already in [-180, 180])."""
        return math.remainder(math.radians(self.yaw_deg), 2 * math.pi)


@dataclass(frozen=True)
class Session:
    name: str
    world: str
    mapping: str
    rviz: bool
    agents: Tuple[Agent, ...]
    source: str = ''
    gui: bool = False
    auto_arm: bool = False             # the planner sessions arm; see the module docstring
    planner: bool = True
    demo: str = 'none'
    odometry: str = 'dlio'             # the default for agents that set none (Agent.odometry is each one's)
    agents_env: str = ''               # the ROVER_SIM_AGENTS this `up` took its agents from

    @property
    def num_agents(self) -> int:
        """num_agents:= for mighty: the highest id, as the launch arg overwrites the overlay."""
        return max(a.id for a in self.agents)

    def by_class(self, cls: str) -> List[Agent]:
        return [a for a in self.agents if a.cls == cls]

    @property
    def truth_agents(self) -> List[str]:
        """The agents on ground-truth odometry (odometry: truth, NOT PARITY)."""
        return [a.name for a in self.agents if a.odometry == 'truth']

    def settings(self) -> 'OrderedDict[str, Any]':
        """The session's own keys, as its YAML spells them (agents aside)."""
        return OrderedDict([('name', self.name), ('world', self.world), ('gui', self.gui), ('mapping', self.mapping),
                            ('odometry', self.odometry), ('rviz', self.rviz), ('auto_arm', self.auto_arm),
                            ('planner', self.planner), ('demo', self.demo)])

    def to_json(self) -> Dict[str, Any]:
        out = self.settings()
        out.update([('num_agents', self.num_agents), ('source', self.source)])
        if self.agents_env:
            out['agents_override'] = self.agents_env
        out['agents'] = [agent_entry(a, with_id=True) for a in self.agents]
        return out


def agent_entry(a: Agent, with_id: bool = False) -> 'OrderedDict[str, Any]':
    """An agent as a session file spells it (validate_session takes it back), with its
    effective odometry; with_id adds mighty's id, as session.json carries it."""
    out = OrderedDict([('name', a.name), ('class', a.cls)])  # type: OrderedDict[str, Any]
    if with_id:
        out['id'] = a.id
    out.update([('pose', [a.x, a.y, a.yaw_deg]), ('goals', [list(g) for g in a.goals]), ('odometry', a.odometry)])
    return out


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_session(data: Any, source: str = '<session>') -> Session:
    """Session from parsed YAML. Collects every problem before failing, so one run
    shows all of them."""
    if not isinstance(data, dict):
        raise SessionError('%s: expected a mapping at the top level' % source)
    errs = []  # type: List[str]
    unknown = sorted(set(data) - set(SESSION_KEYS))
    if unknown:
        errs.append('unknown key(s): %s (allowed: %s)' % (', '.join(unknown), ', '.join(SESSION_KEYS)))
    name, world = data.get('name'), data.get('world')
    if not isinstance(name, str) or not IDENT_RE.match(name):
        errs.append('name: %r is not a plain identifier' % (name,))
    if not isinstance(world, str) or not IDENT_RE.match(world):
        errs.append('world: %r is not a world name' % (world,))
    mapping = data.get('mapping', 'gpu')
    if mapping not in MAPPING_MODES:
        errs.append('mapping: %r is not one of %s' % (mapping, ', '.join(MAPPING_MODES)))
    odometry = data.get('odometry', 'dlio')
    if odometry not in ODOMETRY_MODES:
        errs.append('odometry: %r is not one of %s' % (odometry, ', '.join(ODOMETRY_MODES)))
    flags = OrderedDict((k, data.get(k, d)) for k, d in (('rviz', False), ('gui', False), ('auto_arm', False),
                                                         ('planner', True)))
    for k, v in flags.items():
        if not isinstance(v, bool):
            errs.append('%s: %r is not true/false' % (k, v))
    demo = data.get('demo', 'none')
    if demo not in DEMOS:
        errs.append('demo: %r is not one of %s' % (demo, ', '.join(DEMOS)))
    elif demo != 'none' and flags['planner'] is False:
        errs.append('demo: %s needs the planner (planner: true)' % demo)

    agents = []  # type: List[Agent]
    raw = data.get('agents')
    if not isinstance(raw, list) or not raw:
        errs.append('agents: need a non-empty list')
        raw = []
    names = {}  # type: Dict[str, int]
    ids = {}  # type: Dict[str, str]
    for i, a in enumerate(raw):
        where = 'agents[%d]' % i
        before = len(errs)
        if not isinstance(a, dict):
            errs.append('%s: expected a mapping with name, class, pose' % where)
            continue
        unknown = sorted(set(a) - set(AGENT_KEYS))
        if unknown:
            errs.append('%s: unknown key(s): %s' % (where, ', '.join(unknown)))
        an, cls, pose = a.get('name'), a.get('class'), a.get('pose')
        m = AGENT_NAME_RE.match(an) if isinstance(an, str) else None
        if m is None:
            errs.append('%s.name: %r is not two capital letters and two digits (e.g. JR81)' % (where, an))
        else:
            where = '%s (%s)' % (where, an)
            prefix, num = m.group('prefix'), m.group('num')
            if prefix not in CLASS_PREFIXES.values():
                errs.append('%s: prefix %s is not a robot class (%s)' % (
                    where, prefix, ', '.join(CLASS_PREFIXES.values())))
            if num == '00':
                errs.append('%s: id 00: mighty takes num_agents:= from the highest id' % where)
            if an in names:
                errs.append('%s: duplicate of agents[%d]' % (where, names[an]))
            elif num in ids:
                errs.append('%s: id %s already used by %s: mighty keys agents and /trajs by the '
                            'last two digits, so they must be unique across classes' % (where, num, ids[num]))
            names.setdefault(an, i)
            ids.setdefault(num, an)
        if cls not in CLASS_PREFIXES:
            errs.append('%s.class: %r is not one of %s' % (where, cls, ', '.join(CLASS_PREFIXES)))
        elif m is not None and CLASS_PREFIXES[cls] != m.group('prefix'):
            errs.append('%s: class %s needs the name prefix %s' % (where, cls, CLASS_PREFIXES[cls]))
        if not (isinstance(pose, list) and len(pose) == 3 and all(_is_num(v) for v in pose)):
            errs.append('%s.pose: %r is not [x, y, yaw_deg]' % (where, pose))
        goals = a.get('goals', [])
        if not (isinstance(goals, list) and all(
                isinstance(g, list) and len(g) in (2, 3) and all(_is_num(v) for v in g) for g in goals)):
            errs.append('%s.goals: %r is not a list of [x, y] or [x, y, yaw_deg]' % (where, goals))
        odo = a.get('odometry', odometry if odometry in ODOMETRY_MODES else 'dlio')
        if odo not in ODOMETRY_MODES:
            errs.append('%s.odometry: %r is not one of %s' % (where, odo, ', '.join(ODOMETRY_MODES)))
        if len(errs) == before:
            agents.append(Agent(an, cls, float(pose[0]), float(pose[1]), float(pose[2]),
                                tuple(tuple(float(v) for v in g) for g in goals), odo))
    # goal_relay.py swaps each agent between its start and its goals; an agent without goals
    # stays put (an obstacle to the others). Exploration picks its own goals.
    if demo == 'swap' and agents and not any(a.goals for a in agents):
        errs.append('demo: swap needs goals (each agent drives between its start and its goals)')
    if demo == 'explore' and any(a.goals for a in agents):
        errs.append('demo: explore takes no goals (goal_selector picks frontiers): %s'
                    % ', '.join(a.name for a in agents if a.goals))
    if errs:
        raise SessionError('%s: invalid session:\n  - %s' % (source, '\n  - '.join(errs)))
    return Session(name, world, mapping, flags['rviz'], tuple(agents), source, flags['gui'], flags['auto_arm'],
                   flags['planner'], demo, odometry)


def session_path(arg: str) -> Path:
    if arg.endswith(('.yaml', '.yml')) or os.sep in arg:
        return Path(arg).expanduser()
    return SESSIONS_DIR / (arg + '.yaml')


def load_session(arg: str) -> Session:
    path = session_path(arg)
    if not path.is_file():
        have = sorted(p.stem for p in SESSIONS_DIR.glob('*.yaml')) if SESSIONS_DIR.is_dir() else []
        raise SessionError('no session %s (available: %s)' % (path, ', '.join(have) or 'none'))
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise SessionError('%s: %s' % (path, e))
    try:
        source = str(path.resolve().relative_to(REPO))
    except ValueError:
        source = str(path)
    return validate_session(data, source)


def session_from_json(data: Any, where: str = SESSION_FILE) -> Session:
    """The running session back from <run>/session.json (Session.to_json), validated again."""
    try:
        raw = dict((k, data[k]) for k in SESSION_KEYS if k in data and k != 'agents')
        raw['agents'] = [dict((k, a[k]) for k in AGENT_KEYS if k in a) for a in data['agents']]
    except (KeyError, TypeError) as e:
        raise SessionError('%s: not a session file (%r)' % (where, e))
    s = validate_session(raw, where)
    return dataclasses.replace(s, source=str(data.get('source') or where),
                               agents_env=str(data.get('agents_override', '')))


def load_running_session(run_dir: Path) -> Session:
    sf = run_dir / SESSION_FILE
    try:
        data = json.loads(sf.read_text())
    except (OSError, ValueError) as e:
        raise SimError('no running session (%s: %s): scripts/sim.py up <session>' % (sf, e))
    return session_from_json(data, str(sf))


AGENTS_ENV = 'ROVER_SIM_AGENTS'
# Worlds with a default layout for more agents than a session places: a circle (centre x, y,
# radius, m), the first agent at 45 degrees as in multi_agent_swap, every agent facing the
# centre. The arena's inner obstacles stay 2.2 m or more off a 4 m circle for any count up to
# CIRCLE_MAX_AGENTS (2.1 m apart on it); the office has no free circle.
AGENT_CIRCLES = {'arena': (0.0, 0.0, 4.0), 'empty': (0.0, 0.0, 4.0)}
CIRCLE_MAX_AGENTS = 12


def parse_agents_env(text: str) -> List[Tuple[str, str, Optional[Tuple[float, float, float]]]]:
    """ROVER_SIM_AGENTS: comma-separated NAME:class, or NAME:class:x:y:yaw_deg with a pose of
    its own (world frame). validate_session checks the names and classes."""
    out = []  # type: List[Tuple[str, str, Optional[Tuple[float, float, float]]]]
    for item in (t.strip() for t in text.split(',')):
        if not item:
            continue
        f = item.split(':')
        pose = None  # type: Optional[Tuple[float, float, float]]
        if len(f) == 5:
            try:
                pose = (float(f[2]), float(f[3]), float(f[4]))
            except ValueError:
                pose = None
            if pose is None or not all(math.isfinite(v) for v in pose):
                raise SessionError('%s: %r: x, y and yaw_deg must be numbers' % (AGENTS_ENV, item))
        elif len(f) != 2:
            raise SessionError('%s: %r is not NAME:class or NAME:class:x:y:yaw_deg (e.g. JR81:jr)' % (AGENTS_ENV, item))
        out.append((f[0], f[1], pose))
    if not out:
        raise SessionError('%s is set but names no agent' % AGENTS_ENV)
    return out


def circle_layout(world: str, n: int) -> Optional[List[Tuple[float, float, float]]]:
    """n poses (x, y, yaw_deg) on the world's AGENT_CIRCLES circle, or None."""
    if world not in AGENT_CIRCLES or not 1 <= n <= CIRCLE_MAX_AGENTS:
        return None
    cx, cy, r = AGENT_CIRCLES[world]
    out = []
    for k in range(n):
        ang = math.pi / 4 + 2 * math.pi * k / n
        x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
        out.append((round(x, 2), round(y, 2), round(math.degrees(math.atan2(cy - y, cx - x)), 1)))
    return out


# multi_agent_swap.yaml's: each goal 20 deg past the opposite start, as mighty refuses a terminal goal
# on the cell of the agent parked there (and the paths then pass near the centre, not through it).
SWAP_GOAL_PAST_DEG = 20.0
SWAP_GOAL_CLEAR_M = 1.0                # circle_goal keeps every goal this far from every start


def circle_goal(world: str, pose: Sequence[float], n: int) -> List[float]:
    """The swap goal [x, y] of the agent at pose, one of n on the world's circle: SWAP_GOAL_PAST_DEG
    past the point opposite it, less where n is large enough to bring that within SWAP_GOAL_CLEAR_M of
    a start (for an even n the opposite point is a start, for an odd n it lies midway between two)."""
    cx, cy, r = AGENT_CIRCLES[world]
    spacing = 360.0 / n
    clear = math.degrees(2 * math.asin(SWAP_GOAL_CLEAR_M / (2 * r)))
    past = min(SWAP_GOAL_PAST_DEG, spacing / 2 if n % 2 == 0 else max(0.0, spacing / 2 - clear))
    a = math.atan2(pose[1] - cy, pose[0] - cx) + math.pi + math.radians(past)
    return [round(cx + r * math.cos(a), 2), round(cy + r * math.sin(a), 2)]


def apply_agents_env(session: Session, text: str) -> Session:
    """The session with ROVER_SIM_AGENTS's agents, in its order. A pose given in the variable
    wins (and has no goals). Otherwise, with as many agents as the session places or fewer,
    agent i takes the session's place i and its goals; with more, every one goes on the world's
    circle (circle_layout; swap goals past the far side: circle_goal), and in a world without one the extra
    agents need a pose. Odometry goes by name, not by place: an agent the session names keeps
    its odometry, any other takes the session's default."""
    entries = parse_agents_env(text)
    slots, n = session.agents, len(entries)
    odometry = dict((a.name, a.odometry) for a in slots)
    circle = circle_layout(session.world, n) if n > len(slots) else None
    raw = []
    for i, (name, cls, pose) in enumerate(entries):
        goals = []  # type: List[List[float]]
        if pose is None and circle is not None:
            pose = circle[i]
            if session.demo == 'swap':
                goals = [circle_goal(session.world, pose, n)]
        elif pose is None and i < len(slots):
            pose, goals = (slots[i].x, slots[i].y, slots[i].yaw_deg), [list(g) for g in slots[i].goals]
        elif pose is None:
            raise SessionError('%s: %s has no pose: session %s places %d agent(s), and world %s has no default '
                               'layout for %d (a circle in %s, up to %d agents); give it one as %s:%s:x:y:yaw_deg'
                               % (AGENTS_ENV, name, session.name, len(slots), session.world, n,
                                  ', '.join(sorted(AGENT_CIRCLES)), CIRCLE_MAX_AGENTS, name, cls))
        raw.append({'name': name, 'class': cls, 'pose': list(pose), 'goals': goals})
        if name in odometry:
            raw[-1]['odometry'] = odometry[name]
    data = session.settings()
    data['agents'] = raw
    s = validate_session(data, '%s with %s=%s' % (session.source, AGENTS_ENV, text))
    return dataclasses.replace(s, source=session.source, agents_env=text)


def parse_odometry_arg(text: str) -> Tuple[str, Tuple[str, ...]]:
    """up --odometry MODE[:NS[,NS...]]: (mode, names); no names means every agent."""
    mode, sep, rest = text.partition(':')
    names = tuple(n.strip() for n in rest.split(',') if n.strip())
    if mode not in ODOMETRY_MODES or (sep and not names):
        raise argparse.ArgumentTypeError('%r is not MODE or MODE:NS[,NS...] with MODE one of %s (e.g. truth:JR81)'
                                         % (text, ', '.join(ODOMETRY_MODES)))
    return mode, names


def apply_odometry(session: Session, mode: str, names: Sequence[str] = ()) -> Session:
    """The session with `mode` odometry for the named agents, or, with no names, for every
    agent and as the default (so --odometry dlio also takes a truth session back to DLIO)."""
    unknown = sorted(set(names) - set(a.name for a in session.agents))
    if unknown:
        raise SessionError('--odometry: %s not in session %s (agents: %s)' % (
            ', '.join(unknown), session.name, ', '.join(a.name for a in session.agents)))
    data = session.settings()
    if not names:
        data['odometry'] = mode
    data['agents'] = [agent_entry(dataclasses.replace(a, odometry=mode) if not names or a.name in names else a)
                      for a in session.agents]
    s = validate_session(data, '%s with --odometry %s%s' % (session.source, mode,
                                                             ':' + ','.join(names) if names else ''))
    return dataclasses.replace(s, source=session.source, agents_env=session.agents_env)


def world_problem(world: str, worlds_dir: Path = WORLDS_DIR) -> Optional[str]:
    """world.launch.py loads share/rover_sim/worlds/<world>.world from the gazebo image,
    which is built from this checkout's src/rover_sim: a world missing here is (or will
    be, after the next `make images`) missing there too."""
    if (worlds_dir / (world + '.world')).is_file():
        return None
    have = sorted(p.stem for p in worlds_dir.glob('*.world')) if worlds_dir.is_dir() else []
    return 'world %r is not in %s (have: %s)' % (world, worlds_dir, ', '.join(have) or 'none')


LIDAR_SAMPLES_DEFAULT = 500             # spawn_agent.launch.py's, for a world with no budget


def world_lidar_samples(world: str, path: Path = WORLD_BUDGETS) -> int:
    """MID-360 rays cast per scan in this world (spawn_agent.launch.py lidar_samples):
    src/rover_sim/config/worlds.yaml lidar_samples_4_agents, what 4 agents can cast and keep
    real time (each scan stalls physics while it casts). It is a per-scan cost, so fewer
    agents get the same. The robot casts 20000."""
    try:
        budgets = (yaml.safe_load(path.read_text()) or {}).get('lidar_samples_4_agents') or {}
    except (OSError, yaml.YAMLError) as e:
        raise SimError('%s: %s' % (path, e))
    n = budgets.get(world, LIDAR_SAMPLES_DEFAULT)
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 20000:
        raise SimError('%s: lidar_samples_4_agents.%s = %r is not 1..20000' % (path, world, n))
    return n


def world_walls(world: str, path: Path = WORLD_BUDGETS) -> Optional[List[float]]:
    """[xmin, ymin, xmax, ymax], the inner faces of the world's outer walls (worlds.yaml
    walls), or None for a world that has no such box. spawn.yaml carries it to
    drive_pattern.py, which refuses a plan that leaves it."""
    try:
        walls = (yaml.safe_load(path.read_text()) or {}).get('walls') or {}
    except (OSError, yaml.YAMLError) as e:
        raise SimError('%s: %s' % (path, e))
    box = walls.get(world)
    if box is None:
        return None
    if not (isinstance(box, list) and len(box) == 4 and all(_is_num(v) for v in box)
            and box[0] < box[2] and box[1] < box[3]):
        raise SimError('%s: walls.%s = %r is not [xmin, ymin, xmax, ymax]' % (path, world, box))
    return [float(v) for v in box]


def read_pins(path: Path = STACKS_ENV) -> Dict[str, str]:
    """docker/stacks.env: plain KEY=VALUE lines (see its header)."""
    out = OrderedDict()  # type: Dict[str, str]
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def reference_robot(cls: str, pins: Dict[str, str]) -> str:
    rc = ROBOT_CLASSES[cls]
    ref = pins.get(rc.reference_key, '')
    if not AGENT_NAME_RE.match(ref) or ref[:2] != rc.prefix:
        raise SimError('docker/stacks.env: %s=%r is not a %s robot' % (rc.reference_key, ref, rc.prefix))
    return ref


# ---------------------------------------------------------------------------- containers

ROS_SETUP = '/opt/ros/humble/setup.bash'
SIM_WS = '/home/swarm/code/rover_sim_ws'          # rover_sim's own images (gazebo, elevation)
SIM_PROBE = '/rover_sim/scripts/wait_zenoh_ready.py'
MAPPING_WS = '/home/swarm/code/mapping_ws'         # mapping_ws docker/Dockerfile.mapping ARG WS
# On the Orin, mapping_ws tools/install_orin.sh installs scripts/wait_zenoh_ready.py here and
# config/elevation/ under ELEVATION_OVERLAY_IN; compose.stack.yaml mounts the pinned checkout's.
ORIN_PROBE = '/usr/local/lib/rover/wait_zenoh_ready.py'


@dataclass(frozen=True)
class Container:
    name: str
    setup: Tuple[str, ...]           # sourced in order; a missing one fails the pane
    overlays: Tuple[str, ...] = ()   # sourced when present
    # wait_zenoh_ready.py candidates (first found runs): where each image bakes it,
    # then a rover_sim copy if scripts/ ever carries one.
    probes: Tuple[str, ...] = (SIM_PROBE,)
    role: str = ''


CONTAINERS = OrderedDict((c.name, c) for c in (
    # Gazebo's own setup (GAZEBO_RESOURCE_PATH, plugin paths) for gzserver, livox_ws for
    # the bridges' livox_ros_driver2 messages, the workspace last so its prefix wins.
    Container('sim-gazebo', (ROS_SETUP, '/usr/share/gazebo/setup.bash', '/opt/livox_ws/install/setup.bash',
                             SIM_WS + '/install/setup.bash'),
              probes=(SIM_WS + '/scripts/wait_zenoh_ready.py', SIM_PROBE),
              role='Gazebo, sensor bridges, sim TF, RViz'),
    Container('sim-drive', (ROS_SETUP, '/home/swarm/code/rover_ws/install/setup.bash'),
              probes=('/home/swarm/code/rover_ws/install/share/rover_bringup/scripts/wait_zenoh_ready.py',
                      SIM_PROBE),
              role='zenoh router; safety_node/teleop, rover_description'),
    Container('sim-dlio-jr', (ROS_SETUP, '/home/swarm/code/dlio_ws/install/setup.bash'),
              probes=('/home/swarm/code/dlio_ws/scripts/wait_zenoh_ready.py', SIM_PROBE),
              role='DLIO with the JR02 overlay'),
    Container('sim-dlio-rr', (ROS_SETUP, '/home/swarm/code/dlio_ws/install/setup.bash'),
              probes=('/home/swarm/code/dlio_ws/scripts/wait_zenoh_ready.py', SIM_PROBE),
              role='DLIO with the RR08 overlay'),
    Container('sim-mapping', (ROS_SETUP, MAPPING_WS + '/install/setup.bash'),
              probes=(MAPPING_WS + '/install/share/rover_bringup/scripts/wait_zenoh_ready.py', SIM_PROBE),
              role='global_mapper, map_fusion, persistent_map'),
    Container('sim-elevation', (ROS_SETUP, SIM_WS + '/install/setup.bash'),
              probes=(ORIN_PROBE, SIM_WS + '/scripts/wait_zenoh_ready.py', SIM_PROBE),
              role='elevation_mapping_cupy (GPU)'),
    # mighty-hw bakes no probe: its own mighty_hw.sh gates on the router port only.
    Container('sim-mighty', (ROS_SETUP, '/home/swarm/code/mighty_ws/install/setup.bash'),
              role='mighty_node, convert_odom_to_state, mpc, goal_selector; frame_align'),
    # Gazebo's image (rclpy, numpy, cv2), on demand only (profile viz): `sim.py viz`.
    Container('sim-viz', (ROS_SETUP,), role='sim_viz.py, the 2D viewer (sim.py viz; profile viz)'),
))

# rover_description's share in the drive image (a merged colcon install): the pinned
# sensor_calib files of the reference robots, and where state_publisher.launch.py looks
# for <robot_name>_lidar_mount.yaml and <robot_name>_calib.npy.
RD_SHARE = '/home/swarm/code/rover_ws/install/share/rover_description'
DESCRIPTION_CONTAINER = 'sim-drive'
# The robot's teleop/safety parameters, as the drive image installs them (rover_control).
TELEOP_SRC = '/home/swarm/code/rover_ws/install/share/rover_control/config/teleop_logitech.yaml'

# dlio_ws as its image bakes it. On the robot, the pinned checkout's scripts/ is bind-mounted
# over the image's (dlio_ws docker/compose.yaml); compose.stack.yaml mounts build_images.sh's
# checkout of DLIO_SHA there, read-only.
DLIO_WS = '/home/swarm/code/dlio_ws'
DLIO_SCRIPTS = ('dlio_with_session.launch.py', 'dlio_session.py', 'map_odom_tf.py', 'wait_zenoh_ready.py')
# The cfg DLIO loads in each class's image (dlio.launch.py: cfg/dlio.yaml, then cfg/params.yaml):
# truth_odom.py takes the extrinsics, the crop box and the calibration time from it too.
DLIO_CFG_IN = DLIO_WS + '/install/share/direct_lidar_inertial_odometry/cfg'
TRUTH_ODOM = 'truth_odom.py'         # this repo's scripts/, mounted at SCRIPTS_IN


def sim_only(name: str) -> str:
    """Every docker action on a container goes through this: sim-* registry names only.
    The fleet's containers (acl-c2-*, *-exporter) share this host."""
    if name not in CONTAINERS or not name.startswith('sim-'):
        raise SimError('refusing to act on non-sim container %r' % name)
    return name


def dlio_container(agent: Agent) -> str:
    return 'sim-dlio-' + agent.cls


# ---------------------------------------------------------------------------- panes

@dataclass(frozen=True)
class Pane:
    title: str
    slug: str                          # log file name, unique within its stack
    container: str
    cmd: str                           # in-container command, after setup and gate
    gate: bool = True                  # wait for the router and latched delivery first
    env: Tuple[Tuple[str, str], ...] = ()
    window: str = 'main'
    waits: str = ''                    # the pane's own gates (a TF, another pane), after the router's
    # A regex the pane log shows once its node is really up (after the node marker): `up` waits
    # for it before the wave counts as started, e.g. until mpc has built its NLP.
    ready: str = ''
    # Its wait_for_tf.py gate must pass. The script exits 0 on its timeout ("proceeding anyway
    # (degraded)"), so `up` reads the pane log and stops on that line (settle_wave), and `status`
    # flags the pane.
    strict_tf_gate: bool = False


def require_readable(var: str) -> str:
    """Fail the pane loudly if a config URI env var does not name a readable file. A
    node without its zenoh config would silently fall back to rmw_zenoh's defaults."""
    return ('{ test -r "$%s" || { echo "[rover_sim] %s=${%s:-<unset>} is not readable" >&2; false; }; }'
            % (var, var, var))


def setup_snippet(c: Container) -> str:
    steps = ['source ' + f for f in c.setup]
    steps += ['{ [ ! -f %s ] || source %s; }' % (f, f) for f in c.overlays]
    return ' && '.join(steps)


def tcp_connect(target: str) -> str:
    """A bash test that connects to <ip-or-host>/<port> and closes again without sending
    a byte (the usual `echo >/dev/tcp/...` writes a newline into whatever listens)."""
    return '(exec 3<>/dev/tcp/%s && exec 3>&-)' % target


def tcp_wait(host: str = ROUTER_HOST, port: int = ROUTER_PORT) -> str:
    return ('until %s 2>/dev/null; do echo "waiting for zenoh router (%s:%d)..."; '
            'sleep 2; done' % (tcp_connect('%s/%d' % (host, port)), host, port))


def router_gate(c: Container) -> str:
    """The robot's pane gate: the router's port, then wait_zenoh_ready.py, which proves a
    late joiner receives latched samples (a session formed against a still-starting
    router can stay broken for /tf_static)."""
    probe = ''
    for i, p in enumerate(c.probes):
        probe += '%s [ -f %s ]; then python3 %s; ' % ('if' if i == 0 else 'elif', p, p)
    probe += 'else echo "[rover_sim] no wait_zenoh_ready.py in %s: gated on the router port only"; fi' % c.name
    return ' && '.join([require_readable('ZENOH_SESSION_CONFIG_URI'), tcp_wait(), probe])


# Printed by every pane once its gates have passed, right before its node starts: `up` waits for
# it in every pane log before it unpauses physics (wait_nodes_started). The command line in the
# log has $(date ...) where the pane's output has the time. The containers' clock is UTC and the
# host's lines (bringup.log, arm.log, `up`) are local: both carry their offset.
PANE_TIME = '$(date +%FT%T%z)'
PANE_TIME_RE = r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d{4}'
HOST_TIME_FMT = '%Y-%m-%dT%H:%M:%S%z'
NODE_STARTED_RE = re.compile(r'\[rover_sim\] node starting at ' + PANE_TIME_RE)


def node_marker() -> str:
    return 'echo "[rover_sim] node starting at %s"' % PANE_TIME


def node_started(text: str, ready: str = '') -> bool:
    """A pane log shows its node started: the node marker, and the pane's ready line after the
    last marker (a restarted pane's log holds the earlier runs too)."""
    marks = list(NODE_STARTED_RE.finditer(text))
    if not marks:
        return False
    return not ready or re.search(ready, text[marks[-1].end():]) is not None


# wait_for_tf.py's two outcomes (mighty's and mapping_ws's copies log the same): it exits 0 on
# both, so only its log tells them apart.
WAIT_FOR_TF_RE = re.compile(r'\[wait_for_tf\]: (?:(all transforms available; proceeding)'
                            r'|TIMEOUT after \S+; still missing .*; proceeding anyway \(degraded\))')


# A wait_for_tf.py that has logged its verdict can still exit 1: its rclpy.shutdown() raises when
# zenoh times out closing the session ("Failed to close the session"), which happened on a loaded
# 4-agent graph when every planner pane's gate passed in the same second and 8 sessions closed at
# once (2026-10-20). Its && chain then ends the pane before the node starts. The verdict, not that
# exit, is the gate's outcome (it exits 0 on a timeout too: tf_gate_result reads the log), so the
# panes run the robot's gate inside verdict_gate: as is, but going on after a logged verdict.
VERDICT_GATE_RE = r'\[wait_for_tf\]: .*proceeding'


def verdict_gate(cmd: str) -> str:
    """cmd, a wait_for_tf.py gate, passing when it exits 0 or has logged its verdict (bash, in the
    container; its output still reaches the pane, through tee)."""
    return ('{ g=/tmp/rsim-gate.$$; %s 2>&1 | tee $g; rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] || '
            '{ grep -q \'%s\' $g && echo "[rover_sim] the TF gate exited $rc after its verdict: going on"; }; '
            'ok=$?; rm -f $g; [ $ok -eq 0 ]; }' % (cmd, VERDICT_GATE_RE))


def tf_gate_result(text: str) -> Optional[str]:
    """'ok' or 'degraded' from the last wait_for_tf.py run in a pane log; None if none ran."""
    last = None
    for m in WAIT_FOR_TF_RE.finditer(text):
        last = 'ok' if m.group(1) else 'degraded'
    return last


def pane_inner(p: Pane) -> str:
    c = CONTAINERS[p.container]
    steps = [setup_snippet(c)]
    if p.gate:
        steps.append(router_gate(c))
    if p.waits:
        steps.append(p.waits)
    steps += [node_marker(), p.cmd]
    return ' && '.join(steps)


def pane_command(p: Pane) -> str:
    """The host-side line typed into the pane. The in-container command is one
    single-quoted bash -c argument, so its $VARs expand inside the container."""
    args = ['docker', 'exec', '-it']
    for k, v in p.env:
        args += ['-e', '%s=%s' % (k, v)]
    args += [sim_only(p.container), 'bash', '-c', pane_inner(p)]
    line = shlex_join(args)
    # tmux splits its argv on a trailing ';' (command separator).
    assert not line.endswith(';'), line
    return line


# ---------------------------------------------------------------------------- stacks

@dataclass(frozen=True)
class RunContext:
    run_dir: Path                      # host side of /rover_sim/run
    log_dir: Optional[Path] = None     # this `up`'s pane logs
    rust_log: str = 'zenoh=info'
    gz_spare_cpus: Optional[int] = None  # $ROVER_SIM_GZ_SPARE_CPUS: sim-gazebo cores kept off gzserver


def _always(session: Session) -> bool:
    return True


def _never(session: Session) -> bool:
    return False


@dataclass(frozen=True)
class Stack:
    name: str
    phase: int
    summary: str
    containers: Tuple[str, ...]
    panes: Callable[[Session, RunContext], List[Pane]]
    implemented: bool = False
    wanted: Callable[[Session], bool] = _always
    # Steps that write per-run files under ctx.run_dir (or set up a container) before
    # any pane starts. A step shared by several selected stacks runs once.
    generate: Tuple[Callable[[Session, RunContext], None], ...] = ()
    # Opens windows on $DISPLAY: `up` then grants the containers' root X access.
    needs_display: Callable[[Session], bool] = _never
    # `up` starts the stacks wave by wave, each wave once the one before has settled
    # (settle_wave): stacks of one wave start together.
    wave: int = 0


def zenoh_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """The router: as drive_host_tmux.sh's first pane, but on the sim network. It is the
    one pane with no gate. RUST_LOG has to be passed explicitly: docker exec does not
    carry the host's. At zenoh=info the router logs only its own start and close, not
    each session; raise it with RUST_LOG_ZENOH=<filter> on `sim.py up`."""
    return [Pane(
        title='zenoh router (%s:%d)' % (ROUTER_HOST, ROUTER_PORT), slug='router',
        container=ROUTER_CONTAINER, gate=False, env=(('RUST_LOG', ctx.rust_log),),
        cmd=require_readable('ZENOH_ROUTER_CONFIG_URI')
        + ' && ros2 run rmw_zenoh_cpp rmw_zenohd --ros-args -r __node:=zenoh_router')]


def _not_yet(stack: str) -> SimError:
    s = STACKS[stack]
    return SimError('the %s stack lands in Phase %d (%s)' % (stack, s.phase, s.summary))


def fmt_float(v: float, digits: int = 9) -> str:
    """Fixed point, always with a decimal point: ROS reads `-p tx:=1` as an integer, which
    a double parameter refuses. No exponent, no '-0.0'."""
    s = ('%.*f' % (digits, v)).rstrip('0')
    s = s + '0' if s.endswith('.') else s
    return '0.0' if s == '-0.0' else s


def run_path(*parts: str) -> str:
    """A per-run file as the containers see it."""
    return '/'.join((RUN_DIR_IN,) + parts)


def calib_name(agent: Agent, kind: str) -> str:
    return '%s_%s' % (agent.name, {'mount': 'lidar_mount.yaml', 'npy': 'calib.npy'}[kind])


# gzserver gets most of sim-gazebo's cores (its own cpuset, cpu_budget) and everything else
# in that container (bridges, sim_tf, spawn, status queries, later RViz) the rest, so a busy
# bridge cannot preempt physics: an RTF below 0.95 fails the run. The split is made at run
# time from the container's own affinity, by this script (`sim.py cpus`), which every
# container mounts at /rover_sim/scripts.
SIM_PY_IN = SCRIPTS_IN + '/sim.py'


def split_cpus(cpus: Iterable[int], spare: Optional[int] = None) -> Tuple[List[int], List[int]]:
    """(gzserver's cores, the rest's). The rest gets the top `spare` cores, by default a
    quarter (at least one); gzserver keeps at least one. A single core is shared."""
    c = sorted(set(cpus))
    if len(c) < 2:
        return c, c
    k = min(spare if spare else max(1, len(c) // 4), len(c) - 1)
    return c[:len(c) - k], c[len(c) - k:]


def pin_self(part: str, ctx: RunContext) -> str:
    """Pins the pane's container-side bash, and so all it starts, to gzserver's cores
    ('gz') or the rest ('rest'). taskset prints the old and new affinity into the pane."""
    spare = ' --spare %d' % ctx.gz_spare_cpus if ctx.gz_spare_cpus else ''
    return 'taskset -cp "$(python3 %s cpus %s%s)" $$' % (SIM_PY_IN, part, spare)


def spawn_agent_cmd(a: Agent, lidar_samples: int) -> str:
    """rover_sim spawn_agent.launch.py: xacro to a file in the run dir (kept, for a look at
    what was spawned), then spawn_entity -file at floor height; no robot_state_publisher,
    so the Gazebo model publishes no TF. The lidar pose comes from the agent's generated
    mount file, the one rover_description's base_link->lidar uses too."""
    return ('ros2 launch rover_sim spawn_agent.launch.py namespace:=%s class:=%s x:=%s y:=%s yaw:=%s '
            'lidar_mount_file:=%s lidar_height:=%s lidar_samples:=%d urdf_file:=%s'
            % (a.name, a.cls, fmt_float(a.x), fmt_float(a.y), fmt_float(a.yaw),
               run_path(CALIB_DIR, calib_name(a, 'mount')), fmt_float(ROBOT_CLASSES[a.cls].lidar_height),
               lidar_samples, run_path(URDF_DIR, a.name + '.urdf')))


def spawn_cmd(session: Session, ctx: RunContext) -> str:
    """Into the paused world one by one. The world stays paused until `up` has every stack
    running and unpauses it (bring_up), so no robot moves (or DLIO calibrates on a
    falling IMU) before all are in. The pane then idles: `status` shows it running when
    everything spawned, EXITED when a step failed. Each CLI has a SIGKILL backstop: gzserver
    has stopped answering ROS services after start-up (1 in 10 bring-ups), and a call that
    never returns would leave the pane waiting forever."""
    names = ' '.join(a.name for a in session.agents)
    samples = world_lidar_samples(session.world)
    return ' && '.join(
        [pin_self('rest', ctx), 'echo "[rover_sim] waiting for /spawn_entity (gzserver)..."',
         'until timeout -k 2 15 ros2 service list --no-daemon 2>/dev/null | grep -qx /spawn_entity; do sleep 2; done']
        + [spawn_agent_cmd(a, samples) for a in session.agents]
        + ['echo "[rover_sim] spawned %s into the paused world at %s; scripts/sim.py up unpauses it"'
           % (names, PANE_TIME), 'exec sleep infinity'])


def gazebo_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """sim-gazebo. One gzserver on session.world (rover_sim world.launch.py, ODE at 1 ms,
    real-time capped), started paused; the spawn pane adds every agent, and `up` unpauses it.
    Per agent, livox_sim_bridge.py turns the plugins' sim-stamped sensor streams into the
    robot's /<ns>/livox/{lidar,imu} on wall-clock stamps and reports sim/rtf and
    sim/time_offset. sim_tf.py publishes world-><ns>/map for every agent from
    <run>/spawn.yaml (write_agent_files). With session.gui, gzclient gets a pane of its own,
    off gzserver's cores: the worlds' real-time budgets were measured without it."""
    panes = [
        Pane('gzserver %s' % session.world, 'gzserver', 'sim-gazebo', pin_self('gz', ctx)
             + ' && ros2 launch rover_sim world.launch.py world:=%s gui:=false paused:=true' % session.world,
             window='gazebo'),
        Pane('spawn %s (paused)' % ', '.join(a.name for a in session.agents), 'spawn', 'sim-gazebo',
             spawn_cmd(session, ctx), window='gazebo'),
        Pane('world -> <ns>/map (sim_tf)', 'sim-tf', 'sim-gazebo', pin_self('rest', ctx)
             + ' && python3 %s/sim_tf.py --spawn %s' % (SCRIPTS_IN, run_path(SPAWN_FILE)), window='gazebo'),
    ]
    if session.gui:
        panes.append(Pane('gzclient', 'gzclient', 'sim-gazebo', pin_self('rest', ctx)
                          + ' && ros2 launch rover_sim world.launch.py server:=false gui:=true', window='gazebo'))
    panes += [Pane('%s livox_sim_bridge' % a.name, 'bridge-' + a.name, 'sim-gazebo', pin_self('rest', ctx)
                   + ' && python3 %s/livox_sim_bridge.py --ns %s' % (SCRIPTS_IN, a.name), window='bridges')
              for a in session.agents]
    return panes


def robot_env(a: Agent) -> Tuple[Tuple[str, str], ...]:
    """The identity a rover's /etc/rover/rover.env gives its drive container."""
    return (('VEHTYPE', a.prefix), ('VEHNUM', '%02d' % a.id), ('ROBOT_NAME', a.name))


def teleop_cmd(a: Agent) -> str:
    """The robot's 'teleop + safety' pane with the sim's teleop file. joy_id stays the
    default 0: the containers have no input devices, and joy_node (SDL) then just waits
    for one, as on a robot without its gamepad."""
    return ('ros2 launch rover_control teleop.launch.yaml robot_name:=%s teleop_file:=%s'
            % (a.name, run_path(TELEOP_FILE)))


def drive_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """sim-drive: per agent the robot's 'teleop + safety' and 'description / TF' panes
    (rover drive_host_tmux.sh). teleop.launch.yaml runs joy_node, teleop_twist_joy and
    safety_node in /<ns> with <run>/teleop_sim.yaml (write_teleop_file); safety_node starts
    DISARMED and publishes zero cmd_vel until armed (arm_agents). robot_state_publisher
    puts <ns>/base_link -> <ns>/lidar from the agent's generated mount file (link_calib) on
    /tf_static. No firmware driver: Gazebo's diff drive consumes /<ns>/cmd_vel."""
    return ([Pane('%s teleop + safety' % a.name, 'teleop-safety-' + a.name, 'sim-drive', teleop_cmd(a),
                  env=robot_env(a), window='teleop') for a in session.agents]
            + [Pane('%s description / TF' % a.name, 'description-tf-' + a.name, 'sim-drive',
                    'ros2 launch rover_description state_publisher.launch.py robot_name:=%s ns_tf_frame_ids:=true'
                    % a.name, env=robot_env(a), window='description')
               for a in session.agents])


FRAME_ALIGN_KEYS = ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw')


def frame_align_pairs(session: Session) -> List[Tuple[Agent, Agent]]:
    """Every ordered (ego, other) pair: each mighty_node aligns each other agent itself."""
    return [(e, o) for e in session.agents for o in session.agents if o.name != e.name]


def frame_align_name(ego: Agent, other: Agent) -> str:
    """The name ego's mighty_node subscribes under for `other`: EGO's prefix and the
    other's id (mighty_node.cpp:199-216), so for JR81 the topic of RR83 is
    /frame_align/JR81/JR83."""
    return '%s%02d' % (ego.prefix, other.id)


def frame_align_transform(ego: Agent, other: Agent) -> Tuple[float, ...]:
    """(tx, ty, tz, qx, qy, qz, qw) of inverse(T_world_egomap) * T_world_othermap: it maps
    points in other/map into ego/map, which is how mighty_node applies it to the other's
    trajectories (mighty_node.cpp:1705-1735). Each <ns>/map is the spawn pose (x, y, 0, yaw)."""
    c, s = math.cos(ego.yaw), math.sin(ego.yaw)
    dx, dy = other.x - ego.x, other.y - ego.y
    dyaw = math.atan2(math.sin(other.yaw - ego.yaw), math.cos(other.yaw - ego.yaw))  # qw >= 0
    return (c * dx + s * dy, -s * dx + c * dy, 0.0, 0.0, 0.0, math.sin(dyaw / 2), math.cos(dyaw / 2))


def frame_align_table(session: Session) -> Dict[str, Any]:
    pairs = []
    for e, o in frame_align_pairs(session):
        row = {'ego': e.name, 'other': o.name, 'topic': '/frame_align/%s/%s' % (e.name, frame_align_name(e, o)),
               'node': 'frame_align_%s_%s' % (e.name, o.name)}
        row.update(zip(FRAME_ALIGN_KEYS, frame_align_transform(e, o)))
        pairs.append(row)
    return {'pairs': pairs}


def frame_align_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """sim-mighty. mighty's own scripts/frame_align_publisher.py (installed in mighty-hw as
    lib/mighty/frame_align_publisher.py), one process per ordered pair with a per-pair node
    name, publishing the ground-truth alignment at 10 Hz. mighty_node (use_frame_alignment:
    true) drops another agent's /trajs until it has that agent's alignment. On the robot
    nothing publishes these yet. One window per ego."""
    panes = []
    for ego, other in frame_align_pairs(session):
        name = frame_align_name(ego, other)
        params = ' '.join('-p %s:=%s' % (k, fmt_float(v))
                          for k, v in zip(FRAME_ALIGN_KEYS, frame_align_transform(ego, other)))
        panes.append(Pane('/frame_align/%s/%s (%s)' % (ego.name, name, other.name),
                          'frame-align-%s-%s' % (ego.name, other.name), 'sim-mighty',
                          'ros2 run mighty frame_align_publisher.py --ros-args -r __node:=frame_align_%s_%s '
                          '-p ego_name:=%s -p other_name:=%s %s' % (ego.name, other.name, ego.name, name, params),
                          window='align-' + ego.name))
    return panes


# ---------------------------------------------------------------------------- per-run files

def spawn_table(session: Session, pins: Dict[str, str]) -> Dict[str, Any]:
    """<run>/spawn.yaml: what sim_tf.py and the spawn pane place every agent from, and the
    world's walls (world_walls) for drive_pattern.py. Each agent's odometry is there too, for the
    tools in the containers (truth_odom.py, demo_eval.py, odom_eval.py, sim_viz.py), which read
    this file and not session.json (drive_pattern.spawn_odometry: no key is dlio)."""
    out = {'world': session.world}  # type: Dict[str, Any]
    walls = world_walls(session.world)
    if walls:
        out['walls'] = walls
    out['agents'] = [{'name': a.name, 'class': a.cls, 'id': a.id, 'x': a.x, 'y': a.y, 'yaw': a.yaw,
                      'yaw_deg': a.yaw_deg, 'lidar_height': ROBOT_CLASSES[a.cls].lidar_height,
                      'reference_robot': reference_robot(a.cls, pins), 'odometry': a.odometry}
                     for a in session.agents]
    return out


def generated_yaml(session: Session, what: str, data: Dict[str, Any]) -> str:
    return ('# %s for session %s (%s), generated by scripts/sim.py; rewritten on every `up`.\n'
            % (what, session.name, session.source) + yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def check_mount(data: bytes, where: str) -> None:
    """What state_publisher.launch.py requires of a mount file, checked before anything
    is spawned from it."""
    try:
        m = yaml.safe_load(data.decode('utf-8'))
        xyz, rpy = [float(v) for v in m['xyz']], [float(v) for v in m['rpy']]
    except (yaml.YAMLError, UnicodeDecodeError, KeyError, TypeError, ValueError) as e:
        raise SimError('%s: not a lidar mount file (numeric xyz and rpy): %r' % (where, e))
    if len(xyz) != 3 or len(rpy) != 3:
        raise SimError('%s: xyz and rpy need 3 entries each' % where)


def mount_header(agent: Agent, ref: str, pins: Dict[str, str]) -> bytes:
    """One comment line on top of the reference robot's file, which follows verbatim."""
    return ('# rover_sim: %s (class %s) uses %s\'s lidar mount; below is rover_description/sensor_calib/'
            '%s_lidar_mount.yaml verbatim from %s (rover %s). Generated by scripts/sim.py, do not edit.\n'
            % (agent.name, agent.cls, ref, ref, pins.get('DRIVE_IMAGE', '?'), pins.get('DRIVE_SHA', '?')[:7])
            ).encode()


# A regular file only: link_calib puts the run's own symlinks in the same directory.
_READ_FILE_SH = '[ -f "$1" ] && [ ! -L "$1" ] && base64 -w0 "$1"'


def read_container_file(name: str, path: str) -> bytes:
    p = dexec(name, ['bash', '-c', _READ_FILE_SH, 'read', path], check=False, timeout=30)
    if p.returncode != 0:
        raise SimError('%s: cannot read %s (not a regular file in its image?)%s'
                       % (name, path, ': ' + p.stderr.strip() if (p.stderr or '').strip() else ''))
    return base64.b64decode(p.stdout.strip())


def reference_calib(ref: str) -> Tuple[bytes, bytes]:
    """The reference robot's mount file and camera calibration, from the pinned
    rover_description the drive image installs (not from any local checkout)."""
    d = RD_SHARE + '/' + CALIB_DIR + '/'
    mount = read_container_file(DESCRIPTION_CONTAINER, d + ref + '_lidar_mount.yaml')
    check_mount(mount, '%s:%s%s_lidar_mount.yaml' % (DESCRIPTION_CONTAINER, d, ref))
    npy = read_container_file(DESCRIPTION_CONTAINER, d + ref + '_calib.npy')
    if not npy.startswith(b'\x93NUMPY'):
        raise SimError('%s:%s%s_calib.npy is not a .npy file' % (DESCRIPTION_CONTAINER, d, ref))
    return mount, npy


# Every sim container mounts the run dir read-write and runs as root, so any name in it
# may have been replaced by a symlink to a host path. These never follow one.

def open_own_dir(path: Path) -> int:
    """An fd of the directory at path: a real directory (a symlink as its last component is
    refused, the run dir's own path is check_run_dir's) owned by us."""
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as e:
        raise SimError('%s is not a directory (%s): the run dir is writable from the containers; '
                       'remove it, or set ROVER_SIM_RUN to a fresh dir' % (path, e.strerror))
    uid = os.fstat(fd).st_uid
    if uid != os.getuid():
        os.close(fd)
        raise SimError('%s is owned by uid %d, not you (a container made it?): remove it (sudo rm -r), '
                       'or set ROVER_SIM_RUN to a fresh dir' % (path, uid))
    return fd


def write_file(path: Path, data: bytes) -> None:
    """Atomically: a fresh temp file next to it (O_EXCL: created, never opened through a
    link), renamed over the name (which replaces a link, never writes through it)."""
    fd = open_own_dir(path.parent)
    tmp = '.%s.tmp' % path.name
    try:
        try:
            os.unlink(tmp, dir_fd=fd)
        except FileNotFoundError:
            pass
        out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=fd)
        with os.fdopen(out, 'wb') as f:
            f.write(data)
        os.replace(tmp, path.name, src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        os.close(fd)


def empty_own_dir(path: Path) -> None:
    """path as an empty directory of ours: a symlink there is removed (not followed), a
    missing one made, and every non-directory in it removed, root-owned xacro output too
    (the directory is ours)."""
    try:
        if stat.S_ISLNK(os.lstat(str(path)).st_mode):
            os.unlink(str(path))
    except FileNotFoundError:
        pass
    try:
        os.mkdir(str(path))
    except FileExistsError:
        pass
    fd = open_own_dir(path)
    try:
        for name in os.listdir(fd):
            if not stat.S_ISDIR(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode):
                os.unlink(name, dir_fd=fd)
    finally:
        os.close(fd)


def write_agent_files(session: Session, ctx: RunContext) -> None:
    """Per-run files for the gazebo and drive stacks:
      <run>/spawn.yaml                       agents and spawn poses (spawn_table)
      <run>/sensor_calib/<ns>_lidar_mount.yaml  the reference robot's, plus a header line
      <run>/sensor_calib/<ns>_calib.npy      the reference robot's, byte for byte
      <run>/frame_align.yaml                 what the frame_align panes publish
      <run>/classes.yaml                     the drive limits, for drive_pattern.py / odom_eval.py
                                             in any container (only sim-gazebo mounts src/)
      <run>/urdf/                            the spawn pane's xacro output
    Stale files of an earlier session go first: the containers read these by name."""
    if not running(DESCRIPTION_CONTAINER):
        raise SimError('the reference calibration is read from %s, which is not running '
                       '(scripts/sim.py up without --partial starts it)' % DESCRIPTION_CONTAINER)
    pins = read_pins()
    refs = sorted({reference_robot(a.cls, pins) for a in session.agents})
    calib = dict(zip(refs, parallel(reference_calib, refs)))
    run_dir = ctx.run_dir
    for sub in (CALIB_DIR, URDF_DIR):
        empty_own_dir(run_dir / sub)
    for a in session.agents:
        ref = reference_robot(a.cls, pins)
        mount, npy = calib[ref]
        write_file(run_dir / CALIB_DIR / calib_name(a, 'mount'), mount_header(a, ref, pins) + mount)
        write_file(run_dir / CALIB_DIR / calib_name(a, 'npy'), npy)
    write_file(run_dir / SPAWN_FILE, generated_yaml(session, 'Agents and spawn poses (yaw in radians)',
                                                    spawn_table(session, pins)).encode())
    write_file(run_dir / FRAME_ALIGN_FILE, generated_yaml(
        session, 'Ground-truth frame alignment, inverse(T_world_egomap) * T_world_othermap',
        frame_align_table(session)).encode())
    write_file(run_dir / CLASSES_FILE, CLASSES_YAML.read_bytes())
    info('wrote %s, %s/ (%s), %s and %s' % (SPAWN_FILE, CALIB_DIR, ', '.join(
        '%s from %s' % (a.name, reference_robot(a.cls, pins)) for a in session.agents), FRAME_ALIGN_FILE,
        CLASSES_FILE))


# Links each agent's calib files into the share; refuses to shadow a file the image has.
_LINK_CALIB_SH = r'''d=$1; src=$2; shift 2
for n in "$@"; do
  for f in "${n}_lidar_mount.yaml" "${n}_calib.npy"; do
    [ -f "$src/$f" ] || { echo "$src/$f is missing" >&2; exit 1; }
    if [ -e "$d/$f" ] && [ ! -L "$d/$f" ]; then
      echo "$d/$f is the image's own file: $n is the name of a real robot" >&2; exit 3
    fi
    ln -sfn "$src/$f" "$d/$f" || exit 1
  done
done'''


def link_calib(session: Session, ctx: RunContext) -> None:
    """state_publisher.launch.py takes no path for these: it reads
    sensor_calib/<robot_name>_lidar_mount.yaml from rover_description's share, and
    <robot_name>_calib.npy relative to its own file, which is the same directory in this
    install. So each agent's run-dir files are symlinked into sim-drive's copy of that
    share. That is the container's writable layer, not the image; compose recreating
    the container drops the links and every `up` makes them again. A real robot's own
    file is never replaced."""
    p = dexec(DESCRIPTION_CONTAINER, ['bash', '-c', _LINK_CALIB_SH, 'link', RD_SHARE + '/' + CALIB_DIR,
                                      run_path(CALIB_DIR)] + [a.name for a in session.agents],
              check=False, timeout=30)
    if p.returncode != 0:
        raise SimError('could not link the calibration into %s:%s/%s: %s%s' % (
            DESCRIPTION_CONTAINER, RD_SHARE, CALIB_DIR, (p.stderr or '').strip(),
            '\n  rename that agent: sim agents need names no real robot has (e.g. JR81)' if p.returncode == 3 else ''))


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a repeated mapping key (PyYAML keeps the last silently,
    so a cloned block that repeats a file's key would look fine here)."""


def _unique_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> Dict[Any, Any]:
    keys = set()
    for k, _ in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in keys:
            raise yaml.constructor.ConstructorError(None, None, 'duplicate key %r' % (key,), k.start_mark)
        keys.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def param_blocks(text: str, where: str) -> Dict[str, Any]:
    """The node blocks of a ROS param file: flat, fully qualified keys ('/**/joy_teleop/joy_node',
    '/JR02/safety_node', '/**'), each with only ros__parameters."""
    try:
        doc = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as e:
        raise SimError('%s: %s' % (where, e))
    if not isinstance(doc, dict) or not doc:
        raise SimError('%s: expected node blocks at the top level' % where)
    for k, v in doc.items():
        if not (isinstance(k, str) and k.startswith('/')) or not (isinstance(v, dict) and list(v) == ['ros__parameters']):
            raise SimError('%s: %r is not a flat /<ns>/<node>: {ros__parameters: ...} block' % (where, k))
    return doc


def teleop_sim_yaml(base: str, session: Session, pins: Dict[str, str], source: str) -> str:
    """<run>/teleop_sim.yaml: rover_control's teleop_logitech.yaml verbatim, then for every
    agent a copy of its reference robot's own blocks under the agent's name (JR02's
    /JR02/joy_teleop/teleop_twist_joy_node and /JR02/safety_node become /JR81/...). A robot
    without blocks of its own (RR08) runs on the /** defaults, and so do its agents. The
    copies stay flat, fully qualified keys: the file's own note explains that nesting them
    under a namespace key makes rcl drop siblings. Launch reads the file with allow_substs,
    so nothing generated may contain '$('."""
    blocks = param_blocks(base, source)
    out = ['# rover_sim: %s for session %s, generated by scripts/sim.py; rewritten on every `up`.' % (
        TELEOP_FILE, session.name),
        '# Below: %s verbatim, then each sim agent\'s copy of its reference robot\'s blocks.' % source, '', '']
    tail = ['', '', '# ' + '-' * 75, '# rover_sim agents: copies of their reference robot\'s blocks (flat keys).']
    added = OrderedDict()  # type: Dict[str, Any]
    for a in session.agents:
        if any(k.split('/')[1] == a.name for k in blocks):
            raise SimError('%s has blocks for %s, a real robot: sim agents need names no real robot has '
                           '(e.g. JR81)' % (source, a.name))
        ref = reference_robot(a.cls, pins)
        mine = [(k, v) for k, v in blocks.items() if k.split('/')[1] == ref and k.count('/') > 1]
        if not mine:
            tail.append('# %s: %s has no blocks of its own; the /** defaults apply.' % (a.name, ref))
            continue
        tail.append('# %s (class %s): from %s' % (a.name, a.cls, ref))
        for k, v in mine:
            key = '/%s/%s' % (a.name, k.split('/', 2)[2])
            added[key] = copy.deepcopy(v)
            tail.append(yaml.safe_dump({key: v}, default_flow_style=False, sort_keys=False).rstrip('\n'))
    text = '\n'.join(out) + base.rstrip('\n') + '\n'.join(tail) + '\n'
    if '$(' in '\n'.join(out + tail):
        raise SimError('BUG: generated teleop blocks contain a launch substitution')
    want = OrderedDict(blocks)
    want.update(added)
    if param_blocks(text, TELEOP_FILE) != want:
        raise SimError('BUG: %s does not parse back to the base file plus the agents\' blocks' % TELEOP_FILE)
    return text


def write_teleop_file(session: Session, ctx: RunContext) -> None:
    """<run>/teleop_sim.yaml (teleop_sim_yaml) from the drive image's installed rover_control,
    the pinned rover's: the file safety_node runs with on the robot, not a local copy."""
    if not running(DESCRIPTION_CONTAINER):
        raise SimError('%s is read from %s, which is not running' % (TELEOP_SRC, DESCRIPTION_CONTAINER))
    pins = read_pins()
    src = '%s:%s (rover %s)' % (pins.get('DRIVE_IMAGE', '?'), TELEOP_SRC, pins.get('DRIVE_SHA', '?')[:7])
    try:
        base = read_container_file(DESCRIPTION_CONTAINER, TELEOP_SRC).decode('utf-8')
    except UnicodeDecodeError as e:
        raise SimError('%s: %s' % (src, e))
    write_file(ctx.run_dir / TELEOP_FILE, teleop_sim_yaml(base, session, pins, src).encode())
    info('wrote %s (%s)' % (TELEOP_FILE, ', '.join(
        '%s from %s' % (a.name, reference_robot(a.cls, pins)) for a in session.agents)))


def dlio_scripts_dir(pins: Dict[str, str]) -> Path:
    """build_images.sh's checkout of the dlio_ws pin (.cache/src/dlio_ws@<DLIO_SHA>): its
    scripts/ is what compose.stack.yaml mounts in both DLIO containers."""
    return REPO / '.cache' / 'src' / ('dlio_ws@' + pins.get('DLIO_SHA', '')) / 'scripts'


def dlio_scripts_problem(pins: Dict[str, str]) -> Optional[str]:
    d = dlio_scripts_dir(pins)
    missing = [f for f in DLIO_SCRIPTS if not (d / f).is_file()]
    if not missing:
        return None
    return ('the pinned dlio_ws scripts the DLIO containers mount are missing (%s in %s): '
            'run scripts/build_images.sh dlio-jr' % (', '.join(missing), d))


# The robot's DLIO pane launches dlio_with_session.launch.py, whose dlio_map_node has a fixed
# name and no namespace. Two agents share a sim DLIO container (and four, one graph), so this
# wrapper adds one rcl node-specific remap, dlio_map_node:__node:=dlio_map_node_<ns>, which
# renames that node only (allowed_deviations.yaml rename). Everything else is the robot's file.
# The node's own interfaces keep their root names, so every agent's map node shares them:
# /save_pcd (a call reaches any one of them), /map and the /dlio/odom_node/pointcloud/keyframe
# subscription. JR02 has one of each, and nothing publishes those keyframes there either (the
# odom node's are under /<ns>); allowed_deviations.yaml notes it next to the rename.
DLIO_SIM_LAUNCH = '''"""rover_sim: the robot's DLIO pane launch with a per-agent DLIO map node name.

    ros2 launch {run}/{name} namespace:=JR81 initial_pose_topic:=world two_d_only:=false

Includes {robot_launch} unchanged (the pinned dlio_ws
scripts/, mounted as on the robot), with one rcl node-specific remap in scope:
dlio_map_node:__node:=dlio_map_node_<namespace> renames the unnamespaced dlio_map_node, and only
it, so the agents sharing this container and graph do not share a node name. Its /save_pcd,
/map and keyframe topic stay at the root, shared by every agent's map node.
Generated by rover_sim scripts/sim.py; rewritten on every `up`.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetRemap

ROBOT_LAUNCH = {robot_launch!r}
ARGS = ('namespace', 'initial_pose_topic', 'two_d_only')


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(a) for a in ARGS]
        + [SetRemap(src='dlio_map_node:__node', dst=['dlio_map_node_', LaunchConfiguration('namespace')]),
           IncludeLaunchDescription(PythonLaunchDescriptionSource(ROBOT_LAUNCH),
                                    launch_arguments=[(a, LaunchConfiguration(a)) for a in ARGS])])
'''.format(run=RUN_DIR_IN, name=DLIO_LAUNCH_FILE, robot_launch=DLIO_WS + '/scripts/dlio_with_session.launch.py')


def write_dlio_launch(session: Session, ctx: RunContext) -> None:
    problem = dlio_scripts_problem(read_pins())
    if problem:
        raise SimError(problem)
    write_file(ctx.run_dir / DLIO_LAUNCH_FILE, DLIO_SIM_LAUNCH.encode())


# DLIO registers scans as soon as its 3 s IMU calibration ends, seed or not; a seed that lands
# later leaves odom anchored at z 0 instead of the lidar height (odom.cc:844-853, 916). The
# robot starts its three panes at once and its seed usually wins; here every stack's panes
# start at once, and their gates finish seconds apart. A running seed publisher is no proof of
# delivery either (rmw_zenoh 0.1.9 once delayed it by ~85 s). So the DLIO pane first waits for
# a seed message to arrive in its container (scripts/wait_for_seed.py); the seed pane keeps
# publishing at 2 Hz through DLIO's calibration. Generous: the paused bring-up takes minutes.
SEED_WAIT_S = 300
WAIT_FOR_SEED = 'wait_for_seed.py'   # this repo's scripts/, mounted at SCRIPTS_IN


def seed_wait(a: Agent) -> str:
    return 'python3 %s/%s %s --timeout %d' % (SCRIPTS_IN, WAIT_FOR_SEED, a.name, SEED_WAIT_S)


def dlio_launch_cmd(a: Agent) -> str:
    """dlio_host_tmux.sh pane 0 (JR02 runs two_d_only false), through the wrapper. Its pane
    waits for the seed pose first (seed_wait)."""
    return ('ros2 launch %s namespace:=%s initial_pose_topic:=world two_d_only:=false'
            % (run_path(DLIO_LAUNCH_FILE), a.name))


def seed_pose_cmd(a: Agent) -> str:
    """dlio_host_tmux.sh pane 1: DLIO holds its initialisation until a pose arrives on
    /<ns>/world and anchors odom there, at the robot's DLIO_SEED_Z (the lidar height)."""
    return ('ros2 topic pub -r 2 /%s/world geometry_msgs/msg/PoseStamped "{header: {frame_id: world}, '
            'pose: {position: {z: %s}, orientation: {w: 1}}}"' % (a.name, fmt_float(ROBOT_CLASSES[a.cls].lidar_height)))


def map_odom_tf_cmd(a: Agent) -> str:
    """dlio_host_tmux.sh pane 2. map_odom_tf.py names its node map_odom_tf_broadcaster
    whatever the namespace; its rclpy.init() reads sys.argv, so a __node remap renames it."""
    return ('python3 %s/scripts/map_odom_tf.py %s --ros-args -r __node:=map_odom_tf_broadcaster_%s'
            % (DLIO_WS, a.name, a.name))


def truth_odom_cmd(a: Agent, check: bool = False) -> str:
    """odometry: truth (NOT PARITY): truth_odom.py publishes ground truth under DLIO's names, from
    the image's DLIO cfg (DLIO_CFG_IN) and <run>/spawn.yaml. With check, the pane's gate: it
    validates both, and refuses an agent that spawn.yaml does not mark truth."""
    return 'python3 %s/%s %s --spawn %s --dlio-cfg %s%s' % (SCRIPTS_IN, TRUTH_ODOM, a.name, run_path(SPAWN_FILE),
                                                         DLIO_CFG_IN, ' --check' if check else '')


def dlio_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """sim-dlio-<class> (dlio_container): per agent one window with dlio_host_tmux.sh's three
    panes, the class's image baking its reference robot's cfg_overlay. DLIO calibrates its
    IMU for 3 s after the first sample: safety_node holds the agent still (DISARMED) until
    arm_agents. The kernel balances the DLIO processes within the container's cpuset.
    An agent with odometry: truth (NOT PARITY) gets truth_odom.py in DLIO's place, and no seed
    pose (only DLIO reads it); map->odom is the same."""
    panes = []
    for a in session.agents:
        c = dlio_container(a)
        if a.odometry == 'truth':
            panes.append(Pane('%s truth odometry (ground truth as DLIO), TRUTH: not parity' % a.name,
                              'truth-odom-' + a.name, c, truth_odom_cmd(a), env=robot_env(a), window=a.name,
                              waits=truth_odom_cmd(a, check=True)))
        else:
            panes += [Pane('%s dlio odometry' % a.name, 'dlio-' + a.name, c, dlio_launch_cmd(a), env=robot_env(a),
                           window=a.name, waits=seed_wait(a)),
                      Pane('%s seed pose' % a.name, 'seed-' + a.name, c, seed_pose_cmd(a), env=robot_env(a),
                           window=a.name)]
        panes.append(Pane('%s tf map->odom' % a.name, 'map-odom-tf-' + a.name, c, map_odom_tf_cmd(a),
                          env=robot_env(a), window=a.name))
    return panes


# ---------------------------------------------------------------------------- mapping

# As on the robot, whose mapping_ws compose bind-mounts the pinned checkout's config/mapping at
# /mapping_config (docker/compose.mapping.yaml): compose.stack.yaml mounts build_images.sh's
# checkout of MAPPING_SHA there, read-only, and that checkout's config/elevation where the Orin
# has it (tools/install_orin.sh; elevation_mapper_host_tmux.sh EM_OVERLAY_DIR).
MAPPING_CONFIG_IN = '/mapping_config'
ELEVATION_OVERLAY_IN = '/usr/local/share/rover/elevation'
WAIT_FOR_TF = MAPPING_WS + '/install/share/rover_bringup/scripts/wait_for_tf.py'
PERSISTENT_MAP_PARAMS = 'persistent_map.yaml'
# What compose.stack.yaml binds from that checkout, per container. Compose refuses to start a
# container whose bind source is missing (never created: it would be root's).
MAPPING_MOUNTS = (('sim-mapping', 'config/mapping'), ('sim-elevation', 'config/elevation'),
                  ('sim-elevation', 'scripts/wait_zenoh_ready.py'))


def mapping_checkout(pins: Dict[str, str]) -> Path:
    """build_images.sh's checkout of the mapping_ws pin (.cache/src/mapping_ws@<MAPPING_SHA>)."""
    return REPO / '.cache' / 'src' / ('mapping_ws@' + pins.get('MAPPING_SHA', ''))


def mapping_mounts_problem(pins: Dict[str, str], services: Iterable[str]) -> Optional[str]:
    d, services = mapping_checkout(pins), set(services)
    missing = [rel for c, rel in MAPPING_MOUNTS if c in services and not (d / rel).exists()]
    if not missing:
        return None
    return ('the pinned mapping_ws checkout the mapping containers mount is missing (%s in %s): '
            'run scripts/build_images.sh mapping' % (', '.join(sorted(set(missing))), d))


@dataclass(frozen=True)
class MapperSettings:
    """A class's mapping_host_tmux.sh settings: its reference robot's MAPPER_* values from
    /etc/rover/rover.env, else that script's VEHTYPE defaults. MAPPER_FRAME is DLIO's <ns>/odom
    for both (the frame of its deskewed cloud, and of the elevation map), MAPPER_D455 false
    (no D455 in the sim)."""
    gm_params: str                     # MAPPER_PARAM_FILE, in config/mapping/
    mf_params: str                     # MAP_FUSION_PARAM_FILE
    source: str
    cloud: str = 'dlio/odom_node/pointcloud/deskewed'     # MAPPER_CLOUD_TOPIC
    pose: str = 'dlio/odom_node/pose'                      # MAPPER_POSE_TOPIC
    tracker: bool = False                                  # MAPPER_OBSTACLE_TRACKER
    persistent_map: bool = True                            # MAPPER_PERSISTENT_MAP


MAPPER_SETTINGS = OrderedDict([
    # tools/reference/jr02/rover.env: MAPPER_FRAME=JR02/odom, MAPPER_CLOUD_TOPIC and
    # MAPPER_PARAM_FILE as here, D455 and OBSTACLE_TRACKER false, PERSISTENT_MAP true; no
    # MAP_FUSION_PARAM_FILE, so the JR default.
    ('jr', MapperSettings('global_mapper_junior_dlio.yaml', 'map_fusion_junior.yaml', "JR02's rover.env")),
    # RR08's rover.env was not captured: the RR defaults, except the persistent map, which is
    # off by default and on here, as on JR02 (goal_selector explores on persistent_map/grid).
    ('rr', MapperSettings('global_mapper_red_rover.yaml', 'map_fusion_red_rover.yaml',
                          'the RR defaults, with the persistent map on')),
])

# mapping: fallback (NOT PARITY). On the robot global_mapper's 2D grids only feed map_fusion,
# which makes the planner's grids with the elevation map; the robot files are set for that, so
# without the elevation map and map_fusion the planner would get no usable grid. Each fallback
# param file is the robot's file verbatim with rover_sim's overlay from config/mapping/ appended
# (append_overlay): the robot's launch files take one param_file. persistent_map needs none:
# grid_relay.py publishes map_fusion's live/occ_2d_topic, so it runs the robot's file (pointing
# its window_topic at voxel/occ_2d_topic was tried live: it opened an epoch while paused and
# never saw a dead DLIO).
#   (node, robot file in config/mapping/ (None: the class's global_mapper file), overlay)
FALLBACK_OVERLAYS = (
    ('global_mapper_ros', None, 'global_mapper_fallback_{cls}.yaml'),
)


def fallback_files(a: Agent) -> List[Tuple[str, str, str, str]]:
    """(node, robot file, rover_sim overlay, generated file in <run>/mapping/) of an agent's
    fallback parameters. An overlay's block is for every node of that name, so agents of a
    class share each file."""
    m = MAPPER_SETTINGS[a.cls]
    out = []
    for node, base, overlay in FALLBACK_OVERLAYS:
        overlay = overlay.format(cls=a.cls)
        out.append((node, base or m.gm_params, overlay, overlay))
    return out


def flat_params(d: Dict[str, Any], prefix: str = '') -> 'OrderedDict[str, Any]':
    """Dotted parameter names -> values, as rcl reads a ros__parameters mapping."""
    out = OrderedDict()  # type: OrderedDict[str, Any]
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat_params(v, '%s%s.' % (prefix, k)))
        else:
            out['%s%s' % (prefix, k)] = v
    return out


def param_kind(v: Any) -> str:
    """The parameter type rcl gives a YAML value. A node refuses an override of another type
    than it declared, an integer for a double too."""
    for t, name in ((bool, 'bool'), (int, 'integer'), (float, 'double'), (str, 'string')):
        if isinstance(v, t):
            return name
    if isinstance(v, list):
        return 'array of %s' % (param_kind(v[0]) if v else '?')
    return type(v).__name__


def append_overlay(base_text: str, base_src: str, overlay_text: str, overlay_src: str, node: str) -> str:
    """The robot's param file verbatim, then the overlay verbatim: one /**/<node> block of only
    the keys the sim must change (the robot files are private: none is copied into this repo).
    It comes after the robot's /** block, so it wins for every node of that name: rclpy 3.3.21
    and rclcpp 16.0.19 (this pin's) apply /**, then /**/<node>, then a /<ns>/<node> block
    (checked in throwaway containers). Refused: another block, a key the robot file does not
    set (a typo would change nothing), another type than the robot's (the node would refuse it)."""
    base = param_blocks(base_text, base_src)
    over = param_blocks(overlay_text, overlay_src)
    key = '/**/' + node
    if list(over) != [key]:
        raise SimError('%s: an overlay is one %s block, not %s' % (overlay_src, key, ', '.join(over)))
    if key in base:
        raise SimError('%s has a %s block of its own: the overlay would repeat it' % (base_src, key))
    params = flat_params(over[key]['ros__parameters'] or {})
    if not params:
        raise SimError('%s sets no parameters' % overlay_src)
    have = flat_params((base.get('/**') or {}).get('ros__parameters') or {})
    bad = []
    for k, v in params.items():
        if k not in have:
            bad.append('%s: not in the /** block of %s' % (k, base_src))
        elif v is None or (isinstance(v, float) and not math.isfinite(v)):
            bad.append('%s: %r is not a parameter value' % (k, v))
        elif param_kind(have[k]) != param_kind(v):
            bad.append('%s: %s in %s, %s in the overlay' % (k, param_kind(have[k]), base_src, param_kind(v)))
    if bad:
        raise SimError('%s: refused:\n  - %s' % (overlay_src, '\n  - '.join(bad)))
    text = ('# rover_sim: generated by scripts/sim.py (mapping: fallback, NOT PARITY); rewritten on every `up`.\n'
            '# Below: %s verbatim, then %s verbatim.\n\n' % (base_src, overlay_src)
            + base_text.rstrip('\n') + '\n\n# ' + '-' * 75 + '\n# rover_sim overlay %s\n' % overlay_src
            + overlay_text.rstrip('\n') + '\n')
    want = OrderedDict(base)
    want.update(over)
    if param_blocks(text, 'the file generated from %s' % overlay_src) != want:
        raise SimError('BUG: %s + %s does not parse back to both' % (base_src, overlay_src))
    return text


def mapping_config_problem(session: Session, pins: Dict[str, str]) -> Optional[str]:
    """What the session's mapping panes read and is missing: the robot's files in the pinned
    checkout, and for the fallback rover_sim's overlays."""
    d = mapping_checkout(pins) / 'config' / 'mapping'
    need = set()  # type: set
    for a in session.agents:
        m = MAPPER_SETTINGS[a.cls]
        need |= {m.gm_params} | ({PERSISTENT_MAP_PARAMS} if m.persistent_map else set())
        if session.mapping == 'gpu':
            need.add(m.mf_params)
    missing = sorted(f for f in need if not (d / f).is_file())
    if missing:
        return ('the pinned mapping_ws config the mapping panes read is missing (%s in %s): '
                'run scripts/build_images.sh mapping' % (', '.join(missing), d))
    if session.mapping == 'fallback':
        over = sorted({o for a in session.agents for _, _, o, _ in fallback_files(a)})
        missing = [o for o in over if not (SIM_MAPPING_DIR / o).is_file()]
        if missing:
            return 'mapping: fallback needs the overlay(s) %s in %s' % (', '.join(missing), SIM_MAPPING_DIR)
    return None


def write_mapping_files(session: Session, ctx: RunContext) -> None:
    """<run>/mapping/<overlay> for mapping: fallback (append_overlay); the parity panes read the
    robot's files from /mapping_config as they are. Stale files go first."""
    pins = read_pins()
    problem = mapping_config_problem(session, pins)
    if problem:
        raise SimError(problem)
    out_dir = ctx.run_dir / MAPPING_RUN_DIR
    empty_own_dir(out_dir)
    if session.mapping != 'fallback':
        return
    src = mapping_checkout(pins) / 'config' / 'mapping'
    rel = 'mapping_ws@%s:config/mapping/' % pins.get('MAPPING_SHA', '?')[:7]
    done = OrderedDict()  # type: OrderedDict[str, str]
    for a in session.agents:
        for node, base, overlay, out in fallback_files(a):
            if out not in done:
                write_file(out_dir / out, append_overlay((src / base).read_text(), rel + base,
                                                         (SIM_MAPPING_DIR / overlay).read_text(),
                                                         'rover_sim config/mapping/' + overlay, node).encode())
                done[out] = base
    info('wrote %s/ (mapping: fallback, NOT PARITY): %s' % (
        MAPPING_RUN_DIR, ', '.join('%s (%s + overlay)' % kv for kv in done.items())))


def global_mapper_cmd(a: Agent, param_file: str) -> str:
    """mapping_host_tmux.sh's global_mapper pane, every output under voxel/."""
    m = MAPPER_SETTINGS[a.cls]
    return ('ros2 launch global_mapper_ros global_mapper_node.launch.py hardware:=true ground_robot:=true quad:=%s '
            'global_frame:=%s/odom param_file:=%s use_obstacle_tracker:=%s depth_pointcloud_topic:=%s pose_topic:=%s '
            'occupancy_grid_topic:=voxel/occupancy_grid unknown_grid_topic:=voxel/unknown_grid '
            'frontier_grid_topic:=voxel/frontier_grid occ_2d_topic:=voxel/occ_2d_topic '
            'esdf_2d_topic:=voxel/esdf_2d_topic'
            % (a.name, a.name, param_file, 'true' if m.tracker else 'false', m.cloud, m.pose))


def tf_gate(a: Agent) -> str:
    """The robot's global_mapper gate: DLIO's odometry, <frame> -> <ns>/base_link. It never
    passed while the world was paused (every Phase 4 bring-up: DLIO has had no IMU sample yet
    and stamps its transform 0), so in `up` it waits out its 60 s timeout and proceeds
    "degraded", still paused. global_mapper then starts with rover_description's base_link ->
    lidar, as on the robot, and DLIO's odometry follows once physics runs. Kept as the robot's;
    the cost is 60 s of `up`."""
    return 'python3 %s %s/odom %s/base_link' % (WAIT_FOR_TF, a.name, a.name)


def map_fusion_cmd(a: Agent) -> str:
    """mapping_host_tmux.sh's map_fusion pane: elev/* + voxel/* -> the planner's grids."""
    return ('ros2 launch map_fusion_ros map_fusion.launch.py quad:=%s output_prefix:=none elev_prefix:=elev/ '
            'param_file:=%s/%s' % (a.name, MAPPING_CONFIG_IN, MAPPER_SETTINGS[a.cls].mf_params))


def persistent_map_cmd(a: Agent) -> str:
    """mapping_host_tmux.sh's persistent_map pane (MAPPER_PERSISTENT_MAP), in both modes: the
    robot's file, but save_dir:=none (the launch file's own default: memory only) where JR02
    passes its save directory. So a run never resumes or adds to another run's map, and nothing
    accumulates in sim-mapping's writable layer; a restarted pane starts an empty map
    (allowed_deviations.yaml: persistent_map_node save_dir)."""
    return ('ros2 launch map_fusion_ros persistent_map.launch.py quad:=%s param_file:=%s/%s save_dir:=none'
            % (a.name, MAPPING_CONFIG_IN, PERSISTENT_MAP_PARAMS))


def grid_relay_cmd(a: Agent) -> str:
    """The fallback's stand-in for map_fusion's outputs: scripts/grid_relay.py, node
    /<ns>/grid_relay_<ns> (a *relay* node: allowed_deviations.yaml mapping.fallback sim_only).
    voxel/occ_2d_topic -> occ_2d_topic and live/occ_2d_topic, voxel/esdf_2d_topic ->
    esdf_2d_topic, at map_fusion's 5 Hz, stamped with the newest DLIO cloud's time, nothing
    before the first cloud. mighty_node reads occ_2d_topic and esdf_2d_topic, goal_selector
    occ_2d_topic, persistent_map live/occ_2d_topic; nothing reads planning_occ_2d_topic."""
    return 'python3 %s/grid_relay.py %s' % (SCRIPTS_IN, a.name)


def mapping_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """sim-mapping: per agent one window with mapping_host_tmux.sh's panes, the class's settings
    (MAPPER_SETTINGS). mapping: gpu runs the robot's: global_mapper (voxel/*, gated on DLIO's
    odometry), map_fusion (owns <ns>/{occ_2d,planning_occ_2d,esdf_2d}_topic, from elev/* and
    voxel/*) and persistent_map. mapping: fallback (NOT PARITY) runs global_mapper on its
    fallback file and grid_relay.py in place of map_fusion; persistent_map stays the robot's."""
    panes = []
    for a in session.agents:
        m, env, w = MAPPER_SETTINGS[a.cls], robot_env(a), a.name
        if session.mapping == 'gpu':
            panes += [Pane('%s global_mapper (voxel/*)' % a.name, 'global-mapper-' + a.name, 'sim-mapping',
                           global_mapper_cmd(a, '%s/%s' % (MAPPING_CONFIG_IN, m.gm_params)), env=env, window=w,
                           waits=verdict_gate(tf_gate(a))),
                      Pane('%s map_fusion (elev/* -> grids)' % a.name, 'map-fusion-' + a.name, 'sim-mapping',
                           map_fusion_cmd(a), env=env, window=w)]
        else:
            files = dict((node, run_path(MAPPING_RUN_DIR, out)) for node, _, _, out in fallback_files(a))
            panes += [Pane('%s global_mapper (voxel/*), FALLBACK: not parity' % a.name, 'global-mapper-' + a.name,
                           'sim-mapping', global_mapper_cmd(a, files['global_mapper_ros']), env=env, window=w,
                           waits=verdict_gate(tf_gate(a))),
                      Pane('%s grid_relay (voxel/* -> grids)' % a.name, 'grid-relay-' + a.name, 'sim-mapping',
                           grid_relay_cmd(a), env=env, window=w)]
        if m.persistent_map:
            panes.append(Pane('%s persistent_map (in memory)' % a.name, 'persistent-map-' + a.name, 'sim-mapping',
                              persistent_map_cmd(a), env=env, window=w))
    return panes


# elevation_mapper_host_tmux.sh's grid arguments: ALWAYS elev/* (map_fusion owns the canonical
# names), last on the line so nothing moves them back.
ELEVATION_TOPIC_ARGS = ('occupancy_grid_topic:=elev/occ_2d_topic',
                        'planning_occupancy_grid_topic:=elev/planning_occ_2d_topic',
                        'esdf_grid_topic:=elev/esdf_2d_topic')


def elevation_overlays(a: Agent, pins: Dict[str, str]) -> List[str]:
    """The param overlays the Orin passes (elevation_mapper_host_tmux.sh's OVERLAY_DIR loop), as
    container paths: types/<VEHTYPE>.yaml, then vehicles/<robot>.yaml, those in the pinned
    checkout. The robot is the class's reference robot, which the agent stands in for, not the
    agent. mapping_ws c1233f7 has none for JR or RR (only types/JK.yaml)."""
    d = mapping_checkout(pins) / 'config' / 'elevation'
    rel = ['types/%s.yaml' % a.prefix, 'vehicles/%s.yaml' % reference_robot(a.cls, pins)]
    return ['%s/%s' % (ELEVATION_OVERLAY_IN, r) for r in rel if (d / r).is_file()]


def elevation_cmd(a: Agent, overlays: Sequence[str]) -> str:
    """elevation_mapper_host_tmux.sh's launch line. Its elevation_mapping_cupy takes param_overlays,
    so the line has extra_cloud_exclude:=livox/* (quoted: a glob) and the overlays, if any."""
    args = ['quad:=' + a.name, "'extra_cloud_exclude:=livox/*'"]
    if overlays:
        args.append('param_overlays:=' + ','.join(overlays))
    return 'ros2 launch elevation_mapping_cupy elevation_mapping_dlio.launch.py ' + ' '.join(
        args + list(ELEVATION_TOPIC_ARGS))


# elevation_mapper_host_tmux.sh finds the launch file the same way and greps it for these arguments.
_ELEVATION_LAUNCH_SH = r'''f=$(find -L "$1/install" -name elevation_mapping_dlio.launch.py -print -quit 2>/dev/null)
[ -n "$f" ] || { echo "no elevation_mapping_dlio.launch.py under $1/install"; exit 1; }
for a in param_overlays occupancy_grid_topic; do
  grep -qE "[\"']$a[\"']" "$f" || { echo "$f declares no $a argument"; exit 1; }
done
echo "$f"'''


def check_elevation_launch(session: Session, ctx: RunContext) -> None:
    """Before the elevation panes: does the image's launch file take what elevation_cmd passes?
    The Orin's script passes extra_cloud_exclude (and overlays) only when the launch file
    declares param_overlays, and the elev/* names only when it declares occupancy_grid_topic;
    elevation_cmd always passes both, which is the Orin's line only for such a launch file."""
    p = dexec('sim-elevation', ['bash', '-c', _ELEVATION_LAUNCH_SH, 'check', SIM_WS], check=False, timeout=30)
    out = (p.stdout or '').strip() or (p.stderr or '').strip()
    if p.returncode != 0:
        raise SimError('sim-elevation: %s: the Orin would not pass extra_cloud_exclude:=livox/* and the elev/* '
                       'grid names to this elevation_mapping_cupy; rebuild its image at the pin '
                       '(scripts/build_images.sh elevation)' % (out or 'exit %d' % p.returncode))
    info('elevation launch file takes param_overlays and the grid topic arguments: %s' % out)


# cupy's compiled kernels, kept in a named volume (compose.stack.yaml cupy-kernel-cache).
_KERNEL_CACHE_SH = 'find "${CUPY_CACHE_DIR:-$HOME/.cupy/kernel_cache}" -name "*.cubin" 2>/dev/null | wc -l'


def check_kernel_cache(session: Session, ctx: RunContext) -> None:
    """Warns when cupy has no compiled kernels yet (the first GPU run on this host, or a removed
    volume): the mappers then compile them at their first cloud, right after the unpause, about a
    minute of CPU each on the shared cores, which the unpause is staged to avoid. The next run
    finds them."""
    p = dexec('sim-elevation', ['bash', '-c', _KERNEL_CACHE_SH], check=False, timeout=30)
    n = (p.stdout or '').strip()
    if p.returncode == 0 and n.isdigit() and int(n) > 0:
        info('cupy kernel cache: %s compiled kernels' % n)
        return
    warn('cupy has no compiled kernels in sim-elevation yet: the elevation mappers compile them after the '
         'unpause (about a minute, every mapper at once), and the start-up burst can stall the graph and make '
         'DLIO diverge; if it does, scripts/sim.py down and up again (the kernels are kept)')


def elevation_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """sim-elevation (on the robot, natively on its Orin): per agent one window with the elevation
    mapper on DLIO's deskewed cloud, its grids on <ns>/elev/* for map_fusion. Gated on the router
    and latched delivery like the Orin's pane (ORIN_PROBE); no TF gate, as there."""
    pins = read_pins()
    return [Pane('%s elevation_mapping_cupy -> elev/*' % a.name, 'elevation-' + a.name, 'sim-elevation',
                 elevation_cmd(a, elevation_overlays(a, pins)), env=robot_env(a), window=a.name)
            for a in session.agents]


# ---------------------------------------------------------------------------- planner

# mighty docker/mighty_hw.sh's pane lines (start(): tf_gate, launch_base and the four
# only_nodes:= panes), as JR02 runs them (tools/reference/jr02/processes/docker_top_hw-mighty.txt).
# $ROBOT_NAME expands in the container from the pane's env (robot_env), as rover.env sets it
# there. The sim appends num_agents:= (mighty_cmd) and nothing else; never use_ground_robot:=,
# which switches the base config.
MIGHTY_TF_GATE = 'ros2 run mighty wait_for_tf.py $ROBOT_NAME/map $ROBOT_NAME/odom'
MIGHTY_LAUNCH = ('ros2 launch mighty onboard_mighty.launch.py x:=0.0 y:=0.0 z:=0.0 yaw:=0.0 namespace:=$ROBOT_NAME '
                 'use_hardware:=true use_onboard_localization:=true robot_type:=red_rover depth_camera_name:=d455')


class MightyPane(NamedTuple):
    node: str                          # only_nodes:= key
    slug: str
    title: str                         # mighty_hw.sh's
    tf_gate: bool                      # mighty_hw.sh gates it on MIGHTY_TF_GATE: it resolves <ns>/map
    ready: str = ''


# mpc builds its IPOPT NLP in its constructor (1.1-1.3 s of CPU with its imports, in a throwaway
# of the mighty-hw image) and logs this line after it: its wave settles only after that, so no
# build runs once physics does.
MPC_READY = r'MPC controller started\.'
MIGHTY_PANES = (MightyPane('mighty_node', 'mighty-node', 'MIGHTY planner', True),
                MightyPane('convert_odom_to_state', 'odom-to-state', 'convert_odom_to_state', False),
                MightyPane('mpc', 'mpc', 'MPC', True, MPC_READY),
                MightyPane('goal_selector', 'goal-selector', 'goal_selector', False))


def mighty_cmd(node: str, num_agents: int) -> str:
    """The robot's pane line plus num_agents:=, the one planned planner-parameter deviation: the
    launch argument overwrites any overlay's, and mighty_node subscribes the frame_align topics
    of ids 1..num_agents only (allowed_deviations.yaml all.params.mighty_node.num_agents)."""
    return '%s only_nodes:=%s num_agents:=%d' % (MIGHTY_LAUNCH, node, num_agents)


def mighty_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """sim-mighty: per agent one window with mighty_hw.sh's four panes, gated on the router
    like every pane (mighty_hw.sh waits for the router's port). mighty_node and mpc wait for
    <ns>/map -> <ns>/odom first, as there: dlio's map_odom_tf.py puts it on /tf at wall time,
    so the gate passes while the world is paused, and a timeout means it is missing
    (strict_tf_gate: `up` stops). mpc publishes cmd_vel_auto at 30 Hz from its start (zeros
    until mighty_node sends mpc_waypoints). With ids up to num_agents, each mighty_node
    subscribes num_agents - 1 frame_align topics, of which only the session's are published."""
    panes = []
    for a in session.agents:
        for m in MIGHTY_PANES:
            panes.append(Pane('%s %s' % (a.name, m.title), '%s-%s' % (m.slug, a.name), 'sim-mighty',
                              mighty_cmd(m.node, session.num_agents), env=robot_env(a), window=a.name,
                              waits=verdict_gate(MIGHTY_TF_GATE) if m.tf_gate else '', ready=m.ready,
                              strict_tf_gate=m.tf_gate))
    return panes


# ---------------------------------------------------------------------------- goals

GOAL_TOPIC = '/sim/goal'             # RViz SetGoal (world frame); goal_relay.py forwards it
GOAL_RELAY = 'goal_relay.py'         # this repo's scripts/, mounted at SCRIPTS_IN
GOAL_RELAY_FIRST_DRIVE_M = 1.2       # its --first-drive default (tests/test_sim.py checks they agree)
GOAL_RELAY_ENTITIES = (8, 2)         # goal_relay.py ENTITIES_PER_AGENT, ENTITIES_ROOT: its paced leave's length


def goal_relay_leave_s(n_agents: int) -> float:
    """How long goal_relay.py takes to leave the graph (one entity per STATUS_LEAVE_GAP_S, then the
    pause), plus a margin: what `down` and `start goals` wait before they SIGTERM it."""
    per, root = GOAL_RELAY_ENTITIES
    return (per * n_agents + root) * STATUS_LEAVE_GAP_S + STATUS_LEAVE_PAUSE_S + 3.0


# goal_relay.py --mode per demo (its SESSION_DEMO_MODE): swap drives the session's goals; explore
# and rviz only forward RViz's (goal_selector explores by itself in both).
GOAL_MODES = OrderedDict([('swap', 'swap'), ('explore', 'explore'), ('none', 'rviz')])


def goal_mode(session: Session) -> str:
    return GOAL_MODES[session.demo]


def goal_relay_cmd(session: Session) -> str:
    """goal_selector ignores a goal's frame_id, so goals reach /<ns>/term_goal_rviz already in
    <ns>/map: goal_relay.py turns world-frame goals (the session's, RViz's /sim/goal) into each
    agent's map frame through <run>/spawn.yaml (world -> <ns>/map is the spawn pose, sim_tf.py)."""
    return 'python3 %s/%s --spawn %s --session %s --mode %s' % (SCRIPTS_IN, GOAL_RELAY, run_path(SPAWN_FILE),
                                                               run_path(SESSION_FILE), goal_mode(session))


def check_goal_relay(session: Session, ctx: RunContext) -> None:
    if not (REPO / 'scripts' / GOAL_RELAY).is_file():
        raise SimError('no scripts/%s in this checkout: the goals stack runs it from %s' % (GOAL_RELAY, SCRIPTS_IN))


def goals_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """sim-mighty, one pane for the session, in the last paused wave, with every planner: goal_relay.py
    first drives each agent that is in AUTO off its spot in JOY mode, as an operator would, then
    hands it back to AUTO (its FirstDrive: from a cold start mighty plans no path over floor it never
    saw), and sends nothing before that, so no goal piles up at goal_selector while nothing moves.
    Its container has the planner's messages."""
    what = ('first drive, then the session\'s goals and back to the start, repeatedly; and RViz %s' % GOAL_TOPIC
            if goal_mode(session) == 'swap' else 'first drive, then RViz %s only' % GOAL_TOPIC)
    return [Pane('goal_relay --mode %s (%s)' % (goal_mode(session), what), 'goal-relay', 'sim-mighty',
                 goal_relay_cmd(session), window='goals')]


# ---------------------------------------------------------------------------- rviz

RVIZ_TEMPLATES = REPO / 'rviz' / 'templates'
RVIZ_CONFIG_IN = run_path(RVIZ_RUN_DIR, RVIZ_FILE)
# The parts rviz_config joins: the header (panels, global displays), one agent part per agent
# with the gpu part inside it for mapping: gpu (elev/*), then the footer (tools, view).
RVIZ_PARTS = ('session.rviz.part', 'agent.rviz.part', 'agent_gpu.rviz.part', 'footer.rviz.part')
RVIZ_TOKEN_RE = re.compile(r'@[A-Z_]+@')
# One colour per agent (cloud, goal), in session order; RViz's "r; g; b".
AGENT_COLORS = ('237; 51; 59', '53; 132; 228', '51; 209; 122', '246; 211; 45', '192; 97; 203', '255; 120; 0',
                '0; 200; 200', '154; 153; 150')


def fill_template(text: str, values: Dict[str, str], where: str) -> str:
    for k, v in values.items():
        text = text.replace('@%s@' % k, v)
    left = sorted(set(RVIZ_TOKEN_RE.findall(text)))
    if left:
        raise SimError('%s: no value for %s' % (where, ', '.join(left)))
    return text


def rviz_view(session: Session) -> Tuple[float, float, float]:
    """(focal x, focal y, orbit distance): from above, the centre of the agents' starts, far
    enough to see them all, and the world's walls where it has a box (RViz's default field of
    view shows about 0.8 x the distance across)."""
    xs, ys = [a.x for a in session.agents], [a.y for a in session.agents]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    walls = world_walls(session.world)
    if walls:
        span = max(span, walls[2] - walls[0], walls[3] - walls[1])
    return (sum(xs) / len(xs), sum(ys) / len(ys), max(15.0, 1.25 * span + 2.0))


def rviz_config(session: Session, templates: Path = RVIZ_TEMPLATES) -> str:
    """<run>/rviz/session.rviz from rviz/templates/: Fixed Frame world (sim_tf.py's), per agent
    a display group on the robot's topics in its own frames, SetGoal on GOAL_TOPIC. Displays that
    are off do not subscribe: the ones on are kept few (each is a subscription the graph has to
    undeclare when RViz closes)."""
    def read(name: str) -> str:
        try:
            return (templates / name).read_text()
        except OSError as e:
            raise SimError('RViz template %s: %s' % (templates / name, e.strerror))
    head, agent, gpu, foot = (read(n) for n in RVIZ_PARTS)
    out = [fill_template(head, {'SESSION': '%s (%s)' % (session.name, session.source)}, RVIZ_PARTS[0])]
    for i, a in enumerate(session.agents):
        values = {'NS': a.name, 'COLOR': AGENT_COLORS[i % len(AGENT_COLORS)],
                  'EXPLORE': 'true' if session.demo == 'explore' else 'false'}
        extra = fill_template(gpu, values, RVIZ_PARTS[2]) if session.mapping == 'gpu' else ''
        if agent.count('@MAPPING_DISPLAYS@\n') != 1:
            raise SimError('%s: needs one @MAPPING_DISPLAYS@ line' % RVIZ_PARTS[1])
        out.append(fill_template(agent.replace('@MAPPING_DISPLAYS@\n', extra), values, RVIZ_PARTS[1]))
    fx, fy, dist = rviz_view(session)
    out.append(fill_template(foot, {'GOAL_TOPIC': GOAL_TOPIC, 'FOCAL_X': fmt_float(fx, 2), 'FOCAL_Y': fmt_float(fy, 2),
                                    'DISTANCE': fmt_float(dist, 1)}, RVIZ_PARTS[3]))
    text = ''.join(out)
    try:
        doc = yaml.safe_load(text)
        names = [d['Name'] for d in doc['Visualization Manager']['Displays']]
    except (yaml.YAMLError, KeyError, TypeError) as e:
        raise SimError('rviz/templates/ make no RViz config for session %s: %s' % (session.name, e))
    if names[-len(session.agents):] != [a.name for a in session.agents]:
        raise SimError('rviz/templates/: the agent groups came out as %s' % names)
    return text


def write_rviz_config(session: Session, ctx: RunContext) -> None:
    out = ctx.run_dir / RVIZ_RUN_DIR
    empty_own_dir(out)
    write_file(out / RVIZ_FILE, rviz_config(session).encode())
    info('wrote %s/%s (Fixed Frame world, SetGoal -> %s, %s)' % (RVIZ_RUN_DIR, RVIZ_FILE, GOAL_TOPIC,
                                                               ', '.join(a.name for a in session.agents)))


def rviz_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """sim-gazebo (rviz2 is in its image, X through /tmp/.X11-unix): off gzserver's cores, on
    the rest with the bridges (pin_self('rest', ctx)). Without ROVER_SIM_GAZEBO_RUNTIME=nvidia it
    renders in software, so the config draws at 10 Hz. It starts in a paused wave, before the
    unpause, like every heavy start (module docstring)."""
    return [Pane('RViz %s (Fixed Frame world; SetGoal -> %s)' % (RVIZ_CONFIG_IN, GOAL_TOPIC), 'rviz', 'sim-gazebo',
                 pin_self('rest', ctx) + ' && rviz2 -d %s' % RVIZ_CONFIG_IN, window='rviz')]


# Start order is registry order. Waves (Stack.wave): 0 the router and Gazebo, so the spawns run
# on a quiet graph (gzserver's ROS services have hung under a storm of graph events); 1 what DLIO
# needs before it can run; 2 mapping, which reads DLIO; 3 elevation, the heaviest start (cupy on
# the GPU); 4 the planner, which reads all of them (and declares num_agents - 1 frame_align
# subscriptions per agent); 5 RViz and the goals, which talk to the planner.
STACKS = OrderedDict((s.name, s) for s in (
    Stack('zenoh', 1, 'zenoh router in sim-drive', ('sim-drive',), zenoh_panes, implemented=True),
    Stack('gazebo', 2, 'Gazebo world, spawns, sensor bridges, world-><ns>/map', ('sim-gazebo',), gazebo_panes,
          implemented=True, generate=(write_agent_files,), needs_display=lambda s: s.gui),
    Stack('drive', 2, 'teleop + safety_node and rover_description per agent', ('sim-drive',), drive_panes,
          implemented=True, generate=(write_agent_files, link_calib, write_teleop_file), wave=1),
    Stack('frame_align', 2, 'ground-truth frame_align per ordered agent pair', ('sim-mighty',),
          frame_align_panes, implemented=True, wanted=lambda s: len(s.agents) > 1, wave=1),
    Stack('dlio', 3, 'DLIO (truth_odom.py for odometry: truth), seed pose and map->odom per agent',
          ('sim-dlio-jr', 'sim-dlio-rr'), dlio_panes,
          implemented=True, generate=(write_dlio_launch,), wave=1),
    Stack('mapping', 4, 'global_mapper, map_fusion, persistent_map per agent', ('sim-mapping',),
          mapping_panes, implemented=True, generate=(write_mapping_files,), wave=2),
    Stack('elevation', 4, 'GPU elevation mapping per agent', ('sim-elevation',), elevation_panes,
          implemented=True, wanted=lambda s: s.mapping == 'gpu', generate=(check_elevation_launch, check_kernel_cache),
          wave=3),
    Stack('mighty', 5, 'mighty_node, convert_odom_to_state, mpc, goal_selector per agent',
          ('sim-mighty',), mighty_panes, implemented=True, wanted=lambda s: s.planner, wave=4),
    Stack('rviz', 5, 'RViz on $DISPLAY, the last paused wave', ('sim-gazebo',), rviz_panes, implemented=True,
          wanted=lambda s: s.rviz, generate=(write_rviz_config,), needs_display=_always, wave=5),
    Stack('goals', 5, 'goal_relay.py: session and RViz goals to term_goal_rviz', ('sim-mighty',), goals_panes,
          implemented=True, wanted=lambda s: s.planner, generate=(check_goal_relay,), wave=5),
))
# Stacks `start` may (re)start alone on a running session, and the command-line text of their
# processes (stop_processes(match=...) stops the old one first).
RESTARTABLE = OrderedDict([('rviz', 'rviz2 -d ' + RVIZ_CONFIG_IN), ('goals', '/' + GOAL_RELAY + ' ')])


# ---------------------------------------------------------------------------- viz (sim_viz.py)

VIZ_CONTAINER = 'sim-viz'
VIZ_PROFILE = 'viz'                  # compose profile of sim-viz: `up` never starts it
VIZ_SCRIPT = 'sim_viz.py'            # this repo's scripts/, mounted at SCRIPTS_IN
VIZ_LAYERS = ('occ', 'orig_hgp', 'hgp', 'traj', 'frontiers', 'pose', 'map')   # sim_viz.py LAYERS
VIZ_MAX_PANELS = 4                   # sim_viz.py MAX_PANELS
# Ctrl-C to the viewer, then this long for its leave (0.5 s per subscription: 4 agents x 7 layers
# take 14.5 s) before SIGTERM. A docker stop of sim-viz gets the same: its command traps the SIGTERM,
# Ctrl-Cs the viewer and waits, within the same 30 s (compose.stack.yaml stop_grace_period).
VIZ_STOP_GRACE_S = 30.0
VIZ_SNAPSHOT_IN = '/tmp/sim_viz_snapshot.png'    # in sim-viz (the run dir is read-only there); docker cp
# Not in STACKS: `up` never starts it, `viz` does, on the running session.
VIZ_STACK = Stack('viz', 5, 'sim_viz.py: the 2D viewer on $DISPLAY (scripts/sim.py viz)', (VIZ_CONTAINER,),
                  lambda session, ctx: viz_panes(), implemented=True, wanted=_never, needs_display=_always)


def viz_cmd(agents: Sequence[str] = (), layers: Sequence[str] = (), follow: bool = False) -> str:
    cmd = ['python3', '%s/%s' % (SCRIPTS_IN, VIZ_SCRIPT), '--spawn', run_path(SPAWN_FILE)]
    if agents:
        cmd += ['--agents', ','.join(agents)]
    if layers:
        cmd += ['--layers', ','.join(layers)]
    if follow:
        cmd.append('--follow')
    return shlex_join(cmd)


def viz_snapshot_argv(agents: Sequence[str], layers: Sequence[str], after: float, follow: bool = False) -> List[str]:
    """docker exec argv in sim-viz: the container's setup sourced (docker exec skips the entrypoint),
    then one frame to VIZ_SNAPSHOT_IN."""
    cmd = '%s --snapshot %s --snapshot-after %s' % (viz_cmd(agents, layers, follow), VIZ_SNAPSHOT_IN,
                                                    fmt_float(after, 3))
    return ['bash', '-c', '%s && exec %s' % (setup_snippet(CONTAINERS[VIZ_CONTAINER]), cmd)]


def viz_panes(agents: Sequence[str] = (), layers: Sequence[str] = (), follow: bool = False) -> List[Pane]:
    """sim-viz, one pane: its cores come from its cpuset (cpu_budget), its X from compose. Gated on the
    router like every pane."""
    return [Pane('sim_viz.py on $DISPLAY (q quits; Ctrl-C here or scripts/sim.py viz --stop: it leaves the '
                 'graph one subscription at a time)', 'viz', VIZ_CONTAINER, viz_cmd(agents, layers, follow),
                 window='viz')]


def stop_viz() -> int:
    """Ctrl-C the viewer, as in its pane, and wait out its paced leave (VIZ_STOP_GRACE_S), then close
    rsim-viz. Returns how many processes there were."""
    n = 0
    if VIZ_CONTAINER in running_sim_containers():
        n = stop_processes(VIZ_CONTAINER, grace_int=VIZ_STOP_GRACE_S, grace_term=5.0)
        if n:
            info('%s: stopped %d process(es); the viewer left the graph one subscription at a time '
                 '(its pane log: "left the graph")' % (VIZ_CONTAINER, n))
    sess = TMUX_PREFIX + VIZ_STACK.name
    if sess in tmux_sessions():
        tmux('kill-session', '-t', '=' + sess, check=False)
        info('closed tmux session %s' % sess)
    return n


def select_stacks(session: Session, requested: Optional[Sequence[str]] = None) -> List[Stack]:
    if requested is None:
        return [s for s in STACKS.values() if s.implemented and s.wanted(session)]
    for name in requested:
        if name not in STACKS:
            raise SimError('unknown stack %r (stacks: %s)' % (name, ', '.join(STACKS)))
        s = STACKS[name]
        if not s.implemented:
            raise _not_yet(name)
        if not s.wanted(session):
            raise SimError('the %s stack does not apply to session %s (mapping: %s, rviz: %s, planner: %s, demo: %s, '
                           '%d agent(s))' % (name, session.name, session.mapping, session.rviz, session.planner,
                                             session.demo, len(session.agents)))
    return [s for s in STACKS.values() if s.name in requested]


def generate_steps(stacks: Sequence[Stack]) -> List[Callable[[Session, RunContext], None]]:
    """The selected stacks' generate steps in start order, each once."""
    out = []  # type: List[Callable[[Session, RunContext], None]]
    for s in stacks:
        out += [g for g in s.generate if g not in out]
    return out


def pending_stacks(session: Session) -> List[str]:
    return ['%s (Phase %d)' % (s.name, s.phase) for s in STACKS.values()
            if not s.implemented and s.wanted(session)]


def session_profiles(session: Session) -> List[str]:
    return [GPU_PROFILE] if session.mapping == 'gpu' else []


def mapping_label(mode: str, runs: bool = True, absent: str = 'not running') -> str:
    """How `up`, its dry run and `status` name a session's mapping mode; only what runs gets
    the label (`up --stacks` without mapping runs none)."""
    if not runs:
        return '%s (session mapping: %s)' % (absent, mode)
    if mode == 'fallback':
        return ('fallback (NOT PARITY: no elevation map, no map_fusion; global_mapper on the sim\'s fallback '
                'config, grid_relay.py; parity_check.py --mapping fallback)')
    return 'gpu (parity: global_mapper, elevation_mapping_cupy, map_fusion, persistent_map)'


def odometry_word(modes: Sequence[str]) -> str:
    """dlio, truth, or mixed: the agents' odometry in one word."""
    have = set(modes)
    return 'mixed' if len(have) > 1 else (have.pop() if have else 'dlio')


def odometry_label(modes: 'OrderedDict[str, str]', runs: bool = True, absent: str = 'not running') -> str:
    """How `up`, its dry run and `status` name the agents' odometry ({agent: mode}); as with
    mapping_label, only what runs gets the label (`up --stacks` without dlio runs none)."""
    word = odometry_word(modes.values())
    if not runs:
        return '%s (session odometry: %s)' % (absent, word)
    truth = [n for n, m in modes.items() if m == 'truth']
    if not truth:
        return 'dlio (parity: DLIO per agent)'
    what = 'ground truth as DLIO (truth_odom.py; parity_check.py takes it from session.json)'
    if word == 'truth':
        return 'truth (NOT PARITY: %s for every agent)' % what
    return 'mixed (NOT PARITY for %s: %s; DLIO for %s)' % (
        ', '.join(truth), what, ', '.join(n for n, m in modes.items() if m != 'truth'))


def session_odometry(session: Session) -> 'OrderedDict[str, str]':
    return OrderedDict((a.name, a.odometry) for a in session.agents)


def json_odometry(data: Dict[str, Any]) -> 'OrderedDict[str, str]':
    """{agent: mode} from a raw session.json; one written before the key existed is all dlio."""
    default = data.get('odometry', 'dlio')
    return OrderedDict((a.get('name'), a.get('odometry', default)) for a in data.get('agents', ())
                       if isinstance(a, dict))


# ---------------------------------------------------------------------------- tmux

def tmux_windows(panes: Sequence[Pane], max_per_window: int = MAX_PANES_PER_WINDOW
                 ) -> List[Tuple[str, List[Pane]]]:
    """Group panes by window (in first-seen order), splitting an oversized window into
    <name>, <name>-2, ..."""
    groups = OrderedDict()  # type: Dict[str, List[Pane]]
    for p in panes:
        groups.setdefault(p.window, []).append(p)
    out = []
    for name, ps in groups.items():
        for k in range(0, len(ps), max_per_window):
            out.append((name if k == 0 else '%s-%d' % (name, k // max_per_window + 1),
                        ps[k:k + max_per_window]))
    return out


def tmux(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return sh(['tmux'] + list(args), check=check)


def tmux_sessions() -> List[str]:
    p = tmux('list-sessions', '-F', '#{session_name}', check=False)  # fails with no server
    return [s for s in p.stdout.split() if s.startswith(TMUX_PREFIX)] if p.returncode == 0 else []


# The panes of a stack start this far apart, not at once: a wave's 12-16 sessions joining together
# on shared cores have made the router block on a peer that did not read (5 s per undeliverable
# declaration, 'Unable to push non droppable network message' in router.log) for minutes, and a
# declaration lost then never reaches the graph (2026-10-20: RR83's map_fusion subscription to
# elev/occ_2d_topic; check_graph catches that).
PANE_START_GAP_S = 0.25


def start_stack(stack: Stack, panes: Sequence[Pane], log_dir: Path) -> str:
    slugs = [p.slug for p in panes]
    if len(set(slugs)) != len(slugs):
        raise SimError('BUG: %s stack has duplicate pane slugs %s' % (stack.name, slugs))
    sess = TMUX_PREFIX + stack.name
    stack_log = log_dir / stack.name
    stack_log.mkdir(parents=True, exist_ok=True)
    tmux('kill-session', '-t', '=' + sess, check=False)    # '=': exact name, not a prefix
    # history-limit only applies to panes created after it is set, and new-session
    # creates one at once. So the session opens on a placeholder window, which is closed
    # once the real windows exist. (tmux 3.0a's set-option rejects '=' targets.)
    tmux('new-session', '-d', '-s', sess, '-n', TMUX_PLACEHOLDER, '-x', TMUX_SIZE[0], '-y', TMUX_SIZE[1], 'cat')
    tmux('set-option', '-t', sess, 'history-limit', str(TMUX_HISTORY))
    windows = tmux_windows(panes)
    started = 0
    for wname, wpanes in windows:
        # Each pane runs a fresh bash, so the typed command lands in its history:
        # Ctrl-C / Up / Enter restarts that node, gate included.
        tmux('new-window', '-d', '-t', '=%s:' % sess, '-n', wname, 'bash')
        target = '=%s:%s' % (sess, wname)
        tmux('set-option', '-w', '-t', target, 'pane-border-status', 'top')
        tmux('set-option', '-w', '-t', target, 'pane-border-format', ' #{pane_index}: #{pane_title} ')
        for _ in wpanes[1:]:
            tmux('split-window', '-t', target, 'bash')
            tmux('select-layout', '-t', target, 'tiled')
        ids = tmux('list-panes', '-t', target, '-F', '#{pane_id}').stdout.split()
        for pid, pane in zip(ids, wpanes):
            tmux('select-pane', '-t', pid, '-T', pane.title)
            tmux('set-option', '-p', '-t', pid, PANE_SLUG_OPTION, pane.slug)
            if pane.strict_tf_gate:
                tmux('set-option', '-p', '-t', pid, PANE_TF_GATE_OPTION, 'strict')
            # pipe-pane taps the pane at the tmux server: nodes run under docker exec, so
            # this is the only full record of their output, across restarts too.
            tmux('pipe-pane', '-o', '-t', pid, 'cat >> ' + shlex.quote(str(stack_log / (pane.slug + '.log'))))
            if started:
                time.sleep(PANE_START_GAP_S)
            tmux('send-keys', '-t', pid, '-l', pane_command(pane))
            tmux('send-keys', '-t', pid, 'C-m')
            started += 1
        tmux('select-pane', '-t', ids[0])
    tmux('kill-window', '-t', '=%s:%s' % (sess, TMUX_PLACEHOLDER))
    if windows:
        tmux('select-window', '-t', '=%s:%s' % (sess, windows[0][0]))
    return sess


def make_log_dir() -> Path:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    d = LOG_ROOT / stamp
    d.mkdir(parents=True, exist_ok=True)
    latest = LOG_ROOT / 'latest'
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(stamp)
    except OSError as e:
        warn('could not update %s: %s' % (latest, e))
    old = sorted(p for p in LOG_ROOT.iterdir()
                 if p.is_dir() and not p.is_symlink() and re.match(r'^\d{8}_\d{6}$', p.name))
    for p in old[:-LOG_KEEP]:
        shutil.rmtree(str(p), ignore_errors=True)
    return d


# ---------------------------------------------------------------------------- docker

class _AnyTagLoader(yaml.SafeLoader):
    """SafeLoader that also reads compose's own tags (!reset, !override)."""


def _any_tag(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


_AnyTagLoader.add_multi_constructor('!', _any_tag)


def stack_services() -> List[str]:
    return list((yaml.safe_load(COMPOSE_FILE.read_text()) or {}).get('services', {}))


# Service keys a local override may not set: each one can open a path to the host or the
# fleet that the internal network does not cover (another network or the host's, the
# host's PID/IPC namespaces, devices, privileges, kernel settings such as the IPv6 kill
# switch, another container's volumes).
OVERRIDE_FORBIDDEN = ('network_mode', 'networks', 'pid', 'ipc', 'userns_mode', 'privileged', 'cap_add',
                      'ports', 'devices', 'sysctls', 'volumes_from')
# A bind of the docker socket, or of a directory holding it, lets a container start a
# host-network container (which reaches the C2 router on localhost:7447).
DOCKER_SOCKET_RE = re.compile(r'(^|/)(docker|containerd)[^/]*\.sock$|^/(var/)?run/(docker|containerd)(/|$)')
DOCKER_SOCKET_DIRS = ('/', '/run', '/var', '/var/run')


def docker_socket_mount(source: str) -> bool:
    src = os.path.normpath(source) if source.startswith('/') else source
    return bool(DOCKER_SOCKET_RE.search(src)) or src in DOCKER_SOCKET_DIRS


def _volume_source(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get('source') or '')
    return str(v).split(':', 1)[0]


def override_problems(doc: Dict[str, Any]) -> List[str]:
    out = []
    if doc.get('networks'):
        out.append('top-level networks (the rover-sim network is defined in compose.stack.yaml only)')
    for name, svc in sorted((doc.get('services') or {}).items()):
        svc = svc or {}
        for key in OVERRIDE_FORBIDDEN:
            if key in svc:
                out.append('%s sets %s' % (name, key))
        for v in svc.get('volumes') or []:
            if docker_socket_mount(_volume_source(v)):
                out.append('%s mounts %s (the docker socket)' % (name, _volume_source(v)))
    return out


def local_override() -> Optional[Path]:
    """docker/compose.local.yaml is shared with the legacy docker/compose.yaml. Append it
    only if it overrides this stack's services: an override of the legacy `rover-sim`
    service would add a broken service to this project. Refuse one that could undo the
    sim's isolation (OVERRIDE_FORBIDDEN)."""
    if not LOCAL_COMPOSE_FILE.is_file():
        return None
    try:
        doc = yaml.load(LOCAL_COMPOSE_FILE.read_text(), Loader=_AnyTagLoader) or {}
    except yaml.YAMLError as e:
        raise SimError('%s: %s' % (LOCAL_COMPOSE_FILE, e))
    foreign = sorted(set((doc.get('services') or {})) - set(stack_services()))
    if foreign:
        warn('not applying %s to the stack: it overrides service(s) %s, which only the legacy '
             'docker/compose.yaml has' % (LOCAL_COMPOSE_FILE.name, ', '.join(foreign)))
        return None
    problems = override_problems(doc)
    if problems:
        raise SimError('refusing %s: it could undo the sim\'s isolation from this host and the fleet:\n  - %s'
                       '\n  (keys a local override may not set: %s, a docker socket volume, top-level networks)'
                       % (LOCAL_COMPOSE_FILE, '\n  - '.join(problems), ', '.join(OVERRIDE_FORBIDDEN)))
    return LOCAL_COMPOSE_FILE


def compose_cmd(profiles: Sequence[str] = ()) -> List[str]:
    cmd = ['docker', 'compose', '-p', PROJECT, '--env-file', str(STACKS_ENV), '-f', str(COMPOSE_FILE)]
    local = local_override()
    if local:
        cmd += ['-f', str(local)]
    for p in profiles:
        cmd += ['--profile', p]
    return cmd


# CPU budget (each service's cpuset in compose.stack.yaml), for hosts with at least
# CPU_BUDGET_MIN cores. The first HOST_CPUS cores are left to the host: X, docker and
# the fleet C2 stack, which the sim must never starve. sim-gazebo gets GAZEBO_CPUS cores
# that no other sim container may use, because physics has to keep real time (RTF
# below 0.95 fails a run); Phase 2 pins gzserver to part of them. The other stacks share
# the remaining cores and the kernel balances them; splitting those further waits for the
# per-stack load measured in Phase 5. ROVER_SIM_CPUSET_<NAME> overrides one service,
# e.g. ROVER_SIM_CPUSET_GAZEBO=2-7, and an empty value leaves it unpinned.
# sim-viz, started on demand while physics runs, gets the top VIZ_CPUS of the shared cores that
# are not SMT siblings of gzserver's: a busy sibling thread slows the core physics runs on, and
# the worlds' real-time budgets were measured without a viewer.
HOST_CPUS = 2
GAZEBO_CPUS = 4
CPU_BUDGET_MIN = 12
VIZ_CPUS = 4


def cpuset_var(service: str) -> str:
    return 'ROVER_SIM_CPUSET_' + service[len('sim-'):].upper().replace('-', '_')


def parse_cpu_list(text: str) -> List[int]:
    """'2,12' or '16-19' (sysfs, cpusets) -> [2, 12] or [16, 17, 18, 19]."""
    out = []  # type: List[int]
    for part in text.strip().split(','):
        if part:
            a, _, b = part.partition('-')
            out += list(range(int(a), int(b or a) + 1))
    return out


def cpu_list(cpus: Iterable[int]) -> str:
    """[16, 17, 18, 19, 21] -> '16-19,21'."""
    runs = []  # type: List[List[int]]
    for c in sorted(set(cpus)):
        if runs and c == runs[-1][-1] + 1:
            runs[-1].append(c)
        else:
            runs.append([c])
    return ','.join('%d-%d' % (r[0], r[-1]) if len(r) > 1 else '%d' % r[0] for r in runs)


def half_siblings(ncpu: int) -> Callable[[int], Iterable[int]]:
    """Linux's usual numbering of 2-way SMT: CPU n and n + ncpu/2 share a core."""
    half = ncpu // 2
    return lambda c: {c + half if c < half else c - half}


def host_siblings(cpu: int) -> Iterable[int]:
    """This host's SMT siblings of a CPU (sysfs), else half_siblings's guess."""
    try:
        text = Path('/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list' % cpu).read_text()
        return set(parse_cpu_list(text)) - {cpu}
    except (OSError, ValueError):
        return half_siblings(docker_ncpu())(cpu)


def viz_cpus(ncpu: int, siblings: Callable[[int], Iterable[int]]) -> List[int]:
    gz = set(range(HOST_CPUS, HOST_CPUS + GAZEBO_CPUS))
    near = gz | {s for c in gz for s in siblings(c)}
    shared = list(range(HOST_CPUS + GAZEBO_CPUS, ncpu))
    return [c for c in shared if c not in near][-VIZ_CPUS:] or shared


def cpu_budget(ncpu: int, services: Iterable[str],
               siblings: Optional[Callable[[int], Iterable[int]]] = None) -> Dict[str, str]:
    """service -> cpuset; {} (unpinned) on a host too small to split. siblings: a CPU's SMT
    siblings (default half_siblings)."""
    if ncpu < CPU_BUDGET_MIN:
        return {}
    gz = '%d-%d' % (HOST_CPUS, HOST_CPUS + GAZEBO_CPUS - 1)
    rest = '%d-%d' % (HOST_CPUS + GAZEBO_CPUS, ncpu - 1)
    out = OrderedDict((s, gz if s == 'sim-gazebo' else rest) for s in services)
    if VIZ_CONTAINER in out:
        out[VIZ_CONTAINER] = cpu_list(viz_cpus(ncpu, siblings or half_siblings(ncpu)))
    return out


_NCPU = []  # type: List[int]


def docker_ncpu() -> int:
    """CPUs of the docker host, which is what a cpuset is checked against."""
    if not _NCPU:
        p = sh(['docker', 'info', '-f', '{{.NCPU}}'], check=False, timeout=20)
        try:
            _NCPU.append(int(p.stdout.strip()))
        except ValueError:
            _NCPU.append(os.cpu_count() or 1)
    return _NCPU[0]


def compose_env(run_dir: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env['ROVER_SIM_RUN'] = str(run_dir)
    for svc, cpus in cpu_budget(docker_ncpu(), CONTAINERS, host_siblings).items():
        env.setdefault(cpuset_var(svc), cpus)
    return env


def docker_inspect(kind: str, name: str) -> Optional[Dict[str, Any]]:
    p = sh(['docker', kind, 'inspect', name], check=False)
    if p.returncode != 0:
        return None
    data = json.loads(p.stdout or '[]')
    return data[0] if data else None


def running(name: str) -> bool:
    c = docker_inspect('container', sim_only(name))
    return bool(c and c['State'].get('Running'))


def running_sim_containers() -> List[str]:
    p = sh(['docker', 'ps', '--filter', 'label=com.docker.compose.project=' + PROJECT,
            '--format', '{{.Names}}'])
    up = set(p.stdout.split())
    return [n for n in CONTAINERS if n in up]


def dexec(name: str, args: Sequence[str], check: bool = True, timeout: Optional[float] = 60,
          env: Sequence[Tuple[str, str]] = (), detach: bool = False) -> subprocess.CompletedProcess:
    cmd = ['docker', 'exec'] + (['-d'] if detach else [])
    for k, v in env:
        cmd += ['-e', '%s=%s' % (k, v)]
    return sh(cmd + [sim_only(name)] + list(args), check=check, timeout=timeout)


def container_ip(name: str) -> Optional[str]:
    c = docker_inspect('container', sim_only(name))
    net = ((c or {}).get('NetworkSettings', {}).get('Networks') or {}).get(NETWORK) or {}
    return net.get('IPAddress') or None


def compose_images(profiles: Sequence[str], run_dir: Path) -> Dict[str, str]:
    p = sh(compose_cmd(profiles) + ['config', '--format', 'json'], env=compose_env(run_dir))
    return OrderedDict((k, v.get('image', '')) for k, v in json.loads(p.stdout)['services'].items())


def check_images(images: Dict[str, str]) -> None:
    missing = ['%s (%s)' % (img, svc) for svc, img in images.items()
               if docker_inspect('image', img) is None]
    if missing:
        raise SimError('missing image(s): %s\n  build them with `make images` '
                       '(scripts/build_images.sh)' % ', '.join(missing))


# build_images.sh labels: rover_sim.source is `pin` for a robot repo built at its pin,
# `rover_sim` for this repo's own targets, `local:<dir>` / `rover_sim+local:<dir>` for a
# --src build. An image without it was retagged (--reuse) or tagged by hand.
PINNED_SOURCES = ('pin', 'rover_sim')


def image_provenance(images: Dict[str, str]) -> List[str]:
    """One line per service image that is not what the pins in docker/stacks.env build."""
    out = []
    for svc, img in images.items():
        d = docker_inspect('image', img)
        if d is None:
            continue
        labels = (d.get('Config') or {}).get('Labels') or {}
        src = labels.get('rover_sim.source')
        if src is None:
            out.append('%s runs %s, which build_images.sh did not build (--reuse or hand-tagged): '
                       'not verified against its pin' % (svc, img))
        elif src not in PINNED_SOURCES:
            out.append('%s runs %s, a LOCAL build (rover_sim.source=%s), not its pin' % (svc, img, src))
        elif labels.get('rover_sim.dirty') == 'true':
            out.append('%s runs %s, built from uncommitted changes (rover_sim.dirty=true, revision %s)'
                       % (svc, img, labels.get('org.opencontainers.image.revision', '?')[:7]))
    return out


def gpu_available() -> bool:
    return shutil.which('nvidia-smi') is not None


def wait_running(names: Sequence[str], timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while True:
        down = [n for n in names if not running(n)]
        if not down:
            return
        if time.time() > deadline:
            raise SimError('container(s) did not come up: %s (docker logs <name>)' % ', '.join(down))
        time.sleep(1)


# ---------------------------------------------------------------------------- run dir

def run_dir_from_env() -> Path:
    return Path(os.environ.get('ROVER_SIM_RUN') or RUN_DIR_DEFAULT)


def mounted_run_dir() -> Optional[Path]:
    c = docker_inspect('container', ROUTER_CONTAINER)
    for m in (c or {}).get('Mounts', []):
        if m.get('Destination') == RUN_DIR_IN:
            return Path(m['Source'])
    return None


def resolve_run_dir() -> Path:
    """The run dir the running containers actually mount (it wins over $ROVER_SIM_RUN)."""
    env, mounted = run_dir_from_env(), mounted_run_dir()
    if mounted and mounted != env:
        warn('the containers mount %s as the run dir, not %s ($ROVER_SIM_RUN); using theirs'
             % (mounted, env))
    return mounted or env


def to_container_path(host_path: Path, run_dir: Path) -> str:
    return RUN_DIR_IN + '/' + str(host_path.relative_to(run_dir)) if host_path != run_dir else RUN_DIR_IN


def run_dir_chain(run_dir: Path) -> List[str]:
    parts = [c for c in str(run_dir).split('/') if c]
    return ['/'] + ['/' + '/'.join(parts[:i + 1]) for i in range(len(parts))]


def check_run_dir(run_dir: Path, docker_made_ok: bool = False) -> bool:
    """Refuse a run dir that another local user could swap or write. Docker follows
    symlinks when it mounts it, and the containers run as root: they write, chown and
    execute files there (the load test's agent script, later the per-run configs). So:
    no symlink on the way; every ancestor owned by root or by us and writable by no one
    else, unless it is a root-owned sticky dir such as /tmp, whose next entry must then be
    ours; the dir itself ours and writable by no one else.

    With docker_made_ok, the dir itself may instead be root-owned and empty (docker
    creates a missing bind source that way): returns True so the caller can chown it."""
    path = os.path.abspath(str(run_dir))
    real = os.path.realpath(path)
    if real != path:
        raise SimError('run dir %s goes through a symlink (it is %s); set ROVER_SIM_RUN to a real path'
                       % (path, real))
    uid, problems, root_empty, after_shared = os.getuid(), [], False, False
    chain = run_dir_chain(Path(path))
    for i, d in enumerate(chain):
        try:
            st = os.lstat(d)
        except OSError as e:
            raise SimError('run dir %s: %s' % (run_dir, e))
        last = i == len(chain) - 1
        shared = bool(st.st_mode & 0o022)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            problems.append('%s is not a directory' % d)
        elif last and docker_made_ok and st.st_uid == 0 and not shared and not os.listdir(d):
            root_empty = True
        elif (last or after_shared) and st.st_uid != uid:
            problems.append('%s is owned by uid %d, not by you (uid %d)' % (d, st.st_uid, uid))
        elif st.st_uid not in (0, uid):
            problems.append('%s is owned by another user (uid %d)' % (d, st.st_uid))
        elif shared and (last or st.st_uid != 0 or not st.st_mode & stat.S_ISVTX):
            problems.append('%s is writable by group or others (mode %o)' % (d, stat.S_IMODE(st.st_mode)))
        after_shared = shared
    if problems:
        raise SimError('unsafe run dir %s:\n  - %s\n  (it is mounted read-write into every sim container, '
                       'where nodes run as root; set ROVER_SIM_RUN to a dir only you can write)'
                       % (run_dir, '\n  - '.join(problems)))
    return root_empty


def prepare_run_dir(run_dir: Path) -> None:
    """Before compose up: create the run dir (ours, mode 700), since docker would create
    a missing bind source as root, and refuse an unsafe one."""
    path = Path(os.path.abspath(str(run_dir)))
    if os.path.realpath(str(path)) != str(path):    # before creating anything through a link
        raise SimError('run dir %s goes through a symlink (it is %s); set ROVER_SIM_RUN to a real path'
                       % (path, os.path.realpath(str(path))))
    for d in run_dir_chain(path):
        if not os.path.lexists(d):
            try:
                os.mkdir(d, 0o700)
                os.chmod(d, 0o700)              # not left to the umask
            except OSError as e:
                raise SimError('cannot create the run dir %s: %s' % (run_dir, e))
    check_run_dir(path, docker_made_ok=True)


def ensure_run_dir(run_dir: Path) -> None:
    """After compose up: take back a run dir that docker created as root and empty (a
    plain `docker compose up` before any sim.py run), from inside a container where we
    are root. Never chown anything else: the path is checked first."""
    prepare_run_dir(run_dir)
    if check_run_dir(run_dir, docker_made_ok=True):
        if mounted_run_dir() != Path(os.path.abspath(str(run_dir))) or not running(ROUTER_CONTAINER):
            raise SimError('run dir %s is owned by root (docker made it); remove it, or bring the stack up '
                           'with scripts/sim.py, which takes it over' % run_dir)
        dexec(ROUTER_CONTAINER, ['chown', '%d:%d' % (os.getuid(), os.getgid()), RUN_DIR_IN])
        dexec(ROUTER_CONTAINER, ['chmod', '700', RUN_DIR_IN])
    check_run_dir(run_dir)
    if not os.access(str(run_dir), os.W_OK):
        raise SimError('run dir %s is not writable' % run_dir)


def write_session_file(session: Session, run_dir: Path) -> Path:
    data = session.to_json()
    data['started'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    path = run_dir / SESSION_FILE
    write_file(path, (json.dumps(data, indent=2) + '\n').encode())
    return path


# ---------------------------------------------------------------------------- isolation

# Per target ip/port: rc and bash's error. Run in parallel so unreachable gateways
# (ARP timeouts) cost one timeout in total. The probe connects and closes without
# writing: if isolation were ever broken, the target is the fleet C2 router.
_PROBE_SH = ('for t in "$@"; do ( err=$(timeout 2 bash -c "exec 3<>/dev/tcp/$t && exec 3>&-" 2>&1); '
             'echo "PROBE $t $? $err" ) & done; wait')


def host_ipv4s() -> List[str]:
    p = sh(['ip', '-4', '-o', 'addr', 'show'], check=False)
    return sorted({ip for ip in re.findall(r'\binet (\d+\.\d+\.\d+\.\d+)/', p.stdout or '')
                   if not ip.startswith('127.')})


def classify_probe(line: str) -> Optional[str]:
    """None when the target is unreachable (the isolated case), else what was reached."""
    parts = line.split(' ', 3)
    if len(parts) < 3 or parts[0] != 'PROBE':
        return None
    target, rc, err = parts[1], parts[2], parts[3] if len(parts) > 3 else ''
    if rc == '0':
        return '%s is open' % target
    if 'refused' in err.lower():
        return '%s answers (connection refused): the host is reachable' % target
    return None


def ipv6_addresses(if_inet6: str) -> List[str]:
    """Non-loopback addresses from /proc/net/if_inet6 (addr ifindex plen scope flags ifname)."""
    out = []
    for line in if_inet6.splitlines():
        f = line.split()
        if len(f) == 6 and f[5] != 'lo':
            out.append('%s%%%s' % (ipaddress.IPv6Address(bytes.fromhex(f[0])), f[5]))
    return out


def default_routes(proc_net_route: str) -> List[str]:
    """'<iface> via <gateway>' for each IPv4 default route in /proc/net/route."""
    out = []
    for line in proc_net_route.splitlines()[1:]:
        f = line.split()
        if len(f) >= 3 and f[1] == '00000000':
            out.append('%s via %s' % (f[0], ipaddress.IPv4Address(bytes.fromhex(f[2])[::-1])))
    return out


def namespace_problems(name: str, hc: Dict[str, Any]) -> List[str]:
    """Host namespaces. A shared PID namespace would also turn `down` into a kill of
    every process on the host (stop_processes refuses it as well)."""
    out = []
    if hc.get('PidMode'):
        out.append('%s shares a PID namespace (pid: %s)' % (name, hc['PidMode']))
    ipc = hc.get('IpcMode') or ''
    if ipc == 'host' or ipc.startswith('container:'):
        out.append('%s shares an IPC namespace (ipc: %s)' % (name, ipc))
    if hc.get('UsernsMode') == 'host':
        out.append('%s runs in the host user namespace (userns_mode: host)' % name)
    return out


_NET_STATE_SH = 'cat /proc/net/if_inet6 2>/dev/null; echo @@route; cat /proc/net/route'


def check_isolation(probe: bool = True) -> List[str]:
    """Problems that break the sim's isolation from the host and the fleet; [] if none."""
    problems = []
    net = docker_inspect('network', NETWORK)
    if net is None:
        return ['network %s does not exist (compose up creates it)' % NETWORK]
    if not net.get('Internal'):
        problems.append('network %s is not internal' % NETWORK)
    if (net.get('Options') or {}).get('com.docker.network.bridge.inhibit_ipv4') != 'true':
        problems.append('network %s has an IPv4 address on its bridge (inhibit_ipv4 unset): '
                        'host listeners are reachable through it' % NETWORK)
    names = running_sim_containers()
    for name in names:
        c = docker_inspect('container', name) or {}
        hc = c.get('HostConfig', {})
        nets = sorted((c.get('NetworkSettings', {}).get('Networks') or {}))
        if hc.get('NetworkMode') == 'host' or nets != [NETWORK]:
            problems.append('%s: network mode %s, networks %s; expected only %s'
                            % (name, hc.get('NetworkMode'), nets, NETWORK))
        if hc.get('Privileged'):
            problems.append('%s is privileged' % name)
        if hc.get('PortBindings'):
            problems.append('%s publishes ports %s' % (name, sorted(hc['PortBindings'])))
        caps = set(hc.get('CapAdd') or [])
        if caps & {'NET_ADMIN', 'CAP_NET_ADMIN', 'SYS_ADMIN', 'CAP_SYS_ADMIN', 'ALL'}:
            problems.append('%s has capabilities %s' % (name, sorted(caps)))
        if hc.get('Devices'):
            problems.append('%s maps host devices %s' % (name, [d.get('PathOnHost') for d in hc['Devices']]))
        problems += namespace_problems(name, hc)
        for m in c.get('Mounts') or []:
            if docker_socket_mount(m.get('Source') or ''):
                problems.append('%s mounts %s at %s: through the docker socket it can start a host-network '
                                'container' % (name, m.get('Source'), m.get('Destination')))
    # The host bridge's fe80:: address is reachable from any container with IPv6 on eth0,
    # and a default route would lead off the subnet (an internal network has none).
    state = parallel(lambda n: dexec(n, ['bash', '-c', _NET_STATE_SH], check=False).stdout or '', names)
    for name, text in zip(names, state):
        if '@@route\n' not in text:
            problems.append('%s: could not read its addresses and routes' % name)
            continue
        if_inet6, _, route = text.partition('@@route\n')
        addrs = ipv6_addresses(if_inet6)
        if addrs:
            problems.append('%s has IPv6 address(es) %s: the host is reachable over link-local'
                            % (name, ', '.join(addrs)))
        routes = default_routes(route)
        if routes:
            problems.append('%s has a default route (%s)' % (name, ', '.join(routes)))
    if probe and running(ROUTER_CONTAINER):
        gws = [c.get('Gateway') for c in ((net.get('IPAM') or {}).get('Config') or []) if c.get('Gateway')]
        targets = ['%s/%d' % (ip, port) for ip in sorted(set(gws) | set(host_ipv4s())) for port in FLEET_PORTS]
        if targets:
            p = dexec(ROUTER_CONTAINER, ['bash', '-c', _PROBE_SH, 'probe'] + targets, check=False, timeout=30)
            for line in (p.stdout or '').splitlines():
                hit = classify_probe(line)
                if hit:
                    problems.append('%s reaches %s' % (ROUTER_CONTAINER, hit))
    return problems


# ---------------------------------------------------------------------------- processes in containers

# [pid, ppid, pgid, starttime, cmdline] of every process but this probe; PID 1 first.
_PS_PY = r'''
import json, os
me = os.getpid()
rows = []
for d in os.listdir('/proc'):
    if not d.isdigit() or int(d) == me:
        continue
    try:
        with open('/proc/%s/cmdline' % d, 'rb') as f:
            cmd = f.read().replace(b'\0', b' ').decode('utf-8', 'replace').strip()
        with open('/proc/%s/stat' % d) as f:
            st = f.read()
    except OSError:
        continue
    if cmd:
        f = st[st.rfind(')') + 2:].split()
        rows.append([int(d), int(f[1]), int(f[2]), int(f[19]), cmd])
print(json.dumps(sorted(rows)))
'''

_KILL_PY = r'''
import os, sys
sig, mode = int(sys.argv[1]), sys.argv[2]
for a in sys.argv[3:]:
    try:
        (os.killpg if mode == 'pg' else os.kill)(int(a), sig)
    except OSError:
        pass
'''


class Proc(NamedTuple):
    pid: int
    ppid: int
    pgid: int
    start: int                         # clock ticks after boot
    cmd: str


def private_pid_namespace(name: str) -> bool:
    """Whether signalling processes inside the container can only reach its own: a
    compose.stack.yaml container with no `pid:` (HostConfig.PidMode empty)."""
    c = docker_inspect('container', sim_only(name))
    if c is None:
        return False
    mode = (c.get('HostConfig') or {}).get('PidMode', None)
    if mode != '':
        warn('%s: not stopping its processes: PID namespace %r is not its own (it would reach '
             'processes outside the container)' % (name, mode))
        return False
    return True


def container_procs(name: str) -> List[Proc]:
    p = dexec(name, ['python3', '-c', _PS_PY], check=False, timeout=20)
    if p.returncode != 0:
        return []
    return [Proc(*r) for r in json.loads(p.stdout or '[]')]


def stop_targets(procs: Sequence[Proc], match: Optional[str] = None) -> Tuple[List[Proc], List[int]]:
    """Processes to stop, and the process groups that may be signalled whole.

    Kept: PID 1 and the container's own command (`sleep infinity`) with anything under
    it. With init: true, PID 1 is docker-init and the command is its oldest child; its
    other children are orphans of finished docker execs (a pane's `&` job), so they go.
    Everything else came from a docker exec.

    Fails closed: unless PID 1 is docker-init (or tini), this is not the private PID
    namespace of a compose.stack.yaml container (in the host's, PID 1 is the host's
    init), and nothing is a target."""
    init = next((p for p in procs if p.pid == 1), None)
    if init is None or not re.search(r'(^|/)(docker-init|tini)( |$)', init.cmd):
        return [], []
    tree = set()  # type: set
    kids = sorted((p for p in procs if p.ppid == 1), key=lambda p: (p.start, p.pid))
    if kids:
        tree = {kids[0].pid}
        while True:
            more = {p.pid for p in procs if p.ppid in tree and p.pid not in tree}
            if not more:
                break
            tree |= more
    keep = tree | {1}
    targets = [p for p in procs if p.pid not in keep and (match is None or match in p.cmd)]
    # Group 0: the leader is outside the container's pid namespace (a non-tty exec).
    keep_groups = {p.pgid for p in procs if p.pid in keep} | {0}
    groups = [] if match else sorted({p.pgid for p in targets} - keep_groups)
    return targets, groups


def stop_processes(name: str, match: Optional[str] = None, grace_int: float = 10.0,
                   grace_term: float = 5.0) -> int:
    """Ctrl-C every pane's process group in the container (as a person would), then
    SIGTERM, then SIGKILL whatever ignores it. Returns how many processes there were.
    Refuses a container that does not have a PID namespace of its own."""
    if not private_pid_namespace(name):
        return 0
    targets, groups = stop_targets(container_procs(name), match)
    if not targets:
        return 0
    pids = [p.pid for p in targets]
    grouped = {p.pid for p in targets if p.pgid in groups}
    for sig, wait in ((2, grace_int), (15, grace_term), (9, 2.0)):
        if sig == 2 and groups:
            dexec(name, ['python3', '-c', _KILL_PY, str(sig), 'pg'] + [str(g) for g in groups], check=False)
            loose = [p for p in pids if p not in grouped]
        else:
            loose = pids
        if loose:
            dexec(name, ['python3', '-c', _KILL_PY, str(sig), 'pid'] + [str(p) for p in loose], check=False)
        deadline = time.time() + wait
        while True:
            alive = {p.pid for p in container_procs(name)}
            pids = [p for p in pids if p in alive]
            if not pids or time.time() > deadline:
                break
            time.sleep(0.5)
        if not pids:
            break
    if pids:
        warn('%s: %d process(es) survived SIGKILL: %s' % (name, len(pids), pids))
    return len(targets)


def stop_run(remove_containers: bool = False) -> None:
    """Stop every ROS process in the sim containers and close the rsim-* sessions.
    Processes first, so the panes still log their shutdown. The viewer first, alone, waiting out
    its paced leave while the router and the stacks are still up (stop_viz)."""
    names = running_sim_containers()
    if VIZ_CONTAINER in names:
        names.remove(VIZ_CONTAINER)
        stop_viz()
    # sim-mighty's goal_relay.py leaves one entity at a time after its Ctrl-C: wait for it.
    try:
        agents = len(json.loads((resolve_run_dir() / SESSION_FILE).read_text())['agents'])
    except (OSError, ValueError, KeyError, TypeError, SimError):
        agents = 4
    grace = dict((n, max(10.0, goal_relay_leave_s(agents)) if n in STACKS['goals'].containers else 10.0)
                 for n in names)
    counts = parallel(lambda n: stop_processes(n, grace_int=grace[n]), names)
    for n, c in zip(names, counts):
        if c:
            info('%s: stopped %d process(es)' % (n, c))
    for s in tmux_sessions():
        tmux('kill-session', '-t', '=' + s, check=False)
        info('closed tmux session %s' % s)
    if remove_containers:
        run_dir = resolve_run_dir()
        sh(compose_cmd([GPU_PROFILE, VIZ_PROFILE]) + ['down', '--remove-orphans'], capture=False,
           env=compose_env(run_dir))


# ---------------------------------------------------------------------------- tcp tables

# Docker's embedded DNS resolver listens in every container's netns; not a zenoh listener.
DOCKER_DNS_IP = '127.0.0.11'
TCP_STATES = {'01': 'ESTABLISHED', '02': 'SYN_SENT', '03': 'SYN_RECV', '06': 'TIME_WAIT', '0A': 'LISTEN'}


def _hex_addr(h: str) -> str:
    raw = bytes.fromhex(h)
    if len(raw) == 4:
        return str(ipaddress.IPv4Address(raw[::-1]))
    # /proc/net/tcp6 prints the address as four host-order (little-endian) 32-bit words.
    ip6 = ipaddress.IPv6Address(b''.join(raw[i:i + 4][::-1] for i in range(0, 16, 4)))
    return str(ip6.ipv4_mapped or ip6)


def parse_proc_net_tcp(text: str) -> List[Tuple[str, int, str, int, str]]:
    """(local ip, local port, remote ip, remote port, state) from /proc/net/tcp{,6}."""
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) < 4 or not f[0].endswith(':') or not f[0][:-1].isdigit():
            continue
        (la, lp), (ra, rp) = f[1].split(':'), f[2].split(':')
        out.append((_hex_addr(la), int(lp, 16), _hex_addr(ra), int(rp, 16), TCP_STATES.get(f[3], f[3])))
    return out


def analyze_links(tables: Dict[str, List[Tuple[str, int, str, int, str]]], router_ip: Optional[str],
                  router_port: int = ROUTER_PORT, router_container: str = ROUTER_CONTAINER) -> Dict[str, Any]:
    """Unique TCP links across the containers' tables: a link between two containers
    shows up in both. Loopback endpoints are per container (each has its own netns)."""
    def ep(c: str, ip: str, port: int) -> Tuple[str, str, int]:
        return (c if ipaddress.ip_address(ip).is_loopback else '', ip, port)

    def is_router(e: Tuple[str, str, int]) -> bool:
        return e[2] == router_port and ((e[0] == '' and e[1] == router_ip) or e[0] == router_container)

    links = set()
    per = OrderedDict()  # type: Dict[str, Dict[str, int]]
    for c, socks in tables.items():
        st = {'established': 0, 'listen': 0, 'syn_sent': 0}
        for lip, lport, rip, rport, state in socks:
            if state == 'ESTABLISHED':
                st['established'] += 1
                links.add(tuple(sorted([ep(c, lip, lport), ep(c, rip, rport)])))
            elif state == 'LISTEN' and lip != DOCKER_DNS_IP:
                st['listen'] += 1
            elif state == 'SYN_SENT':
                st['syn_sent'] += 1
        per[c] = st
    router = sum(1 for a, b in links if is_router(a) or is_router(b))
    return OrderedDict([('total', len(links)), ('router', router), ('p2p', len(links) - router),
                        ('listen', sum(s['listen'] for s in per.values())),
                        ('syn_sent', sum(s['syn_sent'] for s in per.values())),
                        ('per_container', per)])


def link_snapshot(containers: Sequence[str]) -> Dict[str, Any]:
    outs = parallel(lambda c: dexec(c, ['cat', '/proc/net/tcp', '/proc/net/tcp6'], check=False).stdout or '',
                    list(containers))
    tables = OrderedDict((c, parse_proc_net_tcp(o)) for c, o in zip(containers, outs))
    return analyze_links(tables, container_ip(ROUTER_CONTAINER))


# CPU seconds (utime + stime) of every rmw_zenohd in the container. docker stats would
# count the whole of sim-drive, i.e. the load test's own dummy sessions as well.
_ROUTER_CPU_PY = r'''
import os
t = 0
for d in os.listdir('/proc'):
    try:
        if d.isdigit() and open('/proc/%s/comm' % d).read().strip() == 'rmw_zenohd':
            st = open('/proc/%s/stat' % d).read()
            f = st[st.rfind(')') + 2:].split()
            t += int(f[11]) + int(f[12])
    except OSError:
        pass
print(t / os.sysconf('SC_CLK_TCK'))
'''


def router_cpu_seconds() -> Optional[float]:
    p = dexec(ROUTER_CONTAINER, ['python3', '-c', _ROUTER_CPU_PY], check=False, timeout=20)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def router_up() -> bool:
    if not running(ROUTER_CONTAINER):
        return False
    return dexec(ROUTER_CONTAINER, ['bash', '-c', tcp_connect('%s/%d' % (ROUTER_HOST, ROUTER_PORT)) + ' 2>/dev/null'],
                 check=False).returncode == 0


# ---------------------------------------------------------------------------- zenoh configs

_NUM_OR_WORD = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*'
                          r'|[-+]?(?:0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|Infinity|NaN)')


def json5_loads(text: str) -> Any:
    """The JSON5 subset the zenoh configs use: comments, bare keys, single-quoted
    strings, trailing commas, hex and signed numbers."""
    toks = []  # type: List[str]
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
        elif text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            if j < 0:
                raise ValueError('json5: unterminated comment')
            i = j + 2
        elif ch in '{}[]:,':
            toks.append(ch)
            i += 1
        elif ch in '"\'':
            j, buf = i + 1, []
            while True:
                if j >= n:
                    raise ValueError('json5: unterminated string at %d' % i)
                c = text[j]
                if c == '\\':
                    esc = text[j:j + 2]
                    buf.append("'" if esc == "\\'" else esc)
                    j += 2
                    continue
                if c == ch:
                    break
                buf.append('\\"' if c == '"' else c)
                j += 1
            toks.append('"' + ''.join(buf) + '"')
            i = j + 1
        else:
            m = _NUM_OR_WORD.match(text, i)
            if not m:
                raise ValueError('json5: unexpected %r at %d' % (ch, i))
            toks.append(m.group(0))
            i = m.end()
    out = []
    for k, t in enumerate(toks):
        nxt = toks[k + 1] if k + 1 < len(toks) else ''
        if t == ',' and nxt in ('}', ']'):
            continue
        if (t[0].isalpha() or t[0] in '_$') and nxt == ':':
            t = json.dumps(t)
        elif t[0] in '+-.0123456789':
            t = t[1:] if t[0] == '+' else t
            body = t.lstrip('-')
            if body.lower().startswith('0x'):
                t = str(int(t, 16))
            elif body[:1] not in ('I', 'N'):             # Infinity / NaN pass through
                if body.startswith('.'):
                    t = t.replace('.', '0.', 1)
                if t.endswith('.'):
                    t += '0'
        out.append(t)
    return json.loads(' '.join(out))


def _set(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split('.')
    for k in keys[:-1]:
        cfg = cfg.setdefault(k, {})
    cfg[keys[-1]] = value


# Load-test topologies. `shipped` is docker/zenoh/session.json5 as it is; the others are
# derived from it so they differ only in the keys named here.
ZENOH_VARIANTS = OrderedDict([
    ('shipped', 'docker/zenoh/session.json5 as shipped (peer, no peer links: the router brokers)'),
    ('client', 'client mode: one link per session, to the router'),
    ('local-mesh', 'peer, loopback listener, peers autoconnect: a rover-like mesh inside each '
                   'container; dials to loopback locators gossiped from other containers fail'),
    ('mesh', 'peer, listener on the container interface, peers autoconnect: full mesh'),
])


def session_variant(base: Dict[str, Any], variant: str) -> Optional[Dict[str, Any]]:
    """The session config for a load-test variant; None means use the shipped file."""
    if variant not in ZENOH_VARIANTS:
        raise SimError('unknown zenoh variant %r (%s)' % (variant, ', '.join(ZENOH_VARIANTS)))
    if variant == 'shipped':
        return None
    cfg = copy.deepcopy(base)
    if variant == 'client':
        _set(cfg, 'mode', 'client')
        # The shipped values make a client give up at once if the router is not there yet.
        _set(cfg, 'connect.timeout_ms', {'router': -1, 'peer': -1, 'client': -1})
        _set(cfg, 'connect.exit_on_failure', {'router': False, 'peer': False, 'client': False})
    else:
        _set(cfg, 'mode', 'peer')
        _set(cfg, 'listen.endpoints', ['tcp/localhost:0'] if variant == 'local-mesh' else ['tcp/0.0.0.0:0'])
        _set(cfg, 'scouting.gossip.autoconnect', {'router': [], 'peer': ['router', 'peer']})
    return cfg


# ---------------------------------------------------------------------------- load test

# Sessions per agent in each stack's container: about what the robot runs per stack
# (one per node process; mighty is 4 panes, DLIO 3 nodes + seed + map_odom_tf, ...).
LT_PROFILES = OrderedDict([
    ('robot', OrderedDict([('gazebo', 3), ('drive', 4), ('dlio', 5), ('mapping', 3),
                           ('elevation', 1), ('mighty', 4)])),
    ('light', OrderedDict([('gazebo', 1), ('drive', 1), ('dlio', 1), ('mapping', 1),
                           ('elevation', 1), ('mighty', 1)])),
])
# Each stack's first process subscribes to the same agent's upstream stacks, which
# follows the robot's data path: sensors -> odometry -> maps -> planner -> safety -> plant.
LT_UPSTREAM = {'gazebo': ('drive',), 'drive': ('mighty',), 'dlio': ('gazebo',), 'mapping': ('dlio',),
               'elevation': ('dlio',), 'mighty': ('dlio', 'mapping', 'elevation')}
LT_TF_PUBLISHERS = ('gazebo', 'drive', 'dlio')     # /tf is one topic for every agent
LT_TF, LT_TRAJS = '/lt/tf', '/lt/trajs'             # stand-ins for /tf and the fleet-wide /trajs
LT_MARKER = 'lt_agent.py'


def lt_container(stack: str, agent: Agent) -> str:
    return dlio_container(agent) if stack == 'dlio' else 'sim-' + stack


def loadtest_agents(n: int) -> Tuple[Agent, ...]:
    if not 1 <= n <= 19:
        raise SimError('--agents must be 1..19 (ids 81..99)')
    raw = []
    for i in range(n):
        cls = 'jr' if i % 2 == 0 else 'rr'
        raw.append({'name': '%s%d' % (CLASS_PREFIXES[cls], 81 + i), 'class': cls, 'pose': [2.0 * i, 0.0, 0.0]})
    return validate_session({'name': 'loadtest', 'world': 'empty', 'mapping': 'fallback',
                             'agents': raw}, '<loadtest>').agents


def build_loadtest_plan(agents: Sequence[Agent], available: Iterable[str], profile: str = 'robot',
                        topics_per_proc: int = 2, rate_hz: float = 10.0, tf_rate_hz: float = 20.0,
                        payload_bytes: int = 256) -> Dict[str, Any]:
    """One entry per dummy session: what it publishes, and which publishers it must hear
    on each subscription before it counts as discovered. Pure, so it is unit-tested."""
    available = set(available)
    procs, by_key = [], {}  # type: List[Dict[str, Any]], Dict[Tuple[str, str, int], Dict[str, Any]]
    for a in agents:
        for stack, count in LT_PROFILES[profile].items():
            c = lt_container(stack, a)
            if c not in available:
                continue
            for j in range(count):
                pubs = [['/%s/lt/%s%d/t%d' % (a.name, stack, j, k), rate_hz] for k in range(topics_per_proc)]
                if j == 0 and stack in LT_TF_PUBLISHERS:
                    pubs.append([LT_TF, tf_rate_hz])
                if j == 0 and stack == 'mighty':
                    pubs.append([LT_TRAJS, rate_hz])
                p = OrderedDict([('id', '%s.%s%d' % (a.name, stack, j)), ('agent', a.name), ('stack', stack),
                                 ('index', j), ('container', c), ('ns', '/' + a.name),
                                 ('node', 'lt_%s%d' % (stack, j)), ('pubs', pubs), ('subs', {})])
                procs.append(p)
                by_key[(a.name, stack, j)] = p
    tf_pubs = [p['id'] for p in procs if any(t == LT_TF for t, _ in p['pubs'])]
    traj_pubs = [p['id'] for p in procs if any(t == LT_TRAJS for t, _ in p['pubs'])]
    for p in procs:
        subs = OrderedDict()  # type: Dict[str, List[str]]
        for up in LT_UPSTREAM[p['stack']]:
            src = by_key.get((p['agent'], up, 0))
            for t, _ in (src['pubs'] if src else []):
                if not t.startswith('/lt/'):
                    subs[t] = [src['id']]
        if p['index'] == 0:
            subs[LT_TF] = [i for i in tf_pubs if i != p['id']]
            if p['stack'] == 'mighty':
                subs[LT_TRAJS] = [i for i in traj_pubs if i != p['id']]
        p['subs'] = subs
    rates = OrderedDict((t, r) for p in procs for t, r in p['pubs'])
    return OrderedDict([('profile', profile), ('agents', [a.name for a in agents]),
                        ('payload_bytes', payload_bytes), ('topic_rates', rates), ('procs', procs)])


def _pct(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summarize_loadtest(plan: Dict[str, Any], results: Dict[str, Dict[str, Any]], max_discovery: float
                       ) -> Dict[str, Any]:
    """Discovery = from the first session starting to the last one hearing every
    publisher it expects. Delivery = messages received after discovery / expected."""
    procs = {p['id']: p for p in plan['procs']}
    ready = [r for r in results.values() if r.get('t_ready')]
    starts = [r['t_start'] for r in results.values() if r.get('t_start')]
    discovery = (max(r['t_ready'] for r in ready) - min(starts)) if ready and len(ready) == len(procs) else None
    got = exp = 0.0
    for r in ready:
        if r.get('t_end') and r.get('counts_end') and r.get('counts_ready') is not None:
            dt = r['t_end'] - r['t_ready']
            got += sum(r['counts_end'].values()) - sum(r['counts_ready'].values())
            exp += dt * sum(len(set(ids) - {r['id']}) * plan['topic_rates'][t]
                            for t, ids in procs[r['id']]['subs'].items())
    not_ready = sorted(set(procs) - {r['id'] for r in ready})
    out = OrderedDict([
        ('sessions', len(procs)), ('started', len(starts)), ('ready', len(ready)),
        ('discovery_s', discovery),
        ('ready_s', OrderedDict((k, _pct([r['t_ready'] - r['t_start'] for r in ready], q))
                                for k, q in (('p50', .5), ('p95', .95), ('max', 1.0)))),
        ('init_s', OrderedDict((k, _pct([r['t_init'] - r['t_start'] for r in results.values()
                                         if r.get('t_init')], q)) for k, q in (('p50', .5), ('max', 1.0)))),
        ('delivery', (got / exp) if exp else None),
        ('not_ready', not_ready[:20]),
        ('missing', OrderedDict((i, results[i].get('missing')) for i in not_ready[:5] if i in results)),
    ])
    out['pass'] = discovery is not None and discovery <= max_discovery
    return out


LT_AGENT_PY = r'''#!/usr/bin/env python3
"""rover_sim zenoh load-test session. Written into the run dir by scripts/sim.py
loadtest and run inside the sim-* containers (one process = one zenoh session)."""
import json
import os
import subprocess
import sys
import time


def dump(path, obj, plan):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    try:
        os.chown(tmp, plan['uid'], plan['gid'])  # root in here, the host user out there
    except OSError:
        pass
    os.replace(tmp, path)


def launch(plan_path, container):
    with open(plan_path) as f:
        plan = json.load(f)
    ids = [p['id'] for p in plan['procs'] if p['container'] == container]
    dump(os.path.join(plan['dir'], 'launcher-%s.json' % container),
         {'container': container, 'procs': len(ids), 't': time.time()}, plan)
    kids = [subprocess.Popen([sys.executable, os.path.abspath(__file__), 'run', plan_path, i]) for i in ids]
    for k in kids:
        k.wait()


def run(plan_path, pid):
    t_start = time.time()
    with open(plan_path) as f:
        plan = json.load(f)
    me = next(p for p in plan['procs'] if p['id'] == pid)
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    node = Node(me['node'], namespace=me['ns'])
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                     reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
    expected = {t: set(ids) - {pid} for t, ids in me['subs'].items()}
    seen = {t: set() for t in expected}
    counts = {t: 0 for t in expected}

    def on_msg(topic):
        def cb(msg):
            counts[topic] += 1
            seen[topic].add(msg.data.split('|', 1)[0])
        return cb

    subs = [node.create_subscription(String, t, on_msg(t), qos) for t in expected]
    by_rate = {}
    for topic, rate in me['pubs']:
        by_rate.setdefault(rate, []).append(node.create_publisher(String, topic, qos))
    pad = 'x' * plan['payload_bytes']
    seq = [0]

    def ticker(pubs):
        def tick():
            seq[0] += 1
            msg = String()
            msg.data = '%s|%d|%s' % (pid, seq[0], pad)
            for p in pubs:
                p.publish(msg)
        return tick

    timers = [node.create_timer(1.0 / rate, ticker(pubs)) for rate, pubs in by_rate.items()]
    res = {'id': pid, 'container': me['container'], 't_start': t_start, 't_init': time.time(),
           't_ready': None, 'counts_ready': None}
    out = os.path.join(plan['results_dir'], pid + '.json')
    dump(out, res, plan)
    last_poll = 0.0
    try:
        while time.time() < plan['deadline']:
            executor.spin_once(timeout_sec=0.05)
            now = time.time()
            if res['t_ready'] is None and all(seen[t] >= expected[t] for t in expected):
                res['t_ready'], res['counts_ready'] = now, dict(counts)
                dump(out, res, plan)
            if now - last_poll > 0.5:
                last_poll = now
                if os.path.exists(plan['stop_file']):
                    break
    except BaseException as e:  # SIGINT from sim.py's cleanup, or an rclpy shutdown
        res['error'] = repr(e)
    finally:
        res['t_end'] = time.time()
        res['counts_end'] = dict(counts)
        res['missing'] = {t: sorted(expected[t] - seen[t]) for t in expected if expected[t] - seen[t]}
        dump(out, res, plan)
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    if sys.argv[1] == 'launch':
        launch(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[2], sys.argv[3])
'''


def docker_stats(names: Sequence[str]) -> Dict[str, Dict[str, str]]:
    if not names:
        return {}
    p = sh(['docker', 'stats', '--no-stream', '--format', '{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}']
           + [sim_only(n) for n in names], check=False, timeout=30)
    out = OrderedDict()
    for line in (p.stdout or '').splitlines():
        f = line.split('\t')
        if len(f) == 3:
            out[f[0]] = {'cpu': f[1], 'mem': f[2]}
    return out


def run_loadtest_variant(variant: str, agents: Sequence[Agent], available: Sequence[str],
                         args: argparse.Namespace, lt_dir: Path, run_dir: Path,
                         base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    vdir = lt_dir / variant
    (vdir / 'results').mkdir(parents=True, exist_ok=True)
    vdir_in, lt_in = to_container_path(vdir, run_dir), to_container_path(lt_dir, run_dir)
    cfg = session_variant(base_cfg, variant)
    if args.set:
        cfg = copy.deepcopy(cfg if cfg is not None else base_cfg)
        for key, value in args.set:
            _set(cfg, key, value)
    uri = SESSION_URI_IN
    if cfg is not None:
        (vdir / 'session.json').write_text(json.dumps(cfg, indent=2) + '\n')
        uri = vdir_in + '/session.json'
    plan = build_loadtest_plan(agents, available, args.profile, args.topics_per_proc, args.rate,
                               args.tf_rate, args.payload_bytes)
    plan.update(variant=variant, session_uri=uri, dir=vdir_in, results_dir=vdir_in + '/results',
                stop_file=vdir_in + '/stop', deadline=time.time() + args.timeout + args.hold + 30,
                uid=os.getuid(), gid=os.getgid())
    (vdir / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n')
    containers = sorted({p['container'] for p in plan['procs']}, key=list(CONTAINERS).index)
    waited = settle_router()
    if waited is None:
        warn('%s: rmw_zenohd did not go idle within 90 s (another session running?): '
             'session start-up times may be inflated' % variant)
    elif waited > 3:
        info('%s: waited %.0f s for the router to finish cleaning up' % (variant, waited))
    baseline = link_snapshot(available)
    info('%s: %d sessions in %s' % (variant, len(plan['procs']), ', '.join(containers)))

    def launch(c: str) -> None:
        inner = ' && '.join([
            'exec >>%s/launcher-%s.log 2>&1' % (vdir_in, c), setup_snippet(CONTAINERS[c]),
            'exec python3 %s/%s launch %s/plan.json %s' % (lt_in, LT_MARKER, vdir_in, c)])
        dexec(c, ['bash', '-c', inner], env=(('ZENOH_SESSION_CONFIG_URI', uri),), detach=True)

    t_launch = time.time()
    parallel(launch, containers)
    results = {}  # type: Dict[str, Dict[str, Any]]

    def collect() -> None:
        for f in (vdir / 'results').glob('*.json'):
            try:
                results[f.stem] = json.loads(f.read_text())
            except (OSError, ValueError):
                pass  # caught mid-replace; next poll

    deadline = t_launch + args.timeout
    while time.time() < deadline:
        collect()
        if len(plan['procs']) == sum(1 for r in results.values() if r.get('t_ready')):
            break
        time.sleep(0.5)
    dead = [c for c in containers if not (vdir / ('launcher-%s.json' % c)).exists()]
    if dead:
        warn('%s: no launcher started in %s (see %s/launcher-<container>.log)' % (variant, ', '.join(dead), vdir))
    t_hold, cpu0 = time.time(), router_cpu_seconds()
    time.sleep(args.hold)                       # links and data at steady state
    links, cpu = link_snapshot(available), docker_stats(available)
    cpu1 = router_cpu_seconds()
    router_cpu = None
    if cpu0 is not None and cpu1 is not None:
        router_cpu = 100.0 * (cpu1 - cpu0) / (time.time() - t_hold)
    (vdir / 'stop').write_text('')
    end = time.time() + 15
    while time.time() < end:
        collect()
        if all(r.get('t_end') for r in results.values()) and len(results) == len(plan['procs']):
            break
        time.sleep(0.5)
    parallel(lambda c: stop_processes(c, match=LT_MARKER, grace_int=3, grace_term=2), containers)
    summary = summarize_loadtest(plan, results, args.max_discovery)
    summary.update(variant=variant, description=ZENOH_VARIANTS[variant], overrides=dict(args.set),
                   router_settle_s=waited,
                   links=links, router_cpu_pct=router_cpu,
                   baseline_links=baseline, stats=cpu, dir=str(vdir))
    (vdir / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Let the router drop the closed sessions before the next variant measures.
    end = time.time() + 15
    while time.time() < end and link_snapshot([ROUTER_CONTAINER])['router'] > baseline['router']:
        time.sleep(1)
    return summary


def settle_router(timeout: float = 90.0, idle_pct: float = 10.0) -> Optional[float]:
    """Wait until rmw_zenohd is idle: under idle_pct of a core for two 1 s samples in a
    row. After a test's 80 sessions close, the router stays busy for tens of seconds
    cleaning up, and sessions opened meanwhile take 10-40 s to start, whatever their
    mode, so a measurement taken then says nothing about the config under test.
    Returns the seconds waited, None on timeout."""
    t0, quiet = time.time(), 0
    prev = router_cpu_seconds()
    while time.time() - t0 < timeout:
        time.sleep(1.0)
        cur = router_cpu_seconds()
        if prev is None or cur is None:
            return None
        quiet = quiet + 1 if 100.0 * (cur - prev) < idle_pct else 0
        prev = cur
        if quiet >= 2:
            return time.time() - t0
    return None


def _fmt(v: Optional[float], unit: str = 's') -> str:
    return '-' if v is None else '%.1f%s' % (v, unit)


def print_loadtest(summaries: Sequence[Dict[str, Any]], max_discovery: float) -> None:
    head = ('variant', 'sessions', 'ready', 'discovery', 'ready p50/p95', 'links rtr/p2p',
            'listen', 'syn', 'rmw_zenohd cpu', 'sim-drive cpu', 'delivery', 'result')
    rows = [head]
    for s in summaries:
        lk, rt = s['links'], s['stats'].get(ROUTER_CONTAINER, {})
        rows.append((s['variant'], str(s['sessions']), str(s['ready']), _fmt(s['discovery_s']),
                     '%s/%s' % (_fmt(s['ready_s']['p50']), _fmt(s['ready_s']['p95'])),
                     '%d/%d' % (lk['router'], lk['p2p']), str(lk['listen']), str(lk['syn_sent']),
                     _fmt(s.get('router_cpu_pct'), '%'), rt.get('cpu', '-'),
                     '-' if s['delivery'] is None else '%.2f' % s['delivery'],
                     'PASS' if s['pass'] else 'FAIL'))
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    for r in rows:
        print('  ' + '  '.join(c.ljust(w) for c, w in zip(r, widths)))
    print('  (discovery limit %.0f s; links are TCP links at steady state, router = to sim-drive:%d)'
          % (max_discovery, ROUTER_PORT))
    for s in summaries:
        if s['not_ready']:
            print('  %s: not discovered: %s' % (s['variant'], ', '.join(s['not_ready'])))
            for i, m in s['missing'].items():
                print('    %s still waiting for %s' % (i, m))


# ---------------------------------------------------------------------------- commands

def print_plan(session: Session, plan: Sequence[Tuple[Stack, List[Pane]]], arm: bool = False,
               ctx: Optional[RunContext] = None) -> None:
    ctx = ctx or RunContext(run_dir=run_dir_from_env())
    print('session %s (%s): world %s (MID-360 %d rays/scan), gui %s, mapping %s, rviz %s, planner %s, demo %s, '
          'num_agents %d' % (session.name, session.source, session.world, world_lidar_samples(session.world),
                             'on' if session.gui else 'off', session.mapping, 'on' if session.rviz else 'off',
                             'on' if session.planner else 'off', session.demo, session.num_agents))
    names = {stack.name for stack, _ in plan}
    print('mapping: ' + mapping_label(session.mapping, 'mapping' in names, 'not started'))
    print('odometry: ' + odometry_label(session_odometry(session), 'dlio' in names, 'not started'))
    pins = read_pins()
    if session.agents_env:
        print('agents from %s=%s (apply_agents_env)' % (AGENTS_ENV, session.agents_env))
    for a in session.agents:
        print('  %s  %s  id %02d  pose (%.2f, %.2f, %.1f deg)  calibrated from %s, lidar at %.2f m%s%s' % (
            a.name, a.cls, a.id, a.x, a.y, a.yaw_deg, reference_robot(a.cls, pins),
            ROBOT_CLASSES[a.cls].lidar_height,
            '  goals %s' % ' '.join('(%s)' % ', '.join('%.2f' % v for v in g) for g in a.goals) if a.goals else '',
            '  odometry: truth (NOT PARITY)' if a.odometry == 'truth' else ''))
    steps = generate_steps([stack for stack, _ in plan])
    if steps:
        print('before the panes: %s' % ', '.join(g.__name__ for g in steps))
    if 'dlio' in names:
        problem = dlio_scripts_problem(pins)
        print('dlio_ws scripts: %s -> %s/scripts (ro)' % (dlio_scripts_dir(pins), DLIO_WS) if not problem
              else 'WARNING: ' + problem)
    if names & {'mapping', 'elevation'}:
        users = ['sim-mapping'] + (['sim-elevation'] if 'elevation' in names else [])
        problem = mapping_mounts_problem(pins, users)
        where = ['sim-mapping:%s' % MAPPING_CONFIG_IN] + (
            ['sim-elevation:%s and %s' % (ELEVATION_OVERLAY_IN, ORIN_PROBE)] if 'elevation' in names else [])
        print('mapping_ws config: %s -> %s (ro)' % (mapping_checkout(pins), ', '.join(where)) if not problem
              else 'WARNING: ' + problem)
    if 'mapping' in names:
        print('mapping settings: %s' % '; '.join('%s from %s' % (c, MAPPER_SETTINGS[c].source) for c in
                                                 OrderedDict((a.cls, 1) for a in session.agents)))
        problem = mapping_config_problem(session, pins)
        if problem:
            print('WARNING: ' + problem)
        if session.mapping == 'fallback':
            files = OrderedDict()  # type: OrderedDict[str, Tuple[str, str, List[str]]]
            for a in session.agents:
                for node, base, overlay, out in fallback_files(a):
                    files.setdefault(out, (base, overlay, []))[2].append(a.name)
            for out, (base, overlay, who) in files.items():
                print('fallback params: %s = %s/%s + %s/%s (%s)' % (run_path(MAPPING_RUN_DIR, out), MAPPING_CONFIG_IN,
                                                                    base, SIM_MAPPING_IN, overlay, ', '.join(who)))
    if 'elevation' in names:
        print('elevation overlays: %s' % '; '.join('%s %s' % (a.name, ','.join(elevation_overlays(a, pins)) or 'none')
                                                   for a in session.agents))
    if 'mighty' in names:
        print('planner: mighty_hw.sh\'s four panes per agent, num_agents:=%d (the highest id; JR02 runs the launch '
              'default 10); mighty_node and mpc wait for <ns>/map -> <ns>/odom (a timeout stops `up`); mpc\'s wave '
              'waits for "%s"' % (session.num_agents, MPC_READY.replace('\\', '')))
    if 'rviz' in names:
        print('rviz: %s from %s/ on $DISPLAY (%s)' % (RVIZ_CONFIG_IN, RVIZ_TEMPLATES.relative_to(REPO),
                                                     os.environ.get('DISPLAY') or 'unset: it will not open'))
    if 'goals' in names and not (REPO / 'scripts' / GOAL_RELAY).is_file():
        print('WARNING: no scripts/%s yet: the goals stack would fail' % GOAL_RELAY)
    print('start in waves; after each, wait until its panes have started their nodes (up to %d s) and the sim '
          'containers are idle (rmw_zenohd under %.0f %% and all together under %.0f %% of a core, %d samples in a '
          'row; up to %d s)' % (NODES_WAIT_S, SETTLE_ROUTER_PCT, SETTLE_STACKS_PCT_PER_AGENT * len(session.agents),
                                SETTLE_QUIET_SAMPLES, SETTLE_TIMEOUT_S))

    def show(stack: Stack, panes: Sequence[Pane]) -> None:
        print('stack %s -> tmux session %s%s' % (stack.name, TMUX_PREFIX, stack.name))
        for wname, wpanes in tmux_windows(panes):
            print('  window %s' % wname)
            for p in wpanes:
                print('    [%s] %s%s\n      %s' % (p.slug, p.title, '  (ready: %s)' % p.ready if p.ready else '',
                                                  pane_command(p)))
    for k, wave in enumerate(start_waves(plan)):
        print('wave %d: %s' % (k, ', '.join(s.name for s, _ in wave)))
        for stack, panes in wave:
            show(stack, panes)
        if any(s.name == 'gazebo' for s, _ in wave):
            print('wave %d settles once the spawn pane has spawned every agent into the paused world (up to %d s; '
                  'fails if it fails)' % (k, SPAWN_WAIT_S))
    if arm and session.planner:
        print('after the last wave, before the unpause: arm %s: /<ns>/auto_mode at once (auto_arm; --no-arm '
              'skips): nothing commands motion until DLIO tracks, and the query leaves a graph with no sensor data'
              % ', '.join(a.name for a in session.agents))
    if 'gazebo' in names:
        print('after the last wave: unpause physics over gazebo transport (up to %d tries; fails if the world stays '
              'paused):\n      %s' % (UNPAUSE_TRIES, shlex_join(['docker', 'exec', 'sim-gazebo', 'bash', '-c',
                                                                  unpause_cmd(ctx)])))
    if arm and not session.planner:
        print('after the panes: arm %s: /<ns>/auto_mode once its DLIO registers scans (auto_arm; '
              '--no-arm skips)' % ', '.join(a.name for a in session.agents))
    elif session.auto_arm:
        print('not arming: auto_arm needs the dlio and drive stacks, and mighty in a planner session (or --no-arm '
              'was given)')
    else:
        print('not arming: auto_arm is off, so every safety_node stays DISARMED (scripts/sim.py arm)')
    if 'goals' in names:
        print('goal_relay.py drives each agent in AUTO %.1f m ahead in JOY mode first%s%s' % (
            GOAL_RELAY_FIRST_DRIVE_M, ', then starts the swap' if goal_mode(session) == 'swap' else '',
            '' if arm else ' (after scripts/sim.py arm)'))
    pending = pending_stacks(session)
    if pending:
        print('not started yet (later phases): ' + ', '.join(pending))


def grant_x_access() -> None:
    """Let the containers' root use $DISPLAY over /tmp/.X11-unix, and no one else.
    `xhost +local:` would open the display to every local user and container on this
    shared host, keystrokes and all. Root can read our X cookie anyway, so this grants
    nothing new."""
    if not os.environ.get('DISPLAY'):
        warn('no DISPLAY: windows from the sim containers will not open')
        return
    if shutil.which('xhost') is None:
        warn('xhost not found: cannot grant the sim containers X access (they run as root)')
        return
    p = sh(['xhost', '+SI:localuser:root'], check=False)
    if p.returncode != 0:
        warn('xhost +SI:localuser:root failed: %s' % (p.stderr or '').strip())


def busy_containers(names: Sequence[str]) -> List[str]:
    found = parallel(lambda n: bool(stop_targets(container_procs(n))[0]), list(names))
    return [n for n, f in zip(names, found) if f]


def gz_spare_from_env() -> Optional[int]:
    v = os.environ.get('ROVER_SIM_GZ_SPARE_CPUS', '').strip()
    if not v:
        return None
    if not v.isdigit() or int(v) < 1:
        raise SimError('ROVER_SIM_GZ_SPARE_CPUS=%r: expected a number of cores (>= 1)' % v)
    return int(v)


def rotate_rtf_log(run_dir: Path) -> None:
    """A violation belongs to the run whose bridge logged it: each `up` starts a fresh log
    (the previous run's stays as rtf_violations.prev.log)."""
    log = run_dir / RTF_LOG
    if log.exists() or log.is_symlink():
        log.replace(log.with_name('rtf_violations.prev.log'))


_CLEAR_ROS_LOGS_SH = 'rm -rf -- "${ROS_LOG_DIR:-${ROS_HOME:-$HOME/.ros}/log}"'


def clear_ros_logs(names: Sequence[str]) -> None:
    """Each node also logs to its container's ROS log dir, in the writable layer that outlives
    `up`: sim-drive's grew 26 MB in 25 minutes of safety_nodes in AUTO with no cmd_vel_auto.
    The pane logs have the same lines, so each run starts with an empty one."""
    parallel(lambda n: dexec(n, ['sh', '-c', _CLEAR_ROS_LOGS_SH], check=False, timeout=30), list(names))


# The spawn pane's last line once every agent is in the paused world (spawn_cmd); the command
# line the pane log also holds has $(date ...) where this has the time.
SPAWNED_RE = re.compile(r'\[rover_sim\] spawned .* into the paused world at ' + PANE_TIME_RE)
SPAWN_WAIT_S = 300            # gzserver's start, then up to 125 s per stuck spawn_entity (spawn_agent.launch.py)
SPAWN_PANE_GRACE_S = 10       # before a pane's docker exec runs, tmux reports its bash


def spawn_pane_exited() -> bool:
    """The spawn pane is found by its slug (start_stack's pane option), not its title: a
    shell prompt may retitle the pane (an xterm PS1 does)."""
    p = tmux('list-panes', '-s', '-t', '=%sgazebo' % TMUX_PREFIX, '-F',
             '#{%s}\t#{pane_current_command}' % PANE_SLUG_OPTION, check=False)
    rows = [l.split('\t') for l in (p.stdout or '').splitlines() if l.split('\t')[0] == 'spawn']
    return p.returncode != 0 or not rows or rows[0][1] != 'docker'


def wait_spawned(log_dir: Path, timeout: float = SPAWN_WAIT_S) -> float:
    """Until the spawn pane has put every agent in the paused world. Once in 10 bring-ups
    gzserver never answered a /spawn_entity call (nor any other ROS service after it): the pane
    then fails after spawn_agent's timeout, and this says so instead of arming waiting 3 minutes
    for IMU data.
    Raises when the pane fails or takes longer than timeout; returns the seconds waited."""
    log, t0 = log_dir / 'gazebo' / 'spawn.log', time.time()
    while True:
        try:
            if SPAWNED_RE.search(log.read_text(errors='replace')):
                return time.time() - t0
        except FileNotFoundError:
            pass
        waited = time.time() - t0
        if waited > SPAWN_PANE_GRACE_S and spawn_pane_exited():
            raise SimError('the spawn pane failed (%s): an agent was not spawned; '
                           'scripts/sim.py down, then up again' % log)
        if waited > timeout:
            raise SimError('the agents were not all spawned within %.0f s (%s; scripts/sim.py attach gazebo): '
                           'scripts/sim.py down, then up again' % (timeout, log))
        time.sleep(1)


# Physics starts only once every wave has settled. What went wrong otherwise, with 4 agents:
# - Phase 3: a start-up burst in the first seconds of physics (~40 processes importing rclpy on
#   the shared cores, 12-15 among them, the SMT siblings of gzserver's) cost an RTF violation
#   once in 3 runs;
# - Phase 4, mapping panes started after the unpause (all at once, or one agent at a time):
#   zenoh delivery stalls of 0.5-1.2 s, which the sensor bridges take for sim time jumps, and
#   parked DLIOs diverged;
# - Phase 4, every stack at once into the paused world: /spawn_entity never answered (2 of 2),
#   or the router timed out declarations (DeclareFinal) and two DLIOs never got their seed.
# One wave at a time, each settled, then one unpause, was clean in all 6 runs.
NODES_WAIT_S = 120            # a wave's panes past their gates (global_mapper's TF gate takes 60 s)
SETTLE_TIMEOUT_S = 90         # then idle; past either, `up` warns and goes on
SETTLE_QUIET_SAMPLES = 2      # docker stats samples in a row (about 2 s each) under both limits
SETTLE_ROUTER_PCT = 10.0      # rmw_zenohd, % of a core (settle_router's limit)
# Every sim container together (docker stats). A start-up burst keeps several cores busy; the
# paused stacks need well under half a core per agent (Phase 4, paused, after the last wave: 108-126
# % for 4 agents with elevation, 88 % for 2 in fallback).
SETTLE_STACKS_PCT_PER_AGENT = 50.0
# ...and the samples of a quiet run must agree within this many % of a core (or 15 % of the
# load): two global_mappers still starting read 53 % then 88 %, under smoke's 100 % limit.
SETTLE_STEADY_PCT = 15.0
UNPAUSE_CALL_S = 30           # gz world's SIGKILL backstop
UNPAUSE_TRIES = 2


def pane_commands() -> Optional[Dict[Tuple[str, str], str]]:
    """(tmux session, pane slug) -> the pane's current command: 'docker' while its exec runs.
    None when tmux could not list the panes: then nothing is known about them."""
    p = tmux('list-panes', '-a', '-F', '#{session_name}\t#{%s}\t#{pane_current_command}' % PANE_SLUG_OPTION,
             check=False)
    if p.returncode != 0:
        return None
    rows = [l.split('\t') for l in (p.stdout or '').splitlines()]
    return dict(((r[0], r[1]), r[2]) for r in rows if len(r) == 3 and r[0].startswith(TMUX_PREFIX))


def wait_nodes_started(plan: Sequence[Tuple[Stack, List[Pane]]], log_dir: Path, timeout: float = NODES_WAIT_S
                       ) -> Tuple[float, List[str], List[str]]:
    """Until every pane has started its node: its log has node_marker()'s line, which it prints
    once its gates have passed, and then its Pane.ready line if it has one (node_started). A pane
    whose docker exec has ended is not waited for; while tmux cannot list the panes, none counts
    as ended.
    Returns (seconds, panes that exited, panes still waiting at the timeout) as <stack>/<slug>."""
    t0 = time.time()
    pending = OrderedDict(('%s/%s' % (s.name, p.slug), (log_dir / s.name / (p.slug + '.log'), p.ready))
                          for s, panes in plan for p in panes)
    exited = []  # type: List[str]
    while True:
        for key, (log, ready) in list(pending.items()):
            try:
                if node_started(log.read_text(errors='replace'), ready):
                    del pending[key]
            except FileNotFoundError:
                pass
        waited = time.time() - t0
        states = pane_commands() if pending and waited > SPAWN_PANE_GRACE_S else None
        for key in list(pending) if states is not None else []:
            stack, slug = key.split('/', 1)
            if states.get((TMUX_PREFIX + stack, slug)) != 'docker':
                exited.append(key)
                del pending[key]
        if not pending or waited > timeout:
            return waited, exited, list(pending)
        time.sleep(1)


def _cpu_pct(text: str) -> float:
    try:
        return float(text.strip().rstrip('%'))
    except ValueError:
        return 0.0


class LoadSampler:
    """Each call: (rmw_zenohd, every given sim container together) in % of one core since the
    previous call, or None when either cannot be read, a docker call that timed out included
    (a busy daemon during a start burst: `up` goes on). A call takes docker stats' own sampling
    time, about 2 s."""

    def __init__(self, names: Sequence[str]):
        self.names = list(names)
        self.prev = (time.time(), self._router())

    @staticmethod
    def _router() -> Optional[float]:
        try:
            return router_cpu_seconds()
        except SimError:
            return None

    def __call__(self) -> Optional[Tuple[float, float]]:
        try:
            stats = docker_stats(self.names)
        except SimError:
            stats = {}
        now, cur = time.time(), self._router()
        (t0, c0), self.prev = self.prev, (now, cur)
        if not stats or cur is None or c0 is None or now <= t0:
            return None
        return 100.0 * (cur - c0) / (now - t0), sum(_cpu_pct(s['cpu']) for s in stats.values())


def wait_idle(sample: Callable[[], Optional[Tuple[float, ...]]], limits: Sequence[float],
              quiet: int = SETTLE_QUIET_SAMPLES, timeout: float = SETTLE_TIMEOUT_S,
              clock: Callable[[], float] = time.time,
              on_sample: Callable[[Optional[Tuple[float, ...]]], None] = lambda s: None,
              sleep: Callable[[float], None] = time.sleep
              ) -> Tuple[Optional[float], Optional[Tuple[float, ...]]]:
    """Until `quiet` samples in a row are each under its limit and steady (their last value,
    the containers' total, within SETTLE_STEADY_PCT of the previous one: a start-up still ramping
    up is not idle), an equivalent of the load test's settle_router over more than the router.
    A sample that cannot be taken (None) is not quiet: it breaks the run, and the next comes
    after the usual 2 s. Returns (seconds waited, the last sample taken); seconds is None after
    the timeout."""
    t0, run, last = clock(), 0, None
    while clock() - t0 < timeout:
        cur = sample()
        on_sample(cur)
        if cur is None:
            run = 0
            sleep(2.0)
            continue
        if not all(v < lim for v, lim in zip(cur, limits)):
            run = 0
        elif run and abs(cur[-1] - last[-1]) > max(SETTLE_STEADY_PCT, 0.15 * last[-1]):
            run = 1                     # this sample starts a new run
        else:
            run += 1
        last = cur
        if run >= quiet:
            return clock() - t0, last
    return None, last


def start_waves(plan: Sequence[Tuple[Stack, List[Pane]]]) -> List[List[Tuple[Stack, List[Pane]]]]:
    """The plan's stacks grouped by Stack.wave, in wave order (registry order within one)."""
    waves = OrderedDict()  # type: OrderedDict[int, List[Tuple[Stack, List[Pane]]]]
    for s, panes in sorted(plan, key=lambda sp: sp[0].wave):
        waves.setdefault(s.wave, []).append((s, panes))
    return list(waves.values())


def degraded_tf_gates(wave: Sequence[Tuple[Stack, List[Pane]]], log_dir: Path) -> List[Tuple[str, bool]]:
    """(<stack>/<slug>, strict) of each pane whose last wait_for_tf.py run timed out."""
    out = []
    for s, panes in wave:
        for p in panes:
            try:
                text = (log_dir / s.name / (p.slug + '.log')).read_text(errors='replace')
            except FileNotFoundError:
                continue
            if tf_gate_result(text) == 'degraded':
                out.append(('%s/%s' % (s.name, p.slug), p.strict_tf_gate))
    return out


def settle_wave(wave: Sequence[Tuple[Stack, List[Pane]]], n_agents: int, log_dir: Path) -> List[Tuple[str, str]]:
    """After a wave's stacks started: until its panes have started their nodes, then until the
    sim containers are idle. Returns (pane, what went wrong) for the panes that ended before
    starting their node and the strict_tf_gate panes whose TF gate timed out, at once (bring_up
    stops there); a pane that still waits, or stacks that stay busy or whose load cannot be read,
    are reported and do not hold `up` back for ever."""
    log, what = log_dir / BRINGUP_LOG, ', '.join(s.name for s, _ in wave)
    waited, exited, missing = wait_nodes_started(wave, log_dir)
    log_line(log, '%s: nodes started after %.1f s; exited %s; still waiting %s' % (what, waited, exited, missing))
    failed = [(k, 'ended before its node started') for k in exited]
    for key, strict in degraded_tf_gates(wave, log_dir):
        # The global_mapper gate always times out while paused (tf_gate); a planner gate never should.
        log_line(log, '%s: its wait_for_tf.py gate timed out and the node started degraded%s'
                 % (key, ' (strict: stopping)' if strict else ''))
        if strict:
            failed.append((key, 'its TF gate timed out (wait_for_tf.py exits 0 and starts the node anyway)'))
    if failed:
        return failed
    if missing:
        warn('pane(s) still waiting on their gates after %.0f s: %s; going on' % (waited, ', '.join(missing)))
    limits = (SETTLE_ROUTER_PCT, SETTLE_STACKS_PCT_PER_AGENT * n_agents)
    took, last = wait_idle(LoadSampler(running_sim_containers()), limits, on_sample=lambda v: log_line(
        log, '%s: load %s' % (what, 'router %.1f %%, sim containers %.0f %% of a core' % tuple(v) if v
                              else 'not readable (a docker call failed or timed out)')))
    log_line(log, '%s: idle check (limits router %.0f %%, sim containers %.0f %%): %s after %s s'
             % ((what,) + limits + ('ok' if took is not None else 'NOT idle', took)))
    if took is None:
        warn('%s: the stacks did not go idle (%s; limits %.0f %% / %.0f %%): going on' % (
            what, 'router %.0f %%, sim containers %.0f %% of a core' % last if last else 'cannot read the load',
            limits[0], limits[1]))
    else:
        info('%s: %s started their nodes after %.0f s, idle %.0f s later (router %.0f %%, sim containers %.0f %% '
             'of a core)' % ((what, 'the other panes' if missing else 'all panes', waited, took) + tuple(last)))
    return []


GZ_SETUP = '. /usr/share/gazebo/setup.sh'


def unpause_cmd(ctx: RunContext) -> str:
    """The one unpause, in sim-gazebo: gz world over gazebo transport, then gz stats until the
    world reports it runs (gz stats prints nothing while paused). Not /unpause_physics:
    gzserver's ROS executor is what stops answering services (rmw_zenoh 0.1.9's lost wakeup;
    Phase 3 and 4 runs), and gazebo transport does not go through it. Not pinned to the rest
    core like the other gazebo panes: gz world and gz stats never return when their affinity
    is one core (any of sim-gazebo's, every try), and the world is paused while gz world runs."""
    del ctx
    return ' && '.join([GZ_SETUP, 'timeout -k 2 %d gz world -p 0' % UNPAUSE_CALL_S,
                        'for i in 1 2 3; do timeout -k 2 6 gz stats -d 2 | grep -m1 -o "Paused\\[F\\]" '
                        '&& exit 0; done; echo "gz stats: the world still reports no Paused[F]"; exit 1'])


def unpause_physics(ctx: RunContext, log: Optional[Path] = None) -> int:
    """Unpauses the world (idempotent), a second time if the first try does not work. Raises
    when neither does; returns the try that worked."""
    for attempt in range(1, UNPAUSE_TRIES + 1):
        t0 = time.time()
        try:
            p = dexec('sim-gazebo', ['bash', '-c', unpause_cmd(ctx)], check=False, timeout=UNPAUSE_CALL_S + 40)
            rc, out = p.returncode, ((p.stdout or '') + (p.stderr or '')).strip()
        except SimError as e:            # the docker exec client hung
            rc, out = None, str(e)
        log_line(log, 'unpause %d: exit %s after %.1f s: %s' % (attempt, rc, time.time() - t0,
                                                                 ' | '.join(out.splitlines()[-3:])))
        if rc == 0:
            return attempt
        warn('unpause %d of %d failed (exit %s): %s' % (attempt, UNPAUSE_TRIES, rc, ''.join(out.splitlines()[-1:])))
    raise SimError('physics was not unpaused: gz world -p 0 did not take (%s); the world stays paused: '
                   'scripts/sim.py down, then up again' % (log or 'no log'))


def bring_up(session: Session, plan: Sequence[Tuple[Stack, List[Pane]]], ctx: RunContext, log_dir: Path,
             before_unpause: Optional[Callable[[], int]] = None) -> int:
    """Starts the stacks wave by wave into the paused world, then unpauses it once (when this
    `up` started gazebo). Wave 0 holds the spawn: it settles once every agent is in. A pane that
    ends before its node starts (a failed gate, a missing file), or a planner pane whose TF gate
    timed out, stops `up` after its wave, before the next wave and the unpause, so a broken
    bring-up never looks like a working one. before_unpause (the planner sessions' arming) runs
    after the last wave; its exit code is returned, and 130 (Ctrl-C) leaves the world paused."""
    log = log_dir / BRINGUP_LOG
    gazebo = any(s.name == 'gazebo' for s, _ in plan)
    waves = start_waves(plan)
    for k, wave in enumerate(waves):
        log_line(log, 'wave %d: starting %s' % (k, ', '.join(s.name for s, _ in wave)))
        for s, panes in wave:
            sess = start_stack(s, panes, log_dir)
            info('%-11s up: tmux attach -t %s   (scripts/sim.py attach %s)' % (s.name, sess, s.name))
        if any(s.name == 'gazebo' for s, _ in wave):
            waited = wait_spawned(log_dir)
            log_line(log, 'every agent spawned after %.1f s' % waited)
            info('every agent spawned into the paused world after %.0f s' % waited)
        failed = settle_wave(wave, len(session.agents), log_dir)
        if failed:
            later = [s.name for w in waves[k + 1:] for s, _ in w]
            what = '; '.join('%s %s' % kw for kw in failed)
            log_line(log, 'stopping: %s' % what)
            raise SimError(
                'pane(s) failed in wave %d: %s (pane logs in %s; scripts/sim.py attach %s). `up` stops here%s%s; '
                'every safety_node stays DISARMED. Fix it, then scripts/sim.py down and up again'
                % (k, what, log_dir, failed[0][0].split('/', 1)[0], ': the world stays paused' if gazebo else '',
                   ', and %s did not start' % ', '.join(later) if later else ''))
    missing = check_graph(session, [s.name for s, _ in plan])
    if missing:
        log_line(log, 'graph check: ' + '; '.join(missing))
        raise SimError(
            'graph check: %d edge(s) of the agents\' data path are missing from the graph:\n  - %s\n'
            'A declaration was lost while the stacks started (router.log has \'Unable to push\' then); those '
            'nodes would never get their input. `up` stops here%s; every safety_node stays DISARMED. '
            'scripts/sim.py down and up again' % (len(missing), '\n  - '.join(missing),
                                                  ': the world stays paused' if gazebo else ''))
    log_line(log, 'graph check: %d edge(s) in the graph' % len(graph_edges(session, [s.name for s, _ in plan])))
    rc = before_unpause() if before_unpause else 0
    if rc == 130:
        log_line(log, 'interrupted before the unpause: the world stays paused')
        return rc
    if gazebo:
        attempt = unpause_physics(ctx, log)
        log_line(log, 'physics unpaused')
        info('physics unpaused at %s%s; DLIO calibrates its IMU now' % (
            time.strftime(HOST_TIME_FMT), '' if attempt == 1 else ' (try %d)' % attempt))
    return rc


# What each agent's stacks need of each other, as (topic under /<ns>/, the subscribing node, the stacks
# both ends run in, the mapping mode it is for or None): the data path, end to end. A declaration
# the router lost while starting (PANE_START_GAP_S) leaves a node unsubscribed for good, and the
# agent never moves while everything looks up; check_graph sees it before the unpause.
GRAPH_EDGES = (
    ('livox/lidar', 'dlio_odom_node', ('gazebo', 'dlio'), None),
    ('dlio/odom_node/odom', 'convert_odom_to_state', ('dlio', 'mighty'), None),
    ('dlio/odom_node/pose', 'mpc', ('dlio', 'mighty'), None),
    ('dlio/odom_node/pose', 'global_mapper_ros', ('dlio', 'mapping'), None),
    ('dlio/odom_node/pointcloud/deskewed', 'global_mapper_ros', ('dlio', 'mapping'), None),
    ('dlio/odom_node/pointcloud/deskewed', 'elevation_mapping_node', ('dlio', 'elevation'), None),
    ('elev/occ_2d_topic', 'map_fusion_node', ('elevation', 'mapping'), 'gpu'),
    ('voxel/occ_2d_topic', 'map_fusion_node', ('mapping',), 'gpu'),
    ('persistent_map/grid', 'map_fusion_node', ('mapping',), 'gpu'),
    ('live/occ_2d_topic', 'persistent_map_node', ('mapping',), None),
    ('occ_2d_topic', 'mighty_node', ('mapping', 'mighty'), None),
    ('occ_2d_topic', 'goal_selector', ('mapping', 'mighty'), None),
    ('esdf_2d_topic', 'mighty_node', ('mapping', 'mighty'), None),
    ('persistent_map/grid', 'goal_selector', ('mapping', 'mighty'), None),
    ('state', 'mighty_node', ('mighty',), None),
    ('term_goal', 'mighty_node', ('mighty',), None),
    ('mpc_waypoints', 'mpc', ('mighty',), None),
    ('cmd_vel_auto', 'safety_node', ('mighty', 'drive'), None),
    ('term_goal_rviz', 'goal_selector', ('mighty', 'goals'), None),
)
GRAPH_CHECK_S = 30            # discovery: entries that are late come within seconds; lost ones never


def graph_edges(session: Session, stacks: Iterable[str]) -> List[Tuple[str, str]]:
    """(topic, subscriber FQN) of every GRAPH_EDGES edge the running stacks make, per agent. An agent
    on odometry: truth has truth_odom.py in DLIO's place: it reads the lidar, and publishes DLIO's
    topics under DLIO's names, so only that one subscriber differs."""
    names = set(stacks)
    out = []
    for a in session.agents:
        for topic, node, need, mode in GRAPH_EDGES:
            if set(need) <= names and mode in (None, session.mapping):
                if node == 'dlio_odom_node' and a.odometry == 'truth':
                    node = 'truth_odom_' + a.name               # truth_odom.node_name
                out.append(('/%s/%s' % (a.name, topic), '/%s/%s' % (a.name, node)))
    return out


# One rclpy node with no publisher or subscription of its own (only the graph cache): leaving costs
# nothing. Arguments <deadline s> then <topic>|<subscriber FQN>; it re-reads until every edge has its
# subscriber and a publisher, or the deadline: '@@ <topic> <subscriber> ok|nosub|nopub' each.
_GRAPH_PY = r"""
import sys, time
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
end = time.time() + float(sys.argv[1])
want = [tuple(a.split('|', 1)) for a in sys.argv[2:]]
rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
node = Node('rover_sim_graph_check', namespace='/sim', enable_rosout=False, start_parameter_services=False)

def state(topic, sub):
    subs = {(i.node_namespace.rstrip('/') + '/' + i.node_name) for i in node.get_subscriptions_info_by_topic(topic)}
    return 'nosub' if sub not in subs else 'ok' if node.get_publishers_info_by_topic(topic) else 'nopub'

seen = {}
while True:
    rclpy.spin_once(node, timeout_sec=0.5)
    seen = dict((w, state(*w)) for w in want)
    if all(v == 'ok' for v in seen.values()) or time.time() > end:
        break
for (topic, sub), v in seen.items():
    print('@@ %s %s %s' % (topic, sub, v), flush=True)
node.destroy_node()
time.sleep(0.5)
rclpy.try_shutdown()
"""


def check_graph(session: Session, stacks: Iterable[str]) -> List[str]:
    """The GRAPH_EDGES that are not in the graph ('<sub> is not subscribed to <topic>', or no
    publisher), from one graph-only rclpy process in sim-drive; [] if every edge is."""
    edges = graph_edges(session, stacks)
    if not edges:
        return []
    inner = ' && '.join([setup_snippet(CONTAINERS[QUERY_CONTAINER]),
                         'exec timeout -k 2 %d python3 -c "$SIM_GRAPH_PY" "$@"' % (GRAPH_CHECK_S + 10)])
    p = dexec(QUERY_CONTAINER, ['bash', '-c', inner, 'graph', str(GRAPH_CHECK_S)] + ['%s|%s' % e for e in edges],
              check=False, timeout=GRAPH_CHECK_S + 20, env=(('SIM_GRAPH_PY', _GRAPH_PY),))
    got = dict(((f[1], f[2]), f[3]) for f in (l.split() for l in (p.stdout or '').splitlines())
               if len(f) == 4 and f[0] == '@@')
    out = []
    for topic, sub in edges:
        v = got.get((topic, sub))
        if v != 'ok':
            out.append('%s is not subscribed to %s' % (sub, topic) if v == 'nosub' else
                       'no publisher of %s (for %s)' % (topic, sub) if v == 'nopub' else
                       '%s on %s: not checked (query exit %s)' % (sub, topic, p.returncode))
    return out


def load_up_session(arg: str) -> Session:
    """The session `up` runs: its file, with ROVER_SIM_AGENTS's agents when that is set."""
    session = load_session(arg)
    agents = os.environ.get(AGENTS_ENV, '').strip()
    return apply_agents_env(session, agents) if agents else session


START_WAIT_S = 60             # `start`: the restarted stack's panes to their node marker (no settle)


def cmd_up(args: argparse.Namespace) -> int:
    session = load_up_session(args.session)
    if args.rviz:
        session = dataclasses.replace(session, rviz=True)       # this run only; session.json says so
    if args.odometry:
        session = apply_odometry(session, *args.odometry)       # this run only; session.json says so
    stacks = select_stacks(session, args.stacks)
    run_dir = run_dir_from_env()
    ctx = RunContext(run_dir=run_dir, rust_log=os.environ.get('RUST_LOG_ZENOH', 'zenoh=info'),
                     gz_spare_cpus=gz_spare_from_env())
    plan = [(s, s.panes(session, ctx)) for s in stacks]
    if any(s.name == 'gazebo' for s in stacks):
        problem = world_problem(session.world)
        if problem:
            warn(problem + ': gzserver will fail to load it')
    arm = will_arm(session, stacks, args.no_arm)
    if args.dry_run:
        print_plan(session, plan, arm, ctx)
        return 0
    for tool in ('docker', 'tmux'):
        if shutil.which(tool) is None:
            raise SimError('%s not found on PATH' % tool)
    profiles = session_profiles(session)
    if profiles and not gpu_available():
        raise SimError('session %s has mapping: gpu but this host has no nvidia-smi; '
                       'use mapping: fallback' % session.name)
    images = compose_images(profiles, run_dir)
    # Every pane gates on the router, and write_agent_files reads from sim-drive.
    services = (sorted({c for s in stacks for c in s.containers} | {ROUTER_CONTAINER}) if args.partial
                else list(images))
    used = OrderedDict((k, v) for k, v in images.items() if k in services)
    check_images(used)
    pins = read_pins()
    # Compose would refuse these mounts.
    for problem in ((dlio_scripts_problem(pins) if any(c.startswith('sim-dlio-') for c in services) else None),
                    mapping_mounts_problem(pins, services)):
        if problem:
            raise SimError(problem)
    if any(s.name == 'mapping' for s in stacks):
        problem = mapping_config_problem(session, pins)
        if problem:
            raise SimError(problem)
    for line in image_provenance(used):
        warn(line)

    if tmux_sessions() or busy_containers(running_sim_containers()):
        info('stopping the previous run first')
        stop_run()
    prepare_run_dir(run_dir)
    sh(compose_cmd(profiles) + ['up', '-d'] + (services if args.partial else []),
       capture=False, env=compose_env(run_dir))
    wait_running(services)
    ensure_run_dir(run_dir)
    problems = check_isolation()
    if problems:
        raise SimError('REFUSING TO START NODES: the sim is not isolated from this host:\n  - '
                       + '\n  - '.join(problems) + '\n  (containers are idle; scripts/sim.py down --containers)')
    info('isolation OK: %s is internal and reaches nothing on this host' % NETWORK)
    if any(s.needs_display(session) for s in stacks):
        grant_x_access()
    write_session_file(session, run_dir)
    rotate_rtf_log(run_dir)
    clear_ros_logs(services)
    log_dir = make_log_dir()
    ctx = dataclasses.replace(ctx, log_dir=log_dir)
    for step in generate_steps(stacks):
        step(session, ctx)
    info('session %s: %s%s' % (session.name, ', '.join('%s (%s)' % (a.name, a.cls) for a in session.agents),
                               ' (%s=%s)' % (AGENTS_ENV, session.agents_env) if session.agents_env else ''))
    if any(s.name == 'mapping' for s in stacks):
        (warn if session.mapping == 'fallback' else info)('mapping: ' + mapping_label(session.mapping))
    if any(s.name == 'dlio' for s in stacks):
        (warn if session.truth_agents else info)('odometry: ' + odometry_label(session_odometry(session)))
    if any(s.name == 'rviz' for s in stacks) and len(session.agents) > 2:
        # 2026-10-20, multi_agent_swap with RViz: demo_eval's probe joining (29 subscriptions) stalled
        # delivery for 29 s, then 58 s, and every DLIO diverged; without RViz the same probe is harmless.
        warn('RViz is on with %d agents: run no probe or graded tool (demo_eval, status, parity_check, '
             'start) while it is: a session joining has stalled the whole graph for tens of seconds then'
             % len(session.agents))
    pending = pending_stacks(session)
    if pending:
        info('not started yet (later phases): ' + ', '.join(pending))
    info('pane logs: %s' % log_dir)
    info('run dir:   %s' % run_dir)
    # The planner sessions arm while paused: the arm query's leave stalls the planner's big graph
    # (STATUS_LEAVE_GAP_S), harmless while nothing flows. It is safe then: nothing commands motion
    # before DLIO tracks (mighty, goal_selector and goal_relay.py act on its estimate, mpc sends zeros
    # without a path), so the agents stay still through DLIO's IMU calibration as before.
    paused = arm and session.planner
    # Gate 'session': the arm records each agent's DLIO session, so a later re-arm in it is not
    # held to the seed height after DLIO has drifted (arm_z_check). wait_for_seed.py has made sure
    # the seed reached DLIO's container before DLIO started, which is what that check guards.
    rc = bring_up(session, plan, ctx, log_dir, before_unpause=(
        lambda: arm_after_up(session, log_dir, 'session', run_dir=run_dir)) if paused else None)
    if rc == 130:
        return rc
    if arm and not paused:
        rc = arm_after_up(session, log_dir, run_dir=run_dir)
    elif session.auto_arm and not args.no_arm and not arm:
        info('not arming: the dlio and drive stacks (and mighty, in a planner session) are not all started '
             '(scripts/sim.py arm)')
    elif not session.auto_arm:
        info('not arming: auto_arm is off; every safety_node stays DISARMED until scripts/sim.py arm')
    if not arm and any(s.name == 'goals' for s in stacks):
        info('goal_relay.py first drives each agent once it is in AUTO, then %s: scripts/sim.py arm'
             % ('starts the swap' if goal_mode(session) == 'swap' else 'relays RViz goals'))
    return rc


def rotate_pane_logs(stack_log: Path, panes: Sequence[Pane]) -> None:
    """A restarted stack's pane logs start empty (the old ones keep their time in the name), so
    its node markers are the new run's."""
    stamp = time.strftime('%Y%m%d_%H%M%S')
    for p in panes:
        f = stack_log / (p.slug + '.log')
        if f.is_file():
            f.rename(stack_log / ('%s.%s.log' % (p.slug, stamp)))


def cmd_start(args: argparse.Namespace) -> int:
    """(Re)start one last-wave stack alone on the running session: its old process is
    stopped, its files regenerated, its panes started in the run's pane-log dir. It joins (and the
    old one leaves) a running graph: the module docstring's stall caveat applies. Fails closed like
    status: with isolation broken it starts nothing (exit 3)."""
    stack = STACKS[args.stack]
    container = stack.containers[0]
    if container not in running_sim_containers():
        raise SimError('%s of project %s is not running (scripts/sim.py up <session>)' % (container, PROJECT))
    run_dir = resolve_run_dir()
    session = load_running_session(run_dir)
    if stack.name == 'rviz':
        session = dataclasses.replace(session, rviz=True)       # asked for: the session's flag aside
    if stack.name == 'goals' and not session.planner:
        raise SimError('session %s runs no planner (planner: false): nothing reads goals' % session.name)
    problems = check_isolation()
    if problems:
        print('isolation: BROKEN\n  - ' + '\n  - '.join(problems))
        print('not starting %s: no ROS session is opened while isolation is broken' % stack.name)
        return 3
    if stack.name == 'goals' and TMUX_PREFIX + 'mighty' not in tmux_sessions():
        warn('no %smighty session: goal_relay.py will have no goal_selector to talk to' % TMUX_PREFIX)
    warn('%s joins the running graph (and its old process leaves it): zenoh delivery may stall for a '
         'moment; do not do this in a window that is being graded' % stack.name)
    n = stop_processes(container, match=RESTARTABLE[stack.name], grace_term=3,
                       grace_int=max(5.0, goal_relay_leave_s(len(session.agents))) if stack.name == 'goals' else 5)
    if n:
        info('%s: stopped %d process(es) of the old %s' % (container, n, stack.name))
    latest = LOG_ROOT / 'latest'
    log_dir = latest.resolve() if latest.is_dir() else make_log_dir()
    ctx = RunContext(run_dir=run_dir, log_dir=log_dir, gz_spare_cpus=gz_spare_from_env())
    for step in stack.generate:
        step(session, ctx)
    if stack.needs_display(session):
        grant_x_access()
    panes = stack.panes(session, ctx)
    (log_dir / stack.name).mkdir(parents=True, exist_ok=True)
    rotate_pane_logs(log_dir / stack.name, panes)
    sess = start_stack(stack, panes, log_dir)
    info('%-11s up: tmux attach -t %s   (scripts/sim.py attach %s)' % (stack.name, sess, stack.name))
    waited, exited, missing = wait_nodes_started([(stack, panes)], log_dir, timeout=START_WAIT_S)
    if exited or missing:
        warn('%s: pane(s) %s %s (pane logs in %s)' % (stack.name, ', '.join(exited + missing),
                                                     'ended before starting their node' if exited
                                                     else 'still waiting after %.0f s' % waited, log_dir / stack.name))
        return 1
    info('%s: its node started after %.0f s' % (stack.name, waited))
    if stack.name == 'rviz' and TMUX_PREFIX + 'goals' not in tmux_sessions():
        info('RViz SetGoal publishes %s: scripts/sim.py start goals relays it to the agents' % GOAL_TOPIC)
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    """Start sim_viz.py on the running session in sim-viz (compose profile viz, so `up` never starts
    it), or stop it (--stop). A running viewer leaves first: one viewer at a time, and compose may
    recreate the container (a new image or $DISPLAY), which would cut a running leave short. Fails
    closed like start: with isolation broken it opens no ROS session (exit 3)."""
    if args.stop:
        if not stop_viz():
            info('no viewer was running')
        return 0
    if ROUTER_CONTAINER not in running_sim_containers():
        raise SimError('%s of project %s is not running (scripts/sim.py up <session>)' % (ROUTER_CONTAINER, PROJECT))
    run_dir = resolve_run_dir()
    names = session_agents(run_dir, args.agents or ())
    if len(names) > VIZ_MAX_PANELS:
        raise SimError('%d agents (%s): pick up to %d with --agents' % (len(names), ', '.join(names), VIZ_MAX_PANELS))
    bad = [l for l in args.layers or () if l not in VIZ_LAYERS + ('default', 'all')]
    if bad:
        raise SimError('unknown layer(s) %s (layers: %s, default, all)' % (', '.join(bad), ', '.join(VIZ_LAYERS)))
    if not (run_dir / SPAWN_FILE).is_file():
        raise SimError('no %s in %s: scripts/sim.py up <session>' % (SPAWN_FILE, run_dir))
    if args.snapshot:
        return viz_snapshot(args, run_dir)
    if not os.environ.get('DISPLAY'):
        raise SimError('no DISPLAY: the viewer opens a window on it (headless: scripts/sim.py viz --snapshot <png>)')
    stop_viz()
    sh(compose_cmd([VIZ_PROFILE]) + ['up', '-d', VIZ_CONTAINER], capture=False, env=compose_env(run_dir))
    wait_running([VIZ_CONTAINER])
    problems = check_isolation()
    if problems:
        print('isolation: BROKEN\n  - ' + '\n  - '.join(problems))
        print('not starting the viewer: no ROS session is opened while isolation is broken')
        return 3
    info('isolation OK: %s is internal and reaches nothing on this host' % NETWORK)
    grant_x_access()
    warn('the viewer joins the running graph (and leaves it on q / viz --stop, one subscription at a time): zenoh '
         'delivery may stall for a moment; do not do this in a window that is being graded')
    latest = LOG_ROOT / 'latest'
    log_dir = latest.resolve() if latest.is_dir() else make_log_dir()
    panes = viz_panes(args.agents or (), args.layers or (), args.follow)
    (log_dir / VIZ_STACK.name).mkdir(parents=True, exist_ok=True)
    rotate_pane_logs(log_dir / VIZ_STACK.name, panes)
    sess = start_stack(VIZ_STACK, panes, log_dir)
    info('%-11s up: tmux attach -t %s   (scripts/sim.py attach viz; scripts/sim.py viz --stop closes it)'
         % (VIZ_STACK.name, sess))
    waited, exited, missing = wait_nodes_started([(VIZ_STACK, panes)], log_dir, timeout=START_WAIT_S)
    if exited or missing:
        warn('viz: pane %s (log %s)' % ('ended before starting the viewer' if exited else
                                         'still waiting after %.0f s' % waited, log_dir / VIZ_STACK.name / 'viz.log'))
        return 1
    info('viz: the viewer started after %.0f s (log %s)' % (waited, log_dir / VIZ_STACK.name / 'viz.log'))
    return 0


def viz_snapshot(args: argparse.Namespace, run_dir: Path) -> int:
    """One frame of the viewer to args.snapshot on this host: in sim-viz (a running viewer's container
    is used as it is: a recreate would stop it), after the same isolation check, then docker cp."""
    if args.snapshot_after < 0:
        raise SimError('--snapshot-after must be >= 0')
    if VIZ_CONTAINER not in running_sim_containers():
        sh(compose_cmd([VIZ_PROFILE]) + ['up', '-d', VIZ_CONTAINER], capture=False, env=compose_env(run_dir))
        wait_running([VIZ_CONTAINER])
    problems = check_isolation()
    if problems:
        print('isolation: BROKEN\n  - ' + '\n  - '.join(problems))
        print('not taking the snapshot: no ROS session is opened while isolation is broken')
        return 3
    info('isolation OK: %s is internal and reaches nothing on this host' % NETWORK)
    warn('the snapshot joins the running graph for %.0f s and leaves it one subscription at a time'
         % args.snapshot_after)
    # The leave is at most VIZ_STOP_GRACE_S; past the timeout only this exec ends, the viewer leaves anyway.
    p = dexec(VIZ_CONTAINER, viz_snapshot_argv(args.agents or (), args.layers or (), args.snapshot_after, args.follow),
              check=False, timeout=args.snapshot_after + VIZ_STOP_GRACE_S + 30)
    sys.stdout.write(p.stdout or '')
    sys.stderr.write(p.stderr or '')
    if p.returncode != 0:
        raise SimError('sim_viz.py --snapshot exited %d' % p.returncode)
    out = Path(args.snapshot).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sh(['docker', 'cp', '%s:%s' % (sim_only(VIZ_CONTAINER), VIZ_SNAPSHOT_IN), str(out)])
    info('viz: wrote %s' % out)
    return 0


def arm_after_up(session: Session, log_dir: Path, gate: str = 'dlio', run_dir: Optional[Path] = None) -> int:
    """auto_arm: every agent's safety_node to AUTO once its DLIO is past the IMU calibration (gate
    dlio), or at once, recording each DLIO session (gate session: the planner sessions, before
    the unpause). Isolation is checked
    again first: no ROS session is opened on a broken network."""
    names = [a.name for a in session.agents]
    problems = check_isolation()
    if problems:
        raise SimError('NOT ARMING: the sim is not isolated from this host:\n  - ' + '\n  - '.join(problems))
    info('arming %s %s (auto_arm; Ctrl-C stops and disarms them; later: scripts/sim.py arm)' % (
        ', '.join(names), 'once each DLIO registers scans' if gate == 'dlio' else
        'now, before the unpause: nothing commands motion until DLIO tracks'))
    try:
        res = arm_agents(names, 'auto_mode', gate, ARM_TIMEOUT_UP_S, log_dir / ARM_LOG, run_dir=run_dir)
    except ArmInterrupted as e:
        report_interrupted(e)
        return 130
    except KeyboardInterrupt:       # a second Ctrl-C while arm_agents was cleaning up
        warn(ARM_STATE_UNKNOWN)
        return 130
    if report_arm(res, 'auto_mode'):
        return 0
    warn('the stacks are up, but not every agent is in AUTO: scripts/sim.py status, scripts/sim.py arm')
    return 1


def cmd_arm(args: argparse.Namespace) -> int:
    """arm: /<ns>/auto_mode once DLIO tracks (--now: at once); disarm: /<ns>/disarm at once.
    Fails closed like status: with isolation broken it opens no ROS session (exit 3)."""
    service = ARM_SERVICES[args.cmd]
    # The query's container must be one check_isolation inspects (the compose project's).
    if QUERY_CONTAINER not in running_sim_containers():
        raise SimError('%s of project %s is not running (scripts/sim.py up <session>)' % (QUERY_CONTAINER, PROJECT))
    run_dir = resolve_run_dir()
    names = session_agents(run_dir, args.agents)
    problems = check_isolation()
    if problems:
        print('isolation: BROKEN\n  - ' + '\n  - '.join(problems))
        print('not calling /<ns>/%s: no ROS session is opened while isolation is broken' % service)
        return 3
    if TMUX_PREFIX + 'drive' not in tmux_sessions():
        warn('no %sdrive session: is safety_node running?' % TMUX_PREFIX)
    latest = LOG_ROOT / 'latest'
    gate = 'dlio' if service == 'auto_mode' and not args.now else 'none'
    z_check = not getattr(args, 'no_z_check', False)
    if not z_check and gate == 'dlio':
        warn('--no-z-check: arming without the DLIO odom height check (logged in %s)' % ARM_LOG)
    try:
        res = arm_agents(names, service, gate, args.timeout, latest / ARM_LOG if latest.is_dir() else None,
                         run_dir=run_dir, z_check=z_check)
    except ArmInterrupted as e:
        report_interrupted(e)
        return 130
    except KeyboardInterrupt:
        warn(ARM_STATE_UNKNOWN)
        return 130
    return 0 if report_arm(res, service) else 1


def cmd_down(args: argparse.Namespace) -> int:
    stop_run(remove_containers=args.containers)
    info('down%s' % (' (containers removed)' if args.containers else '; containers left idle '
                     '(--containers removes them)'))
    return 0


def cmd_attach(args: argparse.Namespace) -> int:
    sessions = tmux_sessions()
    if args.stack is None:
        if len(sessions) != 1:
            info('sessions: %s' % (', '.join(sessions) or 'none (scripts/sim.py up <session>)'))
            return 0 if sessions else 1
        sess = sessions[0]
    else:
        sess = TMUX_PREFIX + args.stack
        if sess not in sessions:
            raise SimError('no tmux session %s (running: %s)' % (sess, ', '.join(sessions) or 'none'))
    if os.environ.get('TMUX'):
        os.execvp('tmux', ['tmux', 'switch-client', '-t', '=' + sess])
    os.execvp('tmux', ['tmux', 'attach', '-t', '=' + sess])
    return 0  # not reached


def pane_gate_note(stack: str, slug: str) -> str:
    """For a strict_tf_gate pane: its last TF gate's outcome, from this run's pane log (a pane
    restarted later may have proceeded degraded, which `up` cannot see)."""
    try:
        result = tf_gate_result((LOG_ROOT / 'latest' / stack / (slug + '.log')).read_text(errors='replace'))
    except OSError:
        return ''
    if result == 'degraded':
        return '  (TF gate TIMED OUT: the node started without <ns>/map -> <ns>/odom)'
    return '  (TF gate passed)' if result == 'ok' else ''


def cmd_status(args: argparse.Namespace) -> int:
    up = running_sim_containers()
    print('containers (project %s, network %s):' % (PROJECT, NETWORK))
    images = OrderedDict()  # type: Dict[str, str]
    for name, c in CONTAINERS.items():
        d = docker_inspect('container', name)
        state = d['State']['Status'] if d else 'absent'
        image = d['Config']['Image'] if d else '-'
        cpus = ((d or {}).get('HostConfig') or {}).get('CpusetCpus') or ('any' if d else '-')
        if d:
            images[name] = image
        print('  %-14s %-8s %-32s cpus %-6s %s' % (name, state, image, cpus, c.role))
    for line in image_provenance(images):
        print('  WARNING: ' + line)
    problems = check_isolation() if up else []
    if up:
        print('isolation: %s' % ('OK' if not problems else 'BROKEN\n  - ' + '\n  - '.join(problems)))
    if ROUTER_CONTAINER in up:
        links = link_snapshot(up)
        print('zenoh router %s:%d: %s; TCP links: %d to the router, %d peer-to-peer, %d listening, %d dialing'
              % (ROUTER_HOST, ROUTER_PORT, 'listening' if router_up() else 'DOWN', links['router'],
                 links['p2p'], links['listen'], links['syn_sent']))
    p = tmux('list-panes', '-a', '-F', '#{session_name}\t#{window_name}\t#{pane_index}\t#{pane_title}\t'
             '#{pane_current_command}\t#{%s}\t#{%s}' % (PANE_SLUG_OPTION, PANE_TF_GATE_OPTION), check=False)
    rows = [(l.split('\t') + ['', ''])[:7] for l in (p.stdout or '').splitlines() if l.startswith(TMUX_PREFIX)]
    print('tmux:%s' % ('' if rows else ' no rsim-* sessions'))
    for sess, win, idx, title, cmd, slug, gate in rows:
        # The pane's bash is back at its prompt once its docker exec has ended.
        print('  %s:%s.%s  %-9s %s%s' % (sess, win, idx, 'running' if cmd == 'docker' else 'EXITED', title,
                                         pane_gate_note(sess[len(TMUX_PREFIX):], slug) if gate == 'strict' else ''))
    run_dir = resolve_run_dir() if up else run_dir_from_env()
    sf = run_dir / SESSION_FILE
    s = json.loads(sf.read_text()) if sf.is_file() else None
    stacks = {r[0][len(TMUX_PREFIX):] for r in rows} & {'gazebo', 'dlio', 'drive', 'mapping', 'mighty'}
    if s:
        print('session: %s (%s), world %s, agents %s' % (
            s['name'], s.get('source', ''), s['world'], ', '.join(a['name'] for a in s['agents'])))
        print('mapping: %s' % mapping_label(s.get('mapping', 'gpu'), 'mapping' in stacks))
        print('odometry: %s' % odometry_label(json_odometry(s), 'dlio' in stacks))
    print('run dir: %s' % run_dir)
    if (LOG_ROOT / 'latest').exists():
        print('logs:    %s' % (LOG_ROOT / 'latest').resolve())
    maps = bool(getattr(args, 'maps', False)) and 'mapping' in stacks
    if problems:
        # The query opens a ROS session; `up` starts none on a broken network either.
        print('sim time: not queried (isolation BROKEN)')
    elif s and QUERY_CONTAINER in up and stacks:
        names = [a['name'] for a in s['agents']]
        vals = status_values(status_specs(names, stacks, maps))
        if 'gazebo' in stacks:
            print('sim time (each agent\'s livox_sim_bridge; time_offset = wall - sim):')
            for line in sim_time_lines(names, vals):
                print('  ' + line)
        else:
            print('sim time: n/a (the gazebo stack is not running)')
        if stacks & {'dlio', 'drive', 'mapping', 'mighty'}:
            fallback = s.get('mapping') == 'fallback' and 'mapping' in stacks
            truth = [n for n, m in json_odometry(s).items() if m == 'truth'] if 'dlio' in stacks else []
            print('agents (DLIO%s over %g s; safety_node mode%s%s%s%s%s):' % (
                ' and maps' if maps else '', RATE_WINDOW_S, '' if s.get('auto_arm', False) else '; auto_arm off',
                '; planner: mpc\'s cmd_vel_auto, mighty_node\'s map_hold' if 'mighty' in stacks else '',
                '; maps not read (status --maps)' if 'mapping' in stacks and not maps else '',
                '; maps from the fallback, NOT PARITY' if fallback else '',
                '; odometry from ground truth for %s, NOT PARITY' % ', '.join(truth) if truth else ''))
            for line in agent_lines(names, vals, stacks, maps, truth):
                print('  ' + line)
    else:
        print('sim time: n/a (the gazebo stack is not running)')
    bad = rtf_violations(run_dir)
    if bad:
        stalled = any(l.endswith('rtf=0.000') for l in bad)
        print(red_banner(['RUN INVALID: the real-time factor was below %.2f for over 5 s (%d line(s) in %s):'
                          % (RTF_MIN, len(bad), run_dir / RTF_LOG)] + ['  ' + l for l in bad[-5:]]
                         + ['Physics was slower than the wall clock the stack runs on: results from this run '
                            'do not count. Lighten the load (fewer agents, lidar_samples) and re-run.']
                         + (['rtf=0.000: no IMU reached the bridge at all, so gzserver stalled or was paused, or '
                             'the zenoh graph stalled (rsim-zenoh router.log: "Unable to push").'] if stalled else [])))
    return 3 if problems else 2 if bad else 0


# ---------------------------------------------------------------------------- ROS queries: status, arm

RTF_MIN = 0.95                         # the bridges log a violation below this for > 5 s
ECHO_TIMEOUT_S = 8                     # topics are 1 Hz; rclpy itself takes 1-2 s to start
RATE_WINDOW_S = 3.0                    # DLIO odom (100 Hz) and scan (10 Hz) rates are taken over this
ODOM_MAX_AGE_S = 1.0                   # an older DLIO odom stamp: no IMU is reaching DLIO
# The query leaves the graph one subscription at a time, then pauses before it closes (_STATUS_PY).
# On the loaded 4-agent gpu graph (rmw_zenoh 0.1.9) undeclaring its 20 subscriptions at once, by
# destroy_node or by exiting, stopped every delivery for 0.9-1.0 s (32 with --maps: 1.3-1.4 s);
# all destroyed at once, then a pause: 0.7 s. One every 0.1 s, with no rosout and no parameter
# services on its node (entities its close would otherwise undeclare in a burst): none, the
# clouds' worst inter-arrival 141-205 ms, as without a query (Phase 4 review, 4 runs of each).
# With the planner up the graph is several times bigger (mighty_node alone has ~145 entities, 83 of
# them frame_align subscriptions with ids up to 84), and each undeclaration then holds the router a
# full core for ~0.3 s: clouds 440-530 ms apart for one subscription or client, 820-900 ms for 4
# at 0.1 s, 1.9 s for 8, about 3 s for the arm query's 16 (DLIO diverged). Spread out they do not
# add up: 1 s apart, worst 461 ms for 4; 0.5 s apart (status, parity_check, demo_eval's recorder on
# the driving 4-agent swap), the IMU at most 0.47-0.54 s late, no re-anchor. Client-mode sessions
# cost the same (Phase 5, both measured).
STATUS_LEAVE_GAP_S = 0.5
STATUS_LEAVE_PAUSE_S = 0.5
# Both queries run in sim-drive, on the cores the non-Gazebo stacks share, not on the core of
# sim-gazebo where the sensor bridges run: DLIO odom is 100 Hz per agent.
QUERY_CONTAINER = ROUTER_CONTAINER
# One rclpy session reads every value, rather than a `ros2 topic echo` per topic: 8 CLI
# sessions starting at once on the bridges' one core joined so slowly that the router blocked
# on them (5 s per undeliverable declaration) and stalled the whole sim graph for 3 minutes.
# SIGKILL backs up the timeout: a session stuck in its close keeps the router waiting until
# its socket closes. The timeout covers the paced leave (STATUS_LEAVE_GAP_S): a query killed
# then closes all at once. Arguments are <kind>:<topic>, and each prints one line:
#   f64    std_msgs/Float64, first value                    '@@ <topic> <value>'
#   str    std_msgs/String, last value                      '@@ <topic> <value>'
#   odom   nav_msgs/Odometry: rate, last stamp's age, count  '@@ <topic> <Hz> <age s|inf> <n>'
#   count  sensor_msgs/PointCloud2 (not deserialised): rate  '@@ <topic> <Hz> <n>'
#   grid   nav_msgs/OccupancyGrid: as odom, its stamp read from the serialised header (a grid
#          is not deserialised: the persistent map grows to megabytes)
#   change a grid published only when it changed (persistent_map/grid): as grid, not waited for
#   twist  geometry_msgs/Twist: rate, count, last linear.x and angular.z  '@@ <topic> <Hz> <n> <v> <w>'
#   bool   std_msgs/Bool, last value                        '@@ <topic> true|false'
# or '@@ <topic>' if nothing came. It stops when every one-shot value is in and every rate
# but a `change` one (unless only those were asked for) has RATE_WINDOW_S of samples, or at
# the deadline. Subscriptions are best effort, so the publishers' KEEP_LAST queues never wait
# for this reader; safety_node's mode is TRANSIENT_LOCAL but also re-published at 2 Hz, and
# mighty_node's map_hold at 1 Hz, so no late-joiner delivery is needed.
_STATUS_PY = r"""
import struct, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float64, String
end, window = time.time() + float(sys.argv[1]), float(sys.argv[2])
specs = [a.split(':', 1) for a in sys.argv[3:]]
LEAVE_GAP_S, LEAVE_PAUSE_S = @LEAVE@
rclpy.init(args=[])
node = Node('sim_status_%d' % (time.time_ns() % 1000000), enable_rosout=False, start_parameter_services=False)
qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
TYPES = {'f64': Float64, 'str': String, 'odom': Odometry, 'count': PointCloud2, 'grid': OccupancyGrid,
         'change': OccupancyGrid, 'twist': Twist, 'bool': Bool}
RAW = ('count', 'grid', 'change')
ONCE = ('f64', 'str', 'bool')
times, last = {t: [] for _, t in specs}, {}
def raw_stamp(b):
    # Little-endian CDR: a 4-byte encapsulation header, then header.stamp (int32 sec, uint32 nsec).
    if len(b) < 12 or b[1:2] != b'\x01':
        return 0.0
    sec, nsec = struct.unpack_from('<iI', b, 4)
    return sec + nsec * 1e-9
def cb(kind, topic):
    def f(m):
        times[topic].append(time.time())
        if kind == 'f64':
            last.setdefault(topic, m.data)
        elif kind in ('str', 'bool'):
            last[topic] = m.data
        elif kind == 'odom':
            last[topic] = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        elif kind in ('grid', 'change'):
            last[topic] = raw_stamp(m)
        elif kind == 'twist':
            last[topic] = (m.linear.x, m.angular.z)
    return f
subs = [node.create_subscription(TYPES[k], t, cb(k, t), qos, raw=(k in RAW)) for k, t in specs]
waited = [(k, t) for k, t in specs if k != 'change'] or specs
def complete():
    return all((t in last) if k in ONCE else (times[t] and times[t][-1] - times[t][0] >= window)
               for k, t in waited)
while time.time() < end and not complete():
    rclpy.spin_once(node, timeout_sec=0.05)
now = time.time()
def rate(r):
    return '%.2f' % ((len(r) - 1) / (r[-1] - r[0])) if len(r) > 1 and r[-1] > r[0] else '0'
out = []
for k, t in specs:
    r = times[t]
    if not r:
        out.append('@@ %s' % t)
    elif k == 'bool':
        out.append('@@ %s %s' % (t, 'true' if last[t] else 'false'))
    elif k in ('f64', 'str'):
        out.append('@@ %s %s' % (t, last[t]))
    elif k in ('odom', 'grid', 'change'):
        out.append('@@ %s %s %.3f %d' % (t, rate(r), now - last[t] if last[t] else float('inf'), len(r)))
    elif k == 'twist':
        out.append('@@ %s %s %d %.3f %.3f' % ((t, rate(r), len(r)) + last[t]))
    else:
        out.append('@@ %s %s %d' % (t, rate(r), len(r)))
print('\n'.join(out), flush=True)
for s in subs:
    node.destroy_subscription(s)
    rclpy.spin_once(node, timeout_sec=0)
    time.sleep(LEAVE_GAP_S)
time.sleep(LEAVE_PAUSE_S)
node.destroy_node()
rclpy.try_shutdown()
""".replace('@LEAVE@', '%r, %r' % (STATUS_LEAVE_GAP_S, STATUS_LEAVE_PAUSE_S))
# $1: the timeout, then the specs.
_STATUS_SH = 'exec timeout -k 2 "$1" python3 -c "$SIM_STATUS_PY" %d %g "${@:2}"' % (ECHO_TIMEOUT_S, RATE_WINDOW_S)


def status_budget(n_specs: int) -> int:
    """Seconds the query may take: rclpy's start, its deadline, then its paced leave."""
    return int(math.ceil(ECHO_TIMEOUT_S + 4 + n_specs * STATUS_LEAVE_GAP_S + STATUS_LEAVE_PAUSE_S))


def odom_topic(n: str) -> str:
    return '/%s/dlio/odom_node/odom' % n


def scans_topic(n: str) -> str:
    return '/%s/dlio/odom_node/pointcloud/deskewed' % n


# What the planner reads from the mapping stack (map_fusion, or the fallback's grid_relay.py),
# how old a grid may be (mighty's map-age hold, 1.5 s) and the query kind. persistent_map/grid
# has no limit and is not waited for: it is latched and re-published (every 2 s) only when the
# map changed, so on a parked agent its age just grows, and none may come within the query.
# Read only with `status --maps`: 3 more subscriptions per agent, one of them megabytes (before
# the paced leave, STATUS_LEAVE_GAP_S, that query's end stalled the 4-agent graph for 1.3 s).
MAP_GRIDS = (('occ_2d_topic', 1.5, 'grid'), ('esdf_2d_topic', 1.5, 'grid'), ('persistent_map/grid', None, 'change'))


def grid_topic(n: str, grid: str) -> str:
    return '/%s/%s' % (n, grid)


# The planner's liveness, in two subscriptions per agent and types sim-drive has: mpc's command
# (30 Hz from its start, zero until a path) and mighty_node's map-age hold (1 Hz from its own
# timer, goal or not). Their goal topics are one-shot (term_goal, at a commit) or mighty's own
# message types, which the query container does not have.
def cmd_vel_auto_topic(n: str) -> str:
    return '/%s/cmd_vel_auto' % n


def map_hold_topic(n: str) -> str:
    return '/%s/map_hold' % n


def status_specs(names: Sequence[str], stacks: Iterable[str], maps: bool = False) -> List[str]:
    """What `status` reads for the stacks that run: the bridges' sim/rtf and
    sim/time_offset, DLIO's odom and deskewed scans, safety_node's mode, the planner's
    liveness, and with maps the planner's grids."""
    stacks = set(stacks)
    out = []  # type: List[str]
    for n in names:
        if 'gazebo' in stacks:
            out += ['f64:/%s/sim/rtf' % n, 'f64:/%s/sim/time_offset' % n]
        if 'dlio' in stacks:
            out += ['odom:' + odom_topic(n), 'count:' + scans_topic(n)]
        if 'drive' in stacks:
            out.append('str:/%s/mode' % n)
        if 'mighty' in stacks:
            out += ['twist:' + cmd_vel_auto_topic(n), 'bool:' + map_hold_topic(n)]
        if maps and 'mapping' in stacks:
            out += ['%s:%s' % (kind, grid_topic(n, g)) for g, _, kind in MAP_GRIDS]
    return out


def parse_status(out: str) -> Dict[str, List[str]]:
    """'@@ <topic> <fields...>' lines -> {topic: fields} ([] when nothing came)."""
    vals = {}  # type: Dict[str, List[str]]
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == '@@':
            vals[parts[1]] = parts[2:]
    return vals


def _num(vals: Dict[str, List[str]], topic: str, i: int = 0) -> Optional[float]:
    try:
        return float(vals[topic][i])
    except (KeyError, IndexError, ValueError):
        return None


def status_values(specs: Sequence[str]) -> Dict[str, List[str]]:
    """One sample of every spec (status_specs), read by one rclpy process in sim-drive."""
    inner = ' && '.join([setup_snippet(CONTAINERS[QUERY_CONTAINER]), _STATUS_SH])
    budget = status_budget(len(specs))
    try:
        p = dexec(QUERY_CONTAINER, ['bash', '-c', inner, 'status', str(budget)] + list(specs), check=False,
                  timeout=budget + 16, env=(('SIM_STATUS_PY', _STATUS_PY),))
    except SimError as e:           # a hung query must not hide the violation check after it
        warn('status query: %s' % e)
        return {}
    return parse_status(p.stdout or '')


def sim_time_lines(names: Sequence[str], vals: Dict[str, List[str]]) -> List[str]:
    out = []
    for n in names:
        rtf, off = _num(vals, '/%s/sim/rtf' % n), _num(vals, '/%s/sim/time_offset' % n)
        if rtf is None and off is None:
            out.append('%s  no sim/rtf or sim/time_offset in %d s (is livox_sim_bridge_%s running? '
                       'scripts/sim.py attach gazebo)' % (n, ECHO_TIMEOUT_S, n))
            continue
        out.append('%s  rtf %s%s  time_offset %s' % (
            n, '-' if rtf is None else '%.3f' % rtf, '  LOW' if rtf is not None and rtf < RTF_MIN else '',
            '-' if off is None else '%.3f s' % off))
    return out


def agent_lines(names: Sequence[str], vals: Dict[str, List[str]], stacks: Iterable[str],
                maps: bool = False, truth: Iterable[str] = ()) -> List[str]:
    """Per agent: DLIO's odom rate and stamp age, its scan rate, safety_node's mode, with maps
    the planner's grids (map_part). DLIO publishes odom from a 100 Hz timer from the start, stamped
    with the last IMU sample's time (0 before the first one); deskewed scans only come once its
    IMU calibration is done and it registers scans. The agents in truth (odometry: truth, NOT
    PARITY) have truth_odom.py on the same topics, which publishes only once ground truth flows."""
    stacks, truth = set(stacks), set(truth)
    out = []
    for n in names:
        parts = []
        if 'dlio' in stacks and n in truth:
            hz, age = _num(vals, odom_topic(n)), _num(vals, odom_topic(n), 1)
            if hz is None:
                parts.append('truth odom (NOT PARITY): none yet (ground truth flows only while physics runs; '
                             'scripts/sim.py attach dlio)')
            else:
                parts.append('truth odom (NOT PARITY) %.0f Hz, stamp age %s' % (
                    hz, 'n/a' if age is None or math.isinf(age) else '%.3f s%s' % (
                        age, '  STALE' if age > ODOM_MAX_AGE_S else '')))
            scans = _num(vals, scans_topic(n))
            parts.append('scans %.1f Hz' % scans if scans else 'no scans (IMU calibration time, or the lidar)')
        elif 'dlio' in stacks:
            hz, age = _num(vals, odom_topic(n)), _num(vals, odom_topic(n), 1)
            if hz is None:
                parts.append('DLIO odom: none in %d s (scripts/sim.py attach dlio)' % ECHO_TIMEOUT_S)
            elif age is None or math.isinf(age):
                parts.append('DLIO odom %.0f Hz, unstamped: no IMU has reached DLIO' % hz)
            else:
                parts.append('DLIO odom %.0f Hz, stamp age %.3f s%s' % (hz, age, '  STALE' if age > ODOM_MAX_AGE_S else ''))
            scans = _num(vals, scans_topic(n))
            parts.append('scans %.1f Hz' % scans if scans else 'no scans (IMU calibration, seed pose or lidar)')
        if 'drive' in stacks:
            mode = (vals.get('/%s/mode' % n) or [None])[0]
            parts.append('safety %s' % mode if mode else 'safety: no /%s/mode (scripts/sim.py attach drive)' % n)
        if 'mighty' in stacks:
            parts.append(planner_part(n, vals))
        if maps and 'mapping' in stacks:
            parts.append(map_part(n, vals))
        out.append('%s  %s' % (n, ',  '.join(parts)))
    return out


def planner_part(n: str, vals: Dict[str, List[str]]) -> str:
    """mpc's cmd_vel_auto (rate, last command) and mighty_node's map_hold of one agent."""
    t = cmd_vel_auto_topic(n)
    hz, v, w = _num(vals, t), _num(vals, t, 2), _num(vals, t, 3)
    if hz is None:
        cmd = 'cmd_vel_auto: none in %d s (mpc running? scripts/sim.py attach mighty)' % ECHO_TIMEOUT_S
    else:
        cmd = 'cmd_vel_auto %.0f Hz (v %s m/s, w %s rad/s)' % (hz, '-' if v is None else '%.2f' % v,
                                                              '-' if w is None else '%.2f' % w)
    hold = (vals.get(map_hold_topic(n)) or [None])[0]
    if hold is None:
        held = 'no map_hold in %d s (mighty_node running? scripts/sim.py attach mighty)' % ECHO_TIMEOUT_S
    elif hold == 'true':
        held = 'map_hold ON: mighty holds the agent, its maps are stale'
    else:
        held = 'map_hold off'
    return 'planner: %s, %s' % (cmd, held)


def map_part(n: str, vals: Dict[str, List[str]]) -> str:
    """The planner's grids of one agent: rate and stamp age each, STALE past the age a consumer
    allows. Nothing before physics runs and DLIO registers scans."""
    grids = []
    for g, max_age, _ in MAP_GRIDS:
        topic = grid_topic(n, g)
        hz, age, count = _num(vals, topic), _num(vals, topic, 1), _num(vals, topic, 2)
        when = '' if max_age is not None else ', on change'
        rate = 'once' if count == 1 else '%.1f Hz' % (hz or 0.0)       # one sample has no rate
        if hz is None:
            grids.append('%s none%s' % (g, ' (published on change)' if when else ''))
        elif age is None or math.isinf(age):
            grids.append('%s %s, unstamped' % (g, rate))
        else:
            grids.append('%s %s (age %.2f s%s)' % (g, rate, age, when or (', STALE' if age > max_age else '')))
    return 'maps: ' + ', '.join(grids)


def dlio_session_id(data: str) -> Optional[str]:
    """The id in a /<ns>/dlio/session payload (dlio_ws scripts/dlio_session.py: JSON with an
    "id" that changes exactly when DLIO restarts), or None for anything else. One word: the
    arm query prints it in a line of words."""
    try:
        sid = json.loads(data).get('id')
    except (TypeError, ValueError, AttributeError):
        return None
    return sid if isinstance(sid, str) and sid.split() == [sid] else None


def arm_z_check(n: str, z: float, height: float, z_tol: float, session: Optional[str], armed: Optional[str],
                check: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """(blocker, warning) for agent n's DLIO odom z at an arm. The check catches a seed that
    came too late (odom anchored at z 0), which the first arm of a DLIO session sees; later in
    a session z drifts legitimately (0.51 against 0.36 after a long run), so an agent armed
    before in the same session (armed: the id recorded then, session: the one it publishes
    now) only gets a warning. A new session (DLIO restarted), or none seen, is checked again.
    The arm query runs this source too (_ARM_PY)."""
    off = z - height
    if not check or abs(off) <= z_tol:
        return None, None
    # The late-seed signature first: odom anchored at 0 blocks in any session, also one a paused arm
    # recorded (gate session checks no z), so that record cannot excuse it later.
    if abs(z) <= z_tol:
        return ('DLIO odom z %.2f, expected %.2f: the seed pose came too late; restart its dlio pane '
                '(scripts/sim.py attach dlio: Ctrl-C, Up, Enter)' % (z, height)), None
    if session is not None and session == armed:
        return None, 'odometry z drifted %+.2f m since its DLIO session started; not blocking' % off
    why = ('no /%s/dlio/session seen' % n if session is None else
           'first arm in this DLIO session' if armed is None else 'DLIO restarted since its last arm')
    return ('DLIO odom z %.2f, expected %.2f (offset %+.2f; %s): restart its dlio pane (resets that agent\'s '
            'maps) or scripts/sim.py arm --no-z-check %s' % (z, height, off, why, n)), None


# Arming. One rclpy process for every agent, in sim-drive next to the safety_nodes. For
# auto_mode it waits, per agent, until DLIO is past its IMU calibration (fresh odom stamps
# and at least one deskewed scan: DLIO registers scans only once calibrated), its odom sits
# at the seed height on the agent's first arm in its DLIO session (DLIO registers scans
# seeded or not, odom.cc:913-916, and a seed that came too late leaves odom at z 0;
# arm_z_check) and safety_node reports its mode; an agent already in AUTO is left alone.
# Then it calls the Trigger and waits for safety_node to report the new mode. disarm calls
# at once, and checks no z. A call with no answer is repeated: both services are idempotent.
# Prints '[arm] ...' progress, '@@ <ns> session <id>' with the DLIO session an agent was
# called in, and one '@@ <ns> ok|refused|timeout mode=<mode> <message>' per agent. The marker
# in the first line lets `stop_processes(match=...)` find it: a killed docker exec client
# leaves its process running in the container. It leaves the graph as the status query does
# (no rosout, no parameter services, one entity at a time; up to 20 subscriptions and clients
# with 4 agents), since it runs on a loaded graph too.
ARM_MARKER = 'rover_sim arm query'
_ARM_PY = r"""# rover_sim arm query
import json, signal, sys, time
from typing import Optional, Tuple
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
service, gate, max_age, z_tol = sys.argv[1], sys.argv[2], float(sys.argv[4]), float(sys.argv[5])
z_check = sys.argv[6] == '1'
start = time.time()
end = start + float(sys.argv[3])
agents = [a.split('=', 2) for a in sys.argv[7:]]           # <ns>=<seed height>[=<armed in session>]
height = dict((a[0], float(a[1])) for a in agents)
armed = dict((a[0], a[2]) for a in agents if len(a) > 2)
names = list(height)
WANT = {'auto_mode': 'AUTO', 'disarm': 'DISARMED'}[service]
CALL_S, CONFIRM_S, REPORT_S, SESSION_WAIT_S, SESSION_WAIT_UP_S = 5.0, 2.0, 15.0, 3.0, 15.0
@Z_CHECK@
LEAVE_GAP_S, LEAVE_PAUSE_S = @LEAVE@
# Ctrl-C / SIGTERM end the wait, not the process: the leave below is paced either way.
rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
stop = []
signal.signal(signal.SIGINT, lambda *_: stop.append(1))
signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
node = Node('sim_arm_%d' % (time.time_ns() % 1000000), enable_rosout=False, start_parameter_services=False)
qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
st, subs = {}, []
def setter(s, key, fn):
    def cb(m):
        s[key] = fn(s, m)
    return cb
for n in names:
    s = st[n] = {'mode': None, 'odom': None, 'scans': 0, 'call': None, 'sent': 0.0, 'answer': None, 'why': 'starting',
                 'session': None, 'warn': None, 'warned': False}
    subs.append(node.create_subscription(String, '/%s/mode' % n, setter(s, 'mode', lambda s, m: m.data), qos))
    if gate == 'dlio':
        subs.append(node.create_subscription(Odometry, '/%s/dlio/odom_node/odom' % n, setter(
            s, 'odom', lambda s, m: (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.pose.pose.position.z)), qos))
        subs.append(node.create_subscription(PointCloud2, '/%s/dlio/odom_node/pointcloud/deskewed' % n, setter(
            s, 'scans', lambda s, m: s['scans'] + 1), qos, raw=True))
    if gate in ('dlio', 'session'):
        # 1 Hz, reliable (best effort matches): tells a re-arm from the first arm of a DLIO session.
        subs.append(node.create_subscription(String, '/%s/dlio/session' % n, setter(
            s, 'session', lambda s, m: dlio_session_id(m.data) or s['session']), qos))
    s['client'] = node.create_client(Trigger, '/%s/%s' % (n, service))
def blocker(n, s, now):
    if gate == 'dlio':
        if s['odom'] is None:
            return 'no /%s/dlio/odom_node/odom yet (DLIO not running?)' % n
        stamp, z = s['odom']
        if not stamp:
            return 'DLIO odom unstamped: no IMU has reached DLIO yet'
        if now - stamp > max_age:
            return 'DLIO odom stamp %.1f s old: no IMU is reaching DLIO' % (now - stamp)
        if not s['scans']:
            return 'DLIO registers no scans yet (IMU calibration, seed pose or lidar)'
        why, s['warn'] = arm_z_check(n, z, height[n], z_tol, s['session'], armed.get(n), z_check)
        if why:
            return why
        if s['session'] is None and now - start < SESSION_WAIT_S:
            return 'no /%s/dlio/session yet (the arm records it)' % n
    if gate == 'session' and s['session'] is None and now - start < SESSION_WAIT_UP_S:
        return 'no /%s/dlio/session yet (the arm records it)' % n
    if service == 'auto_mode' and s['mode'] is None:
        return 'no /%s/mode yet (safety_node not running?)' % n
    if not s['client'].service_is_ready():
        return '/%s/%s is not available yet' % (n, service)
    return None
pending, last_report = list(names), time.time()
print('[arm] %s: /<ns>/%s%s, for up to %.0f s' % (' '.join(names), service,
      ' once DLIO registers scans' if gate == 'dlio' else '', end - time.time()), flush=True)
while pending and time.time() < end and not stop:
    rclpy.spin_once(node, timeout_sec=0.05)
    now = time.time()
    for n in list(pending):
        s = st[n]
        if s['answer'] is not None:
            ok, msg, t = s['answer']
            if s['mode'] == WANT or now - t > CONFIRM_S:
                print('@@ %s %s mode=%s %s' % (n, 'ok' if ok else 'refused', s['mode'], msg), flush=True)
                pending.remove(n)
        elif s['call'] is not None:
            if s['call'].done():
                r = s['call'].result()
                s['answer'], s['mode'] = (r.success, r.message, now), None
            elif now - s['sent'] > CALL_S:
                s['call'] = None
        elif service == 'auto_mode' and s['mode'] == 'AUTO':
            print('@@ %s ok mode=AUTO already in AUTO: not called' % n, flush=True)
            pending.remove(n)
        else:
            s['why'] = blocker(n, s, now)
            if s['why'] is None:
                if s['warn'] and not s['warned']:
                    s['warned'] = True
                    print('[arm] %s: WARNING: %s' % (n, s['warn']), flush=True)
                if s['session']:
                    print('@@ %s session %s' % (n, s['session']), flush=True)
                s['call'], s['sent'] = s['client'].call_async(Trigger.Request()), now
    if pending and now - last_report >= REPORT_S:
        last_report = now
        for n in pending:
            print('[arm] %s: waiting: %s' % (n, st[n]['why'] or 'for the answer'), flush=True)
for n in pending:
    print('@@ %s timeout mode=%s %s' % (n, st[n]['mode'], 'interrupted' if stop else st[n]['why'] or 'no answer'),
          flush=True)
for e in subs + [st[n]['client'] for n in names]:
    (node.destroy_subscription if e in subs else node.destroy_client)(e)
    rclpy.spin_once(node, timeout_sec=0)
    time.sleep(LEAVE_GAP_S)
time.sleep(LEAVE_PAUSE_S)
node.destroy_node()
rclpy.try_shutdown()
""".replace('@LEAVE@', '%r, %r' % (STATUS_LEAVE_GAP_S, STATUS_LEAVE_PAUSE_S)).replace(
    '@Z_CHECK@', '\n'.join(inspect.getsource(f) for f in (dlio_session_id, arm_z_check)))
ARM_SERVICES = OrderedDict([('arm', 'auto_mode'), ('disarm', 'disarm')])   # CLI verb -> Trigger
ARM_ENV = (('SIM_ARM_PY', _ARM_PY), ('RUST_LOG', 'zenoh=error'))
ARM_ENTITIES_PER_AGENT = 5   # the arm query's mode, odom, scan and session subscriptions and its client
ARM_TIMEOUT_UP_S = 180       # up's wait: gzserver, spawns, DLIO's start and its 3 s IMU calibration
ARM_TIMEOUT_S = 60           # `sim.py arm`'s
ARM_UNDO_S = 30              # the disarm after a Ctrl-C (ArmInterrupted)
# Seeded odom sits at the lidar height (measured 0.359-0.361 JR, 0.398-0.403 RR); a missed seed
# leaves it at 0. The worlds are flat, but DLIO drifts in z over a long run (+0.15 m seen), so
# the check binds only on an agent's first arm in a DLIO session (arm_z_check).
ARM_Z_TOL_M = 0.1


class ArmResult(NamedTuple):
    name: str
    status: str              # ok | refused | timeout | lost
    mode: str                # safety_node's mode afterwards, as it reported it
    message: str
    session: str = ''        # the DLIO session id it was called in ('' if none was seen)


ALREADY_AUTO = 'already in AUTO: not called'
ARM_STATE_UNKNOWN = 'interrupted: safety_node modes unknown (scripts/sim.py status; scripts/sim.py disarm)'


class ArmInterrupted(KeyboardInterrupt):
    """Ctrl-C during an arm query (arm_agents): what it had reported, and for auto_mode the
    disarm that undid it (None when that was interrupted too)."""

    def __init__(self, service: str, names: Sequence[str], seen: Dict[str, ArmResult],
                 undo: Optional[Dict[str, ArmResult]]):
        super().__init__()
        self.service, self.names, self.seen, self.undo = service, list(names), seen, undo


def parse_arm(out: str) -> Dict[str, ArmResult]:
    res, sessions = OrderedDict(), {}  # type: Dict[str, ArmResult], Dict[str, str]
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 4 and parts[0] == '@@' and parts[2] == 'session':
            sessions[parts[1]] = parts[3]
        elif len(parts) >= 4 and parts[0] == '@@' and parts[3].startswith('mode='):
            res[parts[1]] = ArmResult(parts[1], parts[2], parts[3][len('mode='):], parts[4] if len(parts) > 4 else '')
    return OrderedDict((n, r._replace(session=sessions.get(n, ''))) for n, r in res.items())


def armed_sessions_after(armed: Dict[str, str], res: Dict[str, ArmResult]) -> Dict[str, str]:
    """The record after an arm query: each agent it switched to AUTO, with the DLIO session it saw
    then (z checked, or waived with --no-z-check). An agent found in AUTO was not checked here."""
    out = dict(armed)
    for r in res.values():
        if r.status == 'ok' and r.mode == 'AUTO' and r.message != ALREADY_AUTO and r.session:
            out[r.name] = r.session
    return out


def load_armed_sessions(run_dir: Path) -> Dict[str, str]:
    """<run>/armed_sessions.json. Missing or unreadable is no record: every arm is a first arm."""
    path = run_dir / ARMED_SESSIONS_FILE
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        warn('ignoring %s (%s): every agent gets the first-arm z check' % (path, e))
        return {}
    if not isinstance(data, dict):
        warn('ignoring %s (not an object): every agent gets the first-arm z check' % path)
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def save_armed_sessions(run_dir: Path, armed: Dict[str, str]) -> None:
    try:
        write_file(run_dir / ARMED_SESSIONS_FILE, (json.dumps(armed, indent=1, sort_keys=True) + '\n').encode())
    except (OSError, SimError) as e:
        warn('cannot record the armed DLIO sessions in %s: %s (a re-arm checks z again)' % (run_dir, e))


def lidar_height_of(name: str) -> float:
    """The seed height of an agent, from its class prefix (validate_session checked it)."""
    return next(ROBOT_CLASSES[c].lidar_height for c, p in CLASS_PREFIXES.items() if name.startswith(p))


def arm_budget(timeout: float, n_agents: int) -> int:
    """Seconds the arm query may take in its container: rclpy's start, its own timeout, then its
    paced leave of up to 5 entities per agent (4 subscriptions and a client)."""
    return int(math.ceil(timeout + 6 + 5 * n_agents * STATUS_LEAVE_GAP_S + STATUS_LEAVE_PAUSE_S))


def arm_args(service: str, gate: str, timeout: float, names: Sequence[str],
             armed: Optional[Dict[str, str]] = None, z_check: bool = True) -> List[str]:
    """docker exec arguments of the arm query in sim-drive (the script itself goes in $SIM_ARM_PY):
    each agent as <ns>=<seed height>, and =<session id> when it was armed before (armed)."""
    armed = armed or {}
    run = 'exec timeout -k 2 %d python3 -c "$SIM_ARM_PY" "$@"' % arm_budget(timeout, len(names))
    return (['bash', '-c', ' && '.join([setup_snippet(CONTAINERS[QUERY_CONTAINER]), run]), 'arm', service, gate,
             '%g' % timeout, '%g' % ODOM_MAX_AGE_S, '%g' % ARM_Z_TOL_M, '1' if z_check else '0']
            + ['%s=%s%s' % (n, fmt_float(lidar_height_of(n)), '=' + armed[n] if n in armed else '') for n in names])


def dexec_lines(name: str, args: Sequence[str], timeout: float, env: Sequence[Tuple[str, str]],
                on_line: Callable[[str], None]) -> int:
    """docker exec, handing each output line (stdout and stderr) to on_line as it comes;
    the client is killed after timeout. Returns the exit code."""
    cmd = ['docker', 'exec']
    for k, v in env:
        cmd += ['-e', '%s=%s' % (k, v)]
    with subprocess.Popen(cmd + [sim_only(name)] + list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True) as p:
        timer = threading.Timer(timeout, p.kill)
        timer.start()
        try:
            for line in p.stdout:
                on_line(line.rstrip('\n'))
            return p.wait()
        finally:
            timer.cancel()
            if p.poll() is None:
                p.kill()


def log_line(path: Optional[Path], text: str) -> None:
    if path is None:
        return
    try:
        with open(str(path), 'a') as f:
            f.write('%s %s\n' % (time.strftime(HOST_TIME_FMT), text))
    except OSError as e:
        warn('cannot log to %s: %s' % (path, e))


def arm_agents(names: Sequence[str], service: str, gate: str, timeout: float,
               log: Optional[Path] = None, run_dir: Optional[Path] = None,
               z_check: bool = True) -> Dict[str, ArmResult]:
    """Calls /<ns>/<service> (std_srvs/Trigger) for each agent through the arm query; with
    gate 'dlio', each once its DLIO is past the IMU calibration (and, on its first arm in a DLIO
    session, at the seed height unless z_check is off: arm_z_check). With run_dir it reads and
    updates <run>/armed_sessions.json. The caller has checked
    isolation. Every line goes to log. Ctrl-C stops the query inside the container too and
    raises ArmInterrupted. An auto_mode query fails closed: it may have armed agents by then,
    including some whose answer never reached this client (Ctrl-C ends the docker exec client
    at once, the query only when stop_processes gets to it), so every agent it did not find
    in AUTO already is disarmed again."""
    lines = []  # type: List[str]
    checked = service == 'auto_mode' and gate in ('dlio', 'session')    # records the DLIO session armed in
    armed = load_armed_sessions(run_dir) if checked and run_dir is not None else {}

    def on_line(line: str) -> None:
        lines.append(line)
        log_line(log, line)
        if line.startswith('[arm] '):
            ns, sep, rest = line[len('[arm] '):].partition(': WARNING: ')
            warn(ns + ': ' + rest) if sep else info(line[len('[arm] '):])
        elif not line.startswith('@@ '):
            info('  ' + line)

    log_line(log, 'arm query: %s %s, gate %s, timeout %g s%s' % (
        service, ' '.join(names), gate, timeout, '' if z_check or not checked else ', NO z check (--no-z-check)'))
    try:
        # Its stderr is shown (a traceback matters); zenoh's own start-up warnings are not.
        rc = dexec_lines(QUERY_CONTAINER, arm_args(service, gate, timeout, names, armed, z_check),
                         arm_budget(timeout, len(names)) + 14, ARM_ENV, on_line)
    except KeyboardInterrupt:
        # Ctrl-C reached the query too (the pane's process group): it ends its wait and leaves paced.
        stop_processes(QUERY_CONTAINER, match=ARM_MARKER, grace_term=2,
                       grace_int=ARM_ENTITIES_PER_AGENT * len(names) * STATUS_LEAVE_GAP_S + STATUS_LEAVE_PAUSE_S + 2)
        log_line(log, 'interrupted: the arm query was stopped')
        seen = parse_arm('\n'.join(lines))
        undo = None  # type: Optional[Dict[str, ArmResult]]
        if service == 'auto_mode':
            back = [n for n in names if n not in seen or seen[n].message != ALREADY_AUTO]
            undo = OrderedDict()
            if back:
                info('interrupted: disarming %s again (the query may have armed them; Ctrl-C again to skip)'
                     % ', '.join(back))
                try:
                    undo = arm_agents(back, 'disarm', 'none', ARM_UNDO_S, log)
                except ArmInterrupted:
                    undo = None
        raise ArmInterrupted(service, names, seen, undo)
    got = parse_arm('\n'.join(lines))
    res = OrderedDict((n, got.get(n) or ArmResult(n, 'lost', '?', 'no result from the arm query (exit %d)' % rc))
                      for n in names)
    for r in res.values():
        log_line(log, '%s: %s, mode %s: %s' % (r.name, r.status, r.mode, r.message))
    if checked and run_dir is not None:
        known = load_armed_sessions(run_dir)       # again: another arm may have recorded meanwhile
        after = armed_sessions_after(known, res)
        if after != known:
            save_armed_sessions(run_dir, after)
    return res


def report_arm(res: Dict[str, ArmResult], service: str) -> bool:
    """Prints one line per agent; True when every call took (or was not needed)."""
    want = {'auto_mode': 'AUTO', 'disarm': 'DISARMED'}[service]
    good = True
    for r in res.values():
        ok = r.status == 'ok' and r.mode == want
        good = good and ok
        (info if ok else warn)('%s %s: %s (safety_node mode %s)%s' % (
            service, r.name, r.status, r.mode, ': ' + r.message if r.message else ''))
    return good


def report_interrupted(e: ArmInterrupted) -> None:
    """What a Ctrl-C left each agent in, as far as the queries saw."""
    if e.service == 'disarm':
        done = [n for n in e.names if n in e.seen and e.seen[n].status == 'ok' and e.seen[n].mode == 'DISARMED']
        rest = [n for n in e.names if n not in done]
        warn('disarm interrupted: DISARMED %s; not confirmed: %s (scripts/sim.py status)'
             % (', '.join(done) or 'none', ', '.join(rest) or 'none'))
        return
    kept = [n for n in e.names if n in e.seen and e.seen[n].message == ALREADY_AUTO]
    kept_note = '; %s was already in AUTO and stays so' % ', '.join(kept) if kept else ''
    if e.undo is None:
        warn('arming interrupted, and so was the disarm that undoes it: %s may be in AUTO '
             '(scripts/sim.py status; scripts/sim.py disarm)' % ', '.join(n for n in e.names if n not in kept))
        return
    bad = [n for n, r in e.undo.items() if not (r.status == 'ok' and r.mode == 'DISARMED')]
    if bad:
        warn('arming interrupted: %s did not confirm DISARMED and may be in AUTO: scripts/sim.py disarm %s%s'
             % (', '.join(bad), ' '.join(bad), kept_note))
    else:
        warn('arming interrupted: %s DISARMED%s (scripts/sim.py arm)' % (', '.join(e.undo) or 'nothing', kept_note))


def will_arm(session: Session, stacks: Sequence['Stack'], no_arm: bool = False) -> bool:
    """`up` arms when the session asks for it and starts both the agents' DLIO and their
    safety_nodes, and for a planner session the planner too: armed, safety_node expects commands
    on cmd_vel_auto, which only mpc sends there (odom_eval.py --drive in planner: false)."""
    names = {s.name for s in stacks}
    return (session.auto_arm and not no_arm and {'dlio', 'drive'} <= names
            and (not session.planner or 'mighty' in names))


def session_agents(run_dir: Path, requested: Sequence[str] = ()) -> List[str]:
    """The running session's agents (<run>/session.json), or the requested ones, which must
    be among them: these commands never call a service of a name the sim did not start."""
    sf = run_dir / SESSION_FILE
    try:
        s = json.loads(sf.read_text())
        names = [a['name'] for a in s['agents']]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SimError('no running session (%s: %s): scripts/sim.py up <session>' % (sf, e))
    bad = [n for n in requested if n not in names]
    if bad:
        raise SimError('%s: not an agent of session %s (agents: %s)' % (', '.join(bad), s.get('name'), ', '.join(names)))
    return [n for n in names if n in requested] if requested else names


def rtf_violations(run_dir: Path) -> List[str]:
    """Lines the bridges appended this run (`up` rotates the log). An unreadable log
    counts as a violation: it cannot show the run was valid."""
    try:
        text = (run_dir / RTF_LOG).read_text(errors='replace')
    except FileNotFoundError:
        return []
    except OSError as e:
        return ['(cannot read it: %s)' % e]
    return [l for l in text.splitlines() if l.strip()]


def red_banner(lines: Sequence[str]) -> str:
    bar = '!' * 78
    text = '\n'.join([bar] + ['!! ' + l for l in lines] + [bar])
    return '\033[1;31m%s\033[0m' % text if sys.stdout.isatty() else text


def cmd_loadtest(args: argparse.Namespace) -> int:
    agents = load_session(args.session).agents if args.session else loadtest_agents(args.agents)
    variants = list(ZENOH_VARIANTS) if 'all' in (args.variant or []) else (args.variant or ['shipped'])
    for v in variants:
        session_variant({}, v)  # validates the name before anything starts
    profiles = [GPU_PROFILE] if gpu_available() and not args.no_gpu else []
    run_dir = resolve_run_dir() if running_sim_containers() else run_dir_from_env()
    images = compose_images(profiles, run_dir)
    # Test whatever stacks are built; only the router's image is required.
    have = [s for s, img in images.items() if docker_inspect('image', img) is not None]
    pins = read_pins()
    problem = dlio_scripts_problem(pins)
    if problem and any(s.startswith('sim-dlio-') for s in have):
        warn(problem + ': testing without the DLIO containers')
        have = [s for s in have if not s.startswith('sim-dlio-')]
    blocked = [c for c in have if mapping_mounts_problem(pins, [c])]
    if blocked:
        warn('%s: testing without %s' % (mapping_mounts_problem(pins, blocked), ', '.join(blocked)))
        have = [s for s in have if s not in blocked]
    if ROUTER_CONTAINER not in have:
        check_images(images)
    if len(have) < len(images):
        warn('no image for %s: testing without them' % ', '.join(s for s in images if s not in have))
    todo = [s for s in have if s not in running_sim_containers()]
    if todo:
        prepare_run_dir(run_dir)
        sh(compose_cmd(profiles) + ['up', '-d'] + todo, capture=False, env=compose_env(run_dir))
        wait_running(todo)
    ensure_run_dir(run_dir)
    problems = check_isolation()
    if problems:
        raise SimError('REFUSING TO RUN: the sim is not isolated:\n  - ' + '\n  - '.join(problems))
    others = [s for s in tmux_sessions() if s != TMUX_PREFIX + 'zenoh']
    if others:
        warn('a session is running (%s): its load adds to the measurement' % ', '.join(others))
    if not router_up():
        info('no zenoh router yet: starting the zenoh stack')
        ctx = RunContext(run_dir=run_dir, rust_log=os.environ.get('RUST_LOG_ZENOH', 'zenoh=info'))
        start_stack(STACKS['zenoh'], zenoh_panes(Session('loadtest', 'empty', 'fallback', False, agents), ctx),
                    make_log_dir())
        end = time.time() + 30
        while not router_up():
            if time.time() > end:
                raise SimError('the router did not come up (scripts/sim.py attach zenoh)')
            time.sleep(1)
    available = running_sim_containers()
    if 'sim-elevation' not in available:
        info('sim-elevation is not running (no GPU or --no-gpu): no elevation sessions')
    parallel(lambda c: stop_processes(c, match=LT_MARKER, grace_int=2, grace_term=2), available)
    lt_dir = run_dir / 'loadtest' / time.strftime('%Y%m%d_%H%M%S')
    lt_dir.mkdir(parents=True)
    (lt_dir / LT_MARKER).write_text(LT_AGENT_PY)
    base = json5_loads((ZENOH_DIR / 'session.json5').read_text())
    summaries = [run_loadtest_variant(v, agents, available, args, lt_dir, run_dir, base) for v in variants]
    (lt_dir / 'report.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print('load test, %d agents (%s), profile %s:' % (len(agents), ', '.join(a.name for a in agents), args.profile))
    print_loadtest(summaries, args.max_discovery)
    info('report: %s' % (lt_dir / 'report.json'))
    return 0 if summaries[0]['pass'] else 1


def cmd_cpus(args: argparse.Namespace) -> int:
    """Run inside sim-gazebo by pin_self: this process's affinity is the container's cpuset."""
    gz, rest = split_cpus(os.sched_getaffinity(0), args.spare)
    print(','.join(str(c) for c in (gz if args.part == 'gz' else rest)))
    return 0


def cmd_compose(argv: Sequence[str]) -> int:
    run_dir = resolve_run_dir() if shutil.which('docker') and running_sim_containers() else run_dir_from_env()
    if 'up' in argv:
        prepare_run_dir(run_dir)
    return subprocess.call(compose_cmd() + list(argv), env=compose_env(run_dir))


def _key_value(text: str) -> Tuple[str, Any]:
    key, sep, value = text.partition('=')
    if not sep or not re.match(r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$', key):
        raise argparse.ArgumentTypeError('expected dotted.key=<JSON5 value>, got %r' % text)
    try:
        return key, json5_loads(value)
    except ValueError:
        return key, value                            # a bare word: a string


def _csv(text: str) -> List[str]:
    return [t for t in (s.strip() for s in text.split(',')) if t]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='sim.py', description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', metavar='command')
    sub.required = True
    up = sub.add_parser('up', help='bring up the containers and the stacks of a session')
    up.add_argument('session', help='config/sessions/<name>.yaml, or a path')
    up.add_argument('--stacks', type=_csv, help='comma-separated subset of the stacks (default: all ready)')
    up.add_argument('--partial', action='store_true', help='start only the containers those stacks use')
    up.add_argument('--no-arm', action='store_true', help='leave every safety_node DISARMED (auto_arm off)')
    up.add_argument('--rviz', action='store_true',
                    help="also start RViz on $DISPLAY (as the session's rviz: true), in the last paused wave")
    up.add_argument('--odometry', type=parse_odometry_arg, metavar='MODE[:NS,...]',
                    help="this run's odometry (%s): MODE for every agent, or MODE:JR81,RR83 for those only; "
                         "truth (NOT PARITY) is ground truth under DLIO's names (truth_odom.py)"
                         % '|'.join(ODOMETRY_MODES))
    up.add_argument('--dry-run', action='store_true', help='print the panes and commands; touch nothing')
    down = sub.add_parser('down', help='stop every node and close the rsim-* tmux sessions')
    down.add_argument('--containers', action='store_true', help='also remove the containers and network')
    att = sub.add_parser('attach', help='attach to rsim-<stack>')
    att.add_argument('stack', nargs='?', choices=list(STACKS) + [VIZ_STACK.name])
    st = sub.add_parser('status', help='containers, isolation, router links, panes, sim time, DLIO, safety modes '
                        '(one ROS session, which leaves the graph one subscription at a time: all at once stalled '
                        'a loaded graph for a second)')
    st.add_argument('--maps', action='store_true',
                    help="also read the planner's grids (occ_2d_topic, esdf_2d_topic, persistent_map/grid): 3 more "
                         'subscriptions per agent, one of them the megabyte persistent map')
    sta = sub.add_parser('start', help='(re)start one stack alone on the running session: rviz (RViz on $DISPLAY, '
                         'even if the session has rviz: false) or goals (goal_relay.py); it joins the running graph')
    sta.add_argument('stack', choices=list(RESTARTABLE))
    viz = sub.add_parser('viz', help='open sim_viz.py, the live 2D view of the agents (occ grid, HGP paths, committed '
                         'traj, frontiers, pose), on $DISPLAY from sim-viz; it joins the running graph')
    viz.add_argument('--agents', type=_csv, help='comma-separated agents (default: all, up to %d)' % VIZ_MAX_PANELS)
    viz.add_argument('--layers', type=_csv, help='comma-separated layers to subscribe: %s (default: all but map, '
                     'the megabyte persistent map)' % ', '.join(VIZ_LAYERS))
    viz.add_argument('--follow', action='store_true', help='start in the follow view (f toggles)')
    viz.add_argument('--stop', action='store_true', help='stop the viewer, waiting for its paced leave')
    viz.add_argument('--snapshot', metavar='PNG', help='no window: one frame to PNG on this host')
    viz.add_argument('--snapshot-after', type=float, default=8.0, metavar='S',
                     help='--snapshot: seconds of listening first (default 8: frontiers come every 2 s)')
    for verb, what in (('arm', 'put safety_nodes in AUTO (/<ns>/auto_mode) once their DLIO tracks'),
                       ('disarm', 'put safety_nodes in DISARMED (/<ns>/disarm)')):
        a = sub.add_parser(verb, help=what)
        a.add_argument('agents', nargs='*', metavar='ns', help='agents of the running session (default: all)')
        a.add_argument('--timeout', type=float, default=ARM_TIMEOUT_S, help='seconds (default %d)' % ARM_TIMEOUT_S)
        if verb == 'arm':
            a.add_argument('--now', action='store_true',
                           help='do not wait for DLIO (the agent may then move during its IMU calibration)')
            a.add_argument('--no-z-check', action='store_true',
                           help='arm even if DLIO odom z is off the seed height on the first arm of its DLIO '
                                'session (checked to catch a late seed; logged)')
    lt = sub.add_parser('loadtest', help='zenoh discovery and link-count test with dummy agents')
    lt.add_argument('--agents', type=int, default=4, help='dummy agents JR81, RR82, ... (default 4)')
    lt.add_argument('--session', help='take the agents from this session instead')
    lt.add_argument('--variant', action='append', choices=list(ZENOH_VARIANTS) + ['all'],
                    help='session-config topology to test, repeatable (default shipped)')
    lt.add_argument('--profile', choices=list(LT_PROFILES), default='robot',
                    help='sessions per agent and stack: robot (about the rover\'s count) or light')
    lt.add_argument('--topics-per-proc', type=int, default=2)
    lt.add_argument('--rate', type=float, default=10.0, help='Hz per topic (default 10)')
    lt.add_argument('--tf-rate', type=float, default=20.0, help='Hz of the shared /lt/tf (default 20)')
    lt.add_argument('--payload-bytes', type=int, default=256)
    lt.add_argument('--hold', type=float, default=5.0, help='seconds of traffic after discovery')
    lt.add_argument('--timeout', type=float, default=60.0, help='discovery timeout, seconds')
    lt.add_argument('--max-discovery', type=float, default=10.0, help='pass limit, seconds')
    lt.add_argument('--no-gpu', action='store_true', help='leave sim-elevation out')
    lt.add_argument('--set', action='append', type=_key_value, default=[], metavar='KEY=VALUE',
                    help='override a session.json5 key in every variant, e.g. routing.interests.timeout=1000')
    sub.add_parser('compose', help='docker compose with the stack files: sim.py compose ps', add_help=False)
    cpus = sub.add_parser('cpus', help='(run in sim-gazebo by its panes) print gzserver\'s cores or the rest\'s')
    cpus.add_argument('part', choices=('gz', 'rest'))
    cpus.add_argument('--spare', type=int, help='cores for the rest (default a quarter, at least 1)')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        # compose passes everything through untouched, flags included.
        if argv and argv[0] == 'compose':
            return cmd_compose(argv[1:])
        args = build_parser().parse_args(argv)
        return {'up': cmd_up, 'down': cmd_down, 'attach': cmd_attach, 'status': cmd_status, 'arm': cmd_arm,
                'disarm': cmd_arm, 'start': cmd_start, 'viz': cmd_viz, 'loadtest': cmd_loadtest,
                'cpus': cmd_cpus}[args.cmd](args)
    except SimError as e:
        print('[sim] ERROR: %s' % e, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
