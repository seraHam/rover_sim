#!/usr/bin/env python3
"""rover_sim session runner (host side) for the per-stack sim.

docker/compose.stack.yaml is a set of idle containers, one per robot stack. This
script brings them up and runs every node as a `docker exec` from a host tmux session
per stack (rsim-<stack>), the way the robot's *_host_tmux.sh units do, so Ctrl-C / Up /
Enter in a pane restarts that one node.

  scripts/sim.py up <session> [--stacks a,b] [--partial] [--dry-run]
  scripts/sim.py down [--containers]
  scripts/sim.py attach [<stack>]
  scripts/sim.py status
  scripts/sim.py loadtest [--agents N] [--variant V|all ...]
  scripts/sim.py compose <docker compose args...>

<session> is config/sessions/<name>.yaml, or a path to one:

  name: smoke
  world: empty                  # Gazebo world
  mapping: fallback             # gpu (parity) | fallback (no GPU, "not parity")
  rviz: false
  agents:                       # pose: world frame, [x m, y m, yaw deg]
    - {name: JR81, class: jr, pose: [-2.0, 0.0, 0.0]}
    - {name: RR83, class: rr, pose: [2.0, 0.0, 180.0]}

Agent names are a class prefix (JR, RR) plus two digits. The digits must be unique
across classes: mighty takes its agent id from them.

Stacks are a registry (STACKS below). Each names its containers and builds its panes
from the session. Phase 1 runs only `zenoh`, the router in sim-drive. The others are
placeholders that later phases fill in.

Per-run generated files go to $ROVER_SIM_RUN (default /tmp/rover_sim-<uid>/run), which
is mounted at /rover_sim/run in every container. It must be yours and writable by no one
else: the containers run as root and write, chown and execute files there. Pane logs go
to ~/.local/state/rover_sim/logs/<timestamp>/<stack>/<pane>.log.

Host needs: python3 >= 3.8 with PyYAML, tmux, the docker CLI with compose v2.
"""

import argparse
import copy
import dataclasses
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

LOG_ROOT = Path.home() / '.local' / 'state' / 'rover_sim' / 'logs'
LOG_KEEP = 20                        # newest pane-log dirs kept

# A detached session from a non-tty needs an explicit size; attaching resizes it.
TMUX_SIZE = ('220', '56')
TMUX_HISTORY = 50000                 # scrollback lines per pane (pipe-pane logs keep it all)
TMUX_PLACEHOLDER = '_starting'
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
CLASS_PREFIXES = OrderedDict([('jr', 'JR'), ('rr', 'RR')])  # class -> name prefix
MAPPING_MODES = ('gpu', 'fallback')
IDENT_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
SESSION_KEYS = ('name', 'world', 'mapping', 'rviz', 'agents')
AGENT_KEYS = ('name', 'class', 'pose', 'goals')


@dataclass(frozen=True)
class Agent:
    name: str
    cls: str
    x: float
    y: float
    yaw_deg: float
    goals: Tuple[Tuple[float, ...], ...] = ()

    @property
    def id(self) -> int:
        """mighty's agent id: the last two digits of the namespace (mighty_node.cpp:57-63)."""
        return int(self.name[2:])


@dataclass(frozen=True)
class Session:
    name: str
    world: str
    mapping: str
    rviz: bool
    agents: Tuple[Agent, ...]
    source: str = ''

    @property
    def num_agents(self) -> int:
        """num_agents:= for mighty: the highest id, as the launch arg overwrites the overlay."""
        return max(a.id for a in self.agents)

    def by_class(self, cls: str) -> List[Agent]:
        return [a for a in self.agents if a.cls == cls]

    def to_json(self) -> Dict[str, Any]:
        return OrderedDict([
            ('name', self.name), ('world', self.world), ('mapping', self.mapping),
            ('rviz', self.rviz), ('num_agents', self.num_agents), ('source', self.source),
            ('agents', [OrderedDict([('name', a.name), ('class', a.cls), ('id', a.id),
                                     ('pose', [a.x, a.y, a.yaw_deg]),
                                     ('goals', [list(g) for g in a.goals])])
                        for a in self.agents]),
        ])


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
    rviz = data.get('rviz', False)
    if not isinstance(rviz, bool):
        errs.append('rviz: %r is not true/false' % (rviz,))

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
        if len(errs) == before:
            agents.append(Agent(an, cls, float(pose[0]), float(pose[1]), float(pose[2]),
                                tuple(tuple(float(v) for v in g) for g in goals)))
    if errs:
        raise SessionError('%s: invalid session:\n  - %s' % (source, '\n  - '.join(errs)))
    return Session(name, world, mapping, rviz, tuple(agents), source)


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


# ---------------------------------------------------------------------------- containers

ROS_SETUP = '/opt/ros/humble/setup.bash'
SIM_WS = '/home/swarm/code/rover_sim_ws'          # rover_sim's own images (gazebo, elevation)
SIM_PROBE = '/rover_sim/scripts/wait_zenoh_ready.py'


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
    Container('sim-gazebo', (ROS_SETUP, SIM_WS + '/install/setup.bash'),
              overlays=('/usr/share/gazebo/setup.bash', '/opt/livox_ws/install/setup.bash'),
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
    Container('sim-mapping', (ROS_SETUP, '/home/swarm/code/mapping_ws/install/setup.bash'),
              probes=('/home/swarm/code/mapping_ws/install/share/rover_bringup/scripts/wait_zenoh_ready.py',
                      SIM_PROBE),
              role='global_mapper, map_fusion, persistent_map'),
    Container('sim-elevation', (ROS_SETUP, SIM_WS + '/install/setup.bash'),
              probes=(SIM_WS + '/scripts/wait_zenoh_ready.py', SIM_PROBE),
              role='elevation_mapping_cupy (GPU)'),
    # mighty-hw bakes no probe: its own mighty_hw.sh gates on the router port only.
    Container('sim-mighty', (ROS_SETUP, '/home/swarm/code/mighty_ws/install/setup.bash'),
              role='mighty_node, convert_odom_to_state, mpc, goal_selector'),
))


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


def pane_inner(p: Pane) -> str:
    c = CONTAINERS[p.container]
    steps = [setup_snippet(c)]
    if p.gate:
        steps.append(router_gate(c))
    steps.append(p.cmd)
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


def _always(session: Session) -> bool:
    return True


@dataclass(frozen=True)
class Stack:
    name: str
    phase: int
    summary: str
    containers: Tuple[str, ...]
    panes: Callable[[Session, RunContext], List[Pane]]
    implemented: bool = False
    wanted: Callable[[Session], bool] = _always
    # Writes this stack's per-run files under ctx.run_dir before its panes start.
    generate: Optional[Callable[[Session, RunContext], None]] = None
    # Opens windows on $DISPLAY: `up` then grants the containers' root X access.
    needs_display: bool = False


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


def gazebo_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """TODO(Phase 2), sim-gazebo. One gzserver on session.world (src/rover_sim
    world.launch.py, real-time capped), spawning every agent paused and unpausing once.
    Per agent: livox_sim_bridge.py (sim->wall stamps, sim/rtf) and sim_tf.py
    (world-><ns>/map on /tf, frame_align pairs). Generates <run>/spawn.yaml. gzserver
    gets its own cores out of sim-gazebo's cpuset (cpu_budget), e.g. with taskset, so
    the bridges cannot slow physics. Set needs_display once gzclient runs here."""
    raise _not_yet('gazebo')


def drive_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """TODO(Phase 2/3), sim-drive. Per agent: rover_description state_publisher.launch.py
    robot_name:=<ns> ns_tf_frame_ids:=true (generated <run>/sensor_calib), then
    rover_control teleop.launch.yaml with <run>/teleop_sim.yaml (safety_node). No
    firmware driver: Gazebo's diff drive consumes /<ns>/cmd_vel."""
    raise _not_yet('drive')


def dlio_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """TODO(Phase 3), sim-dlio-<class> (dlio_container). Per agent, as dlio_host_tmux.sh:
    dlio_with_session.launch.py namespace:=<ns>, the seed pose publisher, and
    map_odom_tf.py <ns> with per-agent node names."""
    raise _not_yet('dlio')


def mapping_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """TODO(Phase 4), sim-mapping. Per agent, as mapping_host_tmux.sh: global_mapper
    (voxel/*), map_fusion, persistent_map. session.mapping == 'fallback' switches to
    config/mapping/global_mapper_fallback_<class>.yaml plus the relay."""
    raise _not_yet('mapping')


def elevation_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """TODO(Phase 4), sim-elevation. Per agent: elevation_mapping_dlio.launch.py
    quad:=<ns> with the elev/* remaps (orin/local_mapper_host_tmux.sh)."""
    raise _not_yet('elevation')


def mighty_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """TODO(Phase 5), sim-mighty. Four panes per agent (one window each), as
    mighty_hw.sh: onboard_mighty.launch.py only_nodes:=<mighty_node|convert_odom_to_state|
    mpc|goal_selector> num_agents:=session.num_agents; mighty_node and mpc gated on
    wait_for_tf.py <ns>/map <ns>/odom."""
    raise _not_yet('mighty')


def rviz_panes(session: Session, ctx: RunContext) -> List[Pane]:
    """TODO(Phase 5), sim-gazebo. RViz from a config generated for the session's agents,
    started last (zenoh late-join TF)."""
    raise _not_yet('rviz')


# Start order is registry order. Later phases flip `implemented` and fill in the builder.
STACKS = OrderedDict((s.name, s) for s in (
    Stack('zenoh', 1, 'zenoh router in sim-drive', ('sim-drive',), zenoh_panes, implemented=True),
    Stack('gazebo', 2, 'Gazebo world, spawns, sensor bridges, sim TF', ('sim-gazebo',), gazebo_panes),
    Stack('drive', 2, 'rover_description and safety_node/teleop per agent', ('sim-drive',), drive_panes),
    Stack('dlio', 3, 'DLIO, seed pose and map->odom per agent', ('sim-dlio-jr', 'sim-dlio-rr'), dlio_panes),
    Stack('mapping', 4, 'global_mapper, map_fusion, persistent_map per agent', ('sim-mapping',),
          mapping_panes),
    Stack('elevation', 4, 'GPU elevation mapping per agent', ('sim-elevation',), elevation_panes,
          wanted=lambda s: s.mapping == 'gpu'),
    Stack('mighty', 5, 'mighty_node, convert_odom_to_state, mpc, goal_selector per agent',
          ('sim-mighty',), mighty_panes),
    Stack('rviz', 5, 'RViz, started last', ('sim-gazebo',), rviz_panes, wanted=lambda s: s.rviz,
          needs_display=True),
))


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
            raise SimError('the %s stack does not apply to session %s (mapping: %s, rviz: %s)'
                           % (name, session.name, session.mapping, session.rviz))
    return [s for s in STACKS.values() if s.name in requested]


def pending_stacks(session: Session) -> List[str]:
    return ['%s (Phase %d)' % (s.name, s.phase) for s in STACKS.values()
            if not s.implemented and s.wanted(session)]


def session_profiles(session: Session) -> List[str]:
    return [GPU_PROFILE] if session.mapping == 'gpu' else []


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
            # pipe-pane taps the pane at the tmux server: nodes run under docker exec, so
            # this is the only full record of their output, across restarts too.
            tmux('pipe-pane', '-o', '-t', pid, 'cat >> ' + shlex.quote(str(stack_log / (pane.slug + '.log'))))
            tmux('send-keys', '-t', pid, '-l', pane_command(pane))
            tmux('send-keys', '-t', pid, 'C-m')
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
HOST_CPUS = 2
GAZEBO_CPUS = 4
CPU_BUDGET_MIN = 12


def cpuset_var(service: str) -> str:
    return 'ROVER_SIM_CPUSET_' + service[len('sim-'):].upper().replace('-', '_')


def cpu_budget(ncpu: int, services: Iterable[str]) -> Dict[str, str]:
    """service -> cpuset; {} (unpinned) on a host too small to split."""
    if ncpu < CPU_BUDGET_MIN:
        return {}
    gz = '%d-%d' % (HOST_CPUS, HOST_CPUS + GAZEBO_CPUS - 1)
    rest = '%d-%d' % (HOST_CPUS + GAZEBO_CPUS, ncpu - 1)
    return OrderedDict((s, gz if s == 'sim-gazebo' else rest) for s in services)


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
    for svc, cpus in cpu_budget(docker_ncpu(), CONTAINERS).items():
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
    path = run_dir / 'session.json'
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)
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
    Processes first, so the panes still log their shutdown."""
    names = running_sim_containers()
    counts = parallel(stop_processes, names)
    for n, c in zip(names, counts):
        if c:
            info('%s: stopped %d process(es)' % (n, c))
    for s in tmux_sessions():
        tmux('kill-session', '-t', '=' + s, check=False)
        info('closed tmux session %s' % s)
    if remove_containers:
        run_dir = resolve_run_dir()
        sh(compose_cmd([GPU_PROFILE]) + ['down', '--remove-orphans'], capture=False, env=compose_env(run_dir))


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

def print_plan(session: Session, plan: Sequence[Tuple[Stack, List[Pane]]]) -> None:
    print('session %s (%s): world %s, mapping %s, rviz %s, num_agents %d' % (
        session.name, session.source, session.world, session.mapping,
        'on' if session.rviz else 'off', session.num_agents))
    for a in session.agents:
        print('  %s  %s  id %02d  pose (%.2f, %.2f, %.1f deg)' % (a.name, a.cls, a.id, a.x, a.y, a.yaw_deg))
    for stack, panes in plan:
        print('stack %s -> tmux session %s%s' % (stack.name, TMUX_PREFIX, stack.name))
        for wname, wpanes in tmux_windows(panes):
            print('  window %s' % wname)
            for p in wpanes:
                print('    [%s] %s\n      %s' % (p.slug, p.title, pane_command(p)))
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


def cmd_up(args: argparse.Namespace) -> int:
    session = load_session(args.session)
    stacks = select_stacks(session, args.stacks)
    run_dir = run_dir_from_env()
    ctx = RunContext(run_dir=run_dir, rust_log=os.environ.get('RUST_LOG_ZENOH', 'zenoh=info'))
    plan = [(s, s.panes(session, ctx)) for s in stacks]
    if args.dry_run:
        print_plan(session, plan)
        return 0
    for tool in ('docker', 'tmux'):
        if shutil.which(tool) is None:
            raise SimError('%s not found on PATH' % tool)
    profiles = session_profiles(session)
    if profiles and not gpu_available():
        raise SimError('session %s has mapping: gpu but this host has no nvidia-smi; '
                       'use mapping: fallback' % session.name)
    images = compose_images(profiles, run_dir)
    services = sorted({c for s in stacks for c in s.containers}) if args.partial else list(images)
    used = OrderedDict((k, v) for k, v in images.items() if k in services)
    check_images(used)
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
    if any(s.needs_display for s in stacks):
        grant_x_access()
    write_session_file(session, run_dir)
    log_dir = make_log_dir()
    ctx = dataclasses.replace(ctx, log_dir=log_dir)
    for s in stacks:
        if s.generate:
            s.generate(session, ctx)
    for s, panes in plan:
        sess = start_stack(s, panes, log_dir)
        info('%-8s up: tmux attach -t %s   (scripts/sim.py attach %s)' % (s.name, sess, s.name))
    info('session %s: %s' % (session.name, ', '.join('%s (%s)' % (a.name, a.cls) for a in session.agents)))
    pending = pending_stacks(session)
    if pending:
        info('not started yet (later phases): ' + ', '.join(pending))
    info('pane logs: %s' % log_dir)
    info('run dir:   %s' % run_dir)
    return 0


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
    if up:
        problems = check_isolation()
        print('isolation: %s' % ('OK' if not problems else 'BROKEN\n  - ' + '\n  - '.join(problems)))
    if ROUTER_CONTAINER in up:
        links = link_snapshot(up)
        print('zenoh router %s:%d: %s; TCP links: %d to the router, %d peer-to-peer, %d listening, %d dialing'
              % (ROUTER_HOST, ROUTER_PORT, 'listening' if router_up() else 'DOWN', links['router'],
                 links['p2p'], links['listen'], links['syn_sent']))
    p = tmux('list-panes', '-a', '-F', '#{session_name}\t#{window_name}\t#{pane_index}\t#{pane_title}\t'
             '#{pane_current_command}', check=False)
    rows = [l.split('\t') for l in (p.stdout or '').splitlines() if l.startswith(TMUX_PREFIX)]
    print('tmux:%s' % ('' if rows else ' no rsim-* sessions'))
    for sess, win, idx, title, cmd in rows:
        # The pane's bash is back at its prompt once its docker exec has ended.
        print('  %s:%s.%s  %-9s %s' % (sess, win, idx, 'running' if cmd == 'docker' else 'EXITED', title))
    run_dir = resolve_run_dir() if up else run_dir_from_env()
    sf = run_dir / 'session.json'
    if sf.is_file():
        s = json.loads(sf.read_text())
        print('session: %s (%s), world %s, mapping %s, agents %s' % (
            s['name'], s.get('source', ''), s['world'], s['mapping'], ', '.join(a['name'] for a in s['agents'])))
    print('run dir: %s' % run_dir)
    if (LOG_ROOT / 'latest').exists():
        print('logs:    %s' % (LOG_ROOT / 'latest').resolve())
    print('RTF:     n/a until the gazebo stack (Phase 2) publishes sim/rtf')
    return 0


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
    up.add_argument('--dry-run', action='store_true', help='print the panes and commands; touch nothing')
    down = sub.add_parser('down', help='stop every node and close the rsim-* tmux sessions')
    down.add_argument('--containers', action='store_true', help='also remove the containers and network')
    att = sub.add_parser('attach', help='attach to rsim-<stack>')
    att.add_argument('stack', nargs='?', choices=list(STACKS))
    sub.add_parser('status', help='containers, isolation, router links, panes')
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
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        # compose passes everything through untouched, flags included.
        if argv and argv[0] == 'compose':
            return cmd_compose(argv[1:])
        args = build_parser().parse_args(argv)
        return {'up': cmd_up, 'down': cmd_down, 'attach': cmd_attach, 'status': cmd_status,
                'loadtest': cmd_loadtest}[args.cmd](args)
    except SimError as e:
        print('[sim] ERROR: %s' % e, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
