#!/usr/bin/env python3
"""Diffs one sim agent's ROS params and graph edges against a robot reference.

Usage: parity_check.py <ns> [--ref JR02] [--nodes mighty_node,mpc,...] [--mapping gpu|fallback]
       parity_check.py <ns> --show-ref <node>   print the reference params, renamed to <ns>
       parity_check.py <ns> --selftest          check the checker on the fixtures alone (no docker)
       parity_check.py <ns> --local             query from this process (inside a sim container)

Runs on the host (python3.8 + PyYAML + docker CLI). One rclpy session in one sim container
(--container, default sim-drive) answers every query: the node list, each reference node's
parameters (its list_parameters and get_parameters services, as `ros2 param dump` reads them),
its publishers and subscriptions (the graph, as `ros2 node info`) and each reference topic's
endpoints with their QoS (as `ros2 topic info -v`). The results are diffed against
tools/reference/<ref>/ (see its README.md) with the robot's name (JR02) replaced by <ns>. What
the sim changes on purpose is listed, with the reason, in tools/reference/<ref>/
allowed_deviations.yaml and shows up as ALLOWED.

That session joins and leaves the graph gently: its node has no rosout publisher and no
parameter services, and it closes its service clients one at a time, then waits before it
closes. With rmw_zenoh 0.1.9 a session that undeclares many entities at once stalls zenoh
delivery for everyone; a ros2 CLI per query did that (Phase 4: all four parked DLIOs diverged
after one run). It is still one more session on the graph: run it before or after a window
that is being graded, not during one.

--mapping defaults to the running session's (<run>/session.json, scripts/sim.py's run dir:
$ROVER_SIM_RUN or /tmp/rover_sim-<uid>/run), else $ROVER_SIM_MAPPING, else gpu.

Nodes that are not running (or whose container is down) are reported as NOT RUNNING and do
not fail the check unless --require-running is given, so it can be run while the stacks are
brought up one phase at a time.

Exit status: 0 no unexpected differences; 1 unexpected differences, failed queries (ERROR),
or, with --require-running, nodes not running; 2 bad usage or reference.

It only ever docker-execs into the sim's own containers (sim-* of compose project
rover-sim-stack, the names scripts/sim.py registers); any other --container, --graph-container
or manifest container is refused with exit 2. The fleet's containers share this host, and a
query in one of them (acl-c2-*) would reach the live robots. Never point this at a robot: the
reference was captured without any ros2 CLI call. --local skips docker and queries from this
process: only inside a sim container (ZENOH_SESSION_CONFIG_URI the sim's, config/sim.env).

Reference files from private repos are not copied into tools/reference/ (this repo is public).
manifest.yaml names each by repo, commit and path; they are read from the pinned checkouts
that scripts/build_images.sh keeps under .cache/, or fetched into .cache/reference/ (needs
the GitLab/GitHub key the image builds use), and refused unless their content matches the
hash of the robot's copy.
"""
import argparse
import fnmatch
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, '.cache')
NAME_RE = re.compile(r'^([A-Z]{2})([0-9]{2})$')
# The sim's containers (scripts/sim.py CONTAINERS, compose project rover-sim-stack). Nothing
# else is ever exec'd into: acl-c2-* on this host are attached to the live fleet graph.
SIM_CONTAINERS = ('sim-gazebo', 'sim-drive', 'sim-dlio-jr', 'sim-dlio-rr', 'sim-mapping',
                  'sim-elevation', 'sim-mighty', 'sim-viz')
SIM_PROJECT = 'rover-sim-stack'
# Introspection needs no workspace message types, so whichever install the image has will
# do (mighty_ws, mapping_ws, dlio_ws, rover_ws or rover_sim_ws), else plain Humble.
SETUP = ('for f in /home/swarm/code/*/install/setup.bash /opt/ros/humble/setup.bash; do '
         '[ -f "$f" ] && . "$f" && break; done; export ROS_LOG_DIR=/tmp/rover_sim_parity_log; ')
# Every node has these; they say nothing about parity.
GRAPH_TOPICS_IGNORED = ('/parameter_events', '/rosout')
# The query session (query()): this file as the sim containers mount it, the line its answer is
# on (zenoh's Rust logger writes to the same stdout), and its paced leave (scripts/sim.py
# STATUS_LEAVE_GAP_S, STATUS_LEAVE_PAUSE_S).
QUERY_IN = '/rover_sim/scripts/parity_check.py'
QUERY_MARK = '@@PARITY@@ '
LEAVE_GAP_S, LEAVE_PAUSE_S = 0.5, 0.5
DEFAULT_QUERY_CONTAINER = 'sim-drive'
SIM_SESSION_URI = '/rover_sim/zenoh/session.json5'     # config/sim.env
RUN_DIR_DEFAULT = '/tmp/rover_sim-%d/run' % os.getuid()  # scripts/sim.py's

OK, NOTE, NO_REF, ALLOWED, NOT_RUNNING, ERROR, DIFF = (
    'OK', 'NOTE', 'NO REF', 'ALLOWED', 'NOT RUNNING', 'ERROR', 'DIFF')
RANK = {OK: 0, NOTE: 1, NO_REF: 2, ALLOWED: 3, NOT_RUNNING: 4, ERROR: 5, DIFF: 6}


def die(msg):
    sys.stderr.write('parity_check: %s\n' % msg)
    sys.exit(2)


def sim_container(name):
    """Every docker call goes through this: one of the sim's own containers, or exit 2."""
    if name not in SIM_CONTAINERS:
        die('refusing to query container %r: only the sim containers (%s)' % (name, ', '.join(SIM_CONTAINERS)))
    return name


# ---------------------------------------------------------------------------------- YAML
class Loader(yaml.SafeLoader):
    """SafeLoader that also reads YAML 1.2 floats such as 1e-9, as rcl's parser does."""


Loader.add_implicit_resolver(
    'tag:yaml.org,2002:float', re.compile(r'^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$'),
    list('-+0123456789'))
# launch_ros writes Python tuples into its /tmp/launch_params_* files this way; rcl ignores the tag.
Loader.add_constructor('tag:yaml.org,2002:python/tuple', lambda loader, node: loader.construct_sequence(node))


def load_yaml(text):
    return yaml.load(text, Loader=Loader)


def load_yaml_file(path):
    with open(path) as f:
        return load_yaml(f)


# ---------------------------------------------------------------------------------- params
def flatten(d, prefix=''):
    """Nested ros__parameters -> {'a.b.c': value}, the names ROS gives them."""
    out = {}
    for k, v in (d or {}).items():
        key = '%s.%s' % (prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def key_matches(key, fqn):
    """rcl's matching of a params-file node key ('/**', '/**/name', '/ns/*', '/ns/name')."""
    key = key if key.startswith('/') else '/' + key
    rx = ''.join({'**': '.*', '*': '[^/]*'}.get(t, re.escape(t)) for t in re.split(r'(\*\*|\*)', key))
    return re.fullmatch(rx.replace('/.*/', '(?:/.*)?/'), fqn) is not None


def merge_param_files(paths, fqn):
    """Effective values of --params-file args, combined the way rcl does it.

    rcl keeps one entry per node key, in order of first appearance (later files update it),
    then applies each matching entry in that order. That differs from plain file order only
    when a later file repeats a wildcard key that an earlier exact block overrode.
    """
    entries = {}
    for path in paths:
        for key, block in (load_yaml_file(path) or {}).items():
            entries.setdefault(key, {}).update(flatten((block or {}).get('ros__parameters')))
    merged = {}
    for key, params in entries.items():
        if key_matches(key, fqn):
            merged.update(params)
    return merged


def dump_params(text):
    """`ros2 param dump` output -> (fqn, flat params)."""
    data = load_yaml(text) or {}
    if len(data) != 1:
        raise ValueError('expected one node in the param dump, got %d' % len(data))
    fqn, block = next(iter(data.items()))
    return fqn, flatten((block or {}).get('ros__parameters'))


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (a != a and b != b) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def show(v):
    s = json.dumps(v, default=str)
    return s if len(s) <= 100 else s[:97] + '...'


def urdf_joints(text):
    joints = {}
    for j in ET.fromstring(text).findall('joint'):
        o = j.find('origin')
        attr = o.attrib if o is not None else {}
        joints[j.get('name')] = {
            'parent': j.find('parent').get('link'), 'child': j.find('child').get('link'),
            'xyz': [float(v) for v in attr.get('xyz', '0 0 0').split()],
            'rpy': [float(v) for v in attr.get('rpy', '0 0 0').split()]}
    return joints


def same_joint(a, b):
    nums = zip(a['xyz'] + a['rpy'], b['xyz'] + b['rpy'])
    return (a['parent'], a['child']) == (b['parent'], b['child']) and all(
        math.isclose(x, y, abs_tol=1e-6) for x, y in nums)


# ---------------------------------------------------------------------------------- external files
def yaml_digest(text):
    """sha256 of a YAML file's parsed content: launch rewrites a params file it passes on
    (comments, quoting), so the robot's copy is compared by content, not bytes."""
    canon = json.dumps(load_yaml(text), sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canon.encode()).hexdigest()


def _git(args, cwd=None, check=True):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    env.setdefault('GIT_SSH_COMMAND', 'ssh -o BatchMode=yes')
    p = subprocess.run(['git'] + args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, env=env)
    if check and p.returncode != 0:
        raise RuntimeError('git %s: %s' % (' '.join(args[:2]), (p.stderr.strip().splitlines() or ['failed'])[-1]))
    return p


def git_repo_with(url, sha):
    """A local clone that has commit sha: build_images.sh's checkouts first, then our own
    blob-less clone under .cache/reference/git/, made on first use."""
    name = os.path.basename(url.rstrip('/'))
    name = name[:-4] if name.endswith('.git') else name
    candidates = (sorted(glob.glob(os.path.join(CACHE, 'src', name + '@*')))
                  + sorted(glob.glob(os.path.join(CACHE, 'ws', name + '@*', 'src', name)))
                  + [os.path.join(CACHE, 'reference', 'git', name + '.git')])
    for d in candidates:
        if os.path.isdir(d) and _git(['-C', d, 'cat-file', '-e', sha + '^{commit}'], check=False).returncode == 0:
            return d
    mine = candidates[-1]
    if not os.path.isdir(mine):
        os.makedirs(os.path.dirname(mine), exist_ok=True)
        tmp = mine + '.tmp%d' % os.getpid()
        _git(['clone', '-q', '--bare', '--filter=blob:none', url, tmp])
        os.rename(tmp, mine)
    if _git(['-C', mine, 'cat-file', '-e', sha + '^{commit}'], check=False).returncode != 0:
        _git(['-C', mine, 'fetch', '-q', 'origin', sha])
    return mine


def fetch_external(spec):
    """Text of spec['path'] at spec['sha'] of spec['url'], checked against yaml_sha256."""
    d = git_repo_with(spec['url'], spec['sha'])
    text = _git(['-C', d, 'show', '%s:%s' % (spec['sha'], spec['path'])]).stdout
    got = yaml_digest(text)
    if got != spec['yaml_sha256']:
        raise RuntimeError('%s@%s:%s does not match the robot\'s copy (yaml_sha256 %s, want %s)'
                           % (spec['url'], spec['sha'][:7], spec['path'], got[:16], spec['yaml_sha256'][:16]))
    return text


# ---------------------------------------------------------------------------------- reference
class Reference:
    """tools/reference/<ref>/ (manifest, captures, whitelist), renamed to one sim agent."""

    def __init__(self, ref, ns, mapping):
        self.dir = os.path.join(REPO, 'tools', 'reference', ref.lower())
        self.external = {}
        if not os.path.isfile(self.path('manifest.yaml')):
            die('no reference %s (%s)' % (ref, self.path('manifest.yaml')))
        m = NAME_RE.match(ns)
        if not m:
            die('%r is not an agent name (2 letters + 2 digits)' % ns)
        self.name = ref.lower()
        self.manifest = load_yaml_file(self.path('manifest.yaml'))
        self.external = self.manifest.get('external') or {}
        allowed = load_yaml_file(self.path('allowed_deviations.yaml')) or {}
        self.ns, self.mapping = ns, mapping
        self.cls = self.manifest['classes'].get(m.group(1))
        if self.cls is None:
            die('no class for prefix %s (known: %s)' % (m.group(1), ', '.join(self.manifest['classes'])))
        tokens = self.manifest['reference_robots'][self.cls]
        self._token_rx = re.compile(r'(?<![A-Za-z0-9])(?:%s)(?![0-9])' % '|'.join(map(re.escape, tokens)))
        self.renames = {k: self.fmt(v) for k, v in (allowed.get('rename') or {}).items()}
        self.patterns = {k: [self.fmt(p) for p in self._layered(allowed, k) or []]
                         for k in ('ignore', 'absent', 'sim_only')}
        self.param_rules = self._layered(allowed, 'params') or {}
        self.topic_rules = self._layered(allowed, 'topics') or {}

    def path(self, rel):
        """A reference file on disk. An `external` one (from a private repo) is fetched and
        verified into .cache/reference/<ref>/ on first use; exit 2 if that fails."""
        if rel not in self.external:
            return os.path.join(self.dir, rel)
        spec = self.external[rel]
        out = os.path.join(CACHE, 'reference', self.name, rel)
        try:
            with open(out) as f:
                if yaml_digest(f.read()) == spec['yaml_sha256']:
                    return out
        except (OSError, yaml.YAMLError):
            pass
        try:
            text = fetch_external(spec)
        except RuntimeError as exc:
            die('reference file %s: %s\n  (it comes from a private repo; parity_check needs read access with '
                'the key `make images` uses)' % (rel, exc))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + '.tmp', 'w') as f:
            f.write(text)
        os.replace(out + '.tmp', out)
        return out

    def files(self, nodes):
        """Every reference file the given manifest entries use (to fetch them up front)."""
        out = ['tf/tf_tree.yaml', 'graph/edges.yaml']
        for e in nodes:
            out += (e.get('params') or {}).get('files') or []
        return out

    def fmt(self, s):
        return s.replace('{ns}', self.ns).replace('{class}', self.cls)

    def norm(self, obj):
        """Replace the reference robot's name by the sim ns, in strings, lists and dict keys."""
        if isinstance(obj, str):
            return self._token_rx.sub(self.ns, obj)
        if isinstance(obj, list):
            return [self.norm(v) for v in obj]
        if isinstance(obj, dict):
            return {self.norm(k): self.norm(v) for k, v in obj.items()}
        return obj

    def _layered(self, section, key):
        """section[key] from the 'all' layer, then this class's, then this mapping mode's.

        Lists concatenate; dicts merge, one level deeper for dict values (node -> {glob: reason})."""
        out = None
        for layer in (section.get('all'), (section.get('class') or {}).get(self.cls),
                      (section.get('mapping') or {}).get(self.mapping)):
            v = (layer or {}).get(key)
            if not v:
                continue
            if isinstance(v, list):
                out = (out or []) + v
            else:
                out = dict(out or {})
                for k, rules in v.items():
                    out[k] = dict(out.get(k) or {}, **rules) if isinstance(rules, dict) else rules
        return out

    def nodes(self):
        """name -> manifest entry with this class's overrides applied and sim names filled in."""
        out = {}
        for name, entry in self.manifest['nodes'].items():
            entry = dict(entry)
            entry.update((entry.pop('by_class', None) or {}).get(self.cls) or {})
            if entry.get('params'):
                params = dict(entry['params'])
                params.update((params.pop('by_class', None) or {}).get(self.cls) or {})
                entry['params'] = params
            entry['container'] = self.fmt(entry['container'])
            entry['sim_fqn'] = self.sim_fqn(entry['fqn'])
            out[name] = entry
        return out

    def sim_fqn(self, ref_fqn):
        return self.renames.get(ref_fqn) or self.norm(ref_fqn)

    def params(self, entry):
        """(kind, flat params) of the reference, renamed. 'dump' = full declared set,
        'files' = only what the robot's param files and launch params set."""
        spec = entry['params']
        paths = [self.path(p) for p in spec['files']]
        if spec['kind'] == 'dump':
            with open(paths[0]) as f:
                flat = dump_params(f.read())[1]
        else:
            flat = merge_param_files(paths, spec.get('ref_fqn', entry['fqn']))
        flat = self.norm(flat)
        flat.pop(spec.get('urdf_key'), None)  # compared joint by joint against tf/tf_tree.yaml
        return spec['kind'], flat

    def tf_joints(self):
        tree = load_yaml_file(self.path('tf/tf_tree.yaml'))
        return {self.norm(j['joint']): self.norm({k: j[k] for k in ('parent', 'child', 'xyz', 'rpy')})
                for j in tree['static'][self.cls]['joints']}

    def edges(self):
        e = self.norm(load_yaml_file(self.path('graph/edges.yaml')))
        e['nodes'] = {self.sim_fqn(k): v for k, v in e['nodes'].items()}
        for t in e['topics'].values():
            for ep in t['endpoints']:
                ep['node'] = self.sim_fqn(ep['node'])
        return e

    def allowed_param(self, node, key):
        for pattern, reason in (self.param_rules.get(node) or {}).items():
            if fnmatch.fnmatchcase(key, self.fmt(pattern)):
                return reason
        return None

    def allowed_topic(self, topic):
        for pattern, reason in self.topic_rules.items():
            if fnmatch.fnmatchcase(topic, self.fmt(pattern)):
                return reason
        return None

    def node_class(self, fqn):
        """'ignore' | 'absent' | 'sim_only' | None, by the node patterns in allowed_deviations."""
        for kind in ('ignore', 'absent', 'sim_only'):
            if any(fnmatch.fnmatchcase(fqn, p) for p in self.patterns[kind]):
                return kind
        return None

    def mine(self, fqn):
        """Whether a sim node belongs to this agent (others share root topics like /trajs)."""
        return fqn.startswith('/%s/' % self.ns) or fqn in self.renames.values()


# ---------------------------------------------------------------------------------- the query session
def param_value(v):
    """rcl_interfaces/ParameterValue -> the value `ros2 param dump` writes (None when not set)."""
    t = v.type
    return {1: lambda: v.bool_value, 2: lambda: v.integer_value, 3: lambda: v.double_value,
            4: lambda: v.string_value, 5: lambda: [int(b) for b in v.byte_array_value],
            6: lambda: list(v.bool_array_value), 7: lambda: list(v.integer_array_value),
            8: lambda: list(v.double_array_value), 9: lambda: list(v.string_array_value)}.get(t, lambda: None)()


def split_fqn(fqn):
    ns, _, name = fqn.rpartition('/')
    return name, ns or '/'


def query(spec, rclpy=None):
    """The sim's answers to spec ({params, info, topics: [...], timeout, spin}) from one rclpy
    session: the node list, the params of the spec's running nodes through their parameter
    services, their pub/sub edges and the topics' endpoints from the graph. The node has no
    rosout and no parameter services; each service client is destroyed right after its call, a
    LEAVE_GAP_S apart, and the node LEAVE_PAUSE_S after the last, so the session never
    undeclares many entities at once (rmw_zenoh 0.1.9 stalls delivery graph-wide for that)."""
    if rclpy is None:
        import rclpy
    from rclpy.node import Node
    from rcl_interfaces.srv import GetParameters, ListParameters

    timeout = float(spec.get('timeout', 30))
    rclpy.init()
    node = Node('parity_check_%d' % os.getpid(), enable_rosout=False, start_parameter_services=False)

    def spin_for(secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, end - time.time())))

    def call(srv, name, request):
        client = node.create_client(srv, name)
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError('no %s service within %.0f s' % (name, timeout))
            fut = client.call_async(request)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
            if not fut.done() or fut.result() is None:
                raise RuntimeError('%s did not answer within %.0f s' % (name, timeout))
            return fut.result()
        finally:
            node.destroy_client(client)
            spin_for(LEAVE_GAP_S)

    out = {'nodes': [], 'params': {}, 'info': {}, 'topics': {}}
    try:
        spin_for(float(spec.get('spin', 2.0)))
        out['nodes'] = sorted({(ns.rstrip('/') + '/' + n) for n, ns in node.get_node_names_and_namespaces()})
        for fqn in spec.get('params') or []:
            if fqn not in out['nodes']:
                continue
            try:
                names = call(ListParameters, fqn + '/list_parameters', ListParameters.Request()).result.names
                req = GetParameters.Request()
                req.names = list(names)
                values = call(GetParameters, fqn + '/get_parameters', req).values
                out['params'][fqn] = {'flat': {n: param_value(v) for n, v in zip(names, values)}}
            except RuntimeError as exc:
                out['params'][fqn] = {'error': str(exc)}
        for fqn in spec.get('info') or []:
            if fqn not in out['nodes']:
                continue
            name, ns = split_fqn(fqn)
            out['info'][fqn] = {
                'publishers': {t: ', '.join(ty) for t, ty in node.get_publisher_names_and_types_by_node(name, ns)},
                'subscribers': {t: ', '.join(ty) for t, ty in node.get_subscriber_names_and_types_by_node(name, ns)}}
        for topic in spec.get('topics') or []:
            eps, types = [], []
            for kind, infos in (('PUBLISHER', node.get_publishers_info_by_topic(topic)),
                                ('SUBSCRIPTION', node.get_subscriptions_info_by_topic(topic))):
                for e in infos:
                    types.append(e.topic_type)
                    eps.append({'node': e.node_namespace.rstrip('/') + '/' + e.node_name, 'kind': kind,
                                'reliability': e.qos_profile.reliability.name,
                                'durability': e.qos_profile.durability.name})
            out['topics'][topic] = {'type': types[0] if types else None, 'endpoints': eps}
    finally:
        time.sleep(LEAVE_PAUSE_S)
        node.destroy_node()
        rclpy.try_shutdown()
    return out


def query_budget(spec):
    """Seconds the query session may take: rclpy's start, discovery, two calls per node (each may wait
    up to timeout for its service, then up to timeout for its answer: call()), the leave."""
    n = len(spec.get('params') or [])
    return int(math.ceil(10 + float(spec.get('spin', 2.0)) + n * (4 * float(spec.get('timeout', 30)) + 2 * LEAVE_GAP_S)
                         + LEAVE_PAUSE_S))


def require_sim_session():
    uri = os.environ.get('ZENOH_SESSION_CONFIG_URI')
    if uri != SIM_SESSION_URI:
        die('--local / --query only inside a sim container (ZENOH_SESSION_CONFIG_URI=%r, the sim\'s is %s)'
            % (uri, SIM_SESSION_URI))


class Sim:
    """The query session, run through `docker exec` in one sim container (or here, local)."""

    def __init__(self, timeout, spin, save_dir=None, local=False):
        self.timeout, self.spin, self.save_dir, self.local = timeout, spin, save_dir, local
        self._up = {}

    def running(self, container):
        """Whether the sim container is up. A container of that name from another compose
        project is not the sim's: exit 2 rather than query it."""
        sim_container(container)
        if self.local:
            return True
        if container not in self._up:
            p = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}} {{index .Config.Labels '
                                '"com.docker.compose.project"}}', container],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
            state, _, project = p.stdout.strip().partition(' ')
            if p.returncode == 0 and project != SIM_PROJECT:
                die('refusing to query container %s: it belongs to compose project %r, not %s'
                    % (container, project, SIM_PROJECT))
            self._up[container] = p.returncode == 0 and state == 'true'
        return self._up[container]

    def collect(self, container, spec):
        """query(spec) in the container: {nodes, params, info, topics} or {error}."""
        spec = dict(spec, timeout=self.timeout, spin=self.spin)
        sim_container(container)
        if self.local:
            out = query(spec)
        elif not self.running(container):
            return {'error': 'container %s is not running' % container}
        else:
            budget = query_budget(spec)
            # `timeout` runs inside the container: killing the docker exec client would leave a hung
            # session behind, which keeps the router waiting until its socket closes.
            cmd = SETUP + 'exec timeout -k 2 %d python3 %s --query "$1"' % (budget, QUERY_IN)
            try:
                p = subprocess.run(['docker', 'exec', container, 'bash', '-c', cmd, 'parity', json.dumps(spec)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                                   timeout=budget + 15)
            except subprocess.TimeoutExpired:
                return {'error': 'docker exec did not return'}
            line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith(QUERY_MARK)), None)
            if line is None:
                err = (p.stderr.strip().splitlines() or ['no answer'])[-1]
                return {'error': ('timed out after %ds' % budget if p.returncode in (124, 137) else
                                  'query failed (exit %d): %s' % (p.returncode, err[-200:]))}
            try:
                out = json.loads(line[len(QUERY_MARK):])
            except ValueError as exc:          # another line interleaved with the answer
                return {'error': 'unparsable answer (%s): %s' % (exc, line[:120])}
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, 'query.json'), 'w') as f:
                json.dump(out, f, indent=1, sort_keys=True)
        return out


class FixtureSim(Sim):
    """--selftest: a "sim" that answers from the reference itself, optionally with drift."""

    def __init__(self, ref, drift):
        super().__init__(0, 0)
        self.ref, self.drift, self.nodes, self.edges = ref, drift, ref.nodes(), ref.edges()

    def running(self, container):
        return True

    def collect(self, container, spec):
        by_fqn = {e['sim_fqn']: (n, e) for n, e in self.nodes.items()}
        out = {'nodes': sorted(by_fqn), 'params': {}, 'info': {}, 'topics': {}}
        for fqn in spec['params']:
            name, entry = by_fqn[fqn]
            flat = dict(self.ref.params(entry)[1])
            if entry['params'].get('urdf_key'):
                flat[entry['params']['urdf_key']] = joints_to_urdf(self.ref.tf_joints())
            if name == 'mighty_node':
                flat['num_agents'] = 84  # whitelisted
                if self.drift:
                    flat['horizon'] += 1  # 1 DIFF
            out['params'][fqn] = {'flat': flat}
        for fqn in spec['info']:
            info = {k: dict(v) for k, v in self.edges['nodes'][fqn].items()}
            if self.drift and fqn.endswith('/mpc'):
                info['publishers'].pop('/%s/cmd_vel_auto' % self.ref.ns)  # 1 DIFF
            out['info'][fqn] = info
        for topic in spec['topics']:
            t = self.edges['topics'][topic]
            eps = [dict(ep) for ep in t['endpoints'] if self.ref.node_class(ep['node']) != 'absent']
            if self.drift and topic.endswith('/term_goal'):
                for ep in eps:
                    ep['durability'] = 'TRANSIENT_LOCAL'  # 2 DIFFs (goal_selector pub, mighty_node sub)
            out['topics'][topic] = {'type': t['type'], 'endpoints': eps}
        return out


FIXTURE_DRIFT_DIFFS = 4


def joints_to_urdf(joints):
    robot = ET.Element('robot', {'name': 'red_rover'})
    for name, j in joints.items():
        el = ET.SubElement(robot, 'joint', {'name': name, 'type': 'fixed'})
        ET.SubElement(el, 'parent', {'link': j['parent']})
        ET.SubElement(el, 'child', {'link': j['child']})
        ET.SubElement(el, 'origin', {'xyz': ' '.join(map(repr, j['xyz'])), 'rpy': ' '.join(map(repr, j['rpy']))})
    return ET.tostring(robot, encoding='unicode')


# ---------------------------------------------------------------------------------- checks
class Result:
    def __init__(self, section, name, status, summary='', lines=()):
        self.section, self.name, self.status, self.summary = section, name, status, summary
        self.lines = list(lines)  # [(status, text)]


def worst(lines, ignore=(NOT_RUNNING, NOTE)):
    statuses = [s for s, _ in lines if s not in ignore]
    return max(statuses, key=RANK.get) if statuses else OK


def check_params(ref, name, entry, sim_flat):
    kind, ref_flat = ref.params(entry)
    urdf_key = entry['params'].get('urdf_key')
    lines, undeclared = [], []

    def add(key, text):
        reason = ref.allowed_param(name, key)
        lines.append((ALLOWED, '%s   [allowed: %s]' % (text, reason)) if reason else (DIFF, text))

    for key in sorted(ref_flat):
        if key not in sim_flat and kind != 'dump' and not ref.allowed_param(name, key):
            # rclcpp drops file params a node never declares, and with no dump of the robot's
            # node we cannot tell whether it declared them; the pinned code is the same.
            undeclared.append(key)
        elif key not in sim_flat:
            add(key, '- %s = %s (missing in sim)' % (key, show(ref_flat[key])))
        elif not same(ref_flat[key], sim_flat[key]):
            add(key, '~ %s: ref %s, sim %s' % (key, show(ref_flat[key]), show(sim_flat[key])))
    extra = sorted(set(sim_flat) - set(ref_flat) - {urdf_key})
    if kind == 'dump':  # a dump is the node's whole declared set, so new params are drift too
        for key in extra:
            add(key, '+ %s = %s (not in reference)' % (key, show(sim_flat[key])))
    if urdf_key:
        want = ref.tf_joints()
        have = urdf_joints(sim_flat[urdf_key]) if sim_flat.get(urdf_key) else {}
        for joint in sorted(set(want) | set(have)):
            key = '%s:%s' % (urdf_key, joint)
            if joint not in have:
                add(key, '- joint %s %s -> %s (missing in sim)' % (joint, want[joint]['parent'], want[joint]['child']))
            elif joint not in want:
                add(key, '+ joint %s %s -> %s (not in reference)' % (joint, have[joint]['parent'], have[joint]['child']))
            elif not same_joint(want[joint], have[joint]):
                add(key, '~ joint %s: ref %s, sim %s' % (joint, show(want[joint]), show(have[joint])))
    if undeclared:
        lines.append((NOTE, '. set in the robot param files but not declared by the sim node (rclcpp '
                            'ignores those; no robot dump to compare): %s' % ', '.join(undeclared)))
    n_diff = sum(s == DIFF for s, _ in lines)
    n_allowed = sum(s == ALLOWED for s, _ in lines)
    summary = '%d compared' % (len(ref_flat) - len(undeclared) + (len(ref.tf_joints()) if urdf_key else 0))
    if kind != 'dump':
        summary += ' (reference = robot param files; %d sim defaults not compared)' % len(extra)
    if n_diff or n_allowed:
        summary += ': %d unexpected, %d allowed' % (n_diff, n_allowed)
    return Result('params', name, worst(lines), summary, lines)


def check_node_edges(ref, name, want_edges, have_edges):
    lines = []
    for sect in ('publishers', 'subscribers'):
        want = {t: ty for t, ty in want_edges[sect].items() if t not in GRAPH_TOPICS_IGNORED}
        have = {t: ty for t, ty in have_edges[sect].items() if t not in GRAPH_TOPICS_IGNORED}
        for topic in sorted(set(want) | set(have)):
            if want.get(topic) == have.get(topic):
                continue
            if topic not in have:
                text = '- %s %s [%s] (missing in sim)' % (sect[:-1], topic, want[topic])
            elif topic not in want:
                text = '+ %s %s [%s] (not in reference)' % (sect[:-1], topic, have[topic])
            else:
                text = '~ %s %s: type ref %s, sim %s' % (sect[:-1], topic, want[topic], have[topic])
            reason = ref.allowed_topic(topic)
            lines.append((ALLOWED, '%s   [allowed: %s]' % (text, reason)) if reason else (DIFF, text))
    n = sum(len(v) for k, v in want_edges.items())
    return Result('graph', 'node ' + name, worst(lines), '%d reference pub/sub edges' % n, lines)


def check_topic(ref, topic, spec, considered, scope, running, sim_type, sim_eps):
    """Endpoints of one topic. considered: the reference endpoints in scope; scope: the sim
    nodes being checked (None = all); running: every sim node."""
    lines = []
    reason = ref.allowed_topic(topic)

    def add(text):
        lines.append((ALLOWED, '%s   [allowed: %s]' % (text, reason)) if reason else (DIFF, text))

    if sim_type and spec['type'] != 'unknown' and sim_type != spec['type']:
        add('~ type: ref %s, sim %s' % (spec['type'], sim_type))
    want = {(ep['node'], ep['kind']): ep for ep in considered}
    have = {(ep['node'], ep['kind']): ep for ep in sim_eps}
    for key, ep in sorted(want.items()):
        if ref.node_class(ep['node']) == 'absent':
            lines.append((ALLOWED, '- %s %s   [allowed: not run in the sim]' % (ep['kind'], ep['node'])))
        elif ep['node'] not in running:
            lines.append((NOT_RUNNING, '  %s %s (node not running)' % (ep['kind'], ep['node'])))
        elif key not in have:
            add('- %s %s (missing in sim)' % (ep['kind'], ep['node']))
        else:
            for qos in ('reliability', 'durability'):
                if have[key][qos] != ep[qos]:
                    add('~ %s %s %s: ref %s, sim %s' % (ep['kind'], ep['node'], qos, ep[qos], have[key][qos]))
    known = {(ep['node'], ep['kind']) for ep in spec['endpoints']}
    for key, ep in sorted(have.items()):
        kind = ref.node_class(ep['node'])
        # Other agents share root topics (/trajs), and --nodes narrows the check further.
        if key in known or kind == 'ignore' or not ref.mine(ep['node']) or (scope and ep['node'] not in scope):
            continue
        if kind == 'sim_only':
            lines.append((ALLOWED, '+ %s %s   [allowed: sim-only node]' % (ep['kind'], ep['node'])))
        else:
            add('+ %s %s (not in reference)' % (ep['kind'], ep['node']))
    idle = sum(s == NOT_RUNNING for s, _ in lines)
    summary = '%d reference endpoints' % len(considered) + (', %d of their nodes not running' % idle if idle else '')
    return Result('graph', topic, worst(lines), summary, lines)


def query_container(args, sim, nodes):
    """Where the query session runs: --container / --graph-container, else sim-drive (with the
    status and arm queries, on the cores the non-Gazebo stacks share), else any sim container up."""
    chosen = args.container or args.graph_container
    if chosen:
        return sim_container(chosen)
    order = [DEFAULT_QUERY_CONTAINER] + sorted({e['container'] for e in nodes.values()} - {DEFAULT_QUERY_CONTAINER})
    return next((c for c in order if sim.running(c)), None)


def run_checks(ref, sim, selected, args):
    nodes = ref.nodes()
    containers = sorted({nodes[n]['container'] for n in selected})
    qc = query_container(args, sim, nodes)
    hdr = ['containers: ' + ('not checked (--local)' if sim.local else
                             ', '.join('%s %s' % (c, 'up' if sim.running(c) else 'DOWN') for c in containers))]
    where = 'this process (--local)' if sim.local else qc
    edges = ref.edges() if args.graph else {'nodes': {}, 'topics': {}}
    results, running, graph_ok, snap = [], set(), False, {}
    if qc is None:
        hdr.append('graph: no sim container is running (stack down: make stack-up)')
    else:
        spec = {'params': [nodes[n]['sim_fqn'] for n in selected if args.params and nodes[n].get('params')],
                'info': [nodes[n]['sim_fqn'] for n in selected if nodes[n]['sim_fqn'] in edges['nodes']],
                'topics': sorted(edges['topics'])}
        snap = sim.collect(qc, spec)
        if 'error' in snap:
            # A down router usually shows up as an empty graph instead: rmw_zenoh gives up
            # on the router after one check and the session then sees only itself.
            msg = '%s (is the zenoh router up in sim-drive? scripts/sim.py status)' % snap['error']
            hdr.append('graph (via %s): ERROR %s' % (where, msg))
            results.append(Result('nodes', 'node list', ERROR, msg))
        else:
            running, graph_ok = set(snap['nodes']), True
            hdr.append('graph (via one query session in %s): %d nodes, %d of them %s%s' % (
                where, len(running), sum(ref.mine(n) for n in running), ref.ns,
                '' if running else ' -- empty: no stack nodes up, or no zenoh router (scripts/sim.py status)'))

    for name in selected:
        e = nodes[name]
        if e['sim_fqn'] not in running:
            absent = ref.node_class(e['sim_fqn']) == 'absent'
            why = ('not run in the sim (%s mapping)' % ref.mapping if absent else
                   'container %s is down' % e['container'] if not sim.running(e['container']) else
                   '%s not in the graph' % e['sim_fqn'] if graph_ok else 'unknown: no node list')
            results.append(Result('nodes', name, ALLOWED if absent else NOT_RUNNING, why))
            continue
        results.append(Result('nodes', name, OK, '%s (%s)' % (e['sim_fqn'], e['container'])))
        if args.params and e.get('params'):
            got = snap['params'].get(e['sim_fqn']) or {'error': 'no answer'}
            results.append(check_params(ref, name, e, got['flat']) if 'flat' in got else
                           Result('params', name, ERROR, 'parameters: %s' % got['error']))
        elif args.params:
            results.append(Result('params', name, NO_REF, e.get('note', 'no param reference')))
        if e['sim_fqn'] in edges['nodes']:
            info = snap['info'].get(e['sim_fqn'])
            results.append(check_node_edges(ref, name, edges['nodes'][e['sim_fqn']], info) if info is not None else
                           Result('graph', 'node ' + name, ERROR, 'no edges from the query'))
    scope = {nodes[n]['sim_fqn'] for n in selected}
    everything = set(selected) == set(nodes)
    for topic, spec in edges['topics'].items():
        considered = [ep for ep in spec['endpoints'] if ref.node_class(ep['node']) != 'ignore'
                      and (everything or ep['node'] in scope)]
        if not any(ep['node'] in running and ref.node_class(ep['node']) != 'absent' for ep in considered):
            continue
        got = snap['topics'].get(topic)
        if got is None:
            results.append(Result('graph', topic, ERROR, 'no endpoints from the query'))
            continue
        results.append(check_topic(ref, topic, spec, considered, None if everything else scope, running,
                                   got['type'], got['endpoints']))
    n_topics = len(edges['topics'])
    if args.graph and n_topics:
        checked = sum(1 for r in results if r.section == 'graph' and not r.name.startswith('node '))
        hdr.append('topics: %d of %d reference topics have a running endpoint node' % (checked, n_topics))
    return hdr, results


def report(results, verbose, out=sys.stdout):
    counts = {}
    for section in ('nodes', 'params', 'graph'):
        rows = sorted((r for r in results if r.section == section), key=lambda r: (-RANK[r.status], r.name))
        if not rows:
            continue
        out.write('\n%s\n' % section.upper())
        width = max(len(r.name) for r in rows)
        for r in rows:
            counts[r.status] = counts.get(r.status, 0) + 1
            out.write('  %-*s  %-11s  %s\n' % (width, r.name, r.status, r.summary))
            for status, text in r.lines:
                if status in (DIFF, ERROR, NOTE) or verbose:
                    out.write('      %s\n' % text)
    return counts


def selftest(ref, selected, args):
    """Fixtures vs themselves must pass; with injected drift exactly that drift must fail."""
    args.container = args.graph_container = None
    ok = True
    for drift in (False, True):
        if drift and set(selected) != set(ref.nodes()):
            print('selftest drift:  skipped (needs all nodes)')
            continue
        _, results = run_checks(ref, FixtureSim(ref, drift), selected, args)
        diffs = ['%s %s: %s' % (r.section, r.name, t) for r in results for s, t in r.lines if s == DIFF]
        errors = ['%s %s: %s' % (r.section, r.name, r.summary) for r in results if r.status == ERROR]
        allowed = sum(s == ALLOWED for r in results for s, _ in r.lines)
        compared = sum(r.section == 'params' and r.status != NO_REF for r in results)
        want = FIXTURE_DRIFT_DIFFS if drift else 0
        good = not errors and len(diffs) == want and compared > 0
        if args.verbose:
            report(results, True)
        print('selftest %-6s %s: %d param sets and %d graph checks; %d unexpected (want %d), %d allowed, %d errors'
              % ('drift:' if drift else 'clean:', 'PASS' if good else 'FAIL', compared,
                 sum(r.section == 'graph' for r in results), len(diffs), want, allowed, len(errors)))
        for line in sorted(diffs) + errors:
            print('    ' + line)
        ok = ok and good
    return 0 if ok else 1


def session_mapping(ns, run_dir=None):
    """(mapping mode, note) of the running session (sim.py's <run>/session.json), or (None, why)."""
    run_dir = run_dir or os.environ.get('ROVER_SIM_RUN') or RUN_DIR_DEFAULT
    path = os.path.join(run_dir, 'session.json')
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        return None, 'no session.json (%s)' % exc
    mode = data.get('mapping')
    if mode not in ('gpu', 'fallback'):
        return None, '%s has no mapping mode' % path
    agents = [a.get('name') for a in data.get('agents') or []]
    note = 'session %s (%s)' % (data.get('name'), path)
    if ns not in agents:
        note += '; %s is not one of its agents (%s)' % (ns, ', '.join(map(str, agents)))
    return mode, note


def query_main(arg):
    """--query: run query() on the JSON spec and print its answer (inside a sim container)."""
    require_sim_session()
    out = query(json.loads(arg))
    print(QUERY_MARK + json.dumps(out, sort_keys=True), flush=True)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['--query']:
        return query_main(argv[1] if len(argv) > 1 else '{}')
    doc_head, doc_rest = __doc__.split('\n', 1)
    p = argparse.ArgumentParser(description=doc_head, epilog=doc_rest,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ns', help='sim agent namespace, e.g. JR81')
    p.add_argument('--ref', default='JR02', help='reference robot, a directory under tools/reference/ (default JR02)')
    p.add_argument('--nodes', action='append', metavar='LIST', help='only these nodes (comma-separated, repeatable)')
    p.add_argument('--mapping', choices=('gpu', 'fallback'), default=None,
                   help="the session's mapping mode; fallback allows its documented non-parity keys (default: the "
                        "running session's, else $ROVER_SIM_MAPPING, else gpu)")
    p.add_argument('--run-dir', help='scripts/sim.py\'s run dir (default $ROVER_SIM_RUN or %s)' % RUN_DIR_DEFAULT)
    p.add_argument('--no-params', dest='params', action='store_false', help='skip the param dumps')
    p.add_argument('--no-graph', dest='graph', action='store_false', help='skip the graph edges')
    p.add_argument('--require-running', action='store_true', help='fail when a reference node is not running')
    p.add_argument('--container', help='the sim container the query session runs in (default sim-drive, else any up)')
    p.add_argument('--graph-container', help='the same as --container (kept for older command lines)')
    p.add_argument('--local', action='store_true', help='run the query session in this process (in a sim container)')
    p.add_argument('--timeout', type=int, default=30, help='seconds per parameter service call (default 30)')
    p.add_argument('--spin-time', type=float, default=2.0, help='graph discovery time before the queries (default 2)')
    p.add_argument('--jobs', type=int, default=1, help='ignored: one query session asks everything in turn')
    p.add_argument('--save', metavar='DIR', help='also write the query session\'s answer under DIR')
    p.add_argument('-v', '--verbose', action='store_true', help='also print allowed and not-running details')
    p.add_argument('--show-ref', metavar='NODE', help='print the reference params of NODE, renamed to <ns>, and exit')
    p.add_argument('--selftest', action='store_true', help='check the checker on the fixtures alone and exit')
    args = p.parse_args(argv)

    mapping_note = 'given'
    if args.mapping is None:
        args.mapping, mapping_note = session_mapping(args.ns, args.run_dir)
        if args.mapping is None:
            env = os.environ.get('ROVER_SIM_MAPPING')
            args.mapping = env if env in ('gpu', 'fallback') else 'gpu'
            mapping_note += '; %s' % ('$ROVER_SIM_MAPPING' if env in ('gpu', 'fallback') else 'default')
    ref = Reference(args.ref, args.ns, args.mapping)
    nodes = ref.nodes()
    if args.show_ref:
        e = nodes.get(args.show_ref)
        if e is None or not e.get('params'):
            die('no param reference for %r (have: %s)'
                % (args.show_ref, ', '.join(n for n, v in nodes.items() if v.get('params'))))
        kind, flat = ref.params(e)
        print('# %s reference (%s) for %s, from %s' % (args.ref, kind, e['sim_fqn'], ', '.join(e['params']['files'])))
        sys.stdout.write(yaml.safe_dump({e['sim_fqn']: {'ros__parameters': flat}}, default_flow_style=False))
        return 0
    selected = [n for arg in args.nodes for n in arg.split(',') if n] if args.nodes else list(nodes)
    unknown = [n for n in selected if n not in nodes]
    if unknown:
        p.error('unknown node(s) %s; reference nodes: %s' % (', '.join(unknown), ', '.join(nodes)))
    if not args.selftest:
        for c in sorted({args.container, args.graph_container} - {None} | {e['container'] for e in nodes.values()}):
            sim_container(c)
        if args.local:
            require_sim_session()
            if not os.path.isdir(os.path.join(REPO, 'tools', 'reference')):
                die('--local needs this checkout\'s tools/reference/, which the sim containers do not mount '
                    '(only scripts/): run parity_check.py on the host, which queries from a sim container')
    for rel in ref.files(nodes[n] for n in selected):
        ref.path(rel)  # fetch private-repo files now, not halfway through the checks
    if args.selftest:
        return selftest(ref, selected, args)

    print('parity_check: %s (class %s, %s mapping: %s) vs %s (%s)'
          % (args.ns, ref.cls, args.mapping, mapping_note, args.ref, os.path.relpath(ref.dir, REPO)))
    hdr, results = run_checks(ref, Sim(args.timeout, args.spin_time, args.save, local=args.local), selected, args)
    for line in hdr:
        print('  ' + line)
    counts = report(results, args.verbose)
    bad = counts.get(DIFF, 0) + counts.get(ERROR, 0) + (counts.get(NOT_RUNNING, 0) if args.require_running else 0)
    print('\nsummary: %s -> %s' % (', '.join('%d %s' % (counts[s], s) for s in sorted(counts, key=lambda s: -RANK[s])),
                                   'FAIL' if bad else 'ok'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
