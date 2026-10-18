#!/usr/bin/env python3
"""Diffs one sim agent's ROS params and graph edges against a robot reference.

Usage: parity_check.py <ns> [--ref JR02] [--nodes mighty_node,mpc,...] [--mapping gpu|fallback]
       parity_check.py <ns> --show-ref <node>   print the reference params, renamed to <ns>
       parity_check.py <ns> --selftest          check the checker on the fixtures alone (no docker)

Runs on the host (python3.8 + PyYAML + docker CLI). For each reference node that is up, it
runs `ros2 param dump` and `ros2 node info` (all with --no-daemon) inside that node's sim
container, and `ros2 topic info -v` for every reference topic one of them touches, then diffs
the results against tools/reference/<ref>/ (see its README.md) with the robot's name (JR02)
replaced by <ns>. What the sim changes on purpose is listed, with the reason, in
tools/reference/<ref>/allowed_deviations.yaml and shows up as ALLOWED.

Nodes that are not running (or whose container is down) are reported as NOT RUNNING and do
not fail the check unless --require-running is given, so it can be run while the stacks are
brought up one phase at a time.

Exit status: 0 no unexpected differences; 1 unexpected differences, failed queries (ERROR),
or, with --require-running, nodes not running; 2 bad usage or reference.

It only ever docker-execs into the sim's own containers (sim-* of compose project
rover-sim-stack, the names scripts/sim.py registers); any other --container, --graph-container
or manifest container is refused with exit 2. The fleet's containers share this host, and a
ros2 CLI call in one of them (acl-c2-*) would query the live robots. Never point this at a
robot: the reference was captured without any ros2 CLI call.

Reference files from private repos are not copied into tools/reference/ (this repo is public).
manifest.yaml names each by repo, commit and path; they are read from the pinned checkouts
that scripts/build_images.sh keeps under .cache/, or fetched into .cache/reference/ (needs
the GitLab/GitHub key the image builds use), and refused unless their content matches the
hash of the robot's copy.
"""
import argparse
import concurrent.futures
import fnmatch
import glob
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, '.cache')
NAME_RE = re.compile(r'^([A-Z]{2})([0-9]{2})$')
# The sim's containers (scripts/sim.py CONTAINERS, compose project rover-sim-stack). Nothing
# else is ever exec'd into: acl-c2-* on this host are attached to the live fleet graph.
SIM_CONTAINERS = ('sim-gazebo', 'sim-drive', 'sim-dlio-jr', 'sim-dlio-rr', 'sim-mapping',
                  'sim-elevation', 'sim-mighty')
SIM_PROJECT = 'rover-sim-stack'
# Introspection needs no workspace message types, so whichever install the image has will
# do (mighty_ws, mapping_ws, dlio_ws, rover_ws or rover_sim_ws), else plain Humble.
SETUP = ('for f in /home/swarm/code/*/install/setup.bash /opt/ros/humble/setup.bash; do '
         '[ -f "$f" ] && . "$f" && break; done; export ROS_LOG_DIR=/tmp/rover_sim_parity_log; ')
# Every node has these; they say nothing about parity.
GRAPH_TOPICS_IGNORED = ('/parameter_events', '/rosout')
# zenoh's Rust logger writes to stdout, in colour, interleaved with the CLI's own output.
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
ZENOH_LOG_RE = re.compile(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z\s+(?:TRACE|DEBUG|INFO|WARN|ERROR)\s')

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


# ---------------------------------------------------------------------------------- sim access
def parse_node_list(text):
    return {line.strip() for line in text.splitlines() if line.startswith('/')}


def parse_node_info(text):
    """`ros2 node info` -> {'publishers': {topic: type}, 'subscribers': {topic: type}}."""
    out, sect = {'publishers': {}, 'subscribers': {}}, None
    for line in text.splitlines():
        m = re.match(r'^  (\w[\w ]*):\s*$', line)
        if m:
            sect = {'Subscribers': 'subscribers', 'Publishers': 'publishers'}.get(m.group(1))
            continue
        m = re.match(r'^    (\S+): (.+)$', line)
        if m and sect:
            out[sect][m.group(1)] = m.group(2).strip()
    return out


def parse_topic_info(text):
    """`ros2 topic info -v` -> (type, [{node, kind, reliability, durability}])."""
    t = re.search(r'^Type: (\S+)', text, re.M)
    eps = []
    for block in re.split(r'^(?=Node name:)', text, flags=re.M)[1:]:
        f = dict(re.findall(r'^\s*([A-Za-z][A-Za-z ()]*?):[ \t]*(.*?)[ \t]*$', block, re.M))
        ns = f.get('Node namespace', '/')
        eps.append({'node': ns.rstrip('/') + '/' + f.get('Node name', '?'), 'kind': f.get('Endpoint type', '?'),
                    'reliability': f.get('Reliability', '?'), 'durability': f.get('Durability', '?')})
    return (t.group(1) if t else None), eps


class Sim:
    """ros2 CLI introspection through `docker exec` into the sim containers."""

    def __init__(self, timeout, spin, save_dir=None):
        self.timeout, self.spin, self.save_dir = timeout, spin, save_dir
        self._up = {}

    def running(self, container):
        """Whether the sim container is up. A container of that name from another compose
        project is not the sim's: exit 2 rather than query it."""
        if container not in self._up:
            p = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}} {{index .Config.Labels '
                                '"com.docker.compose.project"}}', sim_container(container)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
            state, _, project = p.stdout.strip().partition(' ')
            if p.returncode == 0 and project != SIM_PROJECT:
                die('refusing to query container %s: it belongs to compose project %r, not %s'
                    % (container, project, SIM_PROJECT))
            self._up[container] = p.returncode == 0 and state == 'true'
        return self._up[container]

    def ros2(self, container, args, save_as):
        """(ok, stdout, stderr). `timeout` runs inside the container because killing the
        docker exec client would leave a hung CLI behind (param dump has hung under zenoh)."""
        if not self.running(container):  # also checks it is the sim's (exit 2 if not)
            return False, '', 'container %s is not running' % container
        args = args[:2] + ['--no-daemon', '--spin-time', str(self.spin)] + args[2:]
        cmd = SETUP + 'exec timeout -k 2 %d ros2 %s' % (self.timeout, ' '.join(map(shlex.quote, args)))
        try:
            p = subprocess.run(['docker', 'exec', sim_container(container), 'bash', '-c', cmd], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True, timeout=self.timeout + 15)
        except subprocess.TimeoutExpired:
            return False, '', 'docker exec did not return'
        if self.save_dir:
            path = os.path.join(self.save_dir, save_as)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(p.stdout)  # raw, logs included
        out = '\n'.join(l for l in ANSI_RE.sub('', p.stdout).splitlines() if not ZENOH_LOG_RE.match(l))
        err = p.stderr.strip()
        if p.returncode in (124, 137):
            err = 'timed out after %ds%s' % (self.timeout, ': ' + err[-200:] if err else '')
        return p.returncode == 0, out + '\n', err

    @staticmethod
    def _file(fqn):
        return fqn.strip('/').replace('/', '__')

    def _call(self, container, args, save_as):
        ok, out, err = self.ros2(container, args, save_as)
        if not ok:
            raise RuntimeError('ros2 %s: %s' % (' '.join(args[:2]), (err or 'failed').splitlines()[-1]))
        return out

    def node_list(self, container):
        return parse_node_list(self._call(container, ['node', 'list'], 'node_list.txt'))

    def param_dump(self, container, fqn):
        return dump_params(self._call(container, ['param', 'dump', fqn], 'params/%s.yaml' % self._file(fqn)))[1]

    def node_info(self, container, fqn):
        return parse_node_info(self._call(container, ['node', 'info', fqn], 'node_info/%s.txt' % self._file(fqn)))

    def topic_info(self, container, topic):
        ok, out, err = self.ros2(container, ['topic', 'info', '-v', topic], 'topic_info/%s.txt' % self._file(topic))
        if not ok and 'Unknown topic' in out + err:
            return None, []  # nobody has it: every live reference endpoint is missing
        if not ok:
            raise RuntimeError('ros2 topic info: %s' % (err or 'failed').splitlines()[-1])
        return parse_topic_info(out)


class FixtureSim(Sim):
    """--selftest: a "sim" that answers from the reference itself, optionally with drift."""

    def __init__(self, ref, drift):
        super().__init__(0, 0)
        self.ref, self.drift, self.nodes, self.edges = ref, drift, ref.nodes(), ref.edges()

    def running(self, container):
        return True

    def node_list(self, container):
        return {e['sim_fqn'] for e in self.nodes.values()}

    def param_dump(self, container, fqn):
        name, entry = next((n, e) for n, e in self.nodes.items() if e['sim_fqn'] == fqn)
        flat = dict(self.ref.params(entry)[1])
        if entry['params'].get('urdf_key'):
            flat[entry['params']['urdf_key']] = joints_to_urdf(self.ref.tf_joints())
        if name == 'mighty_node':
            flat['num_agents'] = 84  # whitelisted
            if self.drift:
                flat['horizon'] += 1  # 1 DIFF
        return flat

    def node_info(self, container, fqn):
        info = {k: dict(v) for k, v in self.edges['nodes'][fqn].items()}
        if self.drift and fqn.endswith('/mpc'):
            info['publishers'].pop('/%s/cmd_vel_auto' % self.ref.ns)  # 1 DIFF
        return info

    def topic_info(self, container, topic):
        spec = self.edges['topics'][topic]
        eps = [dict(ep) for ep in spec['endpoints'] if self.ref.node_class(ep['node']) != 'absent']
        if self.drift and topic.endswith('/term_goal'):
            for ep in eps:
                ep['durability'] = 'TRANSIENT_LOCAL'  # 2 DIFFs (goal_selector pub, mighty_node sub)
        return spec['type'], eps


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


def run_checks(ref, sim, selected, args):
    nodes = ref.nodes()
    if args.container:
        for e in nodes.values():
            e['container'] = args.container
    containers = sorted({nodes[n]['container'] for n in selected})
    graph_c = args.graph_container or next(
        (c for c in ['sim-drive'] + sorted({e['container'] for e in nodes.values()}) if sim.running(c)), None)
    hdr = ['containers: ' + ', '.join('%s %s' % (c, 'up' if sim.running(c) else 'DOWN') for c in containers)]
    results, running, graph_ok = [], set(), False
    if graph_c is None:
        hdr.append('graph: no sim container is running (stack down: make stack-up)')
    else:
        try:
            running, graph_ok = sim.node_list(graph_c), True
            hdr.append('graph (via %s): %d nodes, %d of them %s%s' % (
                graph_c, len(running), sum(ref.mine(n) for n in running), ref.ns,
                '' if running else ' -- empty: no stack nodes up, or no zenoh router (scripts/sim.py status)'))
        except RuntimeError as exc:
            # A down router usually shows up as an empty graph instead: rmw_zenoh gives up
            # on the router after one check and the CLI then sees only itself.
            msg = '%s (is the zenoh router up in sim-drive? scripts/sim.py status)' % exc
            hdr.append('graph (via %s): ERROR %s' % (graph_c, msg))
            results.append(Result('nodes', 'node list', ERROR, msg))
    edges = ref.edges() if args.graph else {'nodes': {}, 'topics': {}}

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs))
    jobs = {}
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
            jobs[pool.submit(sim.param_dump, e['container'], e['sim_fqn'])] = ('params', name, e)
        elif args.params:
            results.append(Result('params', name, NO_REF, e.get('note', 'no param reference')))
        if e['sim_fqn'] in edges['nodes']:
            jobs[pool.submit(sim.node_info, e['container'], e['sim_fqn'])] = ('node', name, edges['nodes'][e['sim_fqn']])
    scope = {nodes[n]['sim_fqn'] for n in selected}
    everything = set(selected) == set(nodes)
    for topic, spec in edges['topics'].items():
        considered = [ep for ep in spec['endpoints'] if ref.node_class(ep['node']) != 'ignore'
                      and (everything or ep['node'] in scope)]
        if any(ep['node'] in running and ref.node_class(ep['node']) != 'absent' for ep in considered):
            jobs[pool.submit(sim.topic_info, graph_c, topic)] = ('topic', topic, (spec, considered))
    scope = None if everything else scope
    for fut in concurrent.futures.as_completed(jobs):
        kind, name, extra = jobs[fut]
        try:
            value = fut.result()
        except Exception as exc:  # report it and keep checking the rest
            label = name if kind != 'node' else 'node ' + name
            results.append(Result('params' if kind == 'params' else 'graph', label, ERROR, str(exc)))
            continue
        if kind == 'params':
            results.append(check_params(ref, name, extra, value))
        elif kind == 'node':
            results.append(check_node_edges(ref, name, extra, value))
        else:
            results.append(check_topic(ref, name, extra[0], extra[1], scope, running, *value))
    pool.shutdown()
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
    args.container, args.graph_container = None, 'fixture'
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


def main(argv=None):
    doc_head, doc_rest = __doc__.split('\n', 1)
    p = argparse.ArgumentParser(description=doc_head, epilog=doc_rest,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ns', help='sim agent namespace, e.g. JR81')
    p.add_argument('--ref', default='JR02', help='reference robot, a directory under tools/reference/ (default JR02)')
    p.add_argument('--nodes', action='append', metavar='LIST', help='only these nodes (comma-separated, repeatable)')
    p.add_argument('--mapping', choices=('gpu', 'fallback'), default=os.environ.get('ROVER_SIM_MAPPING', 'gpu'),
                   help="the session's mapping mode; fallback allows its documented non-parity keys (default gpu)")
    p.add_argument('--no-params', dest='params', action='store_false', help='skip the param dumps')
    p.add_argument('--no-graph', dest='graph', action='store_false', help='skip the graph edges')
    p.add_argument('--require-running', action='store_true', help='fail when a reference node is not running')
    p.add_argument('--container', help='run every query in this one container instead of per node')
    p.add_argument('--graph-container', help='container for node list / topic info (default sim-drive, else any up)')
    p.add_argument('--timeout', type=int, default=30, help='seconds per ros2 CLI call (default 30)')
    p.add_argument('--spin-time', type=float, default=2.0, help='ros2 CLI discovery spin time (default 2)')
    p.add_argument('--jobs', type=int, default=6, help='queries in parallel (default 6)')
    p.add_argument('--save', metavar='DIR', help='also write the raw sim outputs under DIR')
    p.add_argument('-v', '--verbose', action='store_true', help='also print allowed and not-running details')
    p.add_argument('--show-ref', metavar='NODE', help='print the reference params of NODE, renamed to <ns>, and exit')
    p.add_argument('--selftest', action='store_true', help='check the checker on the fixtures alone and exit')
    args = p.parse_args(argv)

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
    for rel in ref.files(nodes[n] for n in selected):
        ref.path(rel)  # fetch private-repo files now, not halfway through the checks
    if args.selftest:
        return selftest(ref, selected, args)

    print('parity_check: %s (class %s, %s mapping) vs %s (%s)'
          % (args.ns, ref.cls, args.mapping, args.ref, os.path.relpath(ref.dir, REPO)))
    hdr, results = run_checks(ref, Sim(args.timeout, args.spin_time, args.save), selected, args)
    for line in hdr:
        print('  ' + line)
    counts = report(results, args.verbose)
    bad = counts.get(DIFF, 0) + counts.get(ERROR, 0) + (counts.get(NOT_RUNNING, 0) if args.require_running else 0)
    print('\nsummary: %s -> %s' % (', '.join('%d %s' % (counts[s], s) for s in sorted(counts, key=lambda s: -RANK[s])),
                                   'FAIL' if bad else 'ok'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
