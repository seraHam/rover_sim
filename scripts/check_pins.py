#!/usr/bin/env python3
"""Check rover_sim's pins against each other and, read-only, against a robot.

  scripts/check_pins.py                       pins vs the pinned repos' own pins (offline)
  scripts/check_pins.py --robot <ssh target>  ... and vs what the robot's containers run
                      [--orin <ssh target>]   ... and the Orin's elevation_mapping_cupy checkout

Offline, from docker/stacks.env, rover_sim.sim.repos and build_images.sh's checkouts of each
pin (.cache/src/<repo>@<sha>; scripts/build_images.sh --prepare-only fetches them):
  - rover_sim.sim.repos takes DecompROS2, livox_ros_driver2 and Livox-SDK2 at mighty.repos'
    commits (the planner's RViz plugins and the Livox message types must match the robot's);
  - ELEVATION_SHA and ROS2_NUMPY_SHA equal mapping_ws orin/orin.repos (what the Orin runs);
  - dynus_interfaces is the same commit in mighty.repos and mapping_ws docker/Dockerfile.mapping
    (one message definition across the stacks);
  - it reports DLIO's commit (dlio_ws dlio.repos) and the mapping image's acl-mapping and
    map_fusion commits. They are part of the pins, not checked against anything else.
The Gazebo plugin repos are expected to differ from mighty.repos: rover_sim.sim.repos takes
the mit-acl copies with the sim-stamp fixes.

--robot runs `docker ps` and `docker image inspect` on the robot over ssh (BatchMode, nothing
else: no ros2 CLI, no writes) and compares each running stack's image revision (the
org.opencontainers.image.revision label, else a commit in its tag) with the pin. --orin runs
`git rev-parse HEAD` in the Orin's elevation_mapping_cupy checkout. A commit read from a tag is
a hint, not proof: an image's own label is.

Exit 0: every check OK. 1: a difference. 2: a setup error (no checkout, ssh failed).
Pure parts (the .repos and Dockerfile parsing, the comparison) import without ssh;
tests/test_check_pins.py covers them.
"""

import argparse
import json
import os
import re
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CACHE = os.path.join(REPO, '.cache', 'src')
SHA_RE = re.compile(r'(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])')
OK, DIFF, INFO, MISSING = 'OK', 'DIFF', 'INFO', 'MISSING'
# Which robot container runs which stack (tools/reference/jr02/processes/docker_top_*.txt).
ROBOT_CONTAINERS = (('hw-mighty', 'MIGHTY_SHA'), ('mapping', 'MAPPING_SHA'), ('dlio', 'DLIO_SHA'),
                    ('drive', 'DRIVE_SHA'))
ORIN_CHECKOUT = '~/ws/elevation_mapping_ws/src/elevation_mapping_cupy'   # mapping_ws orin/orin.repos


class SetupError(Exception):
    pass


def read_env(path):
    """docker/stacks.env: plain KEY=VALUE lines (its own format rules)."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                out[k] = v
    return out


def repos_versions(text):
    """{name: version} of a vcs .repos file."""
    data = yaml.safe_load(text) or {}
    return dict((k, str((v or {}).get('version', ''))) for k, v in (data.get('repositories') or {}).items())


def dockerfile_args(text):
    """{ARG name: default} of a Dockerfile's `ARG NAME=value` and `ARG NAME="a b"` lines."""
    return dict((m.group(1), m.group(2) if m.group(2) is not None else m.group(3))
                for m in re.finditer(r'^ARG\s+(\w+)=(?:"([^"]*)"|(\S+))', text, re.M))


def same(a, b):
    """Two commit ids name the same commit: one is a prefix of the other (7+ hex)."""
    a, b = (a or '').lower(), (b or '').lower()
    return len(a) >= 7 and len(b) >= 7 and (a.startswith(b) or b.startswith(a))


def checkout(name, sha, cache=CACHE):
    path = os.path.join(cache, '%s@%s' % (name, sha))
    if not os.path.isdir(path):
        raise SetupError('no checkout of %s@%s in %s (scripts/build_images.sh --prepare-only fetches it)'
                         % (name, sha[:7], cache))
    return path


def read(path):
    with open(path) as f:
        return f.read()


def offline_checks(pins, sim_repos, cache=CACHE):
    """[(status, what, detail)] of the cross-pin checks."""
    out = []
    mighty = repos_versions(read(os.path.join(checkout('mighty', pins['MIGHTY_SHA'], cache), 'mighty.repos')))
    mapping = checkout('mapping_ws', pins['MAPPING_SHA'], cache)
    orin = repos_versions(read(os.path.join(mapping, 'orin', 'orin.repos')))
    mapping_args = dockerfile_args(read(os.path.join(mapping, 'docker', 'Dockerfile.mapping')))
    dlio = repos_versions(read(os.path.join(checkout('dlio_ws', pins['DLIO_SHA'], cache), 'dlio.repos')))

    def cmp(what, ours, theirs, where):
        out.append((OK if same(ours, theirs) else DIFF, what, '%s, %s %s' % (ours[:7] or '-', where, theirs[:7] or '-')))

    for name in ('DecompROS2', 'livox_ros_driver2', 'Livox-SDK2'):
        cmp(name, sim_repos.get(name, ''), mighty.get(name, ''), 'mighty.repos')
    cmp('elevation_mapping_cupy', pins.get('ELEVATION_SHA', ''), orin.get('elevation_mapping_cupy', ''),
        'orin.repos')
    cmp('ros2_numpy', pins.get('ROS2_NUMPY_SHA', ''), orin.get('ros2_numpy', ''), 'orin.repos')
    cmp('dynus_interfaces (mighty.repos vs Dockerfile.mapping)', mighty.get('dynus_interfaces', ''),
        mapping_args.get('DYNUS_INTERFACES_REF', ''), 'Dockerfile.mapping')
    for name in ('gazebo_ros_pkgs', 'livox_laser_simulation_ros2'):
        out.append((INFO, name, 'sim %s (mit-acl, sim-stamp fixes), mighty.repos %s: expected to differ'
                    % (sim_repos.get(name, '')[:7] or '-', mighty.get(name, '')[:7] or '-')))
    out.append((INFO, 'DLIO', 'dlio_ws dlio.repos %s' % dlio.get('direct_lidar_inertial_odometry', '')[:7]))
    out.append((INFO, 'mapping image', 'acl-mapping %s, map_fusion %s' % (
        mapping_args.get('ACL_MAPPING_REF', '')[:7], mapping_args.get('MAP_FUSION_REF', '')[:7])))
    return out


def image_revision(tags, labels):
    """(commit, source) an image says it was built from: its revision label, else a commit in a tag."""
    for key in ('org.opencontainers.image.revision', 'rover_sim.revision'):
        if (labels or {}).get(key):
            return labels[key], 'label'
    for t in tags or ():
        m = SHA_RE.findall(t.rsplit(':', 1)[-1])
        if m:
            return m[-1], 'tag %s' % t
    return None, None


def robot_checks(pins, ps_text, inspect):
    """[(status, what, detail)]: each stack's running container (`docker ps` lines 'name<TAB>image')
    against its pin. inspect: {image: (tags, labels)}."""
    running = dict(l.split('\t', 1) for l in ps_text.splitlines() if '\t' in l)
    out = []
    for container, key in ROBOT_CONTAINERS:
        image = running.get(container)
        if image is None:
            out.append((MISSING, container, 'not running on the robot'))
            continue
        tags, labels = inspect.get(image, ((), {}))
        rev, how = image_revision(tags, labels)
        pin = pins.get(key, '')
        if rev is None:
            out.append((MISSING, container, '%s carries no revision (label or tag); pin %s' % (image, pin[:7])))
        else:
            out.append((OK if same(rev, pin) else DIFF, container,
                        'robot %s (%s), pin %s %s' % (rev[:7], how, key, pin[:7])))
    return out


def ssh(target, command, timeout=30):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', target, command],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)
    if p.returncode != 0:
        raise SetupError('ssh %s: %s' % (target, (p.stderr.strip().splitlines() or ['exit %d' % p.returncode])[-1]))
    return p.stdout


def read_robot(target):
    """(docker ps text, {image: (tags, labels)}) over ssh: docker ps and docker image inspect only."""
    ps = ssh(target, "docker ps --format '{{.Names}}\t{{.Image}}'")
    images = sorted(set(l.split('\t', 1)[1] for l in ps.splitlines() if '\t' in l))
    inspect = {}
    if images:
        raw = ssh(target, 'docker image inspect --format "{{json .RepoTags}}\t{{json .Config.Labels}}" '
                  + ' '.join("'%s'" % i for i in images))
        for image, line in zip(images, raw.splitlines()):
            tags, labels = line.split('\t', 1)
            inspect[image] = (json.loads(tags) or [], json.loads(labels) or {})
    return ps, inspect


def report(results):
    for status, what, detail in results:
        print('  %-8s %-48s %s' % (status, what, detail))
    return any(s == DIFF for s, _, _ in results)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--robot', metavar='SSH_TARGET', help="the robot's ssh target (e.g. a ~/.ssh/config Host)")
    ap.add_argument('--orin', metavar='SSH_TARGET', help="the robot's Orin's ssh target")
    args = ap.parse_args(argv)
    try:
        pins = read_env(os.path.join(REPO, 'docker', 'stacks.env'))
        sim_repos = repos_versions(read(os.path.join(REPO, 'rover_sim.sim.repos')))
        print('pins against each other (docker/stacks.env, rover_sim.sim.repos, the pinned repos):')
        bad = report(offline_checks(pins, sim_repos, CACHE))
        if args.robot:
            print('pins against %s (docker ps, image labels):' % args.robot)
            bad = report(robot_checks(pins, *read_robot(args.robot))) or bad
        if args.orin:
            head = ssh(args.orin, 'git -C %s rev-parse HEAD' % ORIN_CHECKOUT).strip()
            print('pins against %s (elevation_mapping_cupy checkout):' % args.orin)
            bad = report([(OK if same(head, pins.get('ELEVATION_SHA', '')) else DIFF, 'elevation_mapping_cupy',
                           'orin %s, pin ELEVATION_SHA %s' % (head[:7], pins.get('ELEVATION_SHA', '')[:7]))]) or bad
    except (SetupError, OSError, KeyError, subprocess.TimeoutExpired, ValueError) as e:
        print('check_pins: %s' % e, file=sys.stderr)
        return 2
    print('check_pins: %s' % ('DIFFERENCES found' if bad else 'ok'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
