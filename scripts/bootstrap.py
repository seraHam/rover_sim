#!/usr/bin/env python3
"""Set up rover_sim on a new machine: check it, build the images, start an example.

  scripts/bootstrap.py                the checks, then the images, then a dry run of `smoke`
  scripts/bootstrap.py --run          ... then start `smoke`: JR81 and RR83 swap places in the arena
  scripts/bootstrap.py --check        the checks only (it fetches the pinned sources, builds nothing)
                       [--gpu | --no-gpu]  with or without the elevation image (the gpu sessions);
                                           default: whether docker here can use an NVIDIA GPU
                       [--parallel N]      build N images at once (build_images.sh --parallel)

  1. Host: Python 3.8+ with PyYAML, git, tmux, docker reachable without sudo with its compose and
     buildx plugins, and the ssh key build_images.sh forwards into the builds (an agent holding
     it, or SSH_KEY=<key>). Reported, not required: CPUs, free disk, an NVIDIA GPU, xhost.
  2. Read access (`git ls-remote`) to the stack repos (docker/stacks.env) and the gazebo image's
     sources (rover_sim.sim.repos).
  3. scripts/build_images.sh --prepare-only --force, one unit at a time: each stack repo at its
     pin into .cache/, with what is cloned on the host from there (dlio.repos, rover's
     acl_ctrl_safety submodule).
  4. Read access to every repo those pins clone, on the host or INSIDE a docker build (mighty's
     HW_DEPS in mighty.repos, Dockerfile.mapping's *_REPO), read from the pinned files: a missing
     grant shows here, all of them at once, not part-way through the build.
  5. scripts/build_images.sh: every image, or without a GPU all but elevation (smoke and odom_eval
     run on `mapping: fallback`; multi_agent_swap and office_explore need the GPU).
  6. scripts/sim.py up smoke --dry-run, or with --run `up smoke`.
1 and 2 run together; each later step runs only when everything before it passed. Running it
again is cheap: build_images.sh skips an image whose inputs have not changed.

Exit 0: done. 1: a check failed (its FAIL line says what to do). 2: a build or the session failed.
Pure parts (the repo lists, git's error messages) import without docker; tests/test_bootstrap.py
covers them.
"""

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import sim  # noqa: E402  CPU_BUDGET_MIN
try:
    import yaml
    import check_pins  # noqa: E402  read_env, dockerfile_args, read
except ImportError:  # no PyYAML: host_checks() says how to get it
    yaml = check_pins = None

REPO = os.path.dirname(HERE)
CACHE = os.path.join(REPO, '.cache')
BUILD_IMAGES = os.path.join(HERE, 'build_images.sh')
OK, WARN, FAIL, INFO = 'OK', 'WARN', 'FAIL', 'INFO'
# The build_images.sh units whose repo it fetches on the host, and where it keeps that checkout
# (gazebo and elevation build from this checkout).
FETCHED = (('mighty', 'MIGHTY_SHA', 'src/mighty@%s'), ('mapping', 'MAPPING_SHA', 'src/mapping_ws@%s'),
           ('dlio', 'DLIO_SHA', 'src/dlio_ws@%s'), ('drive', 'DRIVE_SHA', 'ws/rover@%s/src/rover'))
NO_GPU_UNITS = ['mighty', 'mapping', 'dlio', 'drive', 'gazebo']   # every unit but elevation
ELEVATION_REPOS = ('ELEVATION_REPO', 'ROS2_NUMPY_REPO')            # cloned for elevation only
SUBMODULE = 'acl_ctrl_safety'   # the one submodule build_images.sh initialises in rover
EXAMPLE = 'smoke'
# Free space docker's root should have: a cold build measured 24 GB of images (about 16 without
# elevation) and 16 GB of BuildKit cache.
DISK_GB = {True: 40, False: 30}


def run(cmd, timeout=60, **kw):
    """CompletedProcess (text), or None when the program is not installed."""
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=timeout, **kw)
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, '', 'no answer in %d s' % timeout)


def last_line(p):
    return ((p.stderr or p.stdout or '').strip().splitlines() or ['exit %d' % p.returncode])[-1]


# ---------- the repo lists -----------------------------------------------------------------

def repos_urls(text):
    """{name: url} of a vcs .repos file."""
    data = yaml.safe_load(text) or {}
    return dict((k, str((v or {}).get('url', ''))) for k, v in (data.get('repositories') or {}).items())


def stack_repos(pins, sim_repos, elevation):
    """[(where, url)] build_images.sh clones itself or has the gazebo build clone."""
    out = [('stacks.env %s' % k, v) for k, v in pins.items()
           if k.endswith('_REPO') and (elevation or k not in ELEVATION_REPOS)]
    return out + [('rover_sim.sim.repos %s' % n, u) for n, u in sim_repos.items()]


def resolve_url(url, base):
    """A .gitmodules url; a relative one (./x, ../x) is relative to the superproject's url."""
    if not url.startswith(('./', '../')):
        return url
    head = base.rstrip('/')
    for part in url.split('/'):
        if part == '..':
            head = head.rsplit('/', 1)[0] if '/' in head.split(':', 1)[-1] else head.split(':', 1)[0] + ':'
        elif part not in ('.', ''):
            head += part if head.endswith(':') else '/' + part
    return head


def submodule_url(checkout, path, super_url):
    """The url .gitmodules gives the submodule at <path>, or None."""
    p = run(['git', 'config', '-f', os.path.join(checkout, '.gitmodules'), '--get-regexp',
             r'^submodule\..*\.(path|url)$'])
    entries = {}
    for line in (p.stdout if p and p.returncode == 0 else '').splitlines():
        key, _, value = line.partition(' ')
        name, _, field = key[len('submodule.'):].rpartition('.')
        entries.setdefault(name, {})[field] = value
    for e in entries.values():
        if e.get('path') == path and e.get('url'):
            return resolve_url(e['url'], super_url)
    return None


def nested_repos(pins, cache=CACHE):
    """[(where, url)] the pinned checkouts clone further, on the host or inside a docker build, and
    [(status, what, detail)] for what could not be read."""
    out, problems, co = [], [], {}
    for unit, key, rel in FETCHED:
        d = os.path.join(cache, rel % pins.get(key, ''))
        if os.path.isdir(d):
            co[unit] = d
        else:
            problems.append((INFO, unit, 'not checked: no checkout at %s' % os.path.relpath(d, REPO)))
    if 'mighty' in co:
        urls = repos_urls(check_pins.read(os.path.join(co['mighty'], 'mighty.repos')))
        hw = check_pins.read(os.path.join(co['mighty'], 'docker', 'Dockerfile.hw'))
        for name in check_pins.dockerfile_args(hw).get('HW_DEPS', '').split():
            if name in urls:
                out.append(('mighty build (HW_DEPS %s)' % name, urls[name]))
            else:
                problems.append((FAIL, name, 'in mighty Dockerfile.hw HW_DEPS but not in mighty.repos'))
    if 'mapping' in co:
        args = check_pins.dockerfile_args(check_pins.read(os.path.join(co['mapping'], 'docker',
                                                                       'Dockerfile.mapping')))
        out += [('mapping build (%s)' % k, v) for k, v in args.items() if k.endswith('_REPO')]
    if 'dlio' in co:
        urls = repos_urls(check_pins.read(os.path.join(co['dlio'], 'dlio.repos')))
        out += [('dlio_ws dlio.repos %s' % n, u) for n, u in urls.items()]
    if 'drive' in co:
        url = submodule_url(co['drive'], SUBMODULE, pins.get('DRIVE_REPO', ''))
        if url:
            out.append(('rover submodule %s' % SUBMODULE, url))
        else:
            problems.append((FAIL, SUBMODULE, "not a submodule in rover's .gitmodules"))
    return out, problems


def dedupe(pairs, skip=()):
    """[(url, 'where, where')] in first-seen order, leaving out the urls in skip."""
    order, wheres = [], {}
    for where, url in pairs:
        if url in skip:
            continue
        if url not in wheres:
            order.append(url)
            wheres[url] = []
        if where not in wheres[url]:
            wheres[url].append(where)
    return [(u, ', '.join(wheres[u])) for u in order]


# ---------- access -------------------------------------------------------------------------

def url_host(url):
    """github.com from git@github.com:a/b.git, https://github.com/a/b or ssh://git@host/a."""
    m = re.match(r'^[a-z+]+://(?:[^@/]+@)?([^/:]+)', url) or re.match(r'^(?:[^@/]+@)?([^/:]+):', url)
    return m.group(1) if m else url


def access_hint(url, stderr):
    """What to do about a failed `git ls-remote`, from git's and ssh's messages."""
    text = stderr.strip()
    host = url_host(url)
    if 'Host key verification failed' in text:
        return ("ssh does not know %s yet: run `ssh -T git@%s` once and accept its key (compare it with "
                "the host's published fingerprints)" % (host, host))
    if 'Permission denied (publickey' in text:
        return 'your ssh key is not on your %s account: add its public key there' % host
    low = text.lower()
    if 'not found' in low or 'could not be found' in low or 'terminal prompts disabled' in low:
        return 'no read access: ask for read access to %s' % url
    return (text.splitlines() or ['git ls-remote failed'])[-1]


def git_env(env=None):
    """build_images.sh's: host-side git fails rather than prompts, and uses SSH_KEY if set."""
    e = dict(os.environ if env is None else env)
    key = e.get('SSH_KEY')
    e.setdefault('GIT_SSH_COMMAND', 'ssh -o BatchMode=yes' + (' -i %s -o IdentitiesOnly=yes' % key if key else ''))
    e['GIT_TERMINAL_PROMPT'] = '0'
    return e


def ls_remote(url, env, timeout=60):
    """(readable, what to do if not)."""
    p = run(['git', 'ls-remote', '-q', url], timeout=timeout, env=env)
    if p is None:
        return False, 'git is not installed'
    return p.returncode == 0, '' if p.returncode == 0 else access_hint(url, p.stderr)


def access(repos, env):
    """[(status, url, detail)]: `git ls-remote` of each (url, where), several at once."""
    def one(item):
        url, where = item
        ok, why = ls_remote(url, env)
        return (OK, url, where) if ok else (FAIL, url, '%s: %s' % (where, why))
    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(one, repos))


# ---------- the host -----------------------------------------------------------------------

def ssh_key_check(env=None):
    """[(status, what, detail)] for the key build_images.sh forwards into the builds (--ssh)."""
    env = os.environ if env is None else env
    key = env.get('SSH_KEY')
    if key:
        ok = os.access(key, os.R_OK)
        return [(OK if ok else FAIL, 'ssh key', 'SSH_KEY=%s%s' % (key, '' if ok else ': not readable'))]
    sock = env.get('SSH_AUTH_SOCK', '')
    try:
        agent = bool(sock) and stat.S_ISSOCK(os.stat(sock).st_mode)
    except OSError:
        agent = False
    keys = key_hint(env.get('HOME', ''))
    if not agent:
        return [(FAIL, 'ssh key', 'no ssh agent: eval "$(ssh-agent -s)" && ssh-add (loads %s), or run with '
                                  'SSH_KEY=<the one on your GitHub/GitLab accounts> (the builds clone private '
                                  'repos with it)' % keys)]
    p = run(['ssh-add', '-l'], env=env)
    if p is not None and p.returncode == 0:
        return [(OK, 'ssh key', 'the ssh agent holds %d key(s)' % len(p.stdout.splitlines()))]
    return [(FAIL, 'ssh key', 'the ssh agent holds no key: ssh-add (loads %s), or run with SSH_KEY=<the one '
                              'on your GitHub/GitLab accounts>' % keys)]


def key_hint(home):
    """The default private keys in ~/.ssh, which a bare `ssh-add` loads, for the hints. Which of them
    GitHub/GitLab know is not checked here: step 2 shows it."""
    names = [n for n in ('id_ed25519', 'id_ecdsa', 'id_rsa') if home and os.path.isfile(os.path.join(home, '.ssh', n))]
    return ', '.join('~/.ssh/' + n for n in names) if names else 'your default keys; there are none in ~/.ssh'


def gpu_check(docker_ok):
    """(usable, detail): an NVIDIA GPU that docker can give a container."""
    p = run(['nvidia-smi', '-L'])
    if p is None or p.returncode != 0:
        return False, 'none found (nvidia-smi)'
    n = len([l for l in p.stdout.splitlines() if l.startswith('GPU ')])
    if not docker_ok:
        return False, '%d GPU(s); docker is not reachable to check its nvidia runtime' % n
    q = run(['docker', 'info', '--format', '{{json .Runtimes}}'])
    if q is None or 'nvidia' not in q.stdout:
        return False, '%d GPU(s), but docker has no nvidia runtime (install the NVIDIA Container Toolkit)' % n
    return True, '%d GPU(s), docker has the nvidia runtime' % n


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def host_checks(gpu_choice=None):
    """[(status, what, detail)], and whether to build elevation (gpu_choice: --gpu/--no-gpu)."""
    out = []

    def add(status, what, detail):
        out.append((status, what, detail))

    v = sys.version_info[:2]
    old = v < (3, 8)
    add(FAIL if old else OK, 'python3', '%d.%d%s' % (v[0], v[1], ': needs 3.8 or newer' if old else ''))
    add(OK if check_pins else FAIL, 'PyYAML',
        'importable' if check_pins else 'install it: sudo apt install python3-yaml (or pip install pyyaml)')
    for tool in ('git', 'tmux'):
        add(OK if shutil.which(tool) else FAIL, tool, shutil.which(tool) or 'install it: sudo apt install %s' % tool)
    docker_ok = False
    p = run(['docker', 'version', '--format', '{{.Server.Version}}'])
    if p is None:
        add(FAIL, 'docker', 'not installed: https://docs.docker.com/engine/install/')
    elif p.returncode != 0:
        msg = last_line(p)
        if 'permission denied' in msg.lower():
            msg = 'no access to the daemon: sudo usermod -aG docker $USER, then log out and back in'
        add(FAIL, 'docker', msg)
    else:
        docker_ok = True
        add(OK, 'docker', 'engine %s' % p.stdout.strip())
        for plugin in ('compose', 'buildx'):
            q = run(['docker', plugin, 'version'])
            ok = q is not None and q.returncode == 0
            add(OK if ok else FAIL, 'docker %s' % plugin,
                q.stdout.strip().splitlines()[0] if ok and q.stdout.strip() else
                'install the docker-%s-plugin package' % plugin)
    out.extend(ssh_key_check())

    ncpu = len(os.sched_getaffinity(0))
    few = ncpu < sim.CPU_BUDGET_MIN
    add(WARN if few else OK, 'CPUs', '%d%s' % (ncpu, ': under %d, sim.py runs the containers unpinned and physics '
                                               'can fall behind real time' % sim.CPU_BUDGET_MIN if few else ''))
    usable, detail = gpu_check(docker_ok)
    gpu = usable if gpu_choice is None else gpu_choice
    add(INFO, 'NVIDIA GPU', detail)
    forced = '' if gpu_choice is None else (' (--gpu)' if gpu_choice else ' (--no-gpu)')
    add(INFO, 'images', ('all of them, elevation too (the gpu sessions)' if gpu else
                         'all but elevation: smoke and odom_eval run, multi_agent_swap and office_explore '
                         'need the GPU') + forced)
    if docker_ok:
        q = run(['docker', 'info', '--format', '{{.DockerRootDir}}'])
        root = q.stdout.strip() if q is not None and q.returncode == 0 else ''
        try:
            gb = free_gb(root or '/')
            add(OK if gb >= DISK_GB[gpu] else WARN, 'disk', '%.0f GB free under %s%s' % (
                gb, root or '/', '' if gb >= DISK_GB[gpu] else
                ': the images and their build cache need about %d GB' % DISK_GB[gpu]))
        except OSError as e:
            add(WARN, 'disk', 'cannot read the free space under %s: %s' % (root, e))
    add(OK if shutil.which('xhost') else WARN, 'xhost', shutil.which('xhost') or
        'not installed (x11-xserver-utils): the GUI sessions (gui, rviz, sim.py viz) need it')
    return out, gpu


# ---------- the steps ----------------------------------------------------------------------

def failed_url(text):
    """The repo a failed build_images.sh fetch was cloning: a submodule git names, else its last fetch."""
    m = re.search(r"clone of '([^']+)' into submodule", text) or re.search(r"repository '([^']+)' not found", text)
    if m:
        return m.group(1)
    fetches = re.findall(r'^\[images\] fetch (\S+) @ ', text, re.M)
    return fetches[-1] if fetches else None


def fetch_pins():
    """build_images.sh --prepare-only --force, one fetched unit at a time: [(status, unit, detail)]."""
    out = []
    for unit, _, _ in FETCHED:
        p = run([BUILD_IMAGES, '--prepare-only', '--force', unit], timeout=1800)
        if p is None:
            return [(FAIL, unit, '%s is not executable' % BUILD_IMAGES)]
        text = p.stdout + p.stderr
        if p.returncode == 0:
            n = len(re.findall(r'^\[images\] fetch ', text, re.M))
            out.append((OK, unit, 'at its pin%s' % (', fetched %d repo(s)' % n if n else '')))
        else:
            url = failed_url(text)
            out.append((FAIL, unit, '%s: %s' % (url, access_hint(url, text)) if url else last_line(p)))
    return out


def example(start):
    """Dry-run the example session, or start it: [(status, what, detail)]."""
    cmd = [sys.executable, os.path.join(HERE, 'sim.py'), 'up', EXAMPLE]
    if start:
        rc = subprocess.call(cmd)
        return [(OK if rc == 0 else FAIL, EXAMPLE, 'running' if rc == 0 else 'sim.py up exited %d (above)' % rc)]
    p = run(cmd + ['--dry-run'], timeout=300)
    if p is None or p.returncode != 0:
        return [(FAIL, EXAMPLE, 'dry run failed: %s' % (last_line(p) if p else 'no python'))]
    panes = len(re.findall(r'^ {4}\[[^\]]+\] ', p.stdout, re.M))
    waves = len(re.findall(r'^wave \d+:', p.stdout, re.M))
    return [(OK, '%s dry run' % EXAMPLE, '%d panes in %d waves; nothing started' % (panes, waves))]


def report(results):
    width = max([len(w) for _, w, _ in results] or [0])
    for status, what, detail in results:
        print('  %-4s %-*s  %s' % (status, width, what, detail))
    return any(s == FAIL for s, _, _ in results)


def stop(why):
    print('bootstrap: %s; fix the FAIL lines above, then run scripts/bootstrap.py again' % why, file=sys.stderr)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--gpu', dest='gpu', action='store_true', help='build the elevation image even without a GPU')
    g.add_argument('--no-gpu', dest='gpu', action='store_false', help='do not build the elevation image')
    m = ap.add_mutually_exclusive_group()
    m.add_argument('--check', action='store_true', help='the checks only: build nothing')
    m.add_argument('--run', action='store_true', help='start the %s session after the build' % EXAMPLE)
    ap.add_argument('--parallel', type=int, metavar='N', help='build N images at once')
    ap.set_defaults(gpu=None)
    args = ap.parse_args(argv)
    if args.parallel is not None and args.parallel < 1:
        ap.error('--parallel needs a positive number')
    if hasattr(sys.stdout, 'reconfigure'):  # in order with stderr and the builds' output, also in a pipe
        sys.stdout.reconfigure(line_buffering=True)

    print('1/6 host: tools, docker, the ssh key the builds use')
    host, gpu = host_checks(args.gpu)
    bad = report(host)
    if check_pins is None or shutil.which('git') is None:
        return stop('PyYAML and git are needed to read the pins and check access')
    pins = check_pins.read_env(os.path.join(REPO, 'docker', 'stacks.env'))
    sim_repos = repos_urls(check_pins.read(os.path.join(REPO, 'rover_sim.sim.repos')))
    env = git_env()
    print('2/6 read access: the stack repos and the gazebo image\'s sources')
    first = dedupe(stack_repos(pins, sim_repos, gpu))
    bad = report(access(first, env)) or bad
    if bad:
        return stop('the host is not ready')

    print('3/6 the pinned sources, into .cache/ (build_images.sh --prepare-only)')
    bad = report(fetch_pins())
    print('4/6 read access: what the pins clone, on the host or inside a docker build')
    nested, problems = nested_repos(pins)
    bad = report(problems + access(dedupe(nested, skip=set(u for u, _ in first)), env)) or bad
    if bad:
        return stop('some sources cannot be read')
    units = [] if gpu else NO_GPU_UNITS
    if args.check:
        print('bootstrap: the checks passed (--check: nothing built; it would build %s)'
              % (' '.join(units) or 'every image'))
        return 0

    print('5/6 the images: scripts/build_images.sh %s' % ' '.join(units or ['(every image)']))
    cmd = [BUILD_IMAGES] + units + (['--parallel', str(args.parallel)] if args.parallel else [])
    if subprocess.call(cmd) != 0:
        print('bootstrap: the build failed (above). Run scripts/bootstrap.py again to resume: the images '
              'already built are kept', file=sys.stderr)
        return 2
    print('6/6 the example: %s (JR81 and RR83 swap places in the arena)' % EXAMPLE)
    if report(example(args.run)):
        return 2
    if args.run:
        print('bootstrap: %s is running.\n'
              '  scripts/sim.py status                         what runs: containers, panes, sim time, DLIO\n'
              '  scripts/sim.py viz                            the live 2D view (needs a display)\n'
              '  scripts/demo_eval.py swap --duration 120      grade the swap from ground truth\n'
              '  scripts/sim.py down                           stop it (make down also removes the containers)'
              % EXAMPLE)
    else:
        print('bootstrap: ready. Start the example with `scripts/sim.py up %s` (or scripts/bootstrap.py --run)'
              % EXAMPLE)
    return 0


if __name__ == '__main__':
    sys.exit(main())
