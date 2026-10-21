"""Unit tests for scripts/bootstrap.py: the repo lists read from the pinned checkouts, git's error
messages, and the order of the steps (no docker, no network: those parts are mocked).

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('bootstrap', str(REPO / 'scripts' / 'bootstrap.py'))
bs = importlib.util.module_from_spec(_spec)
sys.modules['bootstrap'] = bs
_spec.loader.exec_module(bs)

PINS = {'MIGHTY_REPO': 'git@github.com:a/mighty.git', 'MIGHTY_SHA': '1' * 40,
        'MAPPING_REPO': 'git@gitlab.com:g/mapping_ws.git', 'MAPPING_SHA': '2' * 40,
        'DLIO_REPO': 'git@github.com:a/dlio_ws.git', 'DLIO_SHA': '3' * 40,
        'DRIVE_REPO': 'git@gitlab.com:g/r/rover.git', 'DRIVE_SHA': '4' * 40,
        'ELEVATION_REPO': 'https://github.com/a/eme.git', 'ROS2_NUMPY_REPO': 'https://github.com/b/np.git'}


def repos(**urls):
    return 'repositories:\n' + ''.join('  %s:\n    type: git\n    url: %s\n    version: %s\n' % (k, u, 'a' * 40)
                                        for k, u in urls.items())


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def fake_cache(d, gitmodules=True):
    """A .cache with the four checkouts build_images.sh fetches, as bootstrap reads them."""
    c = Path(d)
    m = c / 'src' / ('mighty@' + '1' * 40)
    write(m / 'mighty.repos', repos(mpc='git@gitlab.com:g/mpc.git', dyn='https://github.com/k/dyn.git',
                                    unused='https://github.com/k/unused.git'))
    write(m / 'docker' / 'Dockerfile.hw', 'ARG WS=/ws\nARG HW_DEPS="dyn mpc"\nFROM x\n')
    write(c / 'src' / ('mapping_ws@' + '2' * 40) / 'docker' / 'Dockerfile.mapping',
          'ARG ACL_MAPPING_REPO=git@gitlab.com:g/acl-mapping.git\nARG ACL_MAPPING_REF=abc\n'
          'ARG DYNUS_INTERFACES_REPO=https://github.com/k/dyn.git\nARG ACL_MAPPING_REPO\n')
    write(c / 'src' / ('dlio_ws@' + '3' * 40) / 'dlio.repos', repos(dlio='git@github.com:a/dlio.git'))
    rover = c / 'ws' / ('rover@' + '4' * 40) / 'src' / 'rover'
    if gitmodules:
        write(rover / '.gitmodules', '[submodule "safety"]\n\tpath = acl_ctrl_safety\n\turl = ../../safety.git\n'
                                     '[submodule "other"]\n\tpath = other\n\turl = git@gitlab.com:g/other.git\n')
    else:
        rover.mkdir(parents=True)
    return str(c)


class RepoLists(unittest.TestCase):
    def test_repos_urls(self):
        self.assertEqual(bs.repos_urls(repos(x='u1', y='u2')), {'x': 'u1', 'y': 'u2'})
        self.assertEqual(bs.repos_urls(''), {})

    def test_stack_repos(self):
        sim_repos = {'gazebo_ros_pkgs': 'https://github.com/m/g.git'}
        without = bs.stack_repos(PINS, sim_repos, elevation=False)
        self.assertEqual([w for w, _ in without], ['stacks.env MIGHTY_REPO', 'stacks.env MAPPING_REPO',
                                                   'stacks.env DLIO_REPO', 'stacks.env DRIVE_REPO',
                                                   'rover_sim.sim.repos gazebo_ros_pkgs'])
        with_elevation = [w for w, _ in bs.stack_repos(PINS, sim_repos, elevation=True)]
        self.assertIn('stacks.env ELEVATION_REPO', with_elevation)
        self.assertIn('stacks.env ROS2_NUMPY_REPO', with_elevation)

    def test_resolve_url(self):
        base = 'git@gitlab.com:mit-acl/ugv/redrover/rover.git'
        self.assertEqual(bs.resolve_url('git@x:y.git', base), 'git@x:y.git')
        self.assertEqual(bs.resolve_url('../safety.git', base), 'git@gitlab.com:mit-acl/ugv/redrover/safety.git')
        self.assertEqual(bs.resolve_url('../../safety.git', base), 'git@gitlab.com:mit-acl/ugv/safety.git')
        self.assertEqual(bs.resolve_url('./sub', 'https://h/a/super'), 'https://h/a/super/sub')
        self.assertEqual(bs.resolve_url('../x.git', 'git@h:super.git'), 'git@h:x.git')

    @unittest.skipIf(shutil.which('git') is None, 'needs git (git config -f)')
    def test_nested_repos(self):
        with tempfile.TemporaryDirectory() as d:
            out, problems = bs.nested_repos(PINS, fake_cache(d))
        self.assertEqual(problems, [])
        self.assertEqual(out, [
            ('mighty build (HW_DEPS dyn)', 'https://github.com/k/dyn.git'),
            ('mighty build (HW_DEPS mpc)', 'git@gitlab.com:g/mpc.git'),
            ('mapping build (ACL_MAPPING_REPO)', 'git@gitlab.com:g/acl-mapping.git'),
            ('mapping build (DYNUS_INTERFACES_REPO)', 'https://github.com/k/dyn.git'),
            ('dlio_ws dlio.repos dlio', 'git@github.com:a/dlio.git'),
            ('rover submodule acl_ctrl_safety', 'git@gitlab.com:g/safety.git')])
        # Merged by url, and without what step 2 already checked.
        self.assertEqual(bs.dedupe(out, skip={'git@github.com:a/dlio.git'})[0],
                         ('https://github.com/k/dyn.git', 'mighty build (HW_DEPS dyn), mapping build (DYNUS_INTERFACES_REPO)'))
        self.assertEqual(len(bs.dedupe(out, skip={'git@github.com:a/dlio.git'})), 4)

    @unittest.skipIf(shutil.which('git') is None, 'needs git (git config -f)')
    def test_nested_repos_problems(self):
        with tempfile.TemporaryDirectory() as d:
            cache = fake_cache(d, gitmodules=False)
            shutil.rmtree(os.path.join(cache, 'src', 'dlio_ws@' + '3' * 40))
            hw = Path(cache, 'src', 'mighty@' + '1' * 40, 'docker', 'Dockerfile.hw')
            hw.write_text('ARG HW_DEPS="mpc typo"\n')
            out, problems = bs.nested_repos(PINS, cache)
        self.assertEqual([(s, w) for s, w, _ in problems],
                         [(bs.INFO, 'dlio'), (bs.FAIL, 'typo'), (bs.FAIL, 'acl_ctrl_safety')])
        self.assertIn('not checked', problems[0][2])
        self.assertNotIn('dlio_ws dlio.repos dlio', [w for w, _ in out])


class Messages(unittest.TestCase):
    def test_url_host(self):
        for url in ('git@github.com:a/b.git', 'https://github.com/a/b.git', 'ssh://git@github.com:22/a/b',
                    'https://user@github.com/a'):
            self.assertEqual(bs.url_host(url), 'github.com', url)

    def test_access_hint(self):
        url = 'git@gitlab.com:g/mpc.git'
        hk = 'Host key verification failed.\nfatal: Could not read from remote repository.\n\nPlease make sure you have the correct access rights'
        self.assertIn('ssh -T git@gitlab.com', bs.access_hint(url, hk))
        self.assertIn('not on your gitlab.com account', bs.access_hint(url, 'git@gitlab.com: Permission denied (publickey).\nfatal: Could not read'))
        gitlab = 'remote: ERROR: The project you were looking for could not be found or you don\'t have permission to view it.'
        github = 'ERROR: Repository not found.\nfatal: Could not read from remote repository.'
        https = "fatal: could not read Username for 'https://github.com': terminal prompts disabled"
        for text in (gitlab, github, https):
            self.assertEqual(bs.access_hint(url, text), 'no read access: ask for read access to %s' % url)
        self.assertEqual(bs.access_hint(url, 'fatal: unable to access x\nssh: connect to host: timed out'),
                         'ssh: connect to host: timed out')

    def test_failed_url(self):
        sub = "[images] fetch git@x:rover.git @ 1234567\nfatal: clone of 'git@x:safety.git' into submodule path '/c/s' failed"
        self.assertEqual(bs.failed_url(sub), 'git@x:safety.git')
        self.assertEqual(bs.failed_url("[images] fetch https://h/a.git @ 1\nfatal: repository 'https://h/a.git' not found"),
                         'https://h/a.git')
        self.assertEqual(bs.failed_url('[images] fetch git@x:a.git @ 1\n[images] fetch git@x:b.git @ 2\nERROR: x'),
                         'git@x:b.git')
        self.assertIsNone(bs.failed_url('[images] error: another build_images.sh is running'))

    def test_git_env(self):
        e = bs.git_env({'PATH': '/bin'})
        self.assertEqual((e['GIT_SSH_COMMAND'], e['GIT_TERMINAL_PROMPT']), ('ssh -o BatchMode=yes', '0'))
        self.assertIn('-i /k -o IdentitiesOnly=yes', bs.git_env({'SSH_KEY': '/k'})['GIT_SSH_COMMAND'])
        self.assertEqual(bs.git_env({'GIT_SSH_COMMAND': 'mine'})['GIT_SSH_COMMAND'], 'mine')

    def test_ssh_key_check(self):
        with tempfile.NamedTemporaryFile() as f:
            self.assertEqual(bs.ssh_key_check({'SSH_KEY': f.name})[0][0], bs.OK)
        self.assertEqual(bs.ssh_key_check({'SSH_KEY': '/nonexistent/key'})[0][0], bs.FAIL)
        status, _, detail = bs.ssh_key_check({'SSH_AUTH_SOCK': '/nonexistent/sock'})[0]
        self.assertEqual(status, bs.FAIL)
        self.assertIn('no ssh agent', detail)
        with tempfile.TemporaryDirectory() as home:
            self.assertIn('there are none in ~/.ssh', bs.ssh_key_check({'HOME': home})[0][2])
            write(Path(home, '.ssh', 'id_rsa'), 'k')
            write(Path(home, '.ssh', 'id_ed25519'), 'k')
            self.assertIn('ssh-add (loads ~/.ssh/id_ed25519, ~/.ssh/id_rsa)', bs.ssh_key_check({'HOME': home})[0][2])


class Steps(unittest.TestCase):
    def run_main(self, argv, host_status=bs.OK, access_status=bs.OK, fetch_status=bs.OK, build_rc=0):
        calls = mock.MagicMock(return_value=build_rc)
        with mock.patch.object(bs, 'host_checks', return_value=([(host_status, 'docker', 'x')], False)), \
                mock.patch.object(bs, 'access', side_effect=lambda repos, env: [(access_status, u, w) for u, w in repos]), \
                mock.patch.object(bs, 'fetch_pins', return_value=[(fetch_status, 'mighty', 'x')]) as fetch, \
                mock.patch.object(bs, 'nested_repos', return_value=([('mighty build (HW_DEPS mpc)', 'git@x:mpc.git')], [])), \
                mock.patch.object(bs, 'example', return_value=[(bs.OK, 'smoke dry run', 'x')]) as example, \
                mock.patch.object(bs.subprocess, 'call', calls), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            rc = bs.main(argv)
        return rc, fetch, calls, example, out.getvalue()

    def test_host_failure_stops_before_fetching(self):
        rc, fetch, calls, _, _ = self.run_main([], host_status=bs.FAIL)
        self.assertEqual(rc, 1)
        fetch.assert_not_called()
        calls.assert_not_called()

    def test_access_failure_stops_before_building(self):
        rc, fetch, calls, _, _ = self.run_main([], access_status=bs.FAIL)
        self.assertEqual(rc, 1)
        fetch.assert_not_called()   # step 2 fails first
        calls.assert_not_called()

    def test_check_builds_nothing(self):
        rc, fetch, calls, example, out = self.run_main(['--check'])
        self.assertEqual(rc, 0)
        fetch.assert_called_once()
        calls.assert_not_called()
        example.assert_not_called()
        self.assertIn('git@x:mpc.git', out)                      # step 4 checked the nested repo
        self.assertIn('mighty mapping dlio drive gazebo', out)   # no GPU: all but elevation

    def test_build_then_example(self):
        rc, _, calls, example, _ = self.run_main(['--parallel', '2'])
        self.assertEqual(rc, 0)
        self.assertEqual(calls.call_args[0][0][1:], bs.NO_GPU_UNITS + ['--parallel', '2'])
        example.assert_called_once_with(False)

    def test_build_failure(self):
        rc, _, _, example, _ = self.run_main([], build_rc=1)
        self.assertEqual(rc, 2)
        example.assert_not_called()


if __name__ == '__main__':
    unittest.main()
