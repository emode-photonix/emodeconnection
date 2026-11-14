"""``emode-install``: the build variant, and installing without write access.

Nothing here touches the network or the real install location. The release
lookup, the EULA, the login and the download are stubbed, the download writes
a small zip shaped like the Linux release (``emode`` and ``install.sh``),
``subprocess.run`` is a recorder, and the install directories live under
``tmp_path``. The OS is forced to Linux (or macOS and Windows for the variant
errors) so the tests run the same on any machine.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pytest

import emode.installer as installer

# The PyPI lookup, before the `downloads` fixture stubs it out.
_LATEST_CLIENT = installer.latest_client

BINARY = b'#!/bin/sh\necho emode\n'
VERSION = '1.0.7'


class _Recorder:
    """Stands in for ``subprocess.run``: records each command, runs none."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[list[str]] = []
        # What the source of each `sudo install` held when it ran, since the
        # temporary file is gone by the time the test looks.
        self.installed: list[bytes] = []

    def __call__(self, cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        self.calls.append([str(c) for c in cmd])
        if cmd[:2] == ['sudo', 'install']:
            self.installed.append(Path(cmd[-2]).read_bytes())
        return subprocess.CompletedProcess(cmd, self.returncode)

    def sudo_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == 'sudo']


class _Stdin:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def _write_release(path: Path) -> None:
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('emode', BINARY)
        zf.writestr('install.sh', '#!/bin/bash\n')


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub every network step, and record what each download asked for."""
    asked: list[dict[str, Any]] = []
    versions = {
        'latest': VERSION,
        'versions': {VERSION: ['Linux', 'Linux-rslab', 'Mac', 'Mac-x86', 'Windows']},
    }
    monkeypatch.setattr(installer, 'get_versions', lambda: versions)
    monkeypatch.setattr(installer, 'handle_eula', lambda: True)
    monkeypatch.setattr(installer, 'prompt_credentials', lambda: ('a@b.c', 'pw'))
    monkeypatch.setattr(installer, 'login', lambda email, password: 'token')
    # No PyPI lookup unless a test asks for one.
    monkeypatch.setattr(installer, 'latest_client', lambda: None)

    def download_release(
        token: str, platform_key: str, version: str, filename: str, dest_dir: Path
    ) -> Path:
        asked.append({'platform_key': platform_key, 'version': version, 'filename': filename})
        dest = dest_dir / filename
        _write_release(dest)
        return dest

    monkeypatch.setattr(installer, 'download_release', download_release)
    return asked


@pytest.fixture
def linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """Linux, with a default install directory and a home under tmp_path."""
    monkeypatch.setattr(installer.platform, 'system', lambda: 'Linux')
    system_dir = tmp_path / 'usr' / 'local' / 'bin'
    system_dir.mkdir(parents=True)
    monkeypatch.setitem(installer.DEFAULT_INSTALL_DIRS, 'Linux', system_dir)
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    monkeypatch.setattr(installer.shutil, 'which', lambda name: f'/usr/bin/{name}')
    return {'system': system_dir, 'home': home, 'user': home / '.local' / 'bin'}


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(installer.subprocess, 'run', recorder)
    return recorder


@pytest.fixture
def read_only(linux: dict[str, Path]) -> Iterator[Path]:
    """The default install directory, without write access."""
    if os.geteuid() == 0:
        pytest.skip('root can write to a read-only directory')
    system_dir = linux['system']
    system_dir.chmod(0o555)
    yield system_dir
    system_dir.chmod(0o755)


def _tty(monkeypatch: pytest.MonkeyPatch, tty: bool) -> None:
    monkeypatch.setattr(installer.sys, 'stdin', _Stdin(tty))


def _is_executable(path: Path) -> bool:
    mode = path.stat().st_mode
    return bool(mode & stat.S_IXUSR and mode & stat.S_IXGRP and mode & stat.S_IXOTH)


# ---------------------------------------------------------------
# --variant
# ---------------------------------------------------------------


def test_the_default_variant_is_unchanged(
    linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder
) -> None:
    installer.install()
    assert downloads == [
        {'platform_key': 'linux', 'version': VERSION, 'filename': f'EMode-{VERSION}-Linux.zip'}
    ]


def test_the_rslab_variant_downloads_the_slim_linux_build(
    linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder
) -> None:
    installer.install(variant='rslab')
    assert downloads == [
        {
            'platform_key': 'linux-rslab',
            'version': VERSION,
            'filename': f'EMode-{VERSION}-Linux-rslab.zip',
        }
    ]
    assert (linux['system'] / 'emode').read_bytes() == BINARY


def test_the_rslab_variant_must_be_listed_for_the_version(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    versions = {'latest': VERSION, 'versions': {VERSION: ['Linux', 'Mac', 'Windows']}}
    monkeypatch.setattr(installer, 'get_versions', lambda: versions)
    with pytest.raises(SystemExit) as exc:
        installer.install(variant='rslab')
    assert exc.value.code == 1
    assert 'Linux-rslab' in capsys.readouterr().out
    assert downloads == []


@pytest.mark.parametrize('system', ['Darwin', 'Windows'])
def test_the_rslab_variant_is_linux_only(
    system: str,
    downloads: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(installer.platform, 'system', lambda: system)
    with pytest.raises(SystemExit) as exc:
        installer.install(variant='rslab')
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert '--variant rslab' in out
    assert 'Linux' in out
    assert downloads == []


def test_the_cli_takes_the_variant(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, 'argv', ['emode-install', '--variant', 'rslab'])
    installer.main()
    assert downloads[0]['platform_key'] == 'linux-rslab'


def test_the_cli_rejects_an_unknown_variant(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, 'argv', ['emode-install', '--variant', 'arm64'])
    with pytest.raises(SystemExit) as exc:
        installer.main()
    assert exc.value.code == 2
    assert 'rslab' in capsys.readouterr().err


def test_the_help_describes_the_variant(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, 'argv', ['emode-install', '--help'])
    with pytest.raises(SystemExit):
        installer.main()
    out = ' '.join(capsys.readouterr().out.split())
    assert '--variant {default,rslab}' in out
    assert 'without MKL' in out


# ---------------------------------------------------------------
# --sha256
# ---------------------------------------------------------------


def _release_digest(tmp_path: Path) -> str:
    """The SHA-256 of the release zip the stubbed download writes."""
    path = tmp_path / 'reference.zip'
    _write_release(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_digests(monkeypatch: pytest.MonkeyPatch, digests: dict[str, str]) -> None:
    versions = {
        'latest': VERSION,
        'versions': {VERSION: ['Linux', 'Linux-rslab', 'Mac', 'Mac-x86', 'Windows']},
        'sha256': digests,
    }
    monkeypatch.setattr(installer, 'get_versions', lambda: versions)


def test_a_matching_sha256_installs(
    linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder, tmp_path: Path
) -> None:
    installer.install(sha256=_release_digest(tmp_path))
    assert (linux['system'] / 'emode').read_bytes() == BINARY


def test_the_sha256_is_not_case_sensitive(
    linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder, tmp_path: Path
) -> None:
    installer.install(sha256=_release_digest(tmp_path).upper())
    assert (linux['system'] / 'emode').exists()


def test_a_mismatched_sha256_refuses_to_install(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        installer.install(sha256='0' * 64)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert 'SHA-256' in out
    assert '0' * 64 in out
    assert not (linux['system'] / 'emode').exists()
    assert run.sudo_calls() == []


@pytest.mark.parametrize('bad', ['abc', 'z' * 64, '0' * 63, ''])
def test_a_malformed_sha256_is_refused_before_any_download(
    bad: str, linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder
) -> None:
    with pytest.raises(SystemExit) as exc:
        installer.install(sha256=bad)
    assert exc.value.code == 2
    assert downloads == []


def test_a_digest_published_for_the_file_is_checked(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish_digests(monkeypatch, {f'EMode-{VERSION}-Linux.zip': 'f' * 64})
    with pytest.raises(SystemExit) as exc:
        installer.install()
    assert exc.value.code == 1
    assert 'SHA-256' in capsys.readouterr().out
    assert not (linux['system'] / 'emode').exists()


def test_a_published_digest_that_matches_installs(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _publish_digests(monkeypatch, {f'EMode-{VERSION}-Linux.zip': _release_digest(tmp_path)})
    installer.install()
    assert (linux['system'] / 'emode').exists()


def test_the_given_sha256_wins_over_a_published_one(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _publish_digests(monkeypatch, {f'EMode-{VERSION}-Linux.zip': 'f' * 64})
    installer.install(sha256=_release_digest(tmp_path))
    assert (linux['system'] / 'emode').exists()


def test_a_digest_published_for_another_file_is_not_applied(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_digests(monkeypatch, {f'EMode-{VERSION}-Mac.pkg': 'f' * 64})
    installer.install()
    assert (linux['system'] / 'emode').exists()


def test_the_cli_takes_the_sha256(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys, 'argv', ['emode-install', '--variant', 'rslab', '--sha256', '0' * 64]
    )
    with pytest.raises(SystemExit) as exc:
        installer.main()
    assert exc.value.code == 1
    assert not (linux['system'] / 'emode').exists()


def test_verify_sha256_hashes_a_large_file_in_chunks(tmp_path: Path) -> None:
    path = tmp_path / 'big.bin'
    data = os.urandom(3 * 1024 * 1024 + 7)
    path.write_bytes(data)
    installer.verify_sha256(path, hashlib.sha256(data).hexdigest())
    with pytest.raises(SystemExit):
        installer.verify_sha256(path, hashlib.sha256(data + b'x').hexdigest())


# ---------------------------------------------------------------
# Installing without write access
# ---------------------------------------------------------------


def test_root_with_a_writable_directory_needs_no_sudo(
    linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder
) -> None:
    """The Colab shape: running as root, the default directory is writable."""
    installer.install()
    exe = linux['system'] / 'emode'
    assert exe.read_bytes() == BINARY
    assert _is_executable(exe)
    assert run.sudo_calls() == []
    assert not linux['user'].exists()


def test_with_a_terminal_sudo_installs_in_one_command(
    read_only: Path,
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tty(monkeypatch, True)
    installer.install()
    dest = read_only / 'emode'
    [call] = run.sudo_calls()
    assert call[:4] == ['sudo', 'install', '-m', '755']
    assert call[5] == str(dest)
    assert len(call) == 6
    assert run.installed == [BINARY]
    out = capsys.readouterr().out
    assert 'administrator' in out
    assert f'Location: {dest}' in out


def test_sudo_creates_a_missing_directory_first(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.geteuid() == 0:
        pytest.skip('root can write to a read-only directory')
    parent = linux['system']
    parent.chmod(0o555)
    missing = parent / 'emode-bin'
    monkeypatch.setitem(installer.DEFAULT_INSTALL_DIRS, 'Linux', missing)
    _tty(monkeypatch, True)
    try:
        installer.install()
    finally:
        parent.chmod(0o755)
    calls = run.sudo_calls()
    assert [c[:3] for c in calls] == [['sudo', 'mkdir', '-p'], ['sudo', 'install', '-m']]
    assert calls[0][3] == str(missing)
    assert calls[1][5] == str(missing / 'emode')


def test_without_a_terminal_it_installs_for_the_user(
    read_only: Path,
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tty(monkeypatch, False)
    installer.install()  # returns, so the exit status is 0
    exe = linux['user'] / 'emode'
    assert exe.read_bytes() == BINARY
    assert _is_executable(exe)
    assert run.sudo_calls() == []
    out = capsys.readouterr().out
    assert str(linux['user']) in out
    assert f'Note: {linux["user"]} is not on your PATH.' in out
    assert f'Location: {exe}' in out


def test_the_path_note_is_left_out_when_the_user_directory_is_on_path(
    read_only: Path,
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('PATH', f'{linux["user"]}:/usr/bin')
    _tty(monkeypatch, False)
    installer.install()
    assert 'not on your PATH' not in capsys.readouterr().out


def test_without_sudo_it_installs_for_the_user(
    read_only: Path,
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(installer.shutil, 'which', lambda name: None)
    _tty(monkeypatch, True)
    installer.install()
    assert (linux['user'] / 'emode').read_bytes() == BINARY
    assert run.sudo_calls() == []


def test_a_declined_sudo_falls_back_to_the_user_directory(
    read_only: Path,
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    declined = _Recorder(returncode=1)
    monkeypatch.setattr(installer.subprocess, 'run', declined)
    _tty(monkeypatch, True)
    installer.install()
    assert len(declined.sudo_calls()) == 1
    exe = linux['user'] / 'emode'
    assert exe.read_bytes() == BINARY
    assert _is_executable(exe)
    out = capsys.readouterr().out
    assert 'sudo did not install EMode' in out
    assert f'Location: {exe}' in out


def test_an_explicit_directory_has_no_fallback(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    run: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.geteuid() == 0:
        pytest.skip('root can write to a read-only directory')
    target = tmp_path / 'opt' / 'bin'
    target.mkdir(parents=True)
    target.chmod(0o555)
    _tty(monkeypatch, True)
    try:
        with pytest.raises(SystemExit) as exc:
            installer.install(install_dir=target)
    finally:
        target.chmod(0o755)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert str(target) in out
    assert 'sudo' in out
    assert '--install-dir' in out
    assert run.sudo_calls() == []
    assert not linux['user'].exists()


def test_a_user_install_counts_as_installed(
    linux: dict[str, Path],
    downloads: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """After the fallback, running emode-install again finds that install."""
    linux['user'].mkdir(parents=True)
    (linux['user'] / 'emode').write_bytes(BINARY)
    with pytest.raises(SystemExit) as exc:
        installer.install()
    assert exc.value.code == 0
    assert f'already installed at {linux["user"] / "emode"}' in capsys.readouterr().out
    assert downloads == []


def test_uninstall_finds_a_user_install(
    linux: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    linux['user'].mkdir(parents=True)
    exe = linux['user'] / 'emode'
    exe.write_bytes(BINARY)
    _tty(monkeypatch, False)
    installer.uninstall()
    assert not exe.exists()


# ---------------------------------------------------------------
# The hint to update the client
# ---------------------------------------------------------------

_HINT = 'pip install -U emode'


def _client_at(monkeypatch: pytest.MonkeyPatch, installed: str, latest: str | None) -> None:
    monkeypatch.setattr(installer, 'client_version', lambda: installed)
    monkeypatch.setattr(installer, 'latest_client', lambda: latest)


def test_a_newer_client_is_pointed_out(
    linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _client_at(monkeypatch, '1.1.20', '1.1.25')
    installer.install()
    out = capsys.readouterr().out
    assert 'installed successfully' in out
    assert _HINT in out
    assert '1.1.25' in out and '1.1.20' in out
    # It comes after the install is reported, as the last thing printed.
    assert out.index(_HINT) > out.index('installed successfully')


def test_the_same_client_says_nothing(
    linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _client_at(monkeypatch, '1.1.25', '1.1.25')
    installer.install()
    assert _HINT not in capsys.readouterr().out


def test_a_development_client_newer_than_pypi_says_nothing(
    linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _client_at(monkeypatch, '1.1.26', '1.1.25')
    installer.install()
    assert _HINT not in capsys.readouterr().out


def test_an_unknown_installed_version_says_nothing(
    linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _client_at(monkeypatch, 'unknown', '1.1.25')
    installer.install()
    assert _HINT not in capsys.readouterr().out


@pytest.mark.parametrize(
    'error',
    [
        installer.http_requests.exceptions.Timeout('slow'),
        installer.http_requests.exceptions.ConnectionError('offline'),
        installer.http_requests.exceptions.HTTPError('503'),
        ValueError('not json'),
    ],
)
def test_a_failed_lookup_says_nothing_and_the_install_succeeds(
    linux: dict[str, Path], downloads: list[dict[str, Any]], run: _Recorder,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception,
) -> None:
    def get(url: str, timeout: float, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(installer, 'latest_client', _LATEST_CLIENT)
    monkeypatch.setattr(installer, 'client_version', lambda: '1.1.20')
    monkeypatch.setattr(installer.http_requests, 'get', get)
    installer.install()
    out = capsys.readouterr().out
    assert 'installed successfully' in out
    assert _HINT not in out
    assert (linux['system'] / 'emode').exists()


class _Reply:
    def __init__(self, body: Any) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self.body


def test_the_lookup_reads_pypi_with_a_short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[str, float]] = []

    def get(url: str, timeout: float, **kwargs: Any) -> _Reply:
        asked.append((url, timeout))
        return _Reply({'info': {'version': '1.1.25'}})

    monkeypatch.setattr(installer.http_requests, 'get', get)
    assert _LATEST_CLIENT() == '1.1.25'
    [(url, timeout)] = asked
    assert url == 'https://pypi.org/pypi/emode/json'
    assert timeout <= 5


@pytest.mark.parametrize('body', [{}, {'info': {}}, [], {'info': {'version': 3}}])
def test_a_malformed_pypi_reply_is_no_version(monkeypatch: pytest.MonkeyPatch, body: Any) -> None:
    monkeypatch.setattr(installer.http_requests, 'get', lambda url, timeout, **kw: _Reply(body))
    assert _LATEST_CLIENT() is None


# ---------------------------------------------------------------
# The macOS a release needs (emodeconnection#137)
# ---------------------------------------------------------------


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A Mac with a given macOS, 1.0.8 and 1.0.9 listed, and a `.pkg` recorder."""

    def set_macos(release: str) -> None:
        monkeypatch.setattr(installer.platform, 'system', lambda: 'Darwin')
        monkeypatch.setattr(installer.platform, 'mac_ver', lambda: (release, ('', '', ''), 'arm64'))

    monkeypatch.setitem(installer.DEFAULT_INSTALL_DIRS, 'Darwin', tmp_path / 'usr-bin')
    versions = {
        'latest': '1.0.9',
        'versions': {'1.0.9': ['Mac', 'Mac-x86'], '1.0.8': ['Mac', 'Mac-x86']},
    }
    monkeypatch.setattr(installer, 'get_versions', lambda: versions)
    monkeypatch.setattr(installer, 'handle_eula', lambda: True)
    monkeypatch.setattr(installer, 'prompt_credentials', lambda: ('a@b.c', 'pw'))
    monkeypatch.setattr(installer, 'login', lambda email, password: 'token')
    monkeypatch.setattr(installer, 'latest_client', lambda: None)
    asked: list[str] = []

    def download_release(token, platform_key, version, filename, dest_dir):
        asked.append(version)
        return dest_dir / filename

    monkeypatch.setattr(installer, 'download_release', download_release)
    installed: list[Path] = []
    monkeypatch.setattr(
        installer, 'install_pkg', lambda pkg, install_dir: installed.append(install_dir) or pkg
    )
    monkeypatch.setattr(installer, 'check_path', lambda install_dir: None)
    monkeypatch.setattr(installer, 'verify_sha256', lambda path, expected: None)
    set_macos.asked = asked  # type: ignore[attr-defined]
    set_macos.installed = installed  # type: ignore[attr-defined]
    return set_macos


def test_macos_13_refuses_1_0_9_and_names_1_0_8(mac, capsys: pytest.CaptureFixture[str]) -> None:
    mac('13.6')
    with pytest.raises(SystemExit) as exc:
        installer.install(version=None)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert 'macOS 14' in out and '13.6' in out
    assert '1.0.8' in out and 'emode-install --version 1.0.8' in out
    assert mac.asked == [], 'nothing is downloaded'
    assert mac.installed == []


def test_a_custom_directory_refuses_the_same_way(mac, tmp_path: Path) -> None:
    mac('13.6')
    with pytest.raises(SystemExit) as exc:
        installer.install(version='1.0.9', install_dir=tmp_path / 'bin')
    assert exc.value.code == 1
    assert mac.asked == []


@pytest.mark.parametrize('release', ['14.0', '14.7.1', '15.1', '26.5'])
def test_macos_14_and_later_install_1_0_9(mac, release: str) -> None:
    mac(release)
    installer.install(version=None)
    assert mac.asked == ['1.0.9']


def test_macos_13_still_installs_1_0_8(mac) -> None:
    mac('13.6')
    installer.install(version='1.0.8')
    assert mac.asked == ['1.0.8']


def test_linux_and_windows_are_unaffected(downloads, linux, run) -> None:
    installer.install(version=None)
    assert [d['version'] for d in downloads] == [VERSION]


def test_the_minimum_macos_of_a_release() -> None:
    assert installer.min_macos('1.0.8') is None
    assert installer.min_macos('1.0.9') == (14, 0)
    assert installer.min_macos('1.1.0') == (14, 0)
    assert installer.min_macos('1.0.9.dev3') == (14, 0)
