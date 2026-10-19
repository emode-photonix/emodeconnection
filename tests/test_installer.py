"""``emode-install``: the build variant, and installing without write access.

Nothing here touches the network or the real install location. The release
lookup, the EULA, the login and the download are stubbed, the download writes
a small zip shaped like the Linux release (``emode`` and ``install.sh``),
``subprocess.run`` is a recorder, and the install directories live under
``tmp_path``. The OS is forced to Linux (or macOS and Windows for the variant
errors) so the tests run the same on any machine.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pytest

import emodeconnection.installer as installer

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
