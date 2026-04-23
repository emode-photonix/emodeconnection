"""
EMode installer — downloads and installs the EMode executable.

Usage:
emode-install                            # Install latest version
emode-install --update                   # Re-download latest version
emode-install --update --version 0.2.5   # Switch to a specific version
emode-install --version 0.2.5            # Install a specific version
emode-install --install-dir /opt/bin     # Custom install directory
emode-install --list-versions            # Show available versions
emode-install --uninstall                # Remove EMode
emode-install --uninstall --install-dir /opt/bin  # Remove from custom location

Environment variables:
EMODE_EULA_ACCEPTED=1    # Accept EULA non-interactively (CI/headless)
EMODE_EMAIL              # Pre-fill email address
EMODE_PASSWORD           # Pre-fill password (use with caution)
EMODE_INTERACTIVE=1      # Use interactive installer GUI
UNINSTALL_EMODE=y        # Accept the uninstallation non-interactively
"""

from __future__ import annotations

import argparse
import getpass
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
import requests as http_requests


API_BASE     = 'https://emodephotonix.com/wp-json/custom/v1'
EULA_URL     = 'https://emodephotonix.com/legal/eula.pdf'

# Default install locations per platform
DEFAULT_INSTALL_DIRS = {
    'Darwin':  Path('/usr/local/bin'),
    'Linux':   Path('/usr/local/bin'),
    'Windows': Path(os.environ.get('PROGRAMFILES', 'C:\\Program Files')) / 'EMode Photonix' / 'EMode',
}

EXECUTABLE_NAMES = {
    'Darwin':  'emode',
    'Linux':   'emode',
    'Windows': 'EMode.exe',
}

# Maps platform.system() to the platform key the server expects
PLATFORM_KEYS = {
    'Darwin':  'mac',
    'Linux':   'linux',
    'Windows': 'windows',
}

# File extensions per platform
# Mac: .pkg (native macOS installer — double-click or headless via `installer`)
# Linux: .zip (contains emode binary + install.sh)
# Windows: .zip (contains Inno Setup .exe installer)
FILE_EXTENSIONS = {
    'Darwin':  '.pkg',
    'Linux':   '.zip',
    'Windows': '.zip',
}


# ---------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------

def get_os() -> str:
    """Return platform.system() and abort if unsupported."""
    os_name = platform.system()
    if os_name not in PLATFORM_KEYS:
        print(f"Error: Unsupported operating system '{os_name}'.")
        sys.exit(1)
    return os_name


def get_default_install_dir(os_name: str) -> Path:
    return DEFAULT_INSTALL_DIRS[os_name]


def get_executable_path(install_dir: Path, os_name: str) -> Path:
    return install_dir / EXECUTABLE_NAMES[os_name]


# ---------------------------------------------------------------
# Progress bar (no dependencies)
# ---------------------------------------------------------------

def print_progress(downloaded: int, total: int, width: int = 40) -> None:
    if total <= 0:
        mb = downloaded / 1_048_576
        print(f"\r  Downloading... {mb:.1f} MB", end='', flush=True)
        return
    fraction = downloaded / total
    filled   = int(width * fraction)
    bar      = '█' * filled + '░' * (width - filled)
    pct      = fraction * 100
    mb_done  = downloaded / 1_048_576
    mb_total = total / 1_048_576
    print(f"\r  [{bar}] {pct:.1f}%  {mb_done:.1f}/{mb_total:.1f} MB",
          end='', flush=True)

# ---------------------------------------------------------------
# API calls
# ---------------------------------------------------------------

def api_get(path: str) -> dict:
    """GET request, returns parsed JSON or raises RuntimeError."""
    url = f"{API_BASE}/{path.lstrip('/')}"
    try:
        resp = http_requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except http_requests.exceptions.HTTPError as e:
        try:
            data = e.response.json()
            raise RuntimeError(data.get('error') or data.get('message') or str(e))
        except (ValueError, AttributeError):
            raise RuntimeError(f"HTTP {e.response.status_code}")
    except http_requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Network error: {e}")
    except http_requests.exceptions.Timeout:
        raise RuntimeError("Request timed out.")


def api_post(path: str, payload: dict) -> dict:
    """POST request with JSON body, returns parsed JSON or raises RuntimeError."""
    url = f"{API_BASE}/{path.lstrip('/')}"
    try:
        resp = http_requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except http_requests.exceptions.HTTPError as e:
        try:
            data = e.response.json()
            raise RuntimeError(data.get('error') or data.get('message') or str(e))
        except (ValueError, AttributeError):
            raise RuntimeError(f"HTTP {e.response.status_code}")
    except http_requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Network error: {e}")
    except http_requests.exceptions.Timeout:
        raise RuntimeError("Request timed out.")


# ---------------------------------------------------------------
# Version resolution
# ---------------------------------------------------------------

def get_versions() -> dict:
    """Returns dict with 'latest' (str) and 'versions' (dict of version -> [platforms])."""
    try:
        return api_get('/versions/')
    except RuntimeError as e:
        print(f"Error fetching available versions: {e}")
        sys.exit(1)


def resolve_version(requested: Optional[str], os_name: str) -> tuple[str, str]:
    """
    Resolve the version to download and build the expected filename.
    Returns (version_string, filename).
    """
    platform_key     = PLATFORM_KEYS[os_name]
    platform_display = {'mac': 'Mac', 'linux': 'Linux', 'windows': 'Windows'}[platform_key]
    ext              = FILE_EXTENSIONS[os_name]

    data = get_versions()

    if not data.get('latest'):
        print("Error: No EMode releases are currently available. Please try again later.")
        sys.exit(1)

    if requested is None or requested == 'latest':
        version = data['latest']
    else:
        if requested not in data.get('versions', {}):
            available = ', '.join(data.get('versions', {}).keys()) or 'none'
            print(f"Error: Version '{requested}' is not available.")
            print(f"Available versions: {available}")
            sys.exit(1)
        version = requested

    # Verify the requested platform is available for this version
    available_platforms = data['versions'].get(version, [])
    if platform_display not in available_platforms:
        print(f"Error: EMode {version} is not yet available for {platform_display}.")
        print(f"Available platforms for {version}: {', '.join(available_platforms) or 'none'}")
        sys.exit(1)

    filename = f"EMode-{version}-{platform_display}{ext}"
    return version, filename


def list_versions() -> None:
    """Print available versions and exit."""
    data = get_versions()
    if not data.get('latest'):
        print("No EMode releases are currently available.")
        return
    print(f"Latest version: {data['latest']}\n")
    print("All available versions:")
    for ver, platforms in data.get('versions', {}).items():
        tag = ' (latest)' if ver == data['latest'] else ''
        print(f"  {ver}{tag}  —  {', '.join(platforms)}")


# ---------------------------------------------------------------
# EULA
# ---------------------------------------------------------------

def handle_eula() -> bool:
    """
    Show EULA, prompt for acceptance.
    Returns True if accepted, False if declined.
    Respects EMODE_EULA_ACCEPTED=1 for non-interactive environments.
    """
    print()
    print("=" * 60)
    print("EMode End User License Agreement")
    print("=" * 60)
    print(f"\n  {EULA_URL}\n")

    if os.environ.get('EMODE_EULA_ACCEPTED') == '1':
        print("\nEULA accepted via EMODE_EULA_ACCEPTED=1 environment variable.")
        return True

    print("Please review the EULA at the link above before continuing.")
    print("=" * 60)
    _try_open_eula()

    if not sys.stdin.isatty():
        print("\nNon-interactive environment detected.")
        print("Set EMODE_EULA_ACCEPTED=1 to accept the EULA non-interactively.")
        return False

    try:
        response = input("\nType 'agree' to accept the EULA and continue, "
                         "or anything else to cancel: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    return response == 'agree'


def _try_open_eula() -> None:
    """Attempt to open the EULA in a browser or PDF viewer. Never raises."""
    try:
        os_name = platform.system()
        if os_name == 'Darwin':
            subprocess.run(['open', EULA_URL], timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os_name == 'Windows':
            os.startfile(EULA_URL)  # type: ignore[attr-defined]
        elif os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            subprocess.run(['xdg-open', EULA_URL], timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ---------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------

def prompt_credentials() -> tuple[str, str]:
    """Prompt for email and password. Respects env vars for automation."""
    print()
    email = os.environ.get('EMODE_EMAIL', '').strip()
    if not email:
        try:
            email = input("Email: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)

    password = os.environ.get('EMODE_PASSWORD', '')
    if not password:
        try:
            password = getpass.getpass("Password: ")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)

    if not email or not password:
        print("Error: Email and password are required.")
        sys.exit(1)

    return email, password


def login(email: str, password: str) -> str:
    """
    Authenticate against the installer-login endpoint.
    Returns the downloadToken string, or exits on failure.
    """
    print("Authenticating...")
    try:
        data = api_post('/installer-login/', {'email': email, 'password': password})
    except RuntimeError as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    token = data.get('downloadToken')
    if not token:
        print("Authentication failed: no download token received.")
        sys.exit(1)

    first = data.get('firstName', '')
    last  = data.get('lastName', '')
    name  = f"{first} {last}".strip()
    if name:
        print(f"Logged in as {name}.")
    return token


# ---------------------------------------------------------------
# Download
# ---------------------------------------------------------------

def download_release(
    token: str,
    platform_key: str,
    version: str,
    filename: str,
    dest_dir: Path,
) -> Path:
    url = (
        f"{API_BASE}/download/"
        f"?token={token}&platform={platform_key}&version={version}"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    print(f"\nDownloading EMode {version} for {platform_key}...")

    try:
        with http_requests.get(
            url,
            headers={'Accept': 'application/octet-stream'},
            stream=True,
            timeout=120,
        ) as resp:
            content_type = resp.headers.get('Content-Type', '')

            # JSON response means an error from the server
            if 'application/json' in content_type or 'text/' in content_type:
                try:
                    data = resp.json()
                    msg = (
                        data.get('error')
                        or (data.get('data') or {}).get('error')
                        or data.get('message')
                        or resp.text[:200]
                    )
                except (ValueError, AttributeError):
                    msg = resp.text[:200]
                print(f"\nError from server: {msg}")
                sys.exit(1)

            total      = int(resp.headers.get('Content-Length', 0))
            downloaded = 0

            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=dest_dir, prefix='.emode_dl_'
            )
            try:
                with os.fdopen(tmp_fd, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            print_progress(downloaded, total)

                print()  # newline after progress bar

                if total > 0 and downloaded < total:
                    os.unlink(tmp_path)
                    print("Error: Download was incomplete. Please try again.")
                    sys.exit(1)

                shutil.move(tmp_path, dest_path)

            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    except http_requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 401:
            print("\nError: Authentication failed — token may have expired. "
                  "Please run emode-install again.")
        elif code == 503:
            try:
                msg = e.response.json().get('error', str(e))
            except ValueError:
                msg = str(e)
            print(f"\nError: {msg}")
        else:
            print(f"\nDownload failed: HTTP {code}")
        sys.exit(1)
    except http_requests.exceptions.ConnectionError as e:
        print(f"\nNetwork error during download: {e}")
        sys.exit(1)
    except http_requests.exceptions.Timeout:
        print("\nDownload timed out. Please try again.")
        sys.exit(1)

    return dest_path

# ---------------------------------------------------------------
# Post-install
# ---------------------------------------------------------------

def install_pkg(pkg_path: Path, install_dir: Path) -> Path:
    """
    Install a macOS .pkg file.

    If install_dir matches the default (/usr/local/bin), run the .pkg
    installer normally via sudo installer or GUI.

    If a custom install_dir is specified, extract the binary directly
    from the .pkg payload — the pkg installer can't install to
    arbitrary locations.

    Returns the expected executable path after installation.
    """
    default_dir = get_default_install_dir('Darwin')
    exe_path    = install_dir / EXECUTABLE_NAMES['Darwin']

    if install_dir != default_dir:
        # Custom install directory — extract binary directly from pkg payload
        print(f"Custom install directory: {install_dir}")
        _install_pkg_custom_dir(pkg_path, install_dir)
        return exe_path

    # Standard install location — run the .pkg installer normally
    if os.environ.get('EMODE_INTERACTIVE', '').strip():
        print("Opening macOS Installer — follow the prompts to complete installation.")
        print("You may be asked for your administrator password.")
        result = subprocess.run(
            ['open', '-W', str(pkg_path)],
            timeout=600,
        )
        if result.returncode != 0:
            print(f"Installer exited with code {result.returncode}.")
            sys.exit(1)
    else:
        print("Installing via sudo installer...")
        result = subprocess.run(
            ['sudo', 'installer', '-pkg', str(pkg_path), '-target', '/'],
            timeout=300,
        )
        if result.returncode != 0:
            print(f"Error: sudo installer failed with exit code {result.returncode}.")
            print("Try running manually: sudo installer -pkg EMode-VERSION-Mac.pkg -target /")
            sys.exit(1)

    return exe_path


def _install_pkg_custom_dir(pkg_path: Path, install_dir: Path) -> None:
    """
    Extract the emode binary from a .pkg payload and place it in install_dir.
    Used when a custom install location is requested on macOS.
    A .pkg is an xar archive whose Payload is a gzipped cpio archive.
    """
    print(f"Extracting binary to {install_dir}...")
    install_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='emode_pkg_') as tmp:
        tmp_path     = Path(tmp)
        expanded_dir = tmp_path / 'expanded'

        # Expand the pkg structure using pkgutil
        result = subprocess.run(
            ['pkgutil', '--expand', str(pkg_path), str(expanded_dir)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"Error expanding .pkg: {result.stderr}")
            sys.exit(1)

        # Find the Payload file inside the expanded pkg
        payload_files = list(expanded_dir.rglob('Payload'))
        if not payload_files:
            print("Error: Could not find Payload inside .pkg archive.")
            sys.exit(1)

        payload      = payload_files[0]
        extract_dir  = tmp_path / 'payload_contents'
        extract_dir.mkdir()

        # The Payload is a gzipped cpio archive — extract with tar
        result = subprocess.run(
            ['tar', '-xzf', str(payload), '-C', str(extract_dir)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"Error extracting payload: {result.stderr}")
            sys.exit(1)

        # Find the emode binary in the extracted payload
        emode_files = list(extract_dir.rglob('emode'))
        if not emode_files:
            print("Error: Could not find 'emode' binary in .pkg payload.")
            sys.exit(1)

        dst = install_dir / 'emode'
        shutil.copy2(emode_files[0], dst)
        dst.chmod(dst.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        # Remove quarantine on the extracted binary
        subprocess.run(
            ['xattr', '-dr', 'com.apple.quarantine', str(dst)],
            timeout=10,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    print(f"EMode installed to {install_dir}")


def install_from_zip(zip_path: Path, install_dir: Path, os_name: str) -> Path:
    """
    Extract and install EMode from a zip archive.

    Windows: extracts the Inno Setup .exe and runs it silently.
             The installer used the install_dir.
             Returns the Windows install path.

    Linux:   extracts only the 'emode' binary from the zip into install_dir.
             Ignores install.sh and other files — the Python installer handles
             everything install.sh would do.
             Returns the path to the installed executable.
    """
    import zipfile

    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()

        if os_name == 'Windows':
            # The installer is versioned (e.g. EMode-0.2.5-Windows.exe)
            # so search for any .exe rather than a fixed name
            exe_members = [
                n for n in names
                if n.endswith('.exe')
                and not n.startswith('__MACOSX')
            ]
            if not exe_members:
                print("Error: Could not find an .exe installer inside the downloaded archive.")
                print(f"Archive contents: {names}")
                sys.exit(1)

            exe_member     = exe_members[0]
            installer_name = Path(exe_member).name
            tmp_dir        = zip_path.parent / '.emode_install_tmp'
            tmp_dir.mkdir(exist_ok=True)
            installer_path = tmp_dir / installer_name

            with zf.open(exe_member) as src, open(installer_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)

            cmd = [
                str(installer_path),
                '/VERYSILENT',
                '/SUPPRESSMSGBOXES',
                '/NORESTART',
            ]
            if install_dir != get_default_install_dir('Windows'):
                cmd.append(f'/DIR={install_dir}')

            result = subprocess.run(cmd, timeout=300)

            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

            if result.returncode != 0:
                print(f"Installer exited with code {result.returncode}.")
                sys.exit(1)

            return get_executable_path(install_dir, 'Windows')

        else:
            # Linux: find the 'emode' binary, ignoring install.sh
            # Handle both flat zips (emode) and zips with a subdirectory
            # (EMode-0.2.5-Linux/emode) for robustness
            exe_name = EXECUTABLE_NAMES[os_name]

            exe_members = [
                n for n in names
                if Path(n).name == exe_name
                and not n.startswith('__MACOSX')
                and not n.endswith('.sh')
            ]
            if not exe_members:
                print(f"Error: Could not find '{exe_name}' inside the downloaded archive.")
                print(f"Archive contents: {names}")
                sys.exit(1)

            exe_member = exe_members[0]
            install_dir.mkdir(parents=True, exist_ok=True)
            dest = install_dir / exe_name

            with zf.open(exe_member) as src, open(dest, 'wb') as dst:
                shutil.copyfileobj(src, dst)

            return dest


def set_permissions(exe_path: Path, os_name: str) -> None:
    """Set executable permissions on Mac/Linux."""
    if os_name != 'Windows':
        exe_path.chmod(exe_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def remove_quarantine(exe_path: Path, os_name: str) -> None:
    """Remove macOS Gatekeeper quarantine attribute."""
    if os_name == 'Darwin':
        try:
            subprocess.run(
                ['xattr', '-dr', 'com.apple.quarantine', str(exe_path)],
                timeout=10,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def check_path(install_dir: Path) -> None:
    """Warn if install_dir is not on PATH."""
    path_dirs = [Path(p) for p in os.environ.get('PATH', '').split(os.pathsep)]
    if install_dir not in path_dirs:
        os_name = platform.system()
        print(f"\nNote: {install_dir} is not on your PATH.")
        if os_name == 'Windows':
            print("  Add it via: System Properties → Environment Variables → Path")
        else:
            print("  Add to your shell profile (~/.zshrc or ~/.bashrc):")
            print(f"    export PATH=\"{install_dir}:$PATH\"")


# ---------------------------------------------------------------
# Main install flow
# ---------------------------------------------------------------

def install(
    version:     Optional[str] = None,
    install_dir: Optional[Path] = None,
    update:      bool = False,
) -> None:
    os_name      = get_os()
    platform_key = PLATFORM_KEYS[os_name]

    if install_dir is None:
        install_dir = get_default_install_dir(os_name)

    exe_path = get_executable_path(install_dir, os_name)

    # Check if already installed
    if exe_path.exists() and not update:
        print(f"EMode is already installed at {exe_path}")
        print("Run 'emode-install --update' to download the latest version.")
        sys.exit(0)

    # Resolve version and build filename
    version, filename = resolve_version(version, os_name)

    # EULA
    if not handle_eula():
        print("\nInstallation cancelled — EULA not accepted.")
        sys.exit(0)

    # Authentication
    email, password = prompt_credentials()
    token = login(email, password)

    # Download to a temp directory
    with tempfile.TemporaryDirectory(prefix='emode_install_') as tmp_dir:
        downloaded_path = download_release(
            token        = token,
            platform_key = platform_key,
            version      = version,
            filename     = filename,
            dest_dir     = Path(tmp_dir),
        )

        print(f"Installing EMode {version}...")

        if os_name == 'Darwin':
            # Mac: run the .pkg installer
            # The .pkg postinstall script handles quarantine removal and
            # permissions internally, so we skip those steps here
            exe_path = install_pkg(downloaded_path, install_dir)
            check_path(install_dir)
        else:
            # Windows and Linux: extract from zip
            try:
                exe_path = install_from_zip(downloaded_path, install_dir, os_name)
            except PermissionError:
                if os_name != 'Windows':
                    print("Permission denied. Trying with sudo...")
                    _install_with_sudo(downloaded_path, install_dir, os_name)
                    exe_path = get_executable_path(install_dir, os_name)
                else:
                    print("Error: Permission denied. Try running as Administrator.")
                    sys.exit(1)

            # Windows PATH is handled by Inno Setup
            if os_name != 'Windows':
                set_permissions(exe_path, os_name)
                check_path(install_dir)

    print(f"\nEMode {version} installed successfully.")
    if os_name != 'Windows':
        print(f"Location: {exe_path}")
    print('Get started by logging in:\n\tpython -c "import emodeconnection as emc; emc.EModeLogin()"')


def _install_with_sudo(zip_path: Path, install_dir: Path, os_name: str) -> None:
    """Extract to a temp location then sudo mv into place. Linux only."""
    import zipfile

    exe_name = EXECUTABLE_NAMES[os_name]
    with tempfile.TemporaryDirectory(prefix='emode_sudo_') as tmp:
        tmp_exe = Path(tmp) / exe_name
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names       = zf.namelist()
            exe_members = [
                n for n in names
                if Path(n).name == exe_name
                and not n.startswith('__MACOSX')
                and not n.endswith('.sh')
            ]
            if not exe_members:
                print(f"Error: Could not find '{exe_name}' in archive.")
                sys.exit(1)
            with zf.open(exe_members[0]) as src, open(tmp_exe, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        tmp_exe.chmod(tmp_exe.stat().st_mode | stat.S_IEXEC)

        result = subprocess.run(
            ['sudo', 'mkdir', '-p', str(install_dir)], timeout=30
        )
        if result.returncode != 0:
            print("Error: sudo mkdir failed.")
            sys.exit(1)

        # Make directory owned by current user so future updates don't need sudo
        result = subprocess.run(
            ['sudo', 'chown', f'{os.getenv("USER", os.getlogin())}', str(install_dir)],
            timeout=10,
        )

        if result.returncode != 0:
            print("Error: sudo mv failed.")
            sys.exit(1)

        subprocess.run(
            ['sudo', 'chmod', '+x', str(install_dir / exe_name)],
            timeout=10,
        )


# ---------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------

def uninstall(install_dir: Optional[Path] = None) -> None:
    os_name = get_os()

    if install_dir is None:
        install_dir = get_default_install_dir(os_name)

    exe_path = get_executable_path(install_dir, os_name)

    if not exe_path.exists():
        print(f"EMode is not installed at {exe_path}")
        sys.exit(0)

    print(f"This will remove EMode from {install_dir}")

    if sys.stdin.isatty():
        try:
            UNINSTALL_EMODE = os.environ.get('UNINSTALL_EMODE', '').strip()
            if UNINSTALL_EMODE == 'y':
                print("Uninstall accepted via UNINSTALL_EMODE=y environment variable.")
                response = UNINSTALL_EMODE
            else:
                response = input("Are you sure? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if response != 'y':
            print("Uninstall cancelled.")
            sys.exit(0)

    if os_name == 'Windows':
        # Use Inno Setup's uninstaller to properly deregister from
        # Programs & Features and remove all installed files
        uninstaller = install_dir / 'unins000.exe'
        if not uninstaller.exists():
            # Fall back to searching for any unins*.exe in the install dir
            uninstallers = list(install_dir.glob('unins*.exe'))
            if uninstallers:
                uninstaller = uninstallers[0]
            else:
                print(f"Warning: Could not find Inno Setup uninstaller in {install_dir}")
                print("Falling back to manual file removal...")
                try:
                    exe_path.unlink()
                    print(f"EMode removed from {exe_path}")
                except PermissionError:
                    print("Error: Permission denied. Try running as Administrator.")
                    sys.exit(1)
                return

        print(f"Running uninstaller: {uninstaller}")
        result = subprocess.run(
            [str(uninstaller), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'],
            timeout=120,
        )
        if result.returncode != 0:
            print(f"Uninstaller exited with code {result.returncode}.")
            sys.exit(1)
        print("EMode uninstalled successfully.")

    else:
        # Mac/Linux: remove the binary directly
        try:
            exe_path.unlink()
            print(f"EMode removed from {exe_path}")
        except PermissionError:
            print("Permission denied. Trying with sudo...")
            result = subprocess.run(['sudo', 'rm', str(exe_path)], timeout=30)
            if result.returncode == 0:
                print(f"EMode removed from {exe_path}")
            else:
                print("Error: sudo rm failed.")
                sys.exit(1)


# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Install or update the EMode executable.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  emode-install                            Install latest version
  emode-install --update                   Re-download latest version
  emode-install --update --version 0.2.5   Switch to a specific version
  emode-install --version 0.2.5            Install a specific version
  emode-install --install-dir /opt/bin     Custom install directory
  emode-install --list-versions            Show available versions
  emode-install --uninstall                Remove EMode
  emode-install --uninstall --install-dir /opt/bin  Remove from custom location

Environment variables:
  EMODE_EULA_ACCEPTED=1    Accept EULA non-interactively
  EMODE_EMAIL              Pre-fill email address
  EMODE_PASSWORD           Pre-fill password (use with caution)
        """,
    )
    parser.add_argument(
        '--version',
        metavar='VERSION',
        default=None,
        help='Version to install (default: latest)',
    )
    parser.add_argument(
        '--install-dir',
        metavar='DIR',
        default=None,
        help='Directory to install EMode into (default: /usr/local/bin on Mac/Linux)',
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Re-download and reinstall even if already installed. '
             'Combine with --version to switch to a specific version.',
    )
    parser.add_argument(
        '--list-versions',
        action='store_true',
        help='List available versions and exit',
    )
    parser.add_argument(
        '--uninstall',
        action='store_true',
        help='Remove the EMode executable.',
    )

    args = parser.parse_args()

    if args.list_versions:
        list_versions()
        sys.exit(0)

    install_dir = Path(args.install_dir) if args.install_dir else None

    if args.uninstall:
        uninstall(install_dir=install_dir)
        sys.exit(0)

    install(
        version     = args.version,
        install_dir = install_dir,
        update      = args.update,
    )


if __name__ == '__main__':
    main()
