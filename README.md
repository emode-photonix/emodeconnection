# emodeconnection

Python interface for [EMode](https://emodephotonix.com) — a photonics simulation
platform for waveguide mode solving, propagation, and analysis.

## Requirements

- Python 3.10 or higher
- An active [EMode account](https://emodephotonix.com/my-account/) and license
- EMode installed on your system (see [Installation](https://docs.emodephotonix.com/installation.html))

## Installation

```bash
pip install emodeconnection
```

## Quick Start

```python
import emodeconnection as emc

# Open the EMode login window (only required on first use)
emc.EModeLogin()

## Connect and initialize EMode
em = emc.EMode(save = False)

## Draw shapes
em.shape(name = 'BOX', material = 'SiO2', height = 800)
em.shape(name = 'core', material = 'Si', width = 600, height = 400)

## Run FDM, display results, and close
em.FDM()
em.report()
em.close()
```

## Features

- Simple Python API for EMode simulation sessions
- Automatic login and license management
- Cross-platform: Windows, macOS, Linux
- Command-line installer via `emode-install`

## Installing EMode

The `emodeconnection` package includes a command-line tool to download and
install the EMode executable:

```bash
emode-install
```

### Options

```bash
emode-install                            # Install latest version
emode-install --help                     # Show all options
emode-install --list-versions            # Show available versions
emode-install --update                   # Update to the latest version
emode-install --update --version 0.2.5   # Switch to a specific version
emode-install --version 0.2.5            # Install a specific version
emode-install --install-dir /opt/bin     # Custom install directory
emode-install --uninstall                # Remove EMode
emode-install --uninstall --install-dir /opt/bin  # Remove from custom location
```

### Environment variables

| Variable | Description |
|---|---|
| `EMODE_EULA_ACCEPTED=1` | Accept the EULA non-interactively (CI/headless) |
| `EMODE_EMAIL` | Pre-fill login email |
| `EMODE_PASSWORD` | Pre-fill password (use with caution) |
| `EMODE_INTERACTIVE=1` | Use interactive installer GUI |
| `UNINSTALL_EMODE=y` | Accept the uninstallation non-interactively |

## Documentation

Full documentation is available at [docs.emodephotonix.com](https://docs.emodephotonix.com).

## License

See [LICENSE](LICENSE) for details.

## MATLAB

A legacy MATLAB interface (`emodeconnection.m`) is included in this repository
for historical reference. It is no longer supported as of EMode 0.2.5 and may
not work with newer versions. No support is provided for MATLAB usage.
