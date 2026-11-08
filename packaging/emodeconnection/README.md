# emodeconnection

`emodeconnection` is the old name of [`emode`](https://pypi.org/project/emode/),
EMode's Python client. This package installs `emode` and re-exports it, so
existing code keeps working. It warns once on import that the name changed.

Switch to the new name:

```bash
pip install emode
```

```python
import emode

em = emode.EMode()
```

`import emodeconnection` and `import emodeconnection.types` (or any other
submodule) give the same modules as `emode` and `emode.types`. The commands
`emode-install` and `emode-mcp` come from the `emode` distribution.

The source lives at <https://github.com/emode-photonix/emodeconnection>.
