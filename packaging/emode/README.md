# emode

Python interface for [EMode](https://emodephotonix.com), a photonics simulation
platform for waveguide mode solving, propagation and analysis, under its short
name.

```bash
pip install emode
```

```python
import emode

em = emode.EMode()
em.shape(name='BOX', material='SiO2', height=800)
em.shape(name='core', material='Si', width=600, height=400)
em.FDM()
em.report()
em.close()
```

This release installs [emodeconnection](https://pypi.org/project/emodeconnection/)
and re-exports it, so `import emode` and `import emodeconnection` give the same
client. A later release of EMode moves the client itself into this package.
`emode-install` and `emode-mcp` come with it. The documentation is at
[docs.emodephotonix.com](https://docs.emodephotonix.com).
