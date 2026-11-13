---
name: emode
description: How to use the EMode photonic mode solver and EME simulator well through its MCP tools. Use it when solving waveguide modes, simulating devices (tapers, bends, couplers, gratings), sweeping or optimizing a design, or checking that a simulation can be trusted.
---

# Using EMode well

EMode solves the modes of a waveguide cross-section and simulates devices by
eigenmode expansion (EME). Lengths are in nm and wavelengths in nm.

## Workflow

1. `emode_open` starts a session and draws the cross-section. List the bottom
   layer first, such as the oxide, then the core. One session is open at a
   time, and `emode_close` releases the license, so close it when done.
2. Run `emode_converge` before trusting any number. It finds the grid
   resolution, window size, PML thickness and mode count the result needs.
   Use `apply=true` to keep what it chose. A result that has not converged is
   a guess.
3. `emode_solve` returns each mode's effective index and loss. Mode 0 is the
   fundamental mode. A mode with a complex effective index is a lossy mode,
   and a large imaginary part means it is not guided. Check the TE fraction
   and loss before naming a mode TE or TM.
4. For a device, declare sections with `emode_section` in the order light
   passes through them, then run `emode_eme`. Refine tapers and bends by their
   own convergence, not by the solver's default.
5. Sweep with `emode_sweep`. Sweeps over more than 20 points and every
   optimization ask for confirmation, so tell the user the cost first and
   repeat the call with `confirm=true`.

## Devices on several cross-sections

A device whose cross-section changes along its length, such as a taper or a
multimode interference coupler, is built from labeled cross-sections:

1. Draw the first cross-section and label it with `emode_profile`, such as
   `input`.
2. Change the shapes with `emode_shape` for the next cross-section, such as a
   wider core or an etched layer with two mask openings for a pair of output
   waveguides (`mask` and `mask_offset`), and label it too.
3. Declare the sections on the labeled cross-sections: a straight section
   names one as its `profile`, a taper names its two ends as `profile` and
   `profile_end`. A taper that widens one shape along a smooth curve names
   its `profile` and the shape's `widths` at equally spaced points instead,
   and `emode_optimize` varies them with the key `'taper, <name>, widths'`
   (or `'taper, <name>, widths[k]'` for one point).

A labeled cross-section keeps its shapes when the current one changes. Change
it with `emode_shape` and `profile`, or vary it in `emode_evaluate`,
`emode_sweep` and `emode_optimize` with the key
`'shape, <profile>, <name>, <property>'`, such as
`'shape, taper_end, core, mask'`.

A labeled cross-section also keeps the settings it was labeled with (grid,
window, modes, boundary condition). `emode_settings` without `profile` changes
only the current cross-section, so the sections would not see it. Converge the
cross-section that needs the most (the widest or most multimode one), then
copy the chosen settings to every cross-section with `emode_settings` and
`profile='all'`. Do the same to switch every cross-section to a coarser grid
for a search and back to the converged one for the check. `emode_get` with
`profile` reads a labeled cross-section's settings.

## Sweeps

- A wavelength sweep of an S-matrix should use `method='interpolated'` with
  `solve_type='EME'`. It solves a few Chebyshev nodes and interpolates between
  them, so it is far cheaper than solving every wavelength. Check the reported
  interpolation residual and add nodes if it is large.
- A length sweep of a section needs no new mode solves after the first point.
- Use `emode_evaluate` for one design point and `emode_optimize` for many.
  Give an optimizer bounds on every parameter.
- Each optimizer parameter takes one value. The exceptions are
  `'poling, <section>, domain_lengths'` and a mask with several openings
  (`'shape, <name>, mask'` or `mask_offset`), which take a list `x0` and one
  `[min, max]` pair per entry in `bounds`, such as `[[600, 1000], [600, 1000]]`
  (or one `[min, max]` for every entry).
- A mask with several openings, such as the two output arms of an MMI, takes
  one value per opening. A single value for it is refused (EMode 1.0.9 and
  later).

## Checking a result

- Check passivity: a passive device's S-matrix has power transmission plus
  reflection at or below one for each input. A value above one means the
  simulation is not converged or a mode was missed.
- Check reciprocity and energy conservation for lossless structures.
- Compare against a known limit when there is one, such as an analytic
  effective index or a straight waveguide's transmission of one.
- Read the warnings in each result. They say what the solver decided.

## Results

Arrays above 1000 elements are saved to `.npy` files, and the tool returns the
file path. Load them with NumPy instead of asking for them inline.
`emode_plot` returns a PNG. Look at the mode fields to confirm a mode is what
you think it is.
