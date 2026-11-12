# P4 round 2: vector port mapping and open-gap correction

**Pre-run declaration, 2026-11-08 WIB.** This is a development-only check. It
does not authorize hybrid optimization or training and cannot close P4 without
Sionna RT fields/direct response and a matching independent reflection NEC
reference.

## Frozen checks

The limits remain the previously declared P4 limits: complex relative error
`<= 0.01`, wrapped phase error `<= 0.01 rad`, co-polar amplitude ratio
`[0.99, 1.01]`, and cross-polar leakage `<= 0.01`. No fitting, per-link
alignment, scale, or tolerance changes are permitted.

For the available geometry-only LOS subchecks, report all numerical metrics
against those limits, but label them `P4_NOT_EVALUATED`: the available target is
NEC, not Sionna RT. For the open-gap diagnostic compare the predicted
`delta_z` against the separately saved monolithic NEC open-minus-free matrix,
and compare `D(Z_EE,free + predicted delta_z)` against the monolithic
open-structure NEC direct channel, where
`D(Z)=-100 inv(Z+50 I)[UE,BS]`. The predictor may read only registered coordinates before
constructing predictions; it may read the NEC matrices only in the subsequent
scoring stage. Preserve the same limits for both calculations. Cross-polar
remains unscored unless a matching independent vector field reference exists.

## Source-derived convention

The installed environment reports `PyNEC 2.3.4`. The corresponding PyPI source
distribution is `pynec-2.3.4.tar.gz`, SHA-256
`d448d038d2bcb4265d511b2440a5d5ce26081269a776d45139fb8bf17a475df4`. Its
`necpp_src/src/electromag.cpp` defines `epsilon=8.854e-12 F/m` and
`mu=4*pi*1e-7 H/m`, deriving `c_NEC = 1/sqrt(mu*epsilon) =
299795637.69321626 m/s`. In `c_geometry.cpp`, NEC forms its spatial scale as
`f/c_NEC`; its structure-current getters expose positions/lengths in
wavelength units. `extract_ris_response` converts those getters using
`lambda_SI=C0/f`, so those returned metre-labelled coordinates and segment
lengths are slightly contracted from physical SI geometry by `C0/c_NEC`.
Using these returned values with the SI Green function reproduces NEC
propagation phase through this cancellation; it is a NEC-consistency
representation, not a physical SI coordinate map. For a physical SI field
map, restore physical coordinates with `c_NEC/C0` and use `C0` in the Green
kernel. For a direct NEC comparison, restore physical coordinates and use
`c_NEC`. Both conventions come from source constants; neither is a fitted
phase adjustment.

The source's `exp(+j omega t)` convention and the outgoing
`exp(-j k r)` Hertzian current-element field give the factor `j*k*eta*I*dl`
in the far field. In the local helper `zdip_E`, its parameter `p` therefore
represents the current moment `I*dl`, despite a misleading docstring calling
it an electric dipole moment. The earlier no-fit subcheck passed `I*dl/j`,
dropping a required factor `j`; its large phase error was partly a convention
error. The no-fit correction in this round uses `I*dl` as derived from the
field expression and the NEC time convention, not from the comparison result.

Primary source references:

- PyPI release file and hash: https://pypi.org/project/PyNEC/2.3.4/
- Pinned package source, `electromag.cpp`:
  https://github.com/tmolteno/necpp/blob/master/src/electromag.cpp
- NEC free-space EFIE convention `exp(jwt)`:
  https://nec2.org/part_1/section2.html
- Hertzian current element fields under `exp(j omega t)`:
  https://phys.libretexts.org/Bookshelves/Electricity_and_Magnetism/Electromagnetics_and_Applications_%28Staelin%29/10%3A_Antennas_and_Radiation/10.02%3A_Short_dipole_antennas

## Planned implementation and one bounded fix

Add only P4-owned vector helpers, a free-space mapping subcheck, and a Munich
open-gap subcheck. Incoming fields are projected onto each NEC segment tangent
and integrated against the unconjugated reciprocal current basis; outgoing
fields sum the vector dyadic Green contribution of each current segment. The
new direct-channel helper adds a predicted open-gap impedance matrix to the
free external block and computes the declared `-100 inv(Z_EE + 50 I)` block.

Two source-derived convention checks are kept separate: first, the Hertzian
current moment must be `I*dl`; second, PyNEC wavelength-normalized coordinates
must be restored to physical metres before using a physical-space Green
kernel. These come from source definitions, not comparison-driven fits. The
original sinusoidal-current assumption remains unchanged. If any declared
comparison fails, preserve the measured result and stop.

## Runtime and evidence

Run only on local CPU with the existing `.venv-em`; no GPU, remote GPU, timing
benchmark, test-set, or frozen file access is part of this track. Outputs go
under `outputs/runs/bench_tracks_p4_20261108/`. Record the installed package
version, source hashes, reference hash, segment count, and every metric. The
full gate remains blocked if RT per-port vector fields/direct H, absolute
Sionna-to-port map, or an independent matching reflection NEC reference is
missing.

## Bounded results

The installed `PyNEC 2.3.4` extension was identified as
`.venv-em/lib/python3.12/site-packages/_PyNEC.cpython-312-x86_64-linux-gnu.so`
(SHA-256 `c0fa8ee0e4b847ad28cda8be18fc4e025545db8415c964221ee2ba79e16dbdd1`).
The official PyPI source distribution was fetched and checked against its
published hash above; the NEC source gives `c_NEC=299795637.69321626 m/s` and
`eta_NEC=376.73430918211017 ohm`. For the stored 271.2745768568 m Munich
BS--UE separation, the source-derived NEC-minus-SI propagation phase is
`0.33346733797 rad`. This is not a fitted bridge. The NEC coordinate getters
are wavelength-scaled; multiplying their normalized coordinates by `C0/f`
contracts them by `C0/c_NEC`. The physical-coordinate runs restore them by
`c_NEC/C0`.

### Current moment and direct port map

The earlier no-fit sinusoidal-current subcheck used `p=I*dl/j` in `zdip_E`,
which its field expression does not support: for the `exp(+j omega t)` and
`exp(-j k r)` convention, this current-element kernel expects `I*dl`. Under
its original exact-SI settings, changing only that fixed factor moves the
historical 148.98% complex error / 1.7988 rad / 0.9012 amplitude ratio to
23.75% / 0.2280 rad / 0.9012. Under physical coordinates and PyNEC's own
source-derived constants, the sinusoidal approximation gives 14.064% / 0.10547
rad / 0.90118. It still fails all three scalar limits; no tolerance or fit was
used. The persistent 9.88% amplitude deficit and remaining phase error show
that a sinusoidal current approximation is inadequate for this 24 mm antenna
in this near-grazing geometry.

Using independently solved PyNEC isolated-wire current distributions instead
of a sinusoid, the source-correct physical-coordinate NEC Green map predicts
mutual impedance with 0.01762% complex relative error, `1.17e-5 rad` maximum
phase error, and amplitude ratio `1.000176`. The NEC-normalized-coordinate
representation with the SI kernel gives a similarly small consistency error
(`0.01656%`, `1.17e-5 rad`): that is cancellation of the wavelength scaling,
not a physical SI port map. Using the exact-SI kernel on restored physical
coordinates against the NEC target gives `33.1965%` relative error and
`0.333479 rad` phase error, consistent with the source-derived speed
difference. This separates the NEC reference convention from the physical
SI convention. Both are recorded; neither is aligned to the other.

The implementation performs the polarization-resolved port integral
`Voc[n] = -sum_m I_mn^(1A) dl_m (t_m dot E_m)` without complex conjugation,
and the outgoing Cartesian dyadic current-element sum with near-, middle-, and
far-field terms. Seven focused tests pass: z-axis formula, Cartesian rotation,
two-axis incoming projection, unconjugated reciprocity, matrix correction,
and invalid input rejection. They validate algebra and conventions only, not
Sionna calibration or cross-polar accuracy.

### Independent RIS-only open-gap prediction

The Munich geometry-only run uses isolated BS and UE current bases, constructs
Cartesian free-space fields at each RIS segment centre, projects onto the
vertical RIS tangents, solves the RIS-alone impressed-segment currents, and
predicts `delta_z` with `open_gap_scatter_impedance`. The monolithic open-minus-
free NEC matrix is read only after the prediction is built. Against that
independent NEC target, predicted `delta_z` has `0.04070%` complex relative
error, `0.005604 rad` worst-entry phase error, and matrix-norm amplitude ratio
`0.999915`; these scalar checks pass the unchanged limits. Applying the
predicted impedance to the separately solved free external block gives
`D(Z_EE,free + delta_z_pred)` with `3.22e-8` relative error, `1.29e-8 rad`
phase error, and amplitude ratio `0.99999997` against monolithic NEC direct H.
The open-gap direct-channel change itself is also scored with the same limits.

This is a free-space NEC consistency subcheck, not a ray-traced P4 score. It
does not insert a correction into Sionna's `H_RT,direct`, because the matching
Sionna per-port vector fields and direct channel are not available on the
local CPU environment. Cross-polar is unscored, and the reflection case still
has no independent matching NEC reference. No GPU or remote compute job was
started; the lab VPN was unavailable during this run. Full P4 remains
`P4_NOT_EVALUATED`.

## Reproduction

Run from `atom-wise-ris/` with the pinned NEC environment:

```bash
PYTHONPATH=src:scripts .venv-em/bin/python3 -m unittest tests.test_p4_vector_mapping -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts .venv-em/bin/python3 scripts/p4_dipole_convention_subcheck.py --out outputs/runs/bench_tracks_p4_20261108/dipole_convention_subcheck_round2.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts .venv-em/bin/python3 scripts/p4_vector_mapping_subcheck.py --segments 81 --out outputs/runs/bench_tracks_p4_20261108/vector_mapping_subcheck_round2.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts .venv-em/bin/python3 scripts/p4_open_gap_subcheck.py --segments 81 --out outputs/runs/bench_tracks_p4_20261108/open_gap_subcheck_round2_source_corrected.json
```
