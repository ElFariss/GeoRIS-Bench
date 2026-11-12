# CoupledRIS-Bench Phase 0 audit design (2026-11-08)

Declare before execution. This is an audit, not Phase 1 or P4. Use Sionna RT
2.0.1 on 3060-1 under the project account. No channels are generated in an urban scene.
No raw mesh, ray path file, or scene render leaves the host; retrieve summary JSON
only. No held-out benchmark data is read.

- **Munich:** compare exact DeepMIMO-export face connectivity with native Sionna
  downward intersections at the historical v3 layout positions. Reproduce the
  disputed footprint test with the lowest horizontal face and report how many
  historical users are geometrically inside its footprint below its top. Compare
  native first-hit height under the original rule (<0.5 m); also report the
  first-hit shape and whether ground is at eye height or below. Render the
  native first-hit height raster on the audit host. For a bounded fix demonstration
  keep the source mesh unchanged: choose the nearest point on a 1 m grid, up to
  45 m in XY, with a ground first hit (<0.5 m and at least 0.2 m below the probe)
  and a qualifying 3×3 ground neighborhood, ties by x then y. Demonstrate for
  the RIS and first invalid historical UE, if one exists. If the old points
  already pass native, do not move them just to match the export; use a 1 m
  grid location inside the disputed footprint that passes native as a check.
  These are geometry diagnostics, not Phase 1 positions or evidence of
  facade suitability, street access, radio blockage quotas, or P4 calibration.
- **RT capabilities:** run 64/128/256 separate physical receiver positions on a
  vertical 8-row lattice (28 mm column, 30 mm row pitch) with two polarization
  ports per receiver and a single dual-polarized source at 5.53 GHz, explicitly
  `synthetic_array=False`. Use a controlled free-space LOS solve, reporting
  raw wall times for three repeats, shape, finite count and co-/cross-polar
  response. This lower bound is not urban scale performance. A pass requires
  every probe represented and one finite nonzero co-polar path per probe.
  Separately use a synthetic thin wall to confirm a blocked LOS path returns
  a diffraction interaction with edge diffraction enabled, and to confirm
  transmission is absent when disabled and responds to two declared slab
  thicknesses (0.05 m and 0.20 m). Inspect the solver's material model and
  whether the native scene supplies per-wall thickness; keep indoor users out
  of scope unless that metadata and physical support are established.
- **Storage:** resolve and install the optional `dataset` extra with the project
  lock, round-trip synthetic complex128 HDF5 and typed per-frame Parquet. Keep
  audit JSON only; no benchmark channels or split files yet.
- **Licensing:** positive evidence for retrace and derived redistribution, checked
  separately from software licensing, is required for every hosted source.
  Candidate licenses are recommendations for acceptance. Missing information
  means unresolved, no data asset ingest and no Phase 1 tracing. A proposed
  retrace-only workflow is separate from authorization to distribute assets.
