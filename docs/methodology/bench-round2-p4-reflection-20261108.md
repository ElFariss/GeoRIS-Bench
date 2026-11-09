# P4 controlled reflection reference — declared 2026-11-08 23:25 WIB

The controlled reflector is a horizontal, perfectly electrically conducting
plane at z=0 with the original registered Munich development port centres
and 5.53 GHz, 81 segments/wire, 24 mm wires and 50 µm radius. The reference is
PyNEC 2.3.4 GN type 1 (perfect ground), independently solved before reading
any Sionna Jones output. Subtract the independent free-space case to
isolate the reflected contribution. PyNEC Cartesian NE electric fields are
normalized to one ampere at the impressed-voltage source feed. Use the same
1% complex, 0.01 rad, [0.99,1.01] amplitude and 1% cross-polar thresholds, and
require all 130 field links and both external mutual directions.

Also solve the complete 66-port PEC reference to record the RIS/external
port-block change against the existing independent free-space monolithic
reference. Score this change separately: the isolated single-bounce
model does not include all PEC–RIS–antenna re-interactions. Its error does
not establish or excuse any threshold change. The controlled PEC model is
not a reference for Munich's native building reflections and cannot close
the originally declared urban gate. There is no fitting or output alignment.

## Results

The independent `p4-pec-reference-20261108` local tmux run wrote `PEC_DONE`
and `outputs/runs/p4_calibrated_20261108/independent_pec_reference.npz`,
SHA-256 `282a9294666287a1568475140133a1cf0e62e1a0e49ab21240e1f331027f75cd`.
All complex values were finite; monolithic 66-port reciprocity residual was
`4.51e-8`. The P4 scorer read this frozen result only after the Sionna
controlled trace completed. It checked that both artifacts use exactly
the registered port centres.

| Controlled plane comparison | Complex relative error | Worst phase error |
|---|---:|---:|
| Reflected field projection, worst of 130 ports | 0.03447% | 0.0003241 rad |
| Reflected external mutual impedance | 0.01726% | 4.04e-6 rad |
| Reflected RIS voltage block vs monolithic PEC/free difference | 0.03914% | 0.0003499 rad |

Worst field cross-polar leakage was **0.01051%**, and all per-port
amplitude ratios and other metrics met the unchanged limits. The
controlled PEC reflection passed. This does not provide an independent
reference for the native urban specular paths.

The local runner is `.cache/p4-calibrated-stage/pec.sh`. The compact
verdict is `outputs/runs/p4_calibrated_20261108/full_gate.json`, SHA-256
`6234fa04d53d99b9852cf3c0ce44dbc5ed82bf9b267ee46b279a6f453461227e`.
The validator `scripts/p4_gate_registered_suite.py` records the controlled
result alongside the registered LOS result and the blocked urban reflection.
No P4 job is still running.
