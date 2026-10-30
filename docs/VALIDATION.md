# Validation

Date: 2026-10-30 (Asia/Shanghai). Evidence review: **SELF_REVIEWED_PROVISIONAL**.

## CPU package checks

Generation passed 22 tests and contracts passed 27 tests against installed wheels.
Both packages built as wheels and source archives; source archives were unpacked
and rebuilt independently. The initial Python 3.14.6 environment reused existing
NumPy/Pydantic/Pillow dependencies. It was an isolated local check, not a
clean-machine installation. Release preparation additionally installed both local source packages, with their
declared test extras, into a fresh Python 3.14.6 virtual environment without system
site-packages. The documented one-checkout install succeeded; **49 tests passed**
(22 Generation + 27 contracts) against the installed packages, and `pip check`
reported no broken requirements. Resolved versions were NumPy 2.5.4, Pydantic
2.14.2, pytest 9.2.0 and Pillow 13.0.0. No Studio package or simulator was installed
in that environment. This is a fresh Python installation check on the existing
host, not a clean-machine simulator/SDK check.

## Fresh-physics batch

The installed Generation/contracts wheels ran through an unchanged, external
GELLO executor on one RTX 5090, Isaac Sim 6.0.1 and PhysX with the official Panda.
No Studio API, database, supervisor or `robot_data_studio` module was used.

| Stage | Attempts | Successes | Failures | Gate |
| --- | ---: | ---: | ---: | --- |
| Nominal replay | 5 | 5 | 0 | All five pass |
| XY retargeting | 10 | 7 | 3 | At least five pass |

Ten retargeted attempts are five sources × **two fixed paired offsets**, not ten
independent positions. The offsets are (-0.011103615, 0.006001827) and
(0.018799013, -0.013020399) metres, sampled within +/-2 cm before launch.
Every candidate ran fresh physics with command-aligned FK targets, IK, control
safety and native evaluation. Success is a closed-gripper lift of at least 8 cm;
hold/release are separate diagnostics. No convergence threshold was relaxed.

All 15 saved Episodes passed full readback: 3,407 frames, 13,628 JPEGs, finite
aligned arrays, monotonic clocks, four-camera alignment, expected parent/seed,
resolved object placement and evaluator/report agreement. All 4,738 original
source files retained their hashes. Failures are retained: two pregrasp timeouts
and one closure without lifting the cube.

Simulator stages took 834.04 s. Peak observed GPU memory was 6,166 MiB at 5 s
samples; instantaneous peak may be higher. The GPU was released. Configured
120 Hz physics / 20 Hz control / 10 Hz recorded timeline do not imply wall-clock
throughput. Public previews use H.264/yuv420p/faststart, 1280x960; frame counts,
durations and complete local decoding were checked. App playback is unconfirmed.

The [public report](../evidence/physics_batch_001/REPORT.md),
[attempts](../evidence/physics_batch_001/attempts.json),
[summary](../evidence/physics_batch_001/summary.json) and
[provenance](../evidence/physics_batch_001/provenance.json) retain the results
without publishing private paths, original Episodes or internal logs.

## Asset compatibility finding

An initial local Franka USD failed pregrasp convergence at 29.34 mm against the
3 mm gate. A matched run using the official Panda asset passed at 1.54 mm final
post-step error and completed grasp/lift/hold/release. The official asset then
passed the full batch above. Only asset/output/audit paths changed in the matched
control. FK targets nearly matched but actual tracking differed. This supports
asset-dependent physical/control behavior; the specific USD property was not
isolated. Use the asset documented in [executor requirements](EXECUTOR.md).

## Limits

This validates the selected executor, official asset, five sources and two offsets
on an existing simulator host. It does not establish improved policy learning,
broad task/position coverage, deformable physics, SDK installation on a clean
machine, hardware safety or real-robot results. No model was trained in this
validation. Historical 79/100 generation and inconclusive policy results remain
separate in the [historical summary](../evidence/historical_summary.json).
