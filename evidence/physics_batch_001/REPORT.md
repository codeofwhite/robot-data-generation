# Standalone generation: bounded physics validation

Date: 2026-10-30 (Asia/Shanghai). Status: **BOUNDED_PHYSICS_BATCH_PASS**.
Evidence review: **SELF_REVIEWED_PROVISIONAL**.

Independent Generation/contracts wheels ran through an external GELLO executor,
without Studio services, on RTX 5090 / Isaac Sim 6.0.1 / PhysX / official Panda.

| Stage | Attempts | Successes | Failures |
| --- | ---: | ---: | ---: |
| Nominal replay | 5 | 5 | 0 |
| Position-retargeted generation | 10 | 7 | 3 |

The registered gate was all five nominal replays, followed by at least five of ten
retargeted successes. All ten attempts remain in the denominator. Retargeting
used five source demonstrations and two pre-resolved paired XY offsets within
+/-2 cm. The two offsets are recorded in summary.json; they are two unique
positions, not ten independent positions.

Each attempt received fresh physics, robot FK/IK, control safety, sensing and
native task evaluation. Success requires closed-gripper lift >=8 cm; hold/release
are separate diagnostics. The unchanged pregrasp gate requires 3 mm convergence
within 2 s. Two attempts timed out there; one closed without lifting the cube.
All failed Episodes were retained. No retry or selective omission was used.

Full readback passed for 15 Episodes, 3,407 frames and 13,628 JPEGs: finite arrays,
frame/camera alignment, monotonic clocks, parent lineage, seeds, resolved initial
placement and evaluator/report agreement. Original source files (4,738) retained
identical hashes. Sanitized per-attempt outcomes are in [attempts.json](attempts.json).

Simulator stages took 834.04 s. Peak observed VRAM was 6,166 MiB at 5 s samples,
with instantaneous peaks possibly higher; the GPU was released. The configured
10 Hz recorded timeline does not measure wall-clock throughput.

## Media

[Success video](randomized-success.mp4) · [Failure video](randomized-failure.mp4)

Both are H.264/yuv420p/faststart, 1280x960, completely decoded locally.
Success: 183 frames / 18.3 s; failure: 338 frames / 33.8 s. App playback is not
user-confirmed. Videos replay saved fresh simulated execution; they do not run
physics in a browser. Camera order: base/top above side/wrist.

## Provenance and scope

[summary.json](summary.json) records counts, conditions and video properties.
[provenance.json](provenance.json) records source revisions, raw-report hashes and
public artifact hashes. Individual manifest/trajectory hashes are in attempts.json.
Internal logs, machine paths, source Episodes, simulator assets and environments
are not published. Checksums identify retained evidence; they do not reconstruct
undistributed original data or SDK assets.

An initial local Franka asset failed pregrasp convergence. A matched official-Panda
replay passed before this batch. The asset-dependent behavior is documented in
[validation](../../docs/VALIDATION.md); its exact USD-property cause is unresolved.
This supports bounded runtime/fresh-physics compatibility, not a new augmentation
method, policy improvement, broad task coverage or real-robot validity.
