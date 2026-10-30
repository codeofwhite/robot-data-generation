# Source configuration

The CPU example writes a complete `config.json` with resolved local paths and UUIDs.
Alternatively copy `rigid_cube_lift.template.json` and replace its explicit path
and UUID placeholders before use; the template is not a runnable data source.
For real sources, create the same configuration with `format_version: 1`,
`scene_id: rigid_cube_lift`, a unique `spec_id`, positive `attempts_per_source`,
`randomization_seed`, `transition_frames >= 2`, `object_xy.range_m` and `sources`.
Each source supplies `path` and `episode_uuid`. Paths are currently resolved against
the invoking process's working directory; prefer absolute paths in local configs.

Sources must contain finite `ee_pose [T,7]`, `object_pose [T,7]`, and
`applied_action [T,8]` arrays. Eligibility checks require original teleoperation
lineage, operator success, native evaluator success, and a unique grasp interval
with approach/release frames and sufficient object lift. External QA acceptance is
not stored or checked by this loader; select reviewed source data explicitly.

`execution` defaults to 20 Hz, pregrasp convergence tolerances and an 8 cm lift
success threshold. Preserve explicit settings for reproducibility. The plan is
not a physics-valid trajectory until an executor runs it and records the outcome.
