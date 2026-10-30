# Executor boundary

The CPU CLI writes a resolved `GenerationPlanV1`. It does not contain a simulator
or command a robot. Each attempt identifies its source Episode, seed, XY offset
and approach transition length. The executor owns the following steps:

1. Load the source and recover controller target poses by FK over `applied_action`
   using the loaded robot and correct hand reference frame. Same-frame observed
   `ee_pose` is not a substitute for command-aligned FK targets.
2. Call `retarget_rigid_cube_lift` with those poses, offset and target control rate.
3. Restore a controlled initial state, apply the resolved object offset and run
   IK, control safety and fresh physics. Delay closure until pregrasp convergence.
4. Write a canonical Episode with parent UUID, `retargeted` lineage, resolved
   seed/offset, action semantics and native evaluator outcome.
5. Persist every attempt via the generation report helpers, including failures;
   resume only against a matching plan digest.

The tested executor is `gello_remote.isaaclab_gello` in trajectory mode, an
optional external implementation. A copied snapshot ran with independently
installed Generation/contracts wheels on RTX 5090 / Isaac Sim 6.0.1 / PhysX:
**5/5 nominal and 7/10 XY-retargeted attempts passed**, using the official Panda,
five source demonstrations and two pre-resolved paired offsets. All 15 retained
Episodes passed readback, including three task failures. This establishes bounded
fresh-physics compatibility for this executor/asset/task combination. Broad task
coverage and a complete standalone executor distribution/install remain unverified.
No Studio checkout, supervisor environment variables or API is required by this
candidate. Integrations may consume the files without becoming core dependencies.

Direct invocation with the tested runtime needs explicit `--rendering_mode balanced`;
an omitted mode became `None` and was rejected by the renderer profile check.
An initial smoke with a local Franka USD failed pregrasp convergence. A matched
control and the subsequent complete batch passed with
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd`
via explicit `--franka-usd`. Their FK targets nearly matched, but actual tracking
did not. The matched result supports asset-dependent physical/control behavior;
the specific differing USD property remains unidentified. Use the verified asset
when reproducing this bounded batch; a similarly named Panda asset is
not sufficient evidence of compatibility.
See [validation](VALIDATION.md) for the evidence boundary.

Training-selection tools create filesystem views with symlinks; they do not export
LeRobot or train policies. Evaluation tools compare existing reports. Neither a
plan nor a successful data read implies physical success or improved learning.

