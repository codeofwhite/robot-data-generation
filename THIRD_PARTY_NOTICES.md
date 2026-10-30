# Third-party notices

Original Robot Data Generation contributions are licensed under [Apache-2.0](LICENSE).
The owner's license selection does not grant rights to third-party code or assets.

## Bundled protocol package

`packages/robot-data-contracts/` is a standalone source snapshot of
`robot-data-contracts` 0.2.0, originally extracted from the public RoboData Studio
contracts package. The existing license, NOTICE and third-party notices are
preserved in that directory. Source revision and file hashes are recorded in its
SOURCE_SNAPSHOT.json. The original suite notices describe their original context;
GELLO, the web console and streaming libraries are not bundled here.

## External dependencies and executors

NumPy, Pydantic, optional test/image packages, build tools and their dependencies
retain their own terms. Dependencies are installed separately, not vendored as
Python environments. No simulator SDK, GELLO executor, robot/scene USD, model,
checkpoint, texture library, calibration or original demonstration dataset is
included. Isaac Sim, IsaacLab and the selected official Panda asset must be
obtained separately under their applicable terms. Product names identify sources
and do not imply sponsorship or endorsement.

## Public validation media

The two MP4 previews and contact sheets under evidence/physics_batch_001/ derive
from this project's recorded simulator runs. They are rendered-output examples,
not redistributions of the underlying USD assets or SDKs. Attribution and the
simulation/evidence limits accompany them in REPORT.md and provenance.json.
The project license grants no rights in underlying third-party materials beyond
those that their owners provide. No real hardware recordings are included.
