# Robot Data Contracts

Standalone robot-data-contracts 0.2.0 source snapshot bundled for installation
without Studio. Import namespace: robot_data_contracts. Episode format: v1.

From the Generation checkout root:

```bash
python -m pip install ./packages/robot-data-contracts .
```

This directory can also be built or installed independently. Its Python source
and tests are unchanged from the recorded source snapshot. See
[SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) for origin/hashes and the enclosing Generation checkout's `docs/CONTRACTS.md` for version/update policy.
That guide is not required to build this standalone package.

NumPy and Pydantic are runtime dependencies. The optional images extra installs
Pillow for JPEG writing; the test extra adds pytest and Pillow. USD helper calls
require an external USD runtime; core Episode access does not import a simulator.

The original Apache-2.0 LICENSE, NOTICE and THIRD_PARTY_NOTICES.md are retained.
They include historical suite attribution, not additional bundled components.
