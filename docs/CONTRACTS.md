# Protocol distribution and compatibility

The required `robot-data-contracts==0.2.0` source is provided under
`packages/robot-data-contracts/` in this repository. It is an independent Python
package; it is not merged into the Generation module or imported from Studio.

From the checkout root:

```bash
python -m pip install ./packages/robot-data-contracts .
```

For tests, install the declared extras into your chosen environment:

```bash
python -m pip install './packages/robot-data-contracts[test]' '.[test]'
python -m pytest tests packages/robot-data-contracts/tests
```

For an offline installation, first build/acquire both wheels and all their
NumPy/Pydantic/build dependencies in a connected environment. Then use a prepared
wheelhouse with `pip --no-index --find-links`. This repository does not contain
third-party dependency wheels and does not promise offline dependency installation.
Installing Generation alone from PyPI is not a supported route in this preview;
no registry publication is assumed.

## Source and update policy

The bundled package is a versioned source snapshot, originally extracted from the
Studio contracts package at the revision recorded in its SOURCE_SNAPSHOT.json.
Its Python source and existing tests retain exactly the recorded hashes. Its
LICENSE, NOTICE and THIRD_PARTY_NOTICES.md preserve upstream attribution. The
additional README/validation text describes this distribution context.

This is one protocol implementation packaged separately, not an alternate Episode
schema. Generation depends on its version, not on Studio's directory layout,
API, database or installation. Studio has not been switched to a different package
by this release preparation.

Treat the bundled snapshot as a pinned dependency: do not independently edit
schema code here. Future protocol changes require a reviewed authoritative
package version, compatibility tests, updated source hashes and an explicit
Generation dependency update. Version 0.2.0 and the `robot_data_contracts` import
namespace stay unchanged; Episode `format_version` remains 1.
