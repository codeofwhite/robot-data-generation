# Public preview scope

This preview includes rigid augmentation/planning code, CPU tests, an explicitly
synthetic example, a standalone bundled Episode protocol source package, licensing,
source hashes and sanitized results from the fresh-physics validation.

Public validation media are saved simulator outputs. No underlying robot USD,
SDK, original demonstration dataset, checkpoint, texture library, private config,
internal log, local environment or historical Git metadata is distributed.
Cloth appearance modules and private renderer integration are excluded.

Original Generation contributions are licensed Apache-2.0 at the owner's request.
The protocol package retains its own existing Apache-2.0 license and all original
attribution. See the root LICENSE, NOTICE and THIRD_PARTY_NOTICES.md.

Install the protocol and Generation from one checkout as documented in README.
The protocol is independently installable and pinned to 0.2.0; Episode v1 and
core algorithms are unchanged. No Studio installation or third repository is
required. NumPy/Pydantic/build dependencies and simulator/executor/assets remain
external. This does not migrate Studio to a new protocol distribution.

The GPU/physics gate passed 5/5 nominal and 7/10 retargeted attempts with the
verified official Panda. Failed/recovered historical gates and evidence limits
remain visible. Validation does not imply policy improvement or real-robot safety.

This is a prepared local release candidate. No GitHub repository creation,
push, publication, PyPI upload or change to an existing public repository is
performed by this preparation. The scoped content review found no matched private machine paths or key patterns,
Markdown file links resolved, source/evidence hashes matched, and the fresh Python
installation/tests passed. Both packages were rebuilt with licensing included;
source distributions also rebuilt independently. This screening is not a blanket
legal/security certification. Destination approval precedes publication. Internal
audit/build logs stay under ignored .verification/.
