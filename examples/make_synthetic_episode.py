"""Write a clearly labeled CPU fixture, never a physical or human demonstration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from robot_data_contracts import (
    AugmentationLineageV1, EpisodeWriter, EvaluatorResultV1, FeatureRole,
    FeatureSpecV1, SceneProvenanceV1, SoftwareProvenanceV1,
)
from robot_data_contracts.profiles.franka import FRANKA_JOINT8_ACTION, franka_feature_specs
from robot_data_generation.models import ObjectXYRandomizationV1, RigidCubeLiftConfigV1, SourceEpisodeRefV1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    features = franka_feature_specs() + tuple(
        FeatureSpecV1(key=key, role=FeatureRole.AUXILIARY, dtype="float32", shape=(7,))
        for key in ("ee_pose", "object_pose")
    )
    writer = EpisodeWriter(
        output / "raw", fps=10, robot_id="franka_panda",
        control_source="synthetic_fixture", features=features,
        action_profile=FRANKA_JOINT8_ACTION, camera_names=(), camera_calibration={},
        scene=SceneProvenanceV1(scene_id="rigid_cube_lift", seed=0,
            parameters={"synthetic_fixture": True, "physics_executed": False}),
        software=SoftwareProvenanceV1(python_version=platform.python_version(),
            platform=platform.system(), engine_id="synthetic_fixture",
            engine_version="none", worker_id="cpu_example"),
        # The source loader requires this label. It is an eligibility fixture,
        # not evidence of teleoperation; control_source and provenance say so.
        lineage=AugmentationLineageV1(generation_method="teleoperation"),
    )
    writer.start("SYNTHETIC FIXTURE: cube lift schema and planning smoke only")
    for frame in range(12):
        values = {spec.key: np.zeros(spec.shape, dtype=spec.dtype) for spec in features}
        width = 0.0 if 4 <= frame < 10 else 0.08
        for key in ("requested_action", "safe_action", "applied_action"):
            values[key][7] = width
        values["ee_pose"][:] = (0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0)
        height = 0.277 + ({5: 0.03, 6: 0.10, 7: 0.12, 8: 0.12, 9: 0.12}.get(frame, 0.0))
        values["object_pose"][:] = (0.5, 0.0, height, 1.0, 0.0, 0.0, 0.0)
        writer.add_frame(images={}, values=values, simulation_time_s=frame / 10,
            capture_timestamp_ns=1_000_000 + frame * 100_000_000, control_sequence=frame)
    episode = writer.finish(operator_success=True, evaluator=EvaluatorResultV1(
        evaluator_id="synthetic_fixture_only", evaluator_version="1", available=True,
        success=True, metrics={"synthetic_fixture": True, "physics_executed": False},
    ))
    manifest = json.loads((episode / "manifest.json").read_text())
    config = RigidCubeLiftConfigV1(spec_id="synthetic_cpu_example", attempts_per_source=4,
        randomization_seed=7, transition_frames=2,
        object_xy=ObjectXYRandomizationV1(range_m=0.02),
        sources=(SourceEpisodeRefV1(path=episode, episode_uuid=manifest["episode_uuid"]),))
    (output / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    print(output / "config.json")


if __name__ == "__main__":
    main()
