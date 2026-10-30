from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .generation_plan import (
    _paired_latin_hypercube,
    load_batch_report,
    load_generation_plan,
)
from .models import EvaluationConditionV1, PolicyEvaluationPlanV1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_generation_offsets(path: str | Path) -> np.ndarray:
    plan = load_generation_plan(path)
    offsets = np.asarray(
        sorted({tuple(attempt.object_xy_offset_m) for attempt in plan.attempts}),
        dtype=np.float64,
    )
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("source generation plan has no valid XY offsets")
    return offsets


def build_evaluation_plan(
    generation_plan_path: str | Path,
    *,
    plan_id: str,
    count: int = 20,
    seed: int = 20261005,
    range_m: float = 0.02,
    minimum_m2_distance_m: float = 0.001,
) -> PolicyEvaluationPlanV1:
    if count <= 0:
        raise ValueError("evaluation condition count must be positive")
    source_path = Path(generation_plan_path).expanduser().resolve()
    training_offsets = unique_generation_offsets(source_path)
    accepted: list[np.ndarray] = []
    candidate_seed = seed
    while len(accepted) < count:
        candidates = _paired_latin_hypercube(
            max(count * 2, 32), seed=candidate_seed, range_m=range_m
        )
        candidate_seed += 1
        for candidate in candidates:
            if np.linalg.norm(candidate) < minimum_m2_distance_m:
                continue
            if np.min(np.linalg.norm(training_offsets - candidate, axis=1)) < minimum_m2_distance_m:
                continue
            if any(np.linalg.norm(existing - candidate) < minimum_m2_distance_m for existing in accepted):
                continue
            accepted.append(candidate.copy())
            if len(accepted) == count:
                break
        if candidate_seed - seed > 10_000:
            raise RuntimeError("could not construct leakage-free held-out XY conditions")
    conditions = tuple(
        EvaluationConditionV1(
            condition_id=f"heldout-{index:03d}",
            rollout_seed=seed + index,
            object_xy_offset_m=(float(offset[0]), float(offset[1])),
        )
        for index, offset in enumerate(accepted)
    )
    return PolicyEvaluationPlanV1(
        plan_id=plan_id,
        source_generation_plan=source_path,
        source_generation_plan_sha256=_sha256(source_path),
        randomization_seed=seed,
        object_xy_range_m=range_m,
        minimum_m2_distance_m=minimum_m2_distance_m,
        conditions=conditions,
    )


def build_training_evaluation_plan(
    generation_plan_path: str | Path,
    generation_batch_report_path: str | Path,
    *,
    plan_id: str,
    count: int = 10,
    seed: int = 20261105,
) -> PolicyEvaluationPlanV1:
    """Select deterministic, distinct M2 positions with a physical success."""
    if count <= 0:
        raise ValueError("evaluation condition count must be positive")
    plan_path = Path(generation_plan_path).expanduser().resolve()
    report_path = Path(generation_batch_report_path).expanduser().resolve()
    generation_plan = load_generation_plan(plan_path)
    report = load_batch_report(report_path, plan=generation_plan)
    successful_ids = {
        result.attempt_id for result in report.results if result.status == "succeeded"
    }
    successful_offsets: list[tuple[float, float]] = []
    for attempt in generation_plan.attempts:
        if attempt.attempt_id not in successful_ids:
            continue
        if attempt.object_xy_offset_m not in successful_offsets:
            successful_offsets.append(attempt.object_xy_offset_m)
    if len(successful_offsets) < count:
        raise ValueError(
            f"M2 has only {len(successful_offsets)} distinct successful positions; requested {count}"
        )
    selected_indices = np.linspace(0, len(successful_offsets) - 1, count, dtype=int)
    selected = [successful_offsets[int(index)] for index in selected_indices]
    range_m = max(max(abs(value) for value in offset) for offset in successful_offsets)
    conditions = tuple(
        EvaluationConditionV1(
            condition_id=f"m2-train-{index:03d}",
            rollout_seed=seed + index,
            object_xy_offset_m=offset,
        )
        for index, offset in enumerate(selected)
    )
    return PolicyEvaluationPlanV1(
        plan_id=plan_id,
        source_generation_plan=plan_path,
        source_generation_plan_sha256=_sha256(plan_path),
        source_generation_batch_report=report_path,
        source_generation_batch_report_sha256=_sha256(report_path),
        position_split="m2_train",
        sampling="m2_successful_subset",
        randomization_seed=seed,
        object_xy_range_m=max(range_m, 1e-9),
        minimum_m2_distance_m=0.001,
        conditions=conditions,
    )


def load_evaluation_plan(path: str | Path) -> PolicyEvaluationPlanV1:
    plan_path = Path(path).expanduser().resolve()
    plan = PolicyEvaluationPlanV1.model_validate_json(plan_path.read_text(encoding="utf-8"))
    if _sha256(plan.source_generation_plan) != plan.source_generation_plan_sha256:
        raise ValueError("source generation plan digest does not match evaluation plan")
    generation_plan = load_generation_plan(plan.source_generation_plan)
    training_offsets = unique_generation_offsets(plan.source_generation_plan)
    if plan.position_split == "heldout":
        for condition in plan.conditions:
            distance = float(
                np.min(
                    np.linalg.norm(
                        training_offsets - np.asarray(condition.object_xy_offset_m), axis=1
                    )
                )
            )
            if distance < plan.minimum_m2_distance_m:
                raise ValueError(
                    f"condition {condition.condition_id} leaks into the M2 position set"
                )
    else:
        assert plan.source_generation_batch_report is not None
        assert plan.source_generation_batch_report_sha256 is not None
        if _sha256(plan.source_generation_batch_report) != plan.source_generation_batch_report_sha256:
            raise ValueError("source generation batch report digest does not match evaluation plan")
        report = load_batch_report(plan.source_generation_batch_report, plan=generation_plan)
        successful_ids = {
            result.attempt_id for result in report.results if result.status == "succeeded"
        }
        successful_offsets = {
            attempt.object_xy_offset_m
            for attempt in generation_plan.attempts
            if attempt.attempt_id in successful_ids
        }
        for condition in plan.conditions:
            if condition.object_xy_offset_m not in successful_offsets:
                raise ValueError(
                    f"condition {condition.condition_id} was not a successful M2 training position"
                )
    return plan


def evaluation_plan_sha256(plan: PolicyEvaluationPlanV1) -> str:
    return hashlib.sha256(plan.model_dump_json().encode("utf-8")).hexdigest()


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _mcnemar_exact(baseline_only: int, augmented_only: int) -> float:
    discordant = baseline_only + augmented_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, augmented_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _trajectory_diagnostics(episode_path: str | Path) -> dict[str, float | int | None]:
    trajectory_path = Path(episode_path) / "trajectory.npz"
    with np.load(trajectory_path) as arrays:
        ee_position = np.asarray(arrays["ee_pose"], dtype=np.float64)[:, :3]
        object_position = np.asarray(arrays["object_pose"], dtype=np.float64)[:, :3]
        closed = np.asarray(arrays["policy_gripper_filtered_closed"], dtype=bool).reshape(-1)
        replan_ids = np.asarray(arrays["policy_replan_id"], dtype=np.int64).reshape(-1)
        requested = np.asarray(arrays["requested_action"], dtype=np.float64)[:, :7]
    delta = ee_position - object_position
    distance_3d = np.linalg.norm(delta, axis=1)
    distance_xy = np.linalg.norm(delta[:, :2], axis=1)
    close_indices = np.flatnonzero(closed)
    first_close = int(close_indices[0]) if len(close_indices) else None
    before_stop = first_close + 1 if first_close is not None else len(distance_3d)
    boundary_indices = np.flatnonzero(replan_ids[1:] != replan_ids[:-1]) + 1
    boundary_jumps = (
        np.max(np.abs(requested[boundary_indices] - requested[boundary_indices - 1]), axis=1)
        if len(boundary_indices)
        else np.empty(0, dtype=np.float64)
    )
    return {
        "first_applied_close_frame": first_close,
        "first_close_hand_object_distance_m": (
            float(distance_3d[first_close]) if first_close is not None else None
        ),
        "first_close_hand_object_xy_distance_m": (
            float(distance_xy[first_close]) if first_close is not None else None
        ),
        "first_close_hand_object_z_delta_m": (
            float(delta[first_close, 2]) if first_close is not None else None
        ),
        "minimum_hand_object_distance_m": float(distance_3d.min()),
        "minimum_preclose_hand_object_distance_m": float(distance_3d[:before_stop].min()),
        "median_replan_boundary_joint_jump_rad": _median(boundary_jumps.tolist()),
        "max_replan_boundary_joint_jump_rad": (
            float(boundary_jumps.max()) if len(boundary_jumps) else None
        ),
    }


def diagnose_evaluation_splits(
    reports: dict[str, str | Path],
    *,
    output_json: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    """Compare nominal, M2-train, and held-out behavior for one checkpoint."""
    required = {"nominal", "m2_train", "heldout"}
    if set(reports) != required:
        raise ValueError(f"diagnostic reports must contain exactly {sorted(required)}")
    payloads = {
        split: json.loads(Path(path).read_text(encoding="utf-8"))
        for split, path in reports.items()
    }
    reference = payloads["heldout"]
    fixed_keys = (
        "scene", "policy_checkpoint", "policy_config", "control_fps",
        "replan_steps", "inference_steps", "full_horizon", "gripper_filter",
    )
    for split, payload in payloads.items():
        if payload.get("status") != "completed":
            raise ValueError(f"{split} report is not completed")
        for key in fixed_keys:
            if payload.get(key) != reference.get(key):
                raise ValueError(f"{split} report differs on fixed setting {key}")
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    stages = ("grasp", "lift", "hold", "release", "success")
    for split in ("nominal", "m2_train", "heldout"):
        split_rows = []
        for rollout in payloads[split]["rollouts"]:
            diagnostics = _trajectory_diagnostics(rollout["episode_path"])
            row = {
                "split": split,
                "condition_id": rollout["condition_id"],
                "object_x_offset_m": rollout["object_xy_offset_m"][0],
                "object_y_offset_m": rollout["object_xy_offset_m"][1],
                **{stage: bool(rollout["metrics"].get(stage, rollout.get(stage))) for stage in stages},
                "raw_gripper_switches": rollout["gripper_diagnostics"]["raw_switches"],
                "applied_gripper_switches": rollout["gripper_diagnostics"]["applied_switches"],
                **diagnostics,
            }
            rows.append(row)
            split_rows.append(row)
        total = len(split_rows)
        summaries[split] = {
            "rollouts": total,
            "stages": {
                stage: {
                    "successes": sum(bool(row[stage]) for row in split_rows),
                    "rate": sum(bool(row[stage]) for row in split_rows) / total,
                }
                for stage in stages
            },
            "median_applied_gripper_switches": _median(
                [float(row["applied_gripper_switches"]) for row in split_rows]
            ),
            "median_first_close_hand_object_distance_m": _median(
                [
                    float(row["first_close_hand_object_distance_m"])
                    for row in split_rows
                    if row["first_close_hand_object_distance_m"] is not None
                ]
            ),
            "median_minimum_preclose_hand_object_distance_m": _median(
                [float(row["minimum_preclose_hand_object_distance_m"]) for row in split_rows]
            ),
            "median_max_replan_boundary_joint_jump_rad": _median(
                [
                    float(row["max_replan_boundary_joint_jump_rad"])
                    for row in split_rows
                    if row["max_replan_boundary_joint_jump_rad"] is not None
                ]
            ),
        }
    train_rate = summaries["m2_train"]["stages"]["success"]["rate"]
    heldout_rate = summaries["heldout"]["stages"]["success"]["rate"]
    diagnosis = (
        "training_action_or_executor_failure"
        if train_rate <= 0.2
        else "position_generalization_failure"
        if train_rate >= 0.8 and heldout_rate <= 0.2
        else "mixed_training_and_generalization_failure"
    )
    result = {
        "status": "completed",
        "review_status": "SELF_REVIEWED_PROVISIONAL",
        "checkpoint": reference["policy_checkpoint"],
        "fixed_settings_verified": True,
        "summary": summaries,
        "diagnosis": diagnosis,
        "rows": rows,
    }
    json_path = Path(output_json).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = Path(output_csv).expanduser().resolve()
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def compare_evaluation_reports(
    baseline_path: str | Path,
    augmented_path: str | Path,
    *,
    output_json: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    augmented = json.loads(Path(augmented_path).read_text(encoding="utf-8"))
    if baseline.get("status") != "completed" or augmented.get("status") != "completed":
        raise ValueError("paired comparison requires two completed evaluation reports")
    keys = (
        "scene",
        "evaluation_plan_sha256",
        "requested_rollouts",
        "policy_config",
        "control_fps",
        "replan_steps",
        "inference_steps",
        "full_horizon",
        "gripper_filter",
    )
    for key in keys:
        if baseline.get(key) != augmented.get(key):
            raise ValueError(f"evaluation reports differ on paired setting {key}")
    baseline_rollouts = {item["condition_id"]: item for item in baseline["rollouts"]}
    augmented_rollouts = {item["condition_id"]: item for item in augmented["rollouts"]}
    if baseline_rollouts.keys() != augmented_rollouts.keys():
        raise ValueError("evaluation reports do not contain identical condition IDs")
    stages = ("grasp", "lift", "hold", "release", "success")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for stage in stages:
        baseline_values = [bool(baseline_rollouts[key]["metrics"].get(stage, baseline_rollouts[key].get(stage))) for key in sorted(baseline_rollouts)]
        augmented_values = [bool(augmented_rollouts[key]["metrics"].get(stage, augmented_rollouts[key].get(stage))) for key in sorted(augmented_rollouts)]
        total = len(baseline_values)
        baseline_successes = sum(baseline_values)
        augmented_successes = sum(augmented_values)
        baseline_only = sum(a and not b for a, b in zip(baseline_values, augmented_values))
        augmented_only = sum(b and not a for a, b in zip(baseline_values, augmented_values))
        summary[stage] = {
            "total": total,
            "baseline_successes": baseline_successes,
            "baseline_rate": baseline_successes / total,
            "baseline_wilson95": _wilson(baseline_successes, total),
            "augmented_successes": augmented_successes,
            "augmented_rate": augmented_successes / total,
            "augmented_wilson95": _wilson(augmented_successes, total),
            "delta_rate": (augmented_successes - baseline_successes) / total,
            "baseline_only": baseline_only,
            "augmented_only": augmented_only,
            "mcnemar_exact_two_sided_p": _mcnemar_exact(baseline_only, augmented_only),
        }
    for condition_id in sorted(baseline_rollouts):
        base = baseline_rollouts[condition_id]
        aug = augmented_rollouts[condition_id]
        if base["object_xy_offset_m"] != aug["object_xy_offset_m"]:
            raise ValueError(f"condition {condition_id} has mismatched object offsets")
        if base["frames"] != aug["frames"]:
            raise ValueError(f"condition {condition_id} has mismatched rollout horizons")
        row: dict[str, Any] = {
            "condition_id": condition_id,
            "object_x_offset_m": base["object_xy_offset_m"][0],
            "object_y_offset_m": base["object_xy_offset_m"][1],
            "frames": base["frames"],
        }
        for stage in stages:
            row[f"baseline_{stage}"] = bool(base["metrics"].get(stage, base.get(stage)))
            row[f"augmented_{stage}"] = bool(aug["metrics"].get(stage, aug.get(stage)))
        rows.append(row)
    success_delta = summary["success"]["delta_rate"]
    success_p = summary["success"]["mcnemar_exact_two_sided_p"]
    direction = (
        "positive" if success_delta > 0 else
        "negative" if success_delta < 0 else "no_difference"
    )
    claim_status = (
        "supported_seed0" if success_delta > 0 and success_p < 0.05 else
        "inconclusive_positive_direction" if success_delta > 0 else
        "not_supported" if success_delta < 0 else "no_difference"
    )
    payload = {
        "status": "completed",
        "review_status": "SELF_REVIEWED_PROVISIONAL",
        "paired_conditions": len(rows),
        "baseline_checkpoint": baseline["policy_checkpoint"],
        "augmented_checkpoint": augmented["policy_checkpoint"],
        "evaluation_plan_sha256": baseline["evaluation_plan_sha256"],
        "summary": summary,
        "observed_direction": direction,
        "claim_c2_seed0": claim_status,
        "replication_gate": (
            "eligible_for_two_additional_training_seeds"
            if success_delta > 0
            else "do_not_expand_training_seeds"
        ),
        "conditions": rows,
    }
    json_output = Path(output_json).expanduser().resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_output = Path(output_csv).expanduser().resolve()
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload
