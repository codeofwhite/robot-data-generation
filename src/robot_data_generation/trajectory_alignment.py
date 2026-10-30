from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation_plan import load_evaluation_plan
from .generation_plan import load_batch_report, load_generation_plan


def _load_trajectory(path: str | Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    trajectory_path = Path(path).expanduser().resolve() / "trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as archive:
        missing = set(keys) - set(archive.files)
        if missing:
            raise ValueError(f"{trajectory_path} is missing arrays: {sorted(missing)}")
        return {key: np.asarray(archive[key]).copy() for key in keys}


def _first_true(values: np.ndarray, *, label: str) -> int:
    indices = np.flatnonzero(np.asarray(values, dtype=bool).reshape(-1))
    if not len(indices):
        raise ValueError(f"trajectory has no {label}")
    return int(indices[0])


def _phase_events(phase_id: np.ndarray) -> dict[str, int]:
    phases = np.asarray(phase_id, dtype=np.int64).reshape(-1)
    return {
        name: _first_true(phases >= value, label=f"{name} phase")
        for name, value in (("close", 1), ("grasp", 2), ("lift", 3), ("release", 4))
    }


def _command_close_frame(applied_action: np.ndarray) -> int:
    actions = np.asarray(applied_action, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 8:
        raise ValueError(f"applied_action must have shape (T, >=8), got {actions.shape}")
    return _first_true(actions[:, 7] <= 0.04, label="gripper close command")


def _angular_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0)))


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def align_m5b_trajectories(
    *,
    generation_plan_path: str | Path,
    generation_report_path: str | Path,
    evaluation_plan_path: str | Path,
    evaluation_report_path: str | Path,
    norm_stats_path: str | Path,
    output_json: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    """Align failed policy rollouts with every successful M2 demo at the same XY."""
    generation_plan = load_generation_plan(generation_plan_path)
    generation_report = load_batch_report(generation_report_path, plan=generation_plan)
    evaluation_plan = load_evaluation_plan(evaluation_plan_path)
    evaluation_report = json.loads(Path(evaluation_report_path).read_text(encoding="utf-8"))
    if evaluation_report.get("status") != "completed":
        raise ValueError("M5b requires a completed policy evaluation report")
    if evaluation_report.get("evaluation_split") != "m2_train":
        raise ValueError("M5b policy report must use the m2_train split")

    successful_results = {
        result.attempt_id: result
        for result in generation_report.results
        if result.status == "succeeded" and result.episode_path is not None
    }
    attempts_by_offset: dict[tuple[float, float], list[Any]] = {}
    for attempt in generation_plan.attempts:
        if attempt.attempt_id in successful_results:
            attempts_by_offset.setdefault(attempt.object_xy_offset_m, []).append(attempt)
    rollouts = {item["condition_id"]: item for item in evaluation_report["rollouts"]}
    if set(rollouts) != {condition.condition_id for condition in evaluation_plan.conditions}:
        raise ValueError("evaluation plan and report condition IDs differ")

    stats = json.loads(Path(norm_stats_path).read_text(encoding="utf-8"))["norm_stats"]["actions"]
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    if q01.shape != (8,) or q99.shape != (8,) or np.any(q99 <= q01):
        raise ValueError("action normalization statistics must contain ordered 8-D q01/q99")

    source_keys = (
        "policy_state", "applied_action", "ee_pose", "object_pose", "trajectory_phase_id"
    )
    policy_keys = (
        "policy_state", "requested_action", "safe_action", "applied_action",
        "model_action_raw", "ee_pose", "object_pose", "policy_gripper_filtered_closed",
        "policy_replan_id",
    )
    pair_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    normalized_actions: list[np.ndarray] = []
    for condition in evaluation_plan.conditions:
        offset = condition.object_xy_offset_m
        matches = attempts_by_offset.get(offset, [])
        if not matches:
            raise ValueError(f"condition {condition.condition_id} has no successful M2 demonstration")
        rollout = rollouts[condition.condition_id]
        policy = _load_trajectory(rollout["episode_path"], policy_keys)
        policy_closed = policy["policy_gripper_filtered_closed"].reshape(-1).astype(bool)
        policy_close = _first_true(policy_closed, label="filtered policy close")
        policy_relative = policy["ee_pose"][:, :3] - policy["object_pose"][:, :3]
        policy_joint = policy["policy_state"][:, :7]
        model_joint = policy["model_action_raw"][:, :7]
        requested_joint = policy["requested_action"][:, :7]
        safe_joint = policy["safe_action"][:, :7]
        applied_joint = policy["applied_action"][:, :7]

        replan_ids = policy["policy_replan_id"].reshape(-1)
        physical = np.empty_like(policy["model_action_raw"], dtype=np.float64)
        physical[:, 7] = policy["model_action_raw"][:, 7]
        for replan_id in np.unique(replan_ids):
            indices = np.flatnonzero(replan_ids == replan_id)
            physical[indices, :7] = model_joint[indices] - policy_joint[indices[0]]
        normalized_actions.append((physical - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0)

        condition_pairs: list[dict[str, Any]] = []
        for attempt in matches:
            source_result = successful_results[attempt.attempt_id]
            source = _load_trajectory(source_result.episode_path, source_keys)
            events = _phase_events(source["trajectory_phase_id"])
            source_close = _command_close_frame(source["applied_action"])
            source_relative = source["ee_pose"][:, :3] - source["object_pose"][:, :3]
            distances = np.linalg.norm(
                policy["ee_pose"][:, :3] - source["ee_pose"][source_close, :3], axis=1
            )
            nearest = int(np.argmin(distances))
            row = {
                "condition_id": condition.condition_id,
                "attempt_id": attempt.attempt_id,
                "source_episode_path": str(source_result.episode_path),
                "policy_episode_path": str(rollout["episode_path"]),
                "source_close_frame": source_close,
                "source_grasp_frame": events["grasp"],
                "source_lift_frame": events["lift"],
                "source_release_frame": events["release"],
                "policy_first_close_frame": policy_close,
                "source_close_relative_z_m": float(source_relative[source_close, 2]),
                "policy_close_relative_z_m": float(policy_relative[policy_close, 2]),
                "close_z_excess_m": float(
                    policy_relative[policy_close, 2] - source_relative[source_close, 2]
                ),
                "source_close_relative_xy_m": float(
                    np.linalg.norm(source_relative[source_close, :2])
                ),
                "policy_close_relative_xy_m": float(
                    np.linalg.norm(policy_relative[policy_close, :2])
                ),
                "source_policy_close_orientation_error_rad": _angular_distance(
                    source["ee_pose"][source_close, 3:7], policy["ee_pose"][policy_close, 3:7]
                ),
                "source_policy_close_actual_joint_l2_rad": float(
                    np.linalg.norm(source["policy_state"][source_close, :7] - policy_joint[policy_close])
                ),
                "source_policy_close_target_joint_l2_rad": float(
                    np.linalg.norm(source["applied_action"][source_close, :7] - model_joint[policy_close])
                ),
                "nearest_source_close_ee_frame": nearest,
                "nearest_source_close_ee_distance_m": float(distances[nearest]),
                "nearest_source_close_ee_policy_closed": bool(policy_closed[nearest]),
                "source_minimum_actual_gripper_width_m": float(
                    source["policy_state"][:, 7].min()
                ),
            }
            pair_rows.append(row)
            condition_pairs.append(row)

        nearest_pair = min(condition_pairs, key=lambda item: item["nearest_source_close_ee_distance_m"])
        source_close_z = [row["source_close_relative_z_m"] for row in condition_pairs]
        close_width_indices = np.flatnonzero(
            (np.arange(len(policy_closed)) >= policy_close) & (policy["policy_state"][:, 7] < 0.075)
        )
        gripper_response_frames = (
            int(close_width_indices[0] - policy_close) if len(close_width_indices) else None
        )
        rollout_rows.append(
            {
                "condition_id": condition.condition_id,
                "successful_source_count": len(condition_pairs),
                "policy_first_close_frame": policy_close,
                "source_close_relative_z_median_m": float(np.median(source_close_z)),
                "policy_close_relative_z_m": float(policy_relative[policy_close, 2]),
                "close_z_excess_vs_source_median_m": float(
                    policy_relative[policy_close, 2] - np.median(source_close_z)
                ),
                "policy_minimum_relative_z_m": float(policy_relative[:, 2].min()),
                "policy_minimum_closed_relative_z_m": float(policy_relative[policy_closed, 2].min()),
                "nearest_source_close_ee_distance_m": nearest_pair[
                    "nearest_source_close_ee_distance_m"
                ],
                "nearest_source_close_ee_policy_closed": nearest_pair[
                    "nearest_source_close_ee_policy_closed"
                ],
                "policy_close_model_actual_joint_linf_rad": float(
                    np.max(np.abs(model_joint[policy_close] - policy_joint[policy_close]))
                ),
                "policy_close_requested_safe_joint_linf_rad": float(
                    np.max(np.abs(requested_joint[policy_close] - safe_joint[policy_close]))
                ),
                "policy_close_applied_actual_joint_linf_rad": float(
                    np.max(np.abs(applied_joint[policy_close] - policy_joint[policy_close]))
                ),
                "maximum_model_requested_joint_difference_rad": float(
                    np.max(np.abs(model_joint - requested_joint))
                ),
                "maximum_requested_safe_joint_difference_rad": float(
                    np.max(np.abs(requested_joint - safe_joint))
                ),
                "gripper_response_frames_to_75mm": gripper_response_frames,
                "policy_minimum_actual_gripper_width_m": float(policy["policy_state"][:, 7].min()),
                "model_closed_fraction_at_first_close": float(
                    policy["model_action_raw"][policy_close, 7]
                ),
                "requested_width_at_first_close_m": float(
                    policy["requested_action"][policy_close, 7]
                ),
                "applied_width_at_first_close_m": float(
                    policy["applied_action"][policy_close, 7]
                ),
            }
        )

    normalized = np.concatenate(normalized_actions, axis=0)
    joint_outside = np.mean(np.abs(normalized[:, :7]) > 1.0, axis=0)
    source_min_widths = [row["source_minimum_actual_gripper_width_m"] for row in pair_rows]
    response_frames = [
        row["gripper_response_frames_to_75mm"]
        for row in rollout_rows
        if row["gripper_response_frames_to_75mm"] is not None
    ]
    contract_ok = (
        max(row["maximum_model_requested_joint_difference_rad"] for row in rollout_rows) <= 1e-6
        and max(row["maximum_requested_safe_joint_difference_rad"] for row in rollout_rows) <= 1e-6
        and float(np.max(joint_outside)) <= 0.05
    )
    executor_ok = (
        _median(rollout_rows, "policy_close_applied_actual_joint_linf_rad") <= 0.05
        and bool(response_frames)
        and float(np.median(response_frames)) <= 2.0
    )
    close_timing_failure = (
        _median(rollout_rows, "close_z_excess_vs_source_median_m") > 0.04
        and sum(bool(row["nearest_source_close_ee_policy_closed"]) for row in rollout_rows) <= 3
        and _median(rollout_rows, "policy_minimum_actual_gripper_width_m") < 0.01
        and float(np.median(source_min_widths)) > 0.04
    )
    diagnosis = (
        "action_contract_or_normalization_failure"
        if not contract_ok
        else "executor_tracking_failure"
        if not executor_ok
        else "policy_close_timing_failure"
        if close_timing_failure
        else "mixed_or_unresolved"
    )
    result = {
        "status": "completed",
        "review_status": "SELF_REVIEWED_PROVISIONAL",
        "inputs": {
            "generation_plan": str(Path(generation_plan_path).expanduser().resolve()),
            "generation_report": str(Path(generation_report_path).expanduser().resolve()),
            "evaluation_plan": str(Path(evaluation_plan_path).expanduser().resolve()),
            "evaluation_report": str(Path(evaluation_report_path).expanduser().resolve()),
            "norm_stats": str(Path(norm_stats_path).expanduser().resolve()),
        },
        "summary": {
            "rollouts": len(rollout_rows),
            "successful_source_comparisons": len(pair_rows),
            "median_source_close_relative_z_m": float(np.median([
                row["source_close_relative_z_m"] for row in pair_rows
            ])),
            "median_policy_close_relative_z_m": _median(
                rollout_rows, "policy_close_relative_z_m"
            ),
            "median_close_z_excess_vs_source_m": _median(
                rollout_rows, "close_z_excess_vs_source_median_m"
            ),
            "median_policy_minimum_relative_z_m": _median(
                rollout_rows, "policy_minimum_relative_z_m"
            ),
            "nearest_source_close_pose_closed_count": sum(
                bool(row["nearest_source_close_ee_policy_closed"]) for row in rollout_rows
            ),
            "median_close_applied_actual_joint_linf_rad": _median(
                rollout_rows, "policy_close_applied_actual_joint_linf_rad"
            ),
            "median_gripper_response_frames_to_75mm": float(np.median(response_frames)),
            "median_successful_source_minimum_gripper_width_m": float(
                np.median(source_min_widths)
            ),
            "median_policy_minimum_gripper_width_m": _median(
                rollout_rows, "policy_minimum_actual_gripper_width_m"
            ),
            "joint_action_outside_training_q01_q99_fraction": joint_outside.tolist(),
            "contract_checks_passed": contract_ok,
            "executor_checks_passed": executor_ok,
            "close_timing_failure_gate": close_timing_failure,
        },
        "diagnosis": diagnosis,
        "rollouts_detail": rollout_rows,
        "source_pairs": pair_rows,
    }
    json_path = Path(output_json).expanduser().resolve()
    csv_path = Path(output_csv).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    return result
