from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation_plan import (
    build_evaluation_plan,
    build_training_evaluation_plan,
    compare_evaluation_reports,
    diagnose_evaluation_splits,
)
from .generation_plan import (
    build_generation_plan,
    build_nominal_plan,
    load_config,
    write_json_atomic,
)
from .source_episode import load_source_trajectory
from .training_selection import prepare_training_selections
from .trajectory_alignment import align_m5b_trajectories


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare physics-verified generation plans")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="validate sources and write a resolved plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--nominal", action="store_true")
    selection = subparsers.add_parser(
        "select-training", help="select accepted human and generated training episodes"
    )
    selection.add_argument("--config", type=Path, required=True)
    selection.add_argument("--batch-report", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--minimum-generated-successes", type=int, default=50)
    evaluation = subparsers.add_parser(
        "plan-evaluation", help="write leakage-checked held-out cube positions"
    )
    evaluation.add_argument("--m2-plan", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--plan-id", required=True)
    evaluation.add_argument("--count", type=int, default=20)
    evaluation.add_argument("--seed", type=int, default=20261005)
    evaluation.add_argument("--range-m", type=float, default=0.02)
    evaluation.add_argument("--minimum-m2-distance-m", type=float, default=0.001)
    training_evaluation = subparsers.add_parser(
        "plan-training-evaluation", help="write a deterministic subset of successful M2 positions"
    )
    training_evaluation.add_argument("--m2-plan", type=Path, required=True)
    training_evaluation.add_argument("--m2-report", type=Path, required=True)
    training_evaluation.add_argument("--output", type=Path, required=True)
    training_evaluation.add_argument("--plan-id", required=True)
    training_evaluation.add_argument("--count", type=int, default=10)
    training_evaluation.add_argument("--seed", type=int, default=20261105)
    comparison = subparsers.add_parser(
        "compare-evaluation", help="compare two completed paired rollout reports"
    )
    comparison.add_argument("--baseline-report", type=Path, required=True)
    comparison.add_argument("--augmented-report", type=Path, required=True)
    comparison.add_argument("--output-json", type=Path, required=True)
    comparison.add_argument("--output-csv", type=Path, required=True)
    diagnosis = subparsers.add_parser(
        "diagnose-m5a", help="compare nominal, M2-train, and held-out reports"
    )
    diagnosis.add_argument("--nominal-report", type=Path, required=True)
    diagnosis.add_argument("--m2-train-report", type=Path, required=True)
    diagnosis.add_argument("--heldout-report", type=Path, required=True)
    diagnosis.add_argument("--output-json", type=Path, required=True)
    diagnosis.add_argument("--output-csv", type=Path, required=True)
    alignment = subparsers.add_parser(
        "align-m5b", help="align successful M2 demonstrations with same-position rollouts"
    )
    alignment.add_argument("--m2-plan", type=Path, required=True)
    alignment.add_argument("--m2-report", type=Path, required=True)
    alignment.add_argument("--evaluation-plan", type=Path, required=True)
    alignment.add_argument("--evaluation-report", type=Path, required=True)
    alignment.add_argument("--norm-stats", type=Path, required=True)
    alignment.add_argument("--output-json", type=Path, required=True)
    alignment.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "plan-evaluation":
        plan = build_evaluation_plan(
            args.m2_plan,
            plan_id=args.plan_id,
            count=args.count,
            seed=args.seed,
            range_m=args.range_m,
            minimum_m2_distance_m=args.minimum_m2_distance_m,
        )
        write_json_atomic(args.output, plan)
        print(args.output.expanduser().resolve())
        return
    if args.command == "plan-training-evaluation":
        plan = build_training_evaluation_plan(
            args.m2_plan,
            args.m2_report,
            plan_id=args.plan_id,
            count=args.count,
            seed=args.seed,
        )
        write_json_atomic(args.output, plan)
        print(args.output.expanduser().resolve())
        return
    if args.command == "diagnose-m5a":
        diagnose_evaluation_splits(
            {
                "nominal": args.nominal_report,
                "m2_train": args.m2_train_report,
                "heldout": args.heldout_report,
            },
            output_json=args.output_json,
            output_csv=args.output_csv,
        )
        print(args.output_json.expanduser().resolve())
        return
    if args.command == "align-m5b":
        align_m5b_trajectories(
            generation_plan_path=args.m2_plan,
            generation_report_path=args.m2_report,
            evaluation_plan_path=args.evaluation_plan,
            evaluation_report_path=args.evaluation_report,
            norm_stats_path=args.norm_stats,
            output_json=args.output_json,
            output_csv=args.output_csv,
        )
        print(args.output_json.expanduser().resolve())
        return
    if args.command == "compare-evaluation":
        compare_evaluation_reports(
            args.baseline_report,
            args.augmented_report,
            output_json=args.output_json,
            output_csv=args.output_csv,
        )
        print(args.output_json.expanduser().resolve())
        return
    if args.command == "select-training":
        result = prepare_training_selections(
            config_path=args.config,
            batch_report_path=args.batch_report,
            output=args.output,
            minimum_generated_successes=args.minimum_generated_successes,
        )
        print(result)
        return
    config = load_config(args.config)
    for source in config.sources:
        load_source_trajectory(
            source,
            grasp_lift_threshold_m=config.execution.grasp_lift_threshold_m,
            task_lift_threshold_m=config.execution.task_lift_threshold_m,
        )
    plan = build_nominal_plan(config) if args.nominal else build_generation_plan(config)
    write_json_atomic(args.output, plan)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
