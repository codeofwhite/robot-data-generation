from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .generation_plan import build_generation_plan, load_batch_report, load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"episode manifest is unavailable: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_episode(
    path: Path,
    *,
    expected_uuid: str | None = None,
    expected_parent_uuid: str | None = None,
) -> dict[str, Any]:
    episode_path = path.expanduser().resolve()
    manifest = _load_manifest(episode_path)
    episode_uuid = str(manifest.get("episode_uuid", ""))
    if not episode_uuid:
        raise ValueError(f"episode has no UUID: {episode_path}")
    if expected_uuid is not None and episode_uuid != expected_uuid:
        raise ValueError(
            f"episode UUID mismatch at {episode_path}: expected {expected_uuid}, got {episode_uuid}"
        )
    evaluator = manifest.get("evaluator") or {}
    if evaluator.get("success") is not True:
        raise ValueError(f"selected episode is not an automatic success: {episode_path}")
    if expected_parent_uuid is not None:
        lineage = manifest.get("lineage") or {}
        if lineage.get("parent_episode_uuid") != expected_parent_uuid:
            raise ValueError(
                f"generated episode parent mismatch at {episode_path}: "
                f"expected {expected_parent_uuid}, got {lineage.get('parent_episode_uuid')}"
            )
    camera_names = tuple(manifest.get("camera_names") or ())
    if not camera_names:
        raise ValueError(f"selected episode has no cameras: {episode_path}")
    fps = manifest.get("fps")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"selected episode has invalid fps: {episode_path}")
    return {
        "path": str(episode_path),
        "episode_uuid": episode_uuid,
        "num_frames": int(manifest["num_frames"]),
        "fps": fps,
        "camera_names": list(camera_names),
        "control_source": manifest.get("control_source"),
        "parent_episode_uuid": (manifest.get("lineage") or {}).get("parent_episode_uuid"),
    }


def _validate_shared_contract(episodes: list[dict[str, Any]]) -> None:
    if not episodes:
        raise ValueError("training selection cannot be empty")
    first = episodes[0]
    uuids = [episode["episode_uuid"] for episode in episodes]
    paths = [episode["path"] for episode in episodes]
    if len(set(uuids)) != len(uuids) or len(set(paths)) != len(paths):
        raise ValueError("training selection contains duplicate episode UUIDs or paths")
    for episode in episodes[1:]:
        if episode["fps"] != first["fps"]:
            raise ValueError("training selection contains mixed fps values")
        if episode["camera_names"] != first["camera_names"]:
            raise ValueError("training selection contains mixed ordered camera sets")


def _stage_view(root: Path, episodes: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True)
    for index, episode in enumerate(episodes):
        source = Path(episode["path"])
        destination = root / f"episode_{index:06d}"
        destination.mkdir()
        for item in source.iterdir():
            (destination / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())


def prepare_training_selections(
    *,
    config_path: str | Path,
    batch_report_path: str | Path,
    output: str | Path,
    minimum_generated_successes: int = 50,
) -> Path:
    if minimum_generated_successes < 1:
        raise ValueError("minimum generated successes must be positive")
    resolved_config = Path(config_path).expanduser().resolve()
    resolved_report = Path(batch_report_path).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")

    config = load_config(resolved_config)
    plan = build_generation_plan(config)
    report = load_batch_report(resolved_report, plan=plan)
    if report.status != "completed":
        raise ValueError("generation batch must be complete before training selection")

    human = [
        _validate_episode(source.path, expected_uuid=source.episode_uuid)
        for source in config.sources
    ]
    attempts = {attempt.attempt_id: attempt for attempt in plan.attempts}
    generated: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for result in report.results:
        metric_success = result.metrics.get("success") is True
        if result.status == "succeeded" and not metric_success:
            raise ValueError(f"successful result lacks success=true: {result.attempt_id}")
        if result.status != "succeeded" or not metric_success:
            excluded.append(
                {
                    "attempt_id": result.attempt_id,
                    "status": result.status,
                    "error": result.error,
                }
            )
            continue
        if result.episode_path is None:
            raise ValueError(f"successful result has no episode path: {result.attempt_id}")
        attempt = attempts[result.attempt_id]
        selected = _validate_episode(
            result.episode_path,
            expected_parent_uuid=attempt.source_episode_uuid,
        )
        selected["attempt_id"] = result.attempt_id
        generated.append(selected)

    if len(generated) < minimum_generated_successes:
        raise ValueError(
            f"generated success gate failed: {len(generated)} < {minimum_generated_successes}"
        )
    combined = [*human, *generated]
    _validate_shared_contract(human)
    _validate_shared_contract(combined)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        _stage_view(temporary / "human5", human)
        _stage_view(temporary / "human5_generated_success", combined)
        payload = {
            "format": "robot_data_training_selection",
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "spec_id": config.spec_id,
            "source_config": str(resolved_config),
            "source_config_sha256": _sha256(resolved_config),
            "generation_batch_report": str(resolved_report),
            "generation_batch_report_sha256": _sha256(resolved_report),
            "requested_attempts": report.requested_attempts,
            "generated_successes": len(generated),
            "excluded_attempts": excluded,
            "variants": {
                "human5": {
                    "view": str(destination / "human5"),
                    "episode_count": len(human),
                    "frame_count": sum(episode["num_frames"] for episode in human),
                    "episodes": human,
                },
                "human5_generated_success": {
                    "view": str(destination / "human5_generated_success"),
                    "episode_count": len(combined),
                    "frame_count": sum(episode["num_frames"] for episode in combined),
                    "episodes": combined,
                },
            },
        }
        (temporary / "selection_report.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "selection_report.json"
