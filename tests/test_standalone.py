"""Verify the installed CPU workflow and the source boundary it promises."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import robot_data_generation


def test_package_has_no_studio_or_simulator_imports():
    forbidden = {"robot_data_studio", "gello_remote", "isaaclab", "isaacsim", "genesis"}
    for path in Path(robot_data_generation.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not forbidden.intersection(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] not in forbidden


def test_installed_cli_reads_fixture_and_preserves_source(tmp_path):
    checkout = Path(__file__).resolve().parents[1]
    output = tmp_path / "synthetic"
    subprocess.run([sys.executable, "-I", str(checkout / "examples/make_synthetic_episode.py"),
                    "--output", str(output)], cwd=tmp_path, check=True, capture_output=True)
    def fingerprint():
        return {p.relative_to(output / "raw").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (output / "raw").rglob("*") if p.is_file()}
    before = fingerprint()
    for name, extra in (("plan", []), ("nominal", ["--nominal"])):
        subprocess.run([sys.executable, "-I", "-m", "robot_data_generation.cli", "plan",
                        "--config", str(output / "config.json"), "--output", str(tmp_path / f"{name}.json"),
                        *extra], cwd=tmp_path, check=True, capture_output=True)
    plan = json.loads((tmp_path / "plan.json").read_text())
    nominal = json.loads((tmp_path / "nominal.json").read_text())
    assert len(plan["attempts"]) == 4
    assert len(nominal["attempts"]) == 1
    subprocess.run([sys.executable, "-I", "-m", "robot_data_generation.cli", "plan-evaluation",
                    "--m2-plan", str(tmp_path / "plan.json"), "--output", str(tmp_path / "heldout.json"),
                    "--plan-id", "standalone-test", "--count", "5"],
                   cwd=tmp_path, check=True, capture_output=True)
    assert len(json.loads((tmp_path / "heldout.json").read_text())["conditions"]) == 5
    assert fingerprint() == before
