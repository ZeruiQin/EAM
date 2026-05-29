"""Open-source friendly wrapper for MCTS self-training."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from eam.config import cfg_get, load_config, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = PROJECT_ROOT / "self_training" / "train" / "MCTS" / "run_pipeline.py"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run EAM MCTS self-training.")
    parser.add_argument("--config", default="configs/eam.example.yaml")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--graph_dir", default=None)
    parser.add_argument("--gold_paths", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--tasks_per_iter", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = _parse_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    config = load_config(args.config)

    graph_dir = resolve_path(args.graph_dir or cfg_get(config, "paths.graph_env_dir"), "artifacts/graph_env")
    gold_paths = resolve_path(args.gold_paths or cfg_get(config, "paths.gold_paths_file"), "artifacts/gold_paths.json")
    output_root = resolve_path(
        args.output_root or cfg_get(config, "paths.training_output_dir"),
        "artifacts/training_runs/eam_mcts",
    )

    cmd = [
        sys.executable,
        str(PIPELINE),
        "--model_path",
        args.model_path,
        "--tasks_file",
        str(gold_paths),
        "--graph_dir",
        str(graph_dir),
        "--output_root",
        str(output_root),
    ]
    if args.resume_from_checkpoint:
        cmd.append("--resume_from_checkpoint")
    if args.num_iterations is not None:
        cmd.extend(["--num_iterations", str(args.num_iterations)])
    if args.tasks_per_iter is not None:
        cmd.extend(["--tasks_per_iter", str(args.tasks_per_iter)])
    if args.dry_run:
        cmd.append("--dry_run")
    cmd.extend(passthrough)

    print(f"$ {shlex.join(cmd)}")
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
