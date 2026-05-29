"""One-shot automatic exploration and KG construction for EAM."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from eam.config import cfg_get, load_config, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(f"$ {shlex.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automatic exploration, KG import, postprocess, and training asset export."
    )
    parser.add_argument("--config", default="configs/eam.example.yaml")
    parser.add_argument("--branch", type=int, default=None, help="Max branching factor during exploration.")
    parser.add_argument("--depth", type=int, default=None, help="Max DFS exploration depth.")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per explored subtask.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_task_combinations", type=int, default=None)
    parser.add_argument("--max_task_index", type=int, default=None)
    parser.add_argument("--package_name", default=None, help="Run a single Android package instead of the full suite.")
    parser.add_argument("--user_task", default=None, help="Single-task goal text.")
    parser.add_argument("--task_dir", default=None, help="Single-task trajectory subdirectory.")
    parser.add_argument("--skip_exploration", action="store_true", help="Use existing/downloaded trajectories.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    trajectories_dir = resolve_path(
        cfg_get(config, "paths.trajectories_dir"),
        "artifacts/trajectories",
    )
    graph_env_dir = resolve_path(
        cfg_get(config, "paths.graph_env_dir"),
        "artifacts/graph_env",
    )
    gold_paths_file = resolve_path(
        cfg_get(config, "paths.gold_paths_file"),
        "artifacts/gold_paths.json",
    )

    branch = args.branch or cfg_get(config, "exploration.branch", 1)
    depth = args.depth or cfg_get(config, "exploration.depth", 10)
    max_steps = args.max_steps or cfg_get(config, "exploration.max_steps", 10)
    seed = args.seed if args.seed is not None else cfg_get(config, "exploration.seed", 2)
    n_task_combinations = (
        args.n_task_combinations
        if args.n_task_combinations is not None
        else cfg_get(config, "exploration.n_task_combinations", 1)
    )
    max_task_index = (
        args.max_task_index
        if args.max_task_index is not None
        else cfg_get(config, "exploration.max_task_index", -1)
    )

    package_name = args.package_name or cfg_get(config, "exploration.package_name")
    user_task = args.user_task or cfg_get(config, "exploration.user_task")
    task_dir = args.task_dir or cfg_get(config, "exploration.task_dir")

    if not args.dry_run:
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        graph_env_dir.mkdir(parents=True, exist_ok=True)
        gold_paths_file.parent.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    if not args.skip_exploration:
        explore_cmd = [
            python,
            "-m",
            "task_explorer.docker.docker_runner",
            "--output_dir",
            str(trajectories_dir),
            "--max_branching_factor",
            str(branch),
            "--max_exploration_depth",
            str(depth),
            "--max_exploration_steps",
            str(max_steps),
            "--seed",
            str(seed),
        ]
        if package_name:
            if not user_task:
                raise ValueError("--user_task is required when --package_name is set")
            explore_cmd.extend(["--package_name", package_name, "--user_task", user_task])
            if task_dir:
                explore_cmd.extend(["--task_dir", task_dir])
        else:
            explore_cmd.extend([
                "--run_suite",
                "--n_task_combinations",
                str(n_task_combinations),
                "--max_task_index",
                str(max_task_index),
            ])
            if cfg_get(config, "exploration.reinitialize", False):
                explore_cmd.append("--reinitialize")
        _run(explore_cmd, dry_run=args.dry_run)

    construct_script = PROJECT_ROOT / "task_explorer" / "utils" / "Construct_KG.py"
    kg_cmd = [python, str(construct_script), "--base_path", str(trajectories_dir)]
    postprocess_cmd = kg_cmd + ["--postprocess"]
    if args.dry_run:
        kg_cmd.append("--dry_run")
        postprocess_cmd.append("--dry_run")

    _run(kg_cmd, dry_run=args.dry_run)
    _run(postprocess_cmd, dry_run=args.dry_run)

    export_graph_script = PROJECT_ROOT / "task_explorer" / "utils" / "KG_to_files.py"
    _run([
        python,
        str(export_graph_script),
        "--base_path",
        str(trajectories_dir),
        "--output_dir",
        str(graph_env_dir),
    ], dry_run=args.dry_run)

    export_gold_script = PROJECT_ROOT / "task_explorer" / "utils" / "export_gold_paths.py"
    snapshot_file = trajectories_dir / "gold_tasks_snapshot.json"
    _run([
        python,
        str(export_gold_script),
        "--snapshot",
        str(snapshot_file),
        "--base_path",
        str(trajectories_dir),
        "--output",
        str(gold_paths_file),
    ], dry_run=args.dry_run)

    print("Offline construction complete.")
    print(f"Trajectories: {trajectories_dir}")
    print(f"Graph env:    {graph_env_dir}")
    print(f"Gold paths:   {gold_paths_file}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()

