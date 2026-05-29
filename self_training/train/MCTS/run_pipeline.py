"""Run iterative MCTS self-training for the GUI reward model.

The pipeline consumes exported graph data plus gold paths, runs MCTS rollouts to
create reward-model training samples, and trains a checkpoint after each round.
It intentionally starts iteration 0 from a base model unless a previous
checkpoint is passed as --model_path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent


def _is_success_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "1", "success", "successful"}
    return False


def _load_success_tasks(tasks_file: Path) -> list[dict[str, Any]]:
    with tasks_file.open("r", encoding="utf-8") as f:
        raw_tasks = json.load(f)
    if not isinstance(raw_tasks, list):
        raise ValueError(f"Expected a JSON list in {tasks_file}")

    tasks = [task for task in raw_tasks if _is_success_value(task.get("success"))]
    if not tasks:
        raise ValueError(f"No successful tasks found in {tasks_file}")
    return tasks


def _select_batch(tasks: list[dict[str, Any]], start: int, size: int) -> list[dict[str, Any]]:
    return [tasks[(start + offset) % len(tasks)] for offset in range(size)]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _merge_jsonl(inputs: Iterable[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as outfile:
        for input_path in inputs:
            with input_path.open("r", encoding="utf-8") as infile:
                outfile.write(infile.read())


def _resolve_model_ref(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.exists() or path.is_absolute() or model_path.startswith((".", "~")):
        return str(path.resolve())
    return model_path


def _run_cmd(cmd: list[str], *, env: dict[str, str], cwd: Path, dry_run: bool) -> None:
    logging.info("Running: %s", shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, env=env, cwd=str(cwd), check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCTS self-training pipeline.")
    parser.add_argument("--model_path", required=True, help="Base model or RM checkpoint path.")
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Treat --model_path as a trained RM checkpoint instead of a base model.",
    )
    parser.add_argument("--tasks_file", required=True, help="Gold-path JSON file.")
    parser.add_argument("--graph_dir", required=True, help="Directory containing app graph subfolders.")
    parser.add_argument("--output_root", required=True, help="Directory for iterative outputs.")

    parser.add_argument("--num_iterations", type=int, default=6)
    parser.add_argument("--tasks_per_iter", type=int, default=None)
    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=1,
        help="Keep only the latest N iteration checkpoint directories; keep logs/data/metrics.",
    )
    parser.add_argument("--accumulate_data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and print commands only.")

    parser.add_argument("--max_depth", type=int, default=60)
    parser.add_argument("--max_iterations", type=int, default=60)
    parser.add_argument("--exploration_constant", type=float, default=1.0)
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", default="5e-6")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--save_limit", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--attn_impl", default="sdpa")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--quick_eval_samples", type=int, default=40)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--enable_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", default="Android-World-RStar")

    parser.add_argument("--rollout_cuda_devices", default="0")
    parser.add_argument("--train_cuda_devices", default=None)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--accelerate_bin", default="accelerate")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    tasks_file = Path(args.tasks_file).resolve()
    graph_dir = Path(args.graph_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not tasks_file.is_file():
        raise FileNotFoundError(f"tasks_file does not exist: {tasks_file}")
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"graph_dir does not exist: {graph_dir}")
    if args.keep_last_checkpoints < 1:
        raise ValueError("--keep_last_checkpoints must be >= 1")

    all_tasks = _load_success_tasks(tasks_file)
    tasks_per_iter = args.tasks_per_iter or len(all_tasks)
    logging.info("Loaded %d successful tasks from %s", len(all_tasks), tasks_file)
    logging.info("Using %d tasks per iteration", tasks_per_iter)

    current_model = _resolve_model_ref(args.model_path)
    current_model_is_trained = args.resume_from_checkpoint
    cumulative_files: list[Path] = []
    saved_checkpoints: list[Path] = []

    for iteration in range(args.num_iterations):
        logging.info("========== Iteration %d / %d ==========", iteration + 1, args.num_iterations)
        iter_dir = output_root / f"iter_{iteration + 1}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        start_idx = (iteration * tasks_per_iter) % len(all_tasks)
        current_batch = _select_batch(all_tasks, start_idx, tasks_per_iter)
        task_file = iter_dir / "batch_tasks.json"
        _write_json(task_file, current_batch)

        rollout_data_path = iter_dir / "train_data.jsonl"
        rollout_env = os.environ.copy()
        rollout_env["CUDA_VISIBLE_DEVICES"] = args.rollout_cuda_devices
        rollout_cmd = [
            args.python_executable,
            "rollout.py",
            "--model_path",
            current_model,
            "--task_file",
            str(task_file.resolve()),
            "--output_file",
            str(rollout_data_path.resolve()),
            "--graph_dir",
            str(graph_dir),
            "--max_iterations",
            str(args.max_iterations),
            "--max_depth",
            str(args.max_depth),
            "--exploration_constant",
            str(args.exploration_constant),
            "--test",
            str(args.test).lower(),
            "--initial_type",
            str(current_model_is_trained),
            "--iter",
            str(iteration if current_model_is_trained else 0),
            "--log_name",
            f"iter_{iteration + 1}",
        ]
        _run_cmd(rollout_cmd, env=rollout_env, cwd=SCRIPT_DIR, dry_run=args.dry_run)

        final_train_path = rollout_data_path
        if args.accumulate_data:
            cumulative_files.append(rollout_data_path)
            final_train_path = iter_dir / "merged_train.jsonl"

        if args.accumulate_data and not args.dry_run:
            _merge_jsonl(cumulative_files, final_train_path)

        model_save_dir = (iter_dir / "checkpoint").resolve()
        metrics_path = (iter_dir / "metrics.json").resolve()
        train_env = os.environ.copy()
        if args.train_cuda_devices is not None:
            train_env["CUDA_VISIBLE_DEVICES"] = args.train_cuda_devices

        train_cmd = [
            args.accelerate_bin,
            "launch",
            f"--num_processes={args.num_processes}",
            "train_regression.py",
            "--model_name_or_path",
            current_model,
            "--train_data_path",
            str(final_train_path.resolve()),
            "--output_dir",
            str(model_save_dir),
            "--initial_type",
            str(current_model_is_trained),
            "--iter",
            str(iteration if current_model_is_trained else 0),
            "--per_device_train_batch_size",
            str(args.batch_size),
            "--gradient_accumulation_steps",
            str(args.grad_acc),
            "--num_train_epochs",
            str(args.num_epochs),
            "--learning_rate",
            args.learning_rate,
            "--evaluation_strategy",
            "steps",
            "--eval_steps",
            str(args.eval_steps),
            "--save_total_limit",
            str(args.save_limit),
            "--logging_steps",
            str(args.logging_steps),
            "--attn_impl",
            args.attn_impl,
            "--dataset_num_proc",
            str(args.dataset_num_proc),
            "--quick_eval_samples",
            str(args.quick_eval_samples),
            "--metrics_path",
            str(metrics_path),
            "--save_strategy",
            args.save_strategy,
        ]
        if args.report_to:
            train_cmd.extend(["--report_to", args.report_to])
        if args.bf16:
            train_cmd.append("--bf16")
        if args.enable_swanlab:
            train_cmd.extend(["--enable_swanlab", "--swanlab_project", args.swanlab_project])

        _run_cmd(train_cmd, env=train_env, cwd=SCRIPT_DIR, dry_run=args.dry_run)
        current_model = str(model_save_dir)
        current_model_is_trained = True
        saved_checkpoints.append(model_save_dir)
        while len(saved_checkpoints) > args.keep_last_checkpoints:
            old_checkpoint = saved_checkpoints.pop(0)
            if args.dry_run:
                logging.info("Would remove old checkpoint: %s", old_checkpoint)
            elif old_checkpoint.exists():
                logging.info("Removing old checkpoint to save disk space: %s", old_checkpoint)
                shutil.rmtree(old_checkpoint)
        logging.info("Iteration %d complete. New model: %s", iteration + 1, current_model)


if __name__ == "__main__":
    main()
