"""Top-level entry script for Docker-based trajectory exploration.

Integrates the HTTP API (port 5000) for AndroidWorld task lifecycle management
with ADB-over-TCP (port 5555) for device interaction and exploration.

Usage examples:

  # Single task exploration (specify package + goal directly)
  python -m task_explorer.docker.docker_runner \
      --package_name com.google.android.contacts \
      --user_task "Create a new contact for Fatima Wang. Their number is +12783095137." \
      --task_dir ContactsAddContact

  # Full suite exploration (iterate all tasks from the HTTP API)
  python -m task_explorer.docker.docker_runner --run_suite
"""

import argparse
import datetime
import io
import json
import os
import sys
import time

from task_explorer.docker.docker_server_client import (
    DockerServerClient,
    wait_for_server,
)
from task_explorer.docker.docker_exploration_and_mining import (
    docker_auto_exploration,
)
from task_mapping import TASK_APP_MAPPING


def task_to_package(task_name: str) -> tuple[str, str]:
  """Resolve a task class name to (package_name, app_name).

  Returns:
      (package_name, app_name) or raises ValueError if not found.
  """
  info = TASK_APP_MAPPING.get(task_name)
  if info and info[0]:
    return info[0], info[1]
  raise ValueError(
      f"Cannot resolve package for task '{task_name}'. "
      f"Add it to task_mapping.TASK_APP_MAPPING."
  )


def run_single_task(args):
  """Run exploration for a single task specified by CLI arguments."""
  usage = {"prompt_tokens": 0, "completion_tokens": 0}
  result = docker_auto_exploration(
      package_name=args.package_name,
      exploration_output_root_dir=args.output_dir,
      docker_host=args.docker_host,
      adb_port=args.adb_port,
      max_exploration_tasks=args.max_branching_factor,
      max_exploration_steps=args.max_exploration_steps,
      max_exploration_depth=args.max_exploration_depth,
      user_task=args.user_task,
      task_dir=args.task_dir,
      usage=usage,
  )
  print(f"Exploration result: {json.dumps(result, indent=2)}")
  return result


def run_suite(args):
  """Iterate over all tasks from the Docker HTTP API and explore each."""
  server = DockerServerClient(
      server_url=f"http://{args.docker_host}:{args.server_port}"
  )
  print("Waiting for server to become healthy...")
  wait_for_server(server, timeout=300)

  server.reset(go_home=True)

  if args.reinitialize:
    print("Reinitializing suite...")
    server.reinitialize_suite(
        n_task_combinations=args.n_task_combinations,
        seed=args.seed,
    )

  task_list = server.get_task_list(max_index=args.max_task_index)
  print(f"Task list ({len(task_list)} tasks): {task_list}")

  # Filter by retry file if provided
  retry_set = None
  if args.retry_file:
    with open(args.retry_file, "r", encoding="utf-8") as rf:
      retry_set = {line.strip() for line in rf if line.strip()}
    task_list = [t for t in task_list if t in retry_set]
    print(f"Retry mode: {len(task_list)} tasks to retry from {args.retry_file}")

  # --- Real-time progress log ---
  os.makedirs(args.output_dir, exist_ok=True)
  log_path = os.path.join(args.output_dir, "suite_progress.log")
  completed = 0
  succeeded = 0
  errored = 0
  skipped = 0

  def _write_log(entry: str) -> None:
    with open(log_path, "a", encoding="utf-8") as lf:
      lf.write(entry + "\n")
      lf.flush()

  _write_log(f"[{datetime.datetime.now().isoformat()}] Suite started, "
             f"{len(task_list)} task types"
             f"{' (retry)' if retry_set else ''}")

  results = {}
  for task_name in task_list:
    num_instances = server.get_task_length(task_name)
    print(f"\n{'='*60}")
    print(f"Task: {task_name} ({num_instances} instances)")
    print(f"{'='*60}")

    for idx in range(num_instances):
      task_key = f"{task_name}_{idx}"
      t_start = time.time()
      try:
        goal = server.get_task_goal(task_name, idx)
        template = server.get_task_template(task_name, idx)
        print(f"\n  Instance {idx}: {goal}")

        # Resolve package name
        if args.package_name:
          pkg = args.package_name
        else:
          try:
            pkg, app_name = task_to_package(task_name)
          except ValueError as e:
            print(f"  Skipping {task_name}: {e}")
            skipped += 1
            _write_log(
                f"[{datetime.datetime.now().isoformat()}] SKIP  "
                f"{task_key} | reason: {e} | "
                f"completed={completed} succeeded={succeeded} "
                f"errored={errored} skipped={skipped}"
            )
            continue

        # Initialize the task via HTTP API
        server.initialize_task(task_name, idx)

        # Run exploration via ADB-over-TCP
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        result = docker_auto_exploration(
            package_name=pkg,
            exploration_output_root_dir=args.output_dir,
            docker_host=args.docker_host,
            adb_port=args.adb_port,
            max_exploration_tasks=args.max_branching_factor,
            max_exploration_steps=args.max_exploration_steps,
            max_exploration_depth=args.max_exploration_depth,
            user_task=goal,
            task_dir=task_name,
            usage=usage,
        )

        # Score and tear down via HTTP API
        score = server.get_task_score(task_name, idx)
        result["score"] = score
        elapsed = time.time() - t_start
        completed += 1
        if score >= 1.0:
          succeeded += 1
        rate = succeeded / completed * 100

        print(f"  Score: {score}")
        print(f"  Success rate: {succeeded}/{completed} ({rate:.1f}%)")
        print(f"  Result: {json.dumps(result, indent=2)}")

        _write_log(
            f"[{datetime.datetime.now().isoformat()}] DONE  "
            f"{task_key} | score={score} | "
            f"time={elapsed:.1f}s | "
            f"success_rate={succeeded}/{completed} ({rate:.1f}%) | "
            f"errored={errored} skipped={skipped}"
        )

        server.tear_down_task(task_name, idx)
        server.reset(go_home=True)

        results[task_key] = result

      except Exception as e:
        elapsed = time.time() - t_start
        completed += 1
        errored += 1
        rate = succeeded / completed * 100

        print(f"  Error on {task_name} idx={idx}: {e}")
        print(f"  Success rate: {succeeded}/{completed} ({rate:.1f}%)")
        print("  Continuing to next task...")

        _write_log(
            f"[{datetime.datetime.now().isoformat()}] ERROR "
            f"{task_key} | error={e} | "
            f"time={elapsed:.1f}s | "
            f"success_rate={succeeded}/{completed} ({rate:.1f}%) | "
            f"errored={errored} skipped={skipped}"
        )

        try:
          server.tear_down_task(task_name, idx)
          server.reset(go_home=True)
        except Exception:
          pass
        continue

  # --- Final summary ---
  total = completed + skipped
  final_rate = (succeeded / completed * 100) if completed > 0 else 0
  summary_line = (
      f"Suite finished: {succeeded}/{completed} succeeded ({final_rate:.1f}%), "
      f"{errored} errors, {skipped} skipped, {total} total"
  )
  print(f"\n{summary_line}")
  _write_log(f"[{datetime.datetime.now().isoformat()}] {summary_line}")

  # Save results summary
  summary_path = os.path.join(args.output_dir, "suite_results_raw.json")
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
  print(f"Suite results saved to {summary_path}")
  print(f"Progress log saved to {log_path}")


def main():
  parser = argparse.ArgumentParser(
      description="Docker-based trajectory exploration for AndroidWorld"
  )

  # Connection settings
  parser.add_argument(
      "--docker_host", default="localhost",
      help="Docker host address (default: localhost)",
  )
  parser.add_argument(
      "--adb_port", type=int, default=5090,
      help="ADB-over-TCP port (default: 5555)",
  )
  parser.add_argument(
      "--server_port", type=int, default=5000,
      help="HTTP API server port (default: 5000)",
  )

  # Exploration parameters
  parser.add_argument(
      "--package_name", default=None,
      help="Android package name (e.g., com.google.android.contacts)",
  )
  parser.add_argument(
      "--user_task", default=None,
      help="User task goal text",
  )
  parser.add_argument(
      "--task_dir", default=None,
      help="Sub-directory name for task output",
  )
  parser.add_argument(
      "--output_dir", default="./docker_exploration_output",
      help="Root output directory (default: ./docker_exploration_output)",
  )
  parser.add_argument(
      "--max_branching_factor", type=int, default=1,
      help="Max sub-tasks to explore at each node (default: 3)",
  )
  parser.add_argument(
      "--max_exploration_steps", type=int, default=10,
      help="Max steps per sub-task (default: 10)",
  )
  parser.add_argument(
      "--max_exploration_depth", type=int, default=10,
      help="Max depth of DFS exploration (default: 10)",
  )

  # Suite mode
  parser.add_argument(
      "--run_suite", action="store_true",
      help="Iterate all tasks from the HTTP API",
  )
  parser.add_argument(
      "--reinitialize", action="store_true",
      help="Reinitialize the task suite before running",
  )
  parser.add_argument(
      "--n_task_combinations", type=int, default=1,
      help="Number of task combinations for suite init (default: 1)",
  )
  parser.add_argument(
      "--seed", type=int, default=2,
      help="Random seed for suite init (default: 2)",
  )
  parser.add_argument(
      "--max_task_index", type=int, default=-1,
      help="Max task index to fetch from suite (-1 for all, default: -1)",
  )
  parser.add_argument(
      "--retry_file", default=None,
      help="Path to a text file with task names (one per line) to retry",
  )

  args = parser.parse_args()

  # Ensure UTF-8 stdout
  if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

  if args.run_suite:
    run_suite(args)
  else:
    if not args.package_name:
      parser.error("--package_name is required when not using --run_suite")
    if not args.user_task:
      parser.error("--user_task is required when not using --run_suite")
    if not args.task_dir:
      args.task_dir = args.package_name.split(".")[-1]
    run_single_task(args)


if __name__ == "__main__":
  main()
