import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Save proxy for LLM API, then clear env vars so Pinecone/local services
# connect directly. LLM calls re-inject proxy explicitly.
LLM_PROXY = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
# Store proxy under a custom key so LLM calls can pick it up
if LLM_PROXY:
    os.environ["LLM_PROXY"] = LLM_PROXY

# Add utils dir FIRST so that `import config` resolves to
# task_explorer/utils/config.py (which has LANGCHAIN_* settings),
# NOT the stripped-down project-root config.py.
UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(UTILS_DIR, "../.."))
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)  # append, not insert — lower priority

import requests as _requests
from action_fuse import (
    KG_Construction,
    Postprocess_KG,
    get_existing_task_names,
)
from traj_to_kg import find_all_task_folders
from task_mapping import PACKAGE_APP_MAPPING
import config


def initialize_feature_model(model_name="resnet50"):
    """Initialize the feature extraction service model (must be called before extract_features)."""
    url = f"{config.Feature_URI}/set_model"
    # Explicitly bypass proxy for local service (Windows may use system proxy)
    resp = _requests.post(url, json={"model_name": model_name},
                          proxies={"http": None, "https": None})
    if resp.status_code == 200:
        print(f"Feature model '{model_name}' initialized: {resp.json()}")
    else:
        raise RuntimeError(f"Feature model init failed: {resp.status_code}, {resp.text}")


def parse_successful_tasks(progress_log_path: str) -> set:
    """Parse suite_progress.log to extract task names with score=1.0."""
    successful = set()
    pattern = re.compile(r'DONE\s+(\S+?)_\d+\s.*?score=([\d.]+)')
    with open(progress_log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                task_name, score = m.group(1), float(m.group(2))
                if score == 1.0:
                    successful.add(task_name)
    return successful


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build Neo4j knowledge graphs from automatic AndroidWorld exploration trajectories."
    )
    parser.add_argument(
        "--base_path", "--base-path",
        dest="base_path",
        default=None,
        help="Automatic exploration output directory. Defaults to docker_exploration_output.",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Run action fuse and node summary for all tasks currently referenced in each app KG.",
    )
    parser.add_argument(
        "--dry_run", "--dry-run",
        dest="dry_run",
        action="store_true",
        help="List packages and selected tasks without writing to Neo4j/Pinecone.",
    )
    return parser.parse_args()


def _build_gold_snapshot(packages, base_path, successful_tasks):
    """Capture successful automatic tasks for later gold-path export."""
    snapshot = {
        "created_at": datetime.now().isoformat(),
        "source": "automatic_exploration_success_tasks",
        "base_path": base_path,
        "apps": {},
    }
    for package_dir in packages:
        package_name = package_dir.name
        app_name = PACKAGE_APP_MAPPING[package_name]
        docker_tasks = sorted(d.name for d in package_dir.iterdir() if d.is_dir())
        gold_tasks = sorted(task for task in docker_tasks if task in successful_tasks)
        app_entry = {
            "database": app_name,
            "gold_tasks": gold_tasks,
            "gold_task_count": len(gold_tasks),
            "docker_task_count": len(docker_tasks),
            "all_tasks": docker_tasks,
            "all_task_count": len(docker_tasks),
        }
        snapshot["apps"][package_name] = app_entry
    return snapshot


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    args = _parse_args()

    # Initialize feature extraction model before any KG construction
    if args.dry_run or args.postprocess:
        print("Skipping feature model initialization.")
    else:
        initialize_feature_model("resnet50")

    default_output = "docker_exploration_output"
    base_path = args.base_path or os.path.join(PROJECT_ROOT, default_output)
    if not os.path.isabs(base_path):
        base_path = os.path.join(PROJECT_ROOT, base_path)
    if args.postprocess:
        token_log_name = "postprocess_token_log.jsonl"
    else:
        token_log_name = "kg_construction_token_log.jsonl"
    token_log_path = os.path.join(base_path, token_log_name)

    successful_tasks = set()
    if not args.postprocess:
        progress_log = os.path.join(base_path, "suite_progress.log")
        successful_tasks = parse_successful_tasks(progress_log) if os.path.exists(progress_log) else set()
        print(f"Found {len(successful_tasks)} successful tasks from progress log:")
        for t in sorted(successful_tasks):
            print(f"  - {t}")
    elif args.postprocess:
        print("Postprocess mode enabled: running action fuse and node summary for all KG tasks.")

    # Global token and duration accumulators (shared across all apps)
    kg_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    mining_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    kg_duration = [0.0]
    mining_duration = [0.0]

    # Iterate over package directories in docker_exploration_output
    packages = sorted([
        d for d in Path(base_path).iterdir()
        if d.is_dir() and d.name in PACKAGE_APP_MAPPING
    ])

    if not args.postprocess:
        gold_snapshot = _build_gold_snapshot(packages, base_path, successful_tasks)
        gold_snapshot_path = os.path.join(base_path, "gold_tasks_snapshot.json")
        if args.dry_run:
            print(f"Dry run enabled: would write gold snapshot to {gold_snapshot_path}")
        else:
            _write_json(gold_snapshot_path, gold_snapshot)
            print(f"Gold task snapshot written to {gold_snapshot_path}")

    print(f"\nFound {len(packages)} app packages to process.")
    for package_dir in packages:
        package_name = package_dir.name
        app_name = PACKAGE_APP_MAPPING[package_name]

        if args.postprocess:
            try:
                kg_tasks = sorted(get_existing_task_names(app_name, app_name))
            except Exception as e:
                print(f"\nSkipping {app_name} ({package_name}): failed to read current KG tasks: {e}")
                continue
            if not kg_tasks:
                print(f"\nSkipping {app_name} ({package_name}): no KG tasks")
                continue

            print(f"\n{'='*80}")
            print(f"Postprocessing {app_name} ({package_name}): {len(kg_tasks)} KG tasks")
            print(f"  Tasks: {kg_tasks}")
            print(f"{'='*80}")
            try:
                Postprocess_KG(
                    database=app_name,
                    index=app_name,
                    task_names=kg_tasks,
                    mining_usage=mining_usage,
                    mining_duration=mining_duration,
                    token_log_path=token_log_path,
                    package=package_name,
                    dry_run=args.dry_run,
                )
            except Exception as e:
                print(f"ERROR postprocessing {app_name} ({package_name}): {e}")
                import traceback
                traceback.print_exc()
            continue

        task_folders = find_all_task_folders(str(package_dir))
        filtered_tasks = task_folders
        if not filtered_tasks:
            print(f"\nSkipping {app_name} ({package_name}): no automatic trajectories")
            continue

        print(f"\n{'='*80}")
        task_label = "automatic tasks"
        print(f"Processing {app_name} ({package_name}): {len(filtered_tasks)} {task_label}")
        print(f"  Tasks: {[t.name for t in filtered_tasks]}")
        print(f"  Successful gold tasks: {sorted(t.name for t in filtered_tasks if t.name in successful_tasks)}")
        print(f"{'='*80}")
        if args.dry_run:
            print("Dry run enabled; no KG writes or chain understanding will be executed.")
            continue

        try:
            KG_Construction(
                database=app_name,
                index=app_name,
                package=package_name,
                base_path=base_path,
                successful_tasks=None,
                kg_usage=kg_usage,
                mining_usage=mining_usage,
                kg_duration=kg_duration,
                mining_duration=mining_duration,
                token_log_path=token_log_path,
            )
        except Exception as e:
            print(f"ERROR processing {app_name} ({package_name}): {e}")
            import traceback
            traceback.print_exc()

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL TOKEN USAGE SUMMARY")
    print(f"{'='*80}")
    print(f"KG Construction:")
    print(f"  Prompt tokens:     {kg_usage['prompt_tokens']:,}")
    print(f"  Completion tokens: {kg_usage['completion_tokens']:,}")
    print(f"  Total duration:    {kg_duration[0]:.1f}s ({kg_duration[0]/60:.1f}min)")
    if args.postprocess:
        mining_label = "action evolution + node summary"
    else:
        mining_label = "chain understanding"
    print(f"Knowledge Mining ({mining_label}):")
    print(f"  Prompt tokens:     {mining_usage['prompt_tokens']:,}")
    print(f"  Completion tokens: {mining_usage['completion_tokens']:,}")
    print(f"  Total duration:    {mining_duration[0]:.1f}s ({mining_duration[0]/60:.1f}min)")
    print(f"Grand Total:")
    print(f"  Prompt tokens:     {kg_usage['prompt_tokens'] + mining_usage['prompt_tokens']:,}")
    print(f"  Completion tokens: {kg_usage['completion_tokens'] + mining_usage['completion_tokens']:,}")
    print(f"  Total duration:    {kg_duration[0] + mining_duration[0]:.1f}s ({(kg_duration[0] + mining_duration[0])/60:.1f}min)")
    print(f"\nDetailed log: {token_log_path}")


if __name__ == "__main__":
    main()
