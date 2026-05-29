import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(UTILS_DIR, "../.."))
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config
from traj_to_kg import TrajectoryToNeo4jImporter


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Export gold paths from current Neo4j KGs using gold_tasks_snapshot.json."
    )
    parser.add_argument(
        "--snapshot",
        default="docker_exploration_output/gold_tasks_snapshot.json",
        help="Path to gold task snapshot JSON.",
    )
    parser.add_argument(
        "--base_path",
        default=None,
        help="Trajectory base path used only to preserve app/task ordering context.",
    )
    parser.add_argument(
        "--output",
        default="data_prepare/path_raw/gold_paths_from_current_kg_with_action.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--no_action_groups",
        action="store_true",
        help="Do not replace element chains with high-level Action nodes.",
    )
    return parser.parse_args()


def _convert_path_entries(entries):
    converted_entries = []
    action_group_count = 0
    for entry in entries:
        converted_entry = dict(entry)
        converted_paths = []
        for path in entry.get("path", []):
            converted_triplets = []
            for triplet in path:
                element_text = triplet.get("element", "Unknown element")
                if "_$Action_" in element_text:
                    action_group_count += 1
                converted_triplets.append([
                    triplet.get("source_page", "Unknown source page"),
                    element_text,
                    triplet.get("target_page", "Unknown target page"),
                ])
            converted_paths.append(converted_triplets)
        converted_entry["path"] = converted_paths
        converted_entries.append(converted_entry)
    return converted_entries, action_group_count


def main():
    args = _parse_args()
    snapshot_path = Path(args.snapshot)
    output_path = Path(args.output)

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    base_path = Path(args.base_path or snapshot.get("base_path", "docker_exploration_output"))
    use_action_groups = not args.no_action_groups

    all_gold_paths = []
    total_action_groups = 0

    for package, info in snapshot.get("apps", {}).items():
        database = info.get("database")
        gold_tasks = info.get("gold_tasks", [])
        if not database or not gold_tasks:
            continue

        print(f"\nProcessing {database} ({package}): {len(gold_tasks)} gold tasks")
        importer = TrajectoryToNeo4jImporter(
            uri=config.Neo4j_URI,
            auth=config.Neo4j_AUTH,
            database=database,
            index=database,
        )

        try:
            entries = importer.find_gold_paths(
                root_path=base_path / package,
                task_names=gold_tasks,
                use_action_groups=use_action_groups,
                app=package,
                success=True,
            )
            converted_entries, action_group_count = _convert_path_entries(entries)
            total_action_groups += action_group_count
            all_gold_paths.extend(converted_entries)

            for entry in converted_entries:
                print(f"  {entry['task']}: {len(entry.get('path', []))} paths")
            print(f"  action groups replaced: {action_group_count}")
        except Exception as exc:
            print(f"  ERROR {database}: {exc}")
        finally:
            importer.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_gold_paths, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(all_gold_paths)} gold task entries to {output_path}")
    print(f"Total action groups replaced: {total_action_groups}")


if __name__ == "__main__":
    main()
