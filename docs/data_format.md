# EAM Data Format

## Trajectories

Place trajectory folders under:

```text
artifacts/trajectories/<package>/<task>/
```

Each task folder should contain the trajectory pickle files produced by the offline explorer, for example `*.pkl.zst`, and the suite metadata files written during exploration.

## Knowledge Graph Export

After construction and postprocessing, export graph files to:

```text
artifacts/graph_env/<app>/<app>_graph.pkl
```

Each graph pickle contains page nodes, element nodes, high-level action nodes, and transition metadata used by MCTS.

## Gold Paths

Gold paths are stored at:

```text
artifacts/gold_paths.json
```

The file is generated from `gold_tasks_snapshot.json` and is used as the task file for MCTS self-training.

## Training Outputs

MCTS rollout data, metrics, logs, and checkpoints are written under:

```text
artifacts/training_runs/eam_mcts/
```

Use `paths.training_output_dir` in `configs/eam.yaml` to change this location.
