# Executable Agentic Memory for GUI Agent

<p align="center">
  <img src="assets/eam_overview.png" alt="Executable Agentic Memory overview" width="90%">
</p>

<p align="center">
  <a href="https://github.com/ZeruiQin/EAM">Code</a>
</p>

## Setup

Install dependencies:

```bash
pip install -r requirements-eam.txt
```

Create local configuration files:

```bash
cp .env.example .env
cp configs/eam.example.yaml configs/eam.yaml
```

Fill in `.env` for Neo4j, LLM, Pinecone, Android SDK/ADB, and feature service settings. Adjust paths and exploration parameters in `configs/eam.yaml`.

Start the feature service:

```bash
docker build -t eam-feature-service feature_service
docker run --rm -p 8001:8001 eam-feature-service
```

Use `FEATURE_URI=http://127.0.0.1:8001` in `.env`.

## Offline Construction

Run automatic exploration, KG construction, postprocessing, and asset export:

```bash
python -m eam.offline.construct \
  --config configs/eam.yaml \
  --branch 3 \
  --depth 10 \
  --max_steps 10 \
  --seed 2
```

Outputs:

```text
artifacts/trajectories/
artifacts/graph_env/
artifacts/gold_paths.json
```

To use existing local trajectories, place them under `artifacts/trajectories/` and run:

```bash
python -m eam.offline.construct --config configs/eam.yaml --skip_exploration
```

## Model Training

Run MCTS self-training:

```bash
python -m eam.training.self_train \
  --config configs/eam.yaml \
  --model_path /path/to/base-or-checkpoint \
  --num_iterations 6 \
  --tasks_per_iter 32 \
  -- --enable_swanlab --swanlab_project EAM
```

Training outputs are written to the `training_output_dir` configured in `configs/eam.yaml`.

## Online Inference

Run the EAM MCTS agent:

```bash
python run.py \
  --suite_family=android_world \
  --agent_name=eam_mcts \
  --tasks=ContactsAddContact \
  --eam_graph_dir=artifacts/graph_env \
  --eam_model_path=/path/to/eam/checkpoint \
  --eam_app_name=contacts \
  --perform_emulator_setup=False
```

The agent retrieves candidate paths with MCTS, sends them to the planner, and executes the generated operation sequence locally.
