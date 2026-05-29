# Repository Guidelines

## Project Structure & Module Organization

This repository is an AndroidWorld-based implementation of Executable Agentic Memory (EAM). Core AndroidWorld code lives in `android_world/`: agents in `android_world/agents/`, device interfaces in `android_world/env/`, tasks in `android_world/task_evals/`, and shared helpers in `android_world/utils/`. EAM wrappers live in `eam/`: `eam/offline/` for automatic exploration and KG construction, `eam/training/` for MCTS self-training, and `eam/online/` for online MCTS path retrieval. KG construction internals remain in `task_explorer/`; reward-model training internals remain in `self_training/train/MCTS/`. Tests should be colocated with source as `*_test.py`.

## Build, Test, and Development Commands

Install dependencies with `pip install -r requirements-eam.txt`. Run unit tests with `pytest android_world task_explorer self_training`. Run automatic offline construction with `python -m eam.offline.construct --config configs/eam.example.yaml --branch 3 --depth 10 --max_steps 10`. Train with `python -m eam.training.self_train --config configs/eam.example.yaml --model_path /path/to/model`. Run online inference with `python run.py --agent_name=eam_mcts --eam_graph_dir=artifacts/graph_env --eam_model_path=/path/to/checkpoint --eam_app_name=contacts`.

Start the bundled visual feature service with `docker build -t eam-feature-service feature_service` and `docker run --rm -p 8001:8001 eam-feature-service`.

## Coding Style & Naming Conventions

Follow the existing Google-style Python conventions: 2-space indentation in AndroidWorld files, 80-character lines where practical, `snake_case` functions/modules, `CamelCase` classes, and `UPPER_CASE` constants. Keep public EAM entrypoints thin and delegate heavy logic to existing modules. Runtime credentials and service URLs should come from environment variables.

## Testing Guidelines

Prefer small unit tests that do not require a live emulator. Add tests for path serialization, action-id parsing, gold-path export filtering, and CLI dry runs. Use emulator, Neo4j, Pinecone, and LLM calls for integration checks.

## Commit & Pull Request Guidelines

Use clear imperative commits with scope, such as `offline: add one-shot construction entrypoint` or `online: add MCTS path retriever`. PRs should describe the affected pipeline stage, list required services, and include commands run.

## Artifact Policy

Place local trajectories, `graph_env`, gold paths, Neo4j dumps, MCTS logs, checkpoints, screenshots, `.env`, and cache directories under ignored local paths such as `artifacts/`.
