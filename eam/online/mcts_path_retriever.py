"""MCTS-based candidate path retrieval for online EAM inference."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MCTS_DIR = PROJECT_ROOT / "self_training" / "train" / "MCTS"
if str(MCTS_DIR) not in sys.path:
    sys.path.insert(0, str(MCTS_DIR))

import rollout  # pylint: disable=wrong-import-position


_ACTION_TOKEN_RE = re.compile(r"_\$((?:Element|Action))_([A-Za-z0-9#_\-]+)\$")


@dataclass
class CandidatePath:
    text: str
    actions: list[dict[str, str]]


class MCTSPathRetriever:
    """Loads graph_env plus a trained reward model and returns top MCTS paths."""

    def __init__(
        self,
        graph_dir: str | Path,
        model_path: str | Path,
        *,
        top_k: int = 5,
        max_depth: int = 60,
        max_iterations: int = 50,
        exploration_constant: float = 3.0,
        trained_checkpoint: bool = True,
    ) -> None:
        self.graph_dir = Path(graph_dir)
        self.model_path = str(model_path)
        self.top_k = top_k
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.trained_checkpoint = trained_checkpoint
        self._graphs: dict[str, Any] | None = None
        self._model = None
        self._tokenizer = None

    def _load_graphs(self) -> dict[str, Any]:
        if self._graphs is None:
            self._graphs = rollout._load_graph_data(str(self.graph_dir))
        return self._graphs

    def _load_model(self):
        if self._model is None or self._tokenizer is None:
            if self.trained_checkpoint:
                self._model, self._tokenizer = rollout.load_model_and_tokenizer(self.model_path)
            else:
                self._model, self._tokenizer = rollout.load_untrained_base_model(self.model_path)
            rollout.tokenizer = self._tokenizer
        return self._model

    @staticmethod
    def _select_start_page(app_graph: Any, start_page_id: str | None) -> str:
        if start_page_id:
            if start_page_id not in app_graph.pages:
                raise KeyError(f"start_page_id not found in graph: {start_page_id}")
            return start_page_id

        for page_id, page in app_graph.pages.items():
            task_steps = getattr(page, "task_steps", None)
            if isinstance(task_steps, dict) and task_steps:
                values = task_steps.values()
                if all(isinstance(v, list) and 0 in v for v in values):
                    return page_id

        if not app_graph.pages:
            raise ValueError(f"Graph {app_graph.app_name} contains no pages")
        return next(iter(app_graph.pages))

    @staticmethod
    def _action_token(action_type: str, action_id: str) -> str:
        label = "Action" if action_type.lower() == "action" else "Element"
        return f"_${label}_{action_id}$"

    @classmethod
    def _transition_to_text(cls, transition: Any, include_from: bool) -> str:
        action_name = str(transition.action_name or "").strip()
        if not _ACTION_TOKEN_RE.search(action_name):
            action_name = f"{action_name}.{cls._action_token(transition.action_type, transition.action_id)}"
        if include_from:
            return f"Page: {transition.from_page} -- Action: {action_name} --> Page: {transition.to_page}"
        return f" -- Action: {action_name} --> Page: {transition.to_page}"

    @classmethod
    def serialize_path(cls, path: list[Any]) -> CandidatePath:
        parts = [cls._transition_to_text(t, include_from=(i == 0)) for i, t in enumerate(path)]
        actions = [
            {"type": str(t.action_type).lower(), "id": str(t.action_id)}
            for t in path
        ]
        return CandidatePath(text="".join(parts), actions=actions)

    def retrieve(
        self,
        goal: str,
        *,
        app_name: str,
        task_name: str | None = None,
        start_page_id: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, str]:
        graphs = self._load_graphs()
        if app_name not in graphs:
            raise KeyError(f"App graph not found: {app_name}. Available: {sorted(graphs.keys())}")
        app_graph = graphs[app_name]
        initial_page_id = self._select_start_page(app_graph, start_page_id)
        model = self._load_model()

        finder = rollout.GUIPathFinder(
            app_graph=app_graph,
            task_description=goal,
            task_name=task_name or goal,
            initial_page_id=initial_page_id,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            exploration_constant=self.exploration_constant,
            GT_paths=[],
        )
        paths, _ = finder.search(top_k or self.top_k, model)
        candidates = [self.serialize_path(path) for path in paths]
        return {f"path_{idx + 1}": candidate.text for idx, candidate in enumerate(candidates)}

