"""Executable Agentic Memory online agent.

This agent keeps RAC's cloud planning and local execution flow, but replaces
the first-step GCR path retrieval with MCTS search over exported EAM graph_env.
"""

from __future__ import annotations

import os

from android_world.agents import RAC_Agent
from android_world.agents import infer
from android_world.env import interface


class EAMAgent(RAC_Agent.RAC):
  """RAC-compatible agent backed by MCTS path retrieval."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      cloud_llm: infer.MultimodalLlmWrapper,
      local_llm: infer.MultimodalLlmWrapper,
      uri: str,
      auth: tuple,
      graph_dir: str,
      model_path: str,
      app_name: str | None = None,
      top_k: int = 5,
      max_depth: int = 60,
      max_iterations: int = 50,
      exploration_constant: float = 3.0,
      trained_checkpoint: bool = True,
      name: str = "EAM",
      wait_after_action_seconds: float = 5.0,
  ):
    super().__init__(
        env=env,
        cloud_llm=cloud_llm,
        local_llm=local_llm,
        uri=uri,
        auth=auth,
        name=name,
        wait_after_action_seconds=wait_after_action_seconds,
    )
    self.app_name = app_name or os.environ.get("EAM_APP_NAME") or os.environ.get("DATABASE")
    if not self.app_name:
      raise ValueError("EAM app name is required. Set --eam_app_name or EAM_APP_NAME/DATABASE.")
    from eam.online.mcts_path_retriever import MCTSPathRetriever
    self.path_retriever = MCTSPathRetriever(
        graph_dir=graph_dir,
        model_path=model_path,
        top_k=top_k,
        max_depth=max_depth,
        max_iterations=max_iterations,
        exploration_constant=exploration_constant,
        trained_checkpoint=trained_checkpoint,
    )

  def _generate_reasoning_paths(self, goal: str, screenshot):
    return self.path_retriever.retrieve(
        goal,
        app_name=self.app_name,
        task_name=os.environ.get("TASK"),
        start_page_id=os.environ.get("EAM_START_PAGE_ID"),
    )
