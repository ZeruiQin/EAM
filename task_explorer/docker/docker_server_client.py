"""DockerServerClient: HTTP API client for AndroidWorld Docker task management."""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

Params = dict[str, int | str]


class DockerServerClient:
  """Client for interacting with the AndroidWorld Docker HTTP server (port 5000)."""

  def __init__(self, server_url: str = "http://localhost:5000"):
    self.base_url = server_url.rstrip("/")
    # Bypass HTTP proxy for local Docker connections
    self.session = requests.Session()
    self.session.trust_env = False

  # ---- Environment management ----

  def health(self) -> bool:
    """Check if the server is healthy."""
    try:
      response = self.session.get(f"{self.base_url}/health", timeout=10)
      response.raise_for_status()
      return True
    except Exception as e:
      logger.warning(f"Health check failed: {e}")
      return False

  def reset(self, go_home: bool = True) -> dict:
    """Reset the environment."""
    response = self.session.post(
        f"{self.base_url}/reset", params={"go_home": go_home}
    )
    response.raise_for_status()
    return response.json()

  def close(self) -> None:
    """Close the environment."""
    response = self.session.post(f"{self.base_url}/close")
    response.raise_for_status()

  # ---- Suite management ----

  def get_task_list(self, max_index: int = -1) -> list[str]:
    """Get the list of available task types."""
    response = self.session.get(
        f"{self.base_url}/suite/task_list",
        params={"max_index": max_index},
    )
    response.raise_for_status()
    return response.json()["task_list"]

  def get_task_length(self, task_type: str) -> int:
    """Get the number of task instances for a given task type."""
    response = self.session.get(
        f"{self.base_url}/suite/task_length",
        params={"task_type": task_type},
    )
    response.raise_for_status()
    return response.json()["length"]

  def reinitialize_suite(
      self,
      n_task_combinations: int = 1,
      seed: int = 2,
      task_family: str = "android_world",
  ) -> dict:
    """Reinitialize the task suite."""
    response = self.session.get(
        f"{self.base_url}/suite/reinitialize",
        params={
            "n_task_combinations": n_task_combinations,
            "seed": seed,
            "task_family": task_family,
        },
    )
    response.raise_for_status()
    return response.json()

  # ---- Single task lifecycle ----

  def initialize_task(self, task_type: str, task_idx: int) -> dict:
    """Initialize a task instance."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self.session.post(
        f"{self.base_url}/task/initialize", params=params
    )
    response.raise_for_status()
    return response.json()

  def tear_down_task(self, task_type: str, task_idx: int) -> dict:
    """Tear down a task instance."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self.session.post(
        f"{self.base_url}/task/tear_down", params=params
    )
    response.raise_for_status()
    return response.json()

  def get_task_score(self, task_type: str, task_idx: int) -> float:
    """Get the score of a task instance."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self.session.get(f"{self.base_url}/task/score", params=params)
    response.raise_for_status()
    return response.json()["score"]

  def get_task_goal(self, task_type: str, task_idx: int) -> str:
    """Get the goal text of a task instance."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self.session.get(f"{self.base_url}/task/goal", params=params)
    response.raise_for_status()
    return response.json()["goal"]

  def get_task_template(self, task_type: str, task_idx: int) -> str:
    """Get the template of a task instance."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self.session.get(f"{self.base_url}/task/template", params=params)
    response.raise_for_status()
    return response.json()["template"]


def wait_for_server(
    server: DockerServerClient,
    timeout: float = 300.0,
    interval: float = 3.0,
) -> None:
  """Block until the server is healthy or timeout."""
  deadline = time.time() + timeout
  while time.time() < deadline:
    if server.health():
      logger.info("Server is healthy.")
      return
    logger.info("Server not ready, retrying...")
    time.sleep(interval)
  raise TimeoutError(
      f"Server at {server.base_url} did not become healthy within {timeout}s."
  )
