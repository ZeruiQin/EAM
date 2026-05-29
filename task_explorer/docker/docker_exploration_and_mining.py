"""Docker version of the exploration pipeline.

Reuses explore_dfs(), task_goal_generator(), state_evaluator() etc. from the
original module, but replaces the environment setup with Docker-compatible
components (DockerDevice, FakeAPK, DockerGUIExplorer).
"""

import json
import os
import sys

from task_explorer.utils.device import _generate_ui_elements_description_list
from task_explorer.exploration_and_mining import (
    explore_dfs,
    task_goal_generator,
    state_evaluator,
    TaskCompletedException,
    is_task_explored,
)
from task_explorer.docker.docker_device import DockerDevice
from task_explorer.docker.app_info_provider import FakeAPK
from task_explorer.docker.docker_explorer import DockerGUIExplorer


def _setup_docker_exploration_env(
    package_name: str,
    root_dir: str,
    docker_host: str = "localhost",
    adb_port: int = 5555,
) -> tuple:
  """Initialize exploration environment for Docker.

  Returns:
      (root_dir, fake_apk, app_info, device, agent)
  """
  root_dir = os.path.abspath(root_dir)
  root_dir = os.path.join(root_dir, package_name)
  os.makedirs(root_dir, exist_ok=True)

  device = DockerDevice(docker_host=docker_host, adb_port=adb_port)
  fake_apk = FakeAPK(package_name, device)

  app_info = {
      "app_name": fake_apk.get_app_name(),
      "app_version": fake_apk.get_androidversion_code(),
      "app_version_name": fake_apk.get_androidversion_name(),
      "app_pkg": fake_apk.get_package(),
      "app_main_activity": fake_apk.get_main_activity(),
  }
  with open(
      os.path.join(root_dir, "app_info.json"), "w", encoding="utf-8"
  ) as f:
    json.dump(app_info, f, indent=2, ensure_ascii=False)

  agent = DockerGUIExplorer(docker_host=docker_host, adb_port=adb_port)

  return root_dir, fake_apk, app_info, device, agent


def docker_auto_exploration(
    package_name: str,
    exploration_output_root_dir: str = "./output",
    docker_host: str = "localhost",
    adb_port: int = 5555,
    max_exploration_tasks: int = 10,
    max_exploration_steps: int = 30,
    max_exploration_depth: int = 5,
    user_task: str = None,
    task_dir: str = None,
    usage: dict[str, int] = None,
) -> dict:
  """Run auto exploration against a Docker-hosted emulator.

  Returns:
      dict with keys:
          exploration_completed (bool)
          num_subtasks_generated (int)
          num_subtasks_explored (int)
          usage (dict)
  """
  if usage is None:
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

  (
      exploration_output_root_dir,
      apk_object,
      app_info,
      device,
      agent,
  ) = _setup_docker_exploration_env(
      package_name, exploration_output_root_dir, docker_host, adb_port
  )

  print("Generating exploration sub-goals.")
  exploration_output_root_dir = os.path.join(
      exploration_output_root_dir, task_dir if task_dir else "default"
  )
  device.home()
  device.launch_app(package_name, front=True)
  initial_elements = device.wait_to_stabilize()
  initial_screen_size = device.get_screen_size()
  initial_elements_list = _generate_ui_elements_description_list(
      initial_elements, initial_screen_size
  )
  screenshot = device.get_screenshot()
  activity = [
      act for act in apk_object.get_activities() if "sdk" not in act.lower()
  ]
  activity_str = "\n".join(activity)
  app_name = apk_object.get_app_name()

  task_list = task_goal_generator(
      screenshot=screenshot,
      package_name=package_name,
      app_name=app_name,
      activity_list=activity_str,
      user_task=user_task,
      element_list=initial_elements_list,
      usage=usage,
  )
  print(f"Generated {len(task_list)} sub-tasks: {task_list}")

  num_subtasks_generated = len(task_list)
  if max_exploration_tasks < len(task_list):
    task_list = task_list[:max_exploration_tasks]
  print(f"Exploring {len(task_list)} sub-tasks: {task_list}")

  exploration_completed = False
  num_subtasks_explored = 0
  try:
    for i, task in enumerate(task_list):
      num_subtasks_explored = i + 1
      print(f"Exploring task {i+1}/{len(task_list)}: {task}")
      explore_dfs(
          current_task=task["directive"],
          current_depth=1,
          exploration_output_root_dir=exploration_output_root_dir,
          max_exploration_tasks=max_exploration_tasks,
          max_exploration_steps=max_exploration_steps,
          agent=agent,
          apk_object=apk_object,
          device_controller=device,
          previous_actions=[],
          previous_subgoals=[],
          package_name=package_name,
          is_first_task=bool(i == 0),
          max_exploration_depth=max_exploration_depth,
          user_task=user_task,
          usage=usage,
      )
    print(
        "Auto exploration finished - all tasks explored but user task not completed."
    )
  except TaskCompletedException as e:
    exploration_completed = True
    print("User task completed successfully! Stopping all exploration.")
    print(f"Task completion detected: {e}")

  return {
      "exploration_completed": exploration_completed,
      "num_subtasks_generated": num_subtasks_generated,
      "num_subtasks_explored": num_subtasks_explored,
      "usage": dict(usage),
  }
