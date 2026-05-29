"""DockerGUIExplorer: GUI_explorer subclass that uses DockerDevice.

Overrides step() to remove RAG (retrieval_batch_api / Ranker) calls so
the exploration pipeline can run without external retrieval services.
"""

import os
from typing import Any

import numpy as np
from dataclasses import asdict

from task_explorer.MLLM_Agent.GUI_explorer import (
    GUI_explorer,
    is_need_stop,
    send_message,
    send_message2,
    _action_selection_prompt,
    _summarize_prompt,
    ask_mllm,
    parse_reason_action_output,
    execute_adb_action,
    extract_json,
)
from task_explorer.MLLM_Agent import json_action
from task_explorer.utils.device import (
    _generate_ui_elements_description_list,
    validate_ui_element,
    add_ui_element_mark,
    add_screenshot_label,
)
from task_explorer.docker.docker_device import DockerDevice


class DockerGUIExplorer(GUI_explorer):
  """GUI_explorer that connects to a Docker-hosted Android emulator
  via ADB-over-TCP instead of a local device serial.

  The step() method is overridden to skip all RAG-related logic
  (cropped image collection, retrieval_batch_api, Ranker) so the
  pipeline works without external retrieval services.
  """

  def __init__(
      self,
      docker_host: str = "localhost",
      adb_port: int = 5555,
      step_interval: float = 2.0,
  ):
    # Skip GUI_explorer.__init__ because it creates Device(device_serial=...)
    # Instead, replicate the parent init logic with DockerDevice.
    self.history = []
    self.device = DockerDevice(docker_host=docker_host, adb_port=adb_port)
    self.step_interval = step_interval
    self.early_stop = False
    self.demo_on = os.getenv("TURN_ON_DEMO_MODE", "False").lower() == "true"

  def step(self, goal: str) -> tuple[bool, dict[str, Any]]:
    """Single exploration step — same as parent but without RAG retrieval."""
    step_data = {
        "goal": goal,
        "raw_screenshot": None,
        "before_screenshot_with_som": None,
        "after_screenshot_with_som": None,
        "action_prompt": None,
        "action_output": None,
        "action_raw_response": None,
        "summary_prompt": None,
        "summary": None,
        "summary_raw_response": None,
        "converted_action": "error_retry",
        "actual_action_coordinates": None,
        "before_screenshot": None,
        "after_screenshot": None,
        "ui_elements": None,
        "top_app_package_name": None,
        "target_element": None,
        "ranker_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
        "logical_screen_size": None,
    }
    print("Step: " + str(len(self.history) + 1))

    self.early_stop = is_need_stop()
    if self.early_stop:
      return (True, step_data)
    before_ui_elements = self.device.wait_to_stabilize()
    orientation = self.device.get_orientation()
    logical_screen_size = self.device.get_screen_size()
    step_data["logical_screen_size"] = logical_screen_size
    physical_frame_boundary = self.device.get_physical_frame_boundary()

    step_data["ui_elements"] = [
        asdict(ui_element) for ui_element in before_ui_elements
    ]
    before_ui_elements_list = _generate_ui_elements_description_list(
        before_ui_elements, logical_screen_size
    )
    pil_before_screenshot = self.device.get_screenshot().convert("RGB")
    before_screenshot = np.array(pil_before_screenshot)
    step_data["raw_screenshot"] = before_screenshot.copy()
    step_data["before_screenshot"] = before_screenshot.copy()
    top_app_package_name = self.device.get_top_package_name()
    step_data["top_app_package_name"] = top_app_package_name

    # SoM annotation — mark valid UI elements on the screenshot
    for index, ui_element in enumerate(before_ui_elements):
      if validate_ui_element(ui_element, logical_screen_size):
        add_ui_element_mark(
            before_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

    self.early_stop = is_need_stop()
    if self.early_stop:
      return (True, step_data)

    step_data["before_screenshot_with_som"] = before_screenshot.copy()

    # NOTE: RAG retrieval and Ranker are intentionally skipped.
    # knowledge_prompt is empty — no retrieval services needed.
    knowledge_prompt = ""

    self.early_stop = is_need_stop()
    if self.early_stop:
      return (True, step_data)
    action_prompt = _action_selection_prompt(
        goal,
        [
            "Step " + str(i + 1) + "- " + step_info["summary"]
            for i, step_info in enumerate(self.history)
        ],
        before_ui_elements_list,
        knowledge_prompt=knowledge_prompt,
    )
    step_data["action_prompt"] = action_prompt
    action_output, raw_response = ask_mllm(
        action_prompt,
        [
            step_data["raw_screenshot"],
            before_screenshot,
        ],
    )

    if not raw_response:
      raise RuntimeError("Error calling LLM in action selection phase.")
    if not action_output or not isinstance(action_output, str):
      print(f"LLM returned empty/invalid action_output: {action_output!r}")
      step_data["summary"] = "LLM returned empty action output, skipping step."
      self.history.append(step_data)
      return (False, step_data)
    step_data["action_output"] = action_output
    step_data["action_raw_response"] = raw_response

    reason, action = parse_reason_action_output(action_output)

    if (not reason) or (not action):
      print("Action prompt output is not in the correct format.")
      step_data["summary"] = (
          "Output for action selection is not in the correct format, so no"
          " action is performed."
      )
      self.history.append(step_data)
      return (False, step_data)

    print(reason)
    print("Action: " + action)
    send_message2(
        {
            "message_type": "reasoning",
            "display_type": "text",
            "message": reason.strip(),
        }
    )

    try:
      converted_action = json_action.JSONAction(
          **extract_json(action),
      )
      step_data["converted_action"] = converted_action
    except Exception as e:
      print("Failed to convert the output to a valid action.")
      print(str(e))
      step_data["summary"] = (
          "Can not parse the output to a valid action. Please make sure to pick"
          " the action from the list with required parameters (if any) in the"
          " correct JSON format!"
      )
      self.history.append(step_data)
      step_data["converted_action"] = "error_retry"
      send_message(
          {
              "message_type": "action",
              "display_type": "text",
              "message": "MLLM responded with an invalid action. Retrying...",
          }
      )
      return (False, step_data)

    if (
        converted_action.action_type
        in ["click", "long_press", "input_text", "scroll"]
        and converted_action.index is not None
    ):
      if converted_action.index >= len(before_ui_elements):
        print("Index out of range.")
        step_data["summary"] = (
            "The parameter index is out of range. Remember the index must be in"
            " the UI element list!"
        )
        self.history.append(step_data)
        step_data["converted_action"] = "error_retry"
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": "MLLM responded with an invalid action. Retrying...",
            }
        )
        return (False, step_data)

      add_ui_element_mark(
          step_data["raw_screenshot"],
          before_ui_elements[converted_action.index],
          converted_action.index,
          logical_screen_size,
          physical_frame_boundary,
          orientation,
      )
      step_data["target_element"] = asdict(
          before_ui_elements[converted_action.index]
      )

    if converted_action.action_type == "status":
      step_data["summary"] = "Agent thinks the request has been completed."
      if converted_action.goal_status == "infeasible":
        print("Agent stopped since it thinks mission impossible.")
        step_data["summary"] = (
            "Agent thinks the mission is infeasible and stopped."
        )
      self.history.append(step_data)
      if converted_action.goal_status == "infeasible":
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": "Task infeasible.",
            }
        )
      else:
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": "Task completed.",
            }
        )
      return (True, step_data)

    if converted_action.action_type == "answer":
      print("Agent answered with: " + converted_action.text)

    try:
      self.early_stop = is_need_stop()
      if self.early_stop:
        return (True, step_data)
      actual_action_coordinates = execute_adb_action(
          converted_action,
          self.device,
          before_ui_elements,
          logical_screen_size,
      )
      step_data["actual_action_coordinates"] = actual_action_coordinates
      self.early_stop = is_need_stop()
      if self.early_stop:
        return (True, step_data)
    except Exception as e:
      print("Failed to execute action.")
      print(str(e))
      step_data["summary"] = (
          "Can not execute the action, make sure to select the action with"
          " the required parameters (if any) in the correct JSON format!"
      )
      step_data["converted_action"] = "error_retry"
      send_message(
          {
              "message_type": "action",
              "display_type": "text",
              "message": "MLLM responded with an invalid action. Retrying...",
          }
      )
      return (False, step_data)

    self.device.wait_to_stabilize()
    self.early_stop = is_need_stop()
    if self.early_stop:
      return (True, step_data)

    orientation = self.device.get_orientation()
    logical_screen_size = self.device.get_screen_size()
    physical_frame_boundary = self.device.get_physical_frame_boundary()

    after_ui_elements = self.device._get_ui_elements()
    after_ui_elements_list = _generate_ui_elements_description_list(
        after_ui_elements, logical_screen_size
    )
    after_screenshot = np.array(self.device.get_screenshot())
    step_data["after_screenshot"] = after_screenshot.copy()
    for index, ui_element in enumerate(after_ui_elements):
      if validate_ui_element(ui_element, logical_screen_size):
        add_ui_element_mark(
            after_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

    add_screenshot_label(step_data["before_screenshot_with_som"], "before")
    add_screenshot_label(after_screenshot, "after")
    step_data["after_screenshot_with_som"] = after_screenshot.copy()

    summary_prompt = _summarize_prompt(
        action,
        reason,
        goal,
        before_ui_elements_list,
        after_ui_elements_list,
    )
    self.early_stop = is_need_stop()
    if self.early_stop:
      return (True, step_data)
    summary, raw_response = ask_mllm(
        summary_prompt,
        [
            before_screenshot,
            after_screenshot,
        ],
    )

    if not raw_response:
      step_data["summary"] = (
          "Some error occurred calling LLM during summarization phase."
      )
      self.history.append(step_data)
      return (False, step_data)

    step_data["summary_prompt"] = summary_prompt
    step_data["summary"] = f"Action selected: {action}. {summary}"
    print("Summary: " + summary)
    send_message(
        {
            "message_type": "summary",
            "display_type": "text",
            "message": summary,
        }
    )
    step_data["summary_raw_response"] = raw_response

    self.history.append(step_data)
    return (False, step_data)
