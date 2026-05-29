from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw

from dataclasses import dataclass, field
# from MobileAgentE.api import encode_image
# from MobileAgentE.controller import tap, swipe, type, back, home, switch_app, enter, save_screenshot_to_file
# from MobileAgentE.text_localization import ocr
import copy
import re
import json
import time
import os


class InfoPool:
    """Keeping track of all information across the agents."""

    # User input / accumulated knowledge
    instruction: str = ""
    # Perception
    width: int = 1080
    height: int = 2340

    # Working memory
    action_history: list = field(default_factory=list)  # List of actions
    error_descriptions: list = field(default_factory=list)

    last_action: str = ""  # Last action

    error_flag_plan: bool = False  # if an error is not solved for multiple attempts with the executor
    error_description_plan: bool = False  # explanation of the error for modifying the plan

    # Planning
    paths: list = field(default_factory=list)
    plan: str = ""
    current_subgoal: str = ""
    prev_subgoal: str = ""

    # future tasks
    future_tasks: list = field(default_factory=list)

class BaseAgent(ABC):
    @abstractmethod
    def init_chat(self) -> list:
        pass
    @abstractmethod
    def get_prompt(self, *args, **kwargs) -> str:
        pass
    @abstractmethod
    def parse_response(self, response: str) -> dict:
        pass


class Planner(BaseAgent):

    def init_chat(self):
        operation_history = []
        system_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to track progress and devise high-level plans to achieve the user's requests. Think as if you are a human user operating the phone."

        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history

    def get_prompt(self, instruction, paths) -> str:
        prompt = """
        Plan the task by analyzing candidate execution paths, extracting reusable operations, and decomposing complex GUI tasks into executable sub-task sequences.

        ## Input Format
        - **Target Task**: [Specific task description to be completed by the user]
        - **Candidate Execution Paths**: [List of relevant execution paths retrieved from the knowledge graph]

        ## Path Format Description
        Each path format: `<PATH>State Description 1 -> Operation Description 1.$action_id$ -> State Description 2 -> Operation Description 2.$action_id$ -> ...</PATH>`

        ## Operation Type Classification (CRITICAL - FOLLOW STRICTLY)

        ### 1. High Level Action
        - **Identifier**: Contains `$action_id$` where action_id is descriptive text like "high_level_action_open_bluetooth_settings"
        - **Source**: Directly extracted from candidate paths WITHOUT ANY MODIFICATION
        - **Characteristics**: Multi-step composite operations, reusable across tasks
        - **Action ID**: Preserve the original `$descriptive_action_id$`
        - **STRICT RULE**: Operation description must be EXACTLY as found in the path

        ### 2. Normal Operation  
        - **Identifier**: Contains `$action_id$` where action_id is a hash code like "a5693944-0c77-4108-8b1c-0913d19e5bde"
        - **Source**: Directly extracted from candidate paths WITHOUT ANY MODIFICATION
        - **Characteristics**: Single atomic operations with hash-based IDs
        - **Action ID**: Preserve the original `$hash_code$`
        - **STRICT RULE**: Operation description must be EXACTLY as found in the path

        ### 3. Exploration Operation (MUST GENERATE WHEN NEEDED)
        - **Trigger Conditions**: 
          - **Parameter Mismatch**: Any operation from paths that requires different parameters (names, categories, values, etc.) than what was originally recorded
          - **Description Modification**: ANY operation that needs to change its description from the original path version
          - **Missing Operations**: Required operations completely absent from all candidate paths
          - **Path Gaps**: Paths end in wrong state and additional steps are needed
          - **Navigation Gaps**: Missing steps between available operations
          - **Task-Specific Adaptations**: Operations that exist in paths but need customization for current task
        - **Characteristics**: 
          - New operations you must invent to bridge gaps
          - Modified versions of existing operations with different parameters
          - Customized operations adapted from paths but with changed descriptions
        - **Action ID**: Always set to `null`
        - **Operation Type**: Always set to `"exploration"`

        ## CRITICAL CLASSIFICATION RULES

        ### When to Use Each Type:
        - **High Level Action / Normal Operation**: ONLY when the operation from the path can be used EXACTLY as-is, with NO changes to description, parameters, or context
        - **Exploration**: ANY time you need to:
          - Change parameter values (names, categories, amounts, etc.)
          - Modify operation descriptions
          - Adapt operation context for current task
          - Fill gaps not covered by existing paths
          - Customize existing operations with different specifics

        ### Examples:
        - **Path has**: "Enter 'John Doe' as name" → **Current task needs**: "Enter 'Jane Smith' as name" → **Classification**: EXPLORATION
        - **Path has**: "Select 'Food' category" → **Current task needs**: "Select 'Travel' category" → **Classification**: EXPLORATION  
        - **Path has**: "Click submit button" → **Current task needs**: "Click submit button" → **Classification**: NORMAL OPERATION (if exact match)

        ## Planning Process

        ### 1. Path Analysis and Operation Extraction
        - Analyze each candidate path individually
        - **STRICT COMPARISON**: Compare each path operation against target task requirements
        - **Parameter Assessment**: Check if any parameters (names, values, categories) need modification
        - **Exact Match Verification**: Only extract operations that can be used without ANY modification
        - Identify operation steps that require customization or parameter changes

        ### 2. Gap Analysis and Exploration (CRITICAL STEP)
        - **MANDATORY**: Identify operations needing parameter modifications or description changes
        - **Parameter Mismatch Detection**: Flag any operation where parameters don't match target task
        - **Customization Requirements**: Identify operations that exist in paths but need adaptation
        - **Missing Operations**: Check for completely absent required operations
        - **Path Completion**: Verify if paths fully accomplish the target task
        - **When gaps/mismatches found**: Generate exploration operations for ALL modifications needed

        ### 3. Operation Sequence Generation
        - Extract only EXACT MATCHES as high_level_action/normal_operation
        - **MANDATORY**: Convert ALL parameter-mismatched operations to exploration operations
        - Generate exploration operations for missing steps and customizations
        - Integrate operations in logical order with proper preconditions
        - Validate that the sequence accomplishes the full target task

        ## Output Requirements
        - **MUST include exploration operations** for parameter mismatches and description modifications
        - **MUST NOT modify** descriptions of high_level_action/normal_operation from their original path versions
        - **ALWAYS analyze** parameter compatibility between paths and target task
        - **Be explicit** about why each operation is classified as exploration vs reuse

        ## Output Format
        ```json
        {
          "gap_analysis": "DETAILED analysis of parameter mismatches, description modifications needed, and missing operations. Explain why specific operations require exploration classification.",
          "selected_operations": [
            {
              "source_path": "Path number or 'exploration'",
              "operation": "Operation description (EXACT for reuse, MODIFIED for exploration)",
              "type": "high_level_action/normal_operation/exploration",
              "action_id": "$action_id$ or null",
              "modification_reason": "Why this became exploration (parameter change, description modification, etc.) or 'none' for exact reuse"
            }
          ],
          "task_execution_sequence": [
            {
              "step": 1,
              "operation_type": "high_level_action/normal_operation/exploration", 
              "description": "Specific operation description",
              "action_id": "$action_id$ or null",
              "expected_state": "Expected interface state after execution",
              "precondition": "Required state/condition before this operation",
              "parameter_customization": "What parameters were changed from original path (if exploration)"
            }
          ]
        }"""
        prompt += "### Target Task ###\n"
        prompt += f"{instruction}\n\n"
        prompt += "### Candidate Execution Paths ###\n"
        prompt += f"{paths}\n\n"
        return prompt

    def parse_response(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Plan ###")[0].replace("\n", " ").replace("  ", " ").strip()
        plan = response.split("### Plan ###")[-1].split("### Current Subgoal ###")[0].replace("\n", " ").replace("  ", " ").strip()
        current_subgoal = response.split("### Current Subgoal ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "plan": plan, "current_subgoal": current_subgoal}


class Critic(BaseAgent):
    # def __init__(self, adb_path):
    #     self.adb = adb_path

    def init_chat(self):
        operation_history = []
        ## Below is the prompt for mobile
        system_prompt = r"""
# State Checker

You are a state checker that determines if a screen is ready for a specific action.

## Task
Look at the screenshot and check if the current screen matches what's needed for the action.

## Input
- Action description and required screen state
- Current screenshot

## Process
1. Look at the screenshot carefully
2. Check if it matches the required state
3. Explain your reasoning briefly
4. Give final answer: true or false

## Output Format
First explain what you see and why it matches/doesn't match.
Then end with: DECISION: true or false

## Rules
- If screen matches requirement: true
- If the element corresponding to the Action Description is visible on the screen: true
- If screen is wrong or unclear: false
- Be simple and direct in your explanation

"""
        # sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to choose the correct actions to complete the user's instruction. Think as if you are a human user operating the phone."
        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history

    def get_prompt(self, action, description) -> str:
        prompt = "## Action to execute\n"
        prompt += f"{action}\n\n"
        prompt += "## Required screen state\n"
        prompt += f"{description}\n\n"
        prompt += "Look at the attached screenshot and determine if the current screen is ready for this action. Explain briefly what you see, then give your decision."

        return prompt

    def get_similarity_prompt(self, info_pool: InfoPool, db_tasks: list) -> str:
        prompt = "## Task Similarity Judgment\n"
        prompt += "Please analyze each task pair and determine if they are identical or different.\n\n"

        for i, task in enumerate(db_tasks):
            prompt += f"**Batch {i + 1}:**\n"
            prompt += f"Task 1: '{info_pool.instruction}'\n"
            prompt += f"Task 2: '{task}'\n"
            prompt += f"\n"

        prompt += "Please provide your judgment for each batch following the specified output format."
        return prompt

    def parse_response(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace("  ",
                                                                                                              " ").strip()
        action = response.split("### Action ###")[-1].split("### Description ###")[0].replace("\n", " ").replace(
            "  ",
            " ").strip()
        description = response.split("### Description ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "action": action, "description": description}


class Operator_local(BaseAgent):
    def __init__(self, adb_path):
        self.adb = adb_path

    def init_chat(self):
        operation_history = []
        ## Below is the prompt for mobile
        system_prompt = r"""You are a GUI agent. You are given a user instruction with screenshots and a list of task related elements in the current screenshots. You need to following the task instruction by performing the next action to complete the task. 

        ## Output Format
        ```\nThought: ...
        Action: ...\n```

        ## Action Space
        click(start_box='<|box_start|>(x1,y1)<|box_end|>')
        long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
        type(content='')
        scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
        press_home()
        press_back()
        finished(content='') # Submit the task regardless of whether it succeeds or fails.

        ## Note
        - Use English in `Thought` part.

        - Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

        - At the end of `Thought` part, state the element you are going to use.
        """
        # sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to choose the correct actions to complete the user's instruction. Think as if you are a human user operating the phone."
        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "## User Instruction\n"
        prompt += f"{info_pool.instruction}\n\n"

        return prompt


    def parse_response(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace("  ",
                                                                                                              " ").strip()
        action = response.split("### Action ###")[-1].split("### Description ###")[0].replace("\n", " ").replace("  ",
                                                                                                                 " ").strip()
        description = response.split("### Description ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "action": action, "description": description}