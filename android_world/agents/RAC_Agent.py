# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Multimodal Autonomous Agent for Android (M3A)."""

import time
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.agents import RAC_build
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
try:
    from GCR.workflow.predict_paths_and_answers import predict_paths
except ImportError:
    predict_paths = None
import re
import copy
import json
from android_world.utils.graph_utils import Neo4jDatabase, extract_features
from android_world.utils.vector_utils import VectorStore
import config
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
from android_world.suite_utils import KnowledgeBaseManager


PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Answer back if the request/goal is a question (or a chat message),'
    ' like user asks "What is my schedule for today?".\n'
    '- Complete some tasks described in the requests/goals by'
    ' performing actions (step by step) on the phone.\n\n'
    'When given a user request, you will try to complete it step by step.'
    ' At each step, you will be given the current screenshot (including the'
    ' original screenshot and the same screenshot with bounding'
    ' boxes and numeric indexes added to some UI elements) and a history of'
    ' what you have done (in text). Based on these pieces of information and'
    ' the goal, you must choose to perform one of the'
    ' action in the following list (action description followed by the JSON'
    ' format) by outputing the action in the correct JSON format.\n'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status:'
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    "- If you think the task is not feasible (including cases like you don't"
    ' have enough information or can not perform some necessary actions),'
    ' finish by using the `status` action with infeasible as goal_status:'
    ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n'
    "- Answer user's question:"
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    '- Click/tap on an element on the screen. We have added marks (bounding'
    ' boxes with numeric indexes on their TOP LEFT corner) to most of the UI'
    ' elements in the screenshot, use the numeric index to indicate which'
    ' element you want to click:'
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    '- Long press on an element on the screen, similar with the click action'
    ' above, use the numeric label on the bounding box to indicate which'
    ' element you want to long press:'
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    '- Type text into a text field (this action contains clicking the text'
    ' field, typing in the text and pressing the enter, so no need to click on'
    ' the target field to start), use the numeric label'
    ' on the bounding box to indicate the target text field:'
    ' `{{"action_type": "input_text", "text": <text_input>,'
    ' "index": <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the same numeric index as above if you want to scroll a'
    ' specific UI element, leave it empty when scroll the whole screen:'
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    '- Open an app (nothing will happen if the app is not'
    ' installed): `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
)


def build_exploration_prompt(user_task, completed_subgoal, remaining_subgoals, task_goal, ui_elements, knowledge="", history="", expected_state=""):
    """
    构建exploration模式的prompt

    Args:
        task_goal (str): 用户的任务目标
        ui_elements (str): 当前屏幕的UI元素信息
        knowledge (str): 专家知识和技巧
        history (str): 操作历史，默认为空

    Returns:
        str: 完整的prompt
    """

#     PROMPT_TEMPLATE = r"""
# ## Role Definition
# You are an Android operation AI that fulfills user requests through precise screen interactions. The current screenshot and the same screenshot with bounding boxes and labels added are also given to you.
#
# ## Action Catalog
# Available actions (STRICT JSON FORMAT REQUIRED):
# 1. Status Operations:
# - Task Complete: {{"action_type": "status", "goal_status": "complete"}}
# - Task Infeasible: {{"action_type": "status", "goal_status": "infeasible"}}
# 2. Information Actions:
# - Answer Question: {{"action_type": "answer", "text": "<answer_text>"}}
# 3. Screen Interactions:
# - Tap Element: {{"action_type": "click", "index": <visible_index>}}
# - Long Press: {{"action_type": "long_press", "index": <visible_index>}}
# - Scroll: Scroll the screen or a specific scrollable UI element. Use the 'index' of the target element if scrolling a specific element, or omit 'index' to scroll the whole screen. "action_type": "scroll", "direction": <"up"|"down"|"left"|"right">, "index": <optional_target_index>
# 4. Input Operations:
# - Text Entry: {{"action_type": "input_text", "text": "<content>", "index": <text_field_index>}}
# - Keyboard Enter: {{"action_type": "keyboard_enter"}}
# 5. Navigation:
# - Home Screen: {{"action_type": "navigate_home"}}
# - Back Navigation: {{"action_type": "navigate_back"}}
# 6. System Actions:
# - Launch App: {{"action_type": "open_app", "app_name": "<exact_name>"}}
# - Wait Refresh: {{"action_type": "wait"}}
#
# ## Current Objective
# User Goal: {task_goal}
#
# ## Expected State
# Completion Statu: {expected_state}
#
# ## Execution Context
# Action History:
# {history}
#
# Visible UI Elements (Only interact with *visible=true elements):
# {ui_elements}
#
# ## Core Strategy(CRITICAL)
# 1. Path Optimization:
# - Prefer direct methods (e.g., open_app > app drawer navigation)
# - ALWAYS use the 'input_text' action for entering text into designated text fields rather than 'type' or 'input'.
# - Verify element visibility ('visible=true') before attempting any interaction (click, long_press, input_text). Do not interact with elements marked 'visible=false'.
# - Use 'scroll' when necessary to bring off-screen elements into view. Prioritize scrolling specific containers ('index' provided) over full-screen scrolls if possible.
# - STRICTLY follow the action format provided in 'Action Catalog'.
#
#
# 2. Error Handling Protocol:
# - Switch approach after ≥ 2 failed attempts
# - Prioritize scrolling ('scroll' action) over force-acting on invisible elements
# - If an element is not visible, use 'scroll' in the likely direction (e.g., 'down' to find elements below the current view).
# - Try opposite scroll direction if initial fails (up/down, left/right)
# - If the 'open_app' action fails to correctly open the app, find the corresponding app in the app drawer and open it.
#
# 3. Information Tasks:
# - MANDATORY: Use answer action for questions
# - Verify data freshness (e.g., check calendar date)
#
# 4. Expected State Compliance:
# - STOP IMMEDIATELY when the current screen state matches the expected completion state
# - DO NOT perform additional actions beyond what is required to reach the expected state
# - VALIDATE before each action: "Does the current state already satisfy the expected state?"
# - If expected state is achieved: Use {{"action_type": "status", "goal_status": "complete"}} immediately
# - Avoid over-execution: Extra actions can break the workflow or cause unintended side effects
#
# ## Expert Techniques
# Here are some tips for you:
# {knowledge}
#
# Now output an action from the above list in the correct JSON format, following the reason why you do that. Your answer should look like:
# Reason: ...
# Action: {{"action_type":...}}
#
# Your Answer:
# """

    PROMPT_TEMPLATE = r"""
    
    
    ## Role Definition
    You are an Android operation AI responsible for executing ONLY THE CURRENT SUB-GOAL. You must STOP immediately after completing your assigned sub-goal.

    ## CRITICAL EXECUTION BOUNDARY 
    **YOUR SOLE RESPONSIBILITY**: Complete the current sub-goal and STOP
    - ALLOWED: Actions that directly contribute to the current sub-goal
    - FORBIDDEN: ANY action beyond the current sub-goal scope
    - MANDATORY STOP: Immediately when current subgoal is achieved

    ## Task Context
    Overall User Task (For context only - NOT your responsibility):
    {user_task}

    Your CURRENT SUBTASK (THIS IS YOUR ONLY JOB):
    **Sub-goal**: {task_goal}
    **Expected State**: {expected_state}

    Progress Information (Read-only context):
    **Already Completed** (DO NOT REPEAT):
    {completed_subgoal}

    **Will be handled later** (DO NOT EXECUTE):
    {remaining_subgoals}
    
    ## Action Catalog
    Available actions (STRICT JSON FORMAT REQUIRED):
    1. Status Operations:
    - Task Complete: {{"action_type": "status", "goal_status": "complete"}}
    - Task Infeasible: {{"action_type": "status", "goal_status": "infeasible"}}
    2. Information Actions:
    - Answer Question: {{"action_type": "answer", "text": "<answer_text>"}}
    3. Screen Interactions:
    - Tap Element: {{"action_type": "click", "index": <visible_index>}}
    - Long Press: {{"action_type": "long_press", "index": <visible_index>}}
    - Scroll: Scroll the screen or a specific scrollable UI element. Use the 'index' of the target element if scrolling a specific element, or omit 'index' to scroll the whole screen. "action_type": "scroll", "direction": <"up"|"down"|"left"|"right">, "index": <optional_target_index>
    4. Input Operations:
    - Text Entry: {{"action_type": "input_text", "text": "<content>", "index": <text_field_index>}}
    - Keyboard Enter: {{"action_type": "keyboard_enter"}}
    5. Navigation:
    - Home Screen: {{"action_type": "navigate_home"}}
    - Back Navigation: {{"action_type": "navigate_back"}}
    6. System Actions:
    - Launch App: {{"action_type": "open_app", "app_name": "<exact_name>"}}
    - Wait Refresh: {{"action_type": "wait"}}

    ## Execution Context
    Recent Actions in Current Subtask:
    {history}

    Visible UI Elements (Only interact with *visible=true elements):
    {ui_elements}

    ## Core Strategy(CRITICAL)
    1. Path Optimization:
    - Prefer direct methods (e.g., open_app > app drawer navigation)
    - ALWAYS use the 'input_text' action for entering text into designated text fields rather than 'type' or 'input'.
    - Verify element visibility ('visible=true') before attempting any interaction (click, long_press, input_text). Do not interact with elements marked 'visible=false'.
    - Use 'scroll' when necessary to bring off-screen elements into view. Prioritize scrolling specific containers ('index' provided) over full-screen scrolls if possible.
    - STRICTLY follow the action format provided in 'Action Catalog'.


    2. Error Handling Protocol:
    - Switch approach after ≥ 2 failed attempts
    - Prioritize scrolling ('scroll' action) over force-acting on invisible elements
    - If an element is not visible, use 'scroll' in the likely direction (e.g., 'down' to find elements below the current view).
    - Try opposite scroll direction if initial fails (up/down, left/right)
    - If the 'open_app' action fails to correctly open the app, find the corresponding app in the app drawer and open it.

    3. Information Tasks:
    - MANDATORY: Use answer action for questions
    - Verify data freshness (e.g., check calendar date)

    4. Expected State Compliance:
    - STOP IMMEDIATELY when the current screen state matches the expected completion state of the current sub-goal
    - DO NOT perform additional actions beyond what is required to reach the expected state
    - VALIDATE before each action: "Does the current state already satisfy the expected state?"
    - If expected state of the current sub-goal is achieved: Use {{"action_type": "status", "goal_status": "complete"}} immediately
    - Avoid over-execution: Extra actions can break the workflow or cause unintended side effects

    ## Expert Techniques
    Here are some tips for you:
    {knowledge}

    Now output an action from the above list in the correct JSON format, following the reason why you do that. Your answer should look like:
    Reason: ...
    Action: {{"action_type":...}}

    Your Answer:
    """

    return PROMPT_TEMPLATE.format(
        user_task=user_task,
        completed_subgoal=completed_subgoal if completed_subgoal else "This is the first sub-goal",
        remaining_subgoals=remaining_subgoals if remaining_subgoals else "This is the final sub-goal",
        task_goal=task_goal,
        ui_elements=ui_elements,
        knowledge=knowledge if knowledge else "No available knowledge.",
        history=history if history else "No previous actions.",
        expected_state=expected_state if expected_state else "No expected state.",
    )

SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is: {goal}\n'
    'Now I want you to summerize the latest step.\n'
    'You will be given the screenshot before you performed the action (which'
    ' has a text label "before" on the bottom right), the action you chose'
    ' (together with the reason) and the screenshot after the action was'
    ' performed (which has a text label "after" on the bottom right).\n'
    'Also here is the list of detailed information for some UI elements'
    ' in the before screenshot:\n{before_elements}\n'
    'Here is the list for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    'By comparing the two screenshots (plus the UI element lists) and the'
    ' action performed, give a brief summary of this step. This summary'
    ' will be added to action history and used in future action selection,'
    ' so try to include essential information you think that will be most'
    ' useful for future action selections like what you'
    ' intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might be'
    ' wrong), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short (better less than 50 words) and in a single line\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)

def _summarize_prompt(
    action: str,
    reason: str,
    goal: str,
    before_elements: str,
    after_elements: str,
) -> str:
  """Generate the prompt for the summarization step.

  Args:
    action: Action picked.
    reason: The reason to pick the action.
    goal: The overall goal.
    before_elements: Information for UI elements on the before screenshot.
    after_elements: Information for UI elements on the after screenshot.

  Returns:
    The text prompt for summarization that will be sent to gpt4v.
  """
  return SUMMARY_PROMPT_TEMPLATE.format(
      goal=goal,
      before_elements=before_elements,
      after_elements=after_elements,
      action=action,
      reason=reason,
  )

def _generate_ui_element_description(
    ui_element: representation_utils.UIElement, index: int
) -> str:
  """Generate a description for a given UI element with important information.

  Args:
    ui_element: UI elements for the current screen.
    index: The numeric index for the UI element.

  Returns:
    The description for the UI element.
  """
  element_description = f'UI element {index}: {{"index": {index}, '
  if ui_element.text:
    element_description += f'"text": "{ui_element.text}", '
  if ui_element.content_description:
    element_description += (
        f'"content_description": "{ui_element.content_description}", '
    )
  if ui_element.hint_text:
    element_description += f'"hint_text": "{ui_element.hint_text}", '
  if ui_element.tooltip:
    element_description += f'"tooltip": "{ui_element.tooltip}", '
  element_description += (
      f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
  )
  element_description += (
      '"is_long_clickable":'
      f' {"True" if ui_element.is_long_clickable else "False"}, '
  )
  element_description += (
      f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
  )
  if ui_element.is_scrollable:
    element_description += '"is_scrollable": True, '
  if ui_element.is_focusable:
    element_description += '"is_focusable": True, '
  element_description += (
      f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
  )
  element_description += (
      f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
  )
  return element_description[:-2] + '}'


def _generate_ui_elements_description_list(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate concise information for a list of UIElement.

  Args:
    ui_elements: UI elements for the current screen.
    screen_width_height_px: The height and width of the screen in pixels.

  Returns:
    Concise information for each UIElement.
  """
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
      tree_info += _generate_ui_element_description(ui_element, index) + '\n'
  return tree_info





def extract_paths_to_json(path_list):
    """
    提取PATH标签内容并转换为简单的JSON格式

    Args:
        path_list: 包含PATH标签的字符串列表

    Returns:
        dict: 格式为 {"path_1": "内容", "path_2": "内容", ...}
    """
    result = {}

    for i, path_string in enumerate(path_list, 1):
        # 使用正则表达式提取<PATH>和</PATH>之间的内容
        match = re.search(r'<PATH>(.*?)</PATH>', path_string, re.DOTALL)

        if match:
            # 提取PATH标签内的内容，去除首尾空格
            path_content = match.group(1).strip()
            result[f"path_{i}"] = path_content
        else:
            # 如果没有PATH标签，将整个字符串作为内容
            result[f"path_{i}"] = path_string.strip()

    return result

def add_response(role, prompt, chat_history):
    new_chat_history = copy.deepcopy(chat_history)
    content = [
            {
            "type": "text",
            "text": prompt
            },
    ]
    new_chat_history.append([role, content])
    return new_chat_history


def parse_task_execution_sequence(model_output: str) -> list:
    """
    提取模型输出中的task_execution_sequence
    """
    # 提取JSON部分
    match = re.search(r'```json\s*(.*?)\s*```', model_output, re.DOTALL)
    if not match:
        # 如果没有```json```标记，尝试整个字符串
        json_str = model_output.strip()
    else:
        json_str = match.group(1).strip()

    # 解析JSON并提取task_execution_sequence
    try:
        data = json.loads(json_str)
        if 'task_execution_sequence' not in data:
            raise ValueError("Missing 'task_execution_sequence' in response")
        return data['task_execution_sequence']
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")


def extract_decision(response_text):
    """
    从state_checker的输出中提取DECISION结果

    Args:
        response_text (str): state_checker的完整输出文本

    Returns:
        bool: True表示可以执行action，False表示不可以，默认返回False
    """
    # 转换为小写便于匹配
    text = response_text.lower()

    # 查找DECISION关键字
    if "decision:" in text:
        # 提取DECISION:后面的内容
        decision_part = text.split("decision:")[1].strip()

        # 检查是true还是false
        if decision_part.startswith("true"):
            return True
        elif decision_part.startswith("false"):
            return False

    # 默认返回False（保守策略）
    return False

def get_relevant_elements_by_intent(
        user_task: str,
        elements: List[Dict[str, Any]],
        min_similarity: float = 0.0,
        model_name: str = 'BAAI/bge-large-en-v1.5'
) -> List[Dict[str, Any]]:
    """
    根据用户任务找出相关的元素，通过比较用户任务与元素意图的语义相似度

    参数:
        user_task (str): 用户任务描述
        elements (List[Dict]): 元素列表，格式为 {"content": content, "intent": user_intent, "description": description}
        min_similarity (float): 最小相似度阈值 (0-1)
        model_name (str): 使用的语义模型名称

    返回:
        List[Dict]: 相关元素列表，按相似度降序排序，每个元素增加similarity字段
    """
    # 参数验证
    if not user_task or not elements:
        return []

    # 加载语义相似度模型
    try:
        model = SentenceTransformer(model_name)
        print(f"Loaded semantic similarity model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to simple keyword matching...")
        return _keyword_fallback(user_task, elements, min_similarity)

    # 编码用户任务
    task_embedding = model.encode([user_task])[0]

    # 准备相似度计算
    relevant_elements = []

    # 处理每个元素
    for element in elements:
        # 获取用户意图 (使用新的数据格式中的intent字段)
        user_intent = element.get("intent", "")
        function = element.get("function", "")
        if not user_intent or not function:
            continue

        query_text = "User Intent: " + user_intent + " function: " + function
        # query_text = task_relation
        # print(f"Query: {query_text}")
        # 计算意图嵌入
        intent_embedding = model.encode([query_text])[0]

        # 计算相似度
        similarity = cosine_similarity(
            task_embedding.reshape(1, -1),
            intent_embedding.reshape(1, -1)
        )[0][0]

        # 添加相似度到元素
        element_with_similarity = element.copy()  # 避免修改原始数据
        element_with_similarity["similarity"] = float(similarity)

        # 只保留相似度高于阈值的结果
        if similarity >= min_similarity:
            relevant_elements.append(element_with_similarity)

    # 按相似度降序排序
    relevant_elements.sort(key=lambda x: x["similarity"], reverse=True)

    return relevant_elements

def _keyword_fallback(user_task: str, elements: List[Dict[str, Any]], min_similarity: float = 0.3) -> List[
    Dict[str, Any]]:
    """
    简单关键词匹配作为备选方案

    参数:
        user_task: 用户任务描述
        elements: 元素列表，格式为 {"content": content, "intent": user_intent, "description": description}
        min_similarity: 最小相似度阈值

    返回:
        相关元素列表
    """
    # 将用户任务转为小写并分词
    task_words = set(user_task.lower().split())

    results = []
    for element in elements:
        # 获取意图 (使用新的数据格式中的intent字段)
        user_intent = element.get("intent", "")
        if not user_intent:
            continue

        # 将意图转为小写并分词
        intent_words = set(user_intent.lower().split())

        # 计算单词重叠率
        if not task_words or not intent_words:
            continue

        common_words = task_words.intersection(intent_words)
        similarity = len(common_words) / max(len(task_words), len(intent_words))

        if similarity >= min_similarity:
            element_copy = element.copy()
            element_copy["similarity"] = similarity
            results.append(element_copy)

    # 按相似度降序排序
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results



def save_screenshot_to_current_dir(before_screenshot, filename="screenshot.png"):
    """将NumPy数组截图保存到当前目录"""
    Image.fromarray(before_screenshot).save(filename)
    return filename

class RAC(base_agent.EnvironmentInteractingAgent):
  """SRoA which stands for screen only agent."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      cloud_llm: infer.MultimodalLlmWrapper,
      local_llm: infer.MultimodalLlmWrapper,
      uri: str,
      auth: tuple,
      name: str = 'SRoA',
      wait_after_action_seconds: float = 5.0,
  ):
    """Initializes a SRoA Agent.

    Args:
      env: The environment.
      llm: The multimodal LLM wrapper.
      name: The agent name.
      wait_after_action_seconds: Seconds to wait for the screen to stablize
        after executing an action
    """
    super().__init__(env, name)
    self.cloud_llm = cloud_llm
    self.local_llm = local_llm
    self.history = []
    self.additional_guidelines = None
    self.wait_after_action_seconds = wait_after_action_seconds
    self.planner = RAC_build.Planner()
    self.critic = RAC_build.Critic()
    self.info = {}
    # self.graph_base = Neo4jDatabase(uri=uri, auth=auth, database=config.Neo4j_DATABASE)
    self.graph_base = Neo4jDatabase(uri=uri, auth=auth)
    # self.vector_base = VectorStore(
    #     api_key=config.PINECONE_API_KEY,
    #     index_name=config.PINECONE_INDEX,
    #     dimension=2048,
    #     batch_size=2,
    # )

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    # Hide the coordinates on screen which might affect the vision model.
    self.env.hide_automation_ui()
    self.history = []

  def _save_planning_data(self, paths, plan, goal):
      """保存 paths 和 plan 到 JSON 文件"""
      import json
      import os
      from pathlib import Path
      from datetime import datetime

      try:
          # 获取任务名称
          task_name = os.environ.get("TASK", "UnknownTask")

          # 获取日志目录
          base_dir = Path(os.environ.get("EAM_PLAN_LOG_DIR", "artifacts/plans"))
          run_folders = sorted([f for f in base_dir.glob("run_*") if f.is_dir()])

          if run_folders:
              log_dir = run_folders[-1] / "runs"
          else:
              timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
              log_dir = base_dir / f"run_{timestamp}" / "runs"

          log_dir.mkdir(parents=True, exist_ok=True)

          # 创建合并数据
          planning_data = {
              'timestamp': datetime.now().isoformat(),
              'task_name': task_name,
              'goal': goal,
              'reasoning_paths': paths,
              'overall_plan': plan
          }

          # 保存为 JSON 文件
          filepath = log_dir / f"{task_name}_PlanningData.json"
          with open(filepath, 'w', encoding='utf-8') as f:
              json.dump(planning_data, f, indent=2, ensure_ascii=False)

          print(f"✅ Planning data saved to: {task_name}_PlanningData.json")

      except Exception as e:
          print(f"❌ Error saving planning data: {e}")

  def _generate_reasoning_paths(self, goal: str, screenshot):
      """Return candidate paths for the planner.

      RAC uses GCR by default. EAM_Agent overrides this method with MCTS path
      retrieval while keeping the planner and executor unchanged.
      """
      if predict_paths is None:
          raise RuntimeError("GCR is not installed; use EAM_Agent or install GCR.")
      kb_file = KnowledgeBaseManager.get_kb()
      return extract_paths_to_json(predict_paths(goal, kb_file)["prediction"])

  @staticmethod
  def _fallback_plan(goal: str, reason: str) -> list:
      return [{
          "step": 1,
          "operation_type": "exploration",
          "description": goal,
          "action_id": None,
          "expected_state": "Complete the user request.",
          "precondition": f"Path-based planning failed: {reason}",
          "parameter_customization": "none",
      }]

  @staticmethod
  def _resolve_plan_action_ref(plan_step: dict) -> tuple[str, str]:
      """Return ('action'|'element', id) from planner action_id fields."""
      raw_id = str(plan_step.get("action_id") or "")
      match = re.search(r"_\$((?:Element|Action))_([A-Za-z0-9#_\-]+)\$", raw_id)
      if match:
          return match.group(1).lower(), match.group(2)
      fallback_type = "action" if plan_step.get("operation_type") == "high_level_action" else "element"
      return fallback_type, raw_id.strip("$")
  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = {
        'raw_screenshot': None,
        'action_prompt': None,
        'action_output': None,
        'action_output_json': None,
        'action_reason': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
        'current_plan': None,
    }
    print('----------step ' + str(len(self.history) + 1))


    state = self.get_post_transition_state()
    step_data['raw_screenshot'] = state.pixels.copy()

    # Generate reasoning paths in the first step.
    if len(self.history) == 0:
        print('Generating reasoning paths based on the extracted graph')
        paths = {}
        try:
            paths = self._generate_reasoning_paths(goal, step_data['raw_screenshot'])
            print(paths)
            print('***************************************************************')
            prompt_planner = self.planner.get_prompt(goal, paths)
            chat_planner = self.planner.init_chat()
            chat_planner = add_response("user", prompt_planner, chat_planner)
            plan, is_safe_plan, raw_response_plan = self.cloud_llm.predict_mm(
                chat_planner,
                [
                    step_data['raw_screenshot'],
                ],
                system=True
            )

            # Mapping the overall plan to the current operation
            self.info['plan'] = parse_task_execution_sequence(plan)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Path planning failed, falling back to exploration: {e}")
            self.info['plan'] = self._fallback_plan(goal, str(e))
        print(f"Overall plan: {self.info['plan']}")
        print('***************************************************************')
        self._save_planning_data(paths, self.info['plan'], goal)
        self.info['current_plan_idx'] = 0

    # determine if the task has been completed
    if self.info['current_plan_idx'] >= len(self.info['plan']):
        print("task has been completed")
        step_data['current_plan'] = 'completed'
        return base_agent.AgentInteractionResult(
                True,
                step_data,
        )
    step_data['current_plan'] = self.info['plan'][self.info['current_plan_idx']]
    need_exploration = False
    exploration_done = False
    execution_failed_reason = ""
    #about to trigger operations listed in the plan
    if step_data['current_plan']['operation_type'] in ['high_level_action','normal_operation']:
        print(f"Checking action {step_data['current_plan']['action_id']} in the {len(self.history) + 1} th step: {step_data['current_plan']['description']} ...... ")
        prompt_critic = self.critic.get_prompt(step_data['current_plan']['description'], step_data['current_plan']['precondition'])
        chat_critic = self.critic.init_chat()
        chat_critic = add_response("user", prompt_critic, chat_critic)
        state_check_ori, is_safe_state, raw_response_state = self.cloud_llm.predict_mm(
            chat_critic,
            [
                step_data['raw_screenshot'],
            ],
            system=True
        )
        state_check = extract_decision(state_check_ori)
        reason = state_check_ori
        # print(f"Check results:{reason}")
        print('***************************************************************')
        if state_check == True:
            print(f"Checking done! Good to extract parameters for execution.")
            print('***************************************************************')
            #extract high_level_action parameters from graph
            action_list = []
            action_ref_type, action_ref_id = self._resolve_plan_action_ref(step_data['current_plan'])
            if action_ref_type == 'action':
                #Get elements sequence
                element_sequence = self.graph_base.get_action_element_sequence(action_ref_id)
                # print(element_sequence)
                for element in element_sequence or []:
                    element_id = element['element_id']
                    action_list.append(self.graph_base.get_element_action_output(element_id))
            #extract element parameters from graph
            else:
                action_list.append(self.graph_base.get_element_action_output(action_ref_id))

            #Determine if need to fall_back to exploration
            if action_list == []:
                need_exploration = True
                execution_failed_reason = "Action list is empty"
                print('falling back to exploration')
            else:
                # execute actions by order
                print(f"Ready to trigger action: {step_data['current_plan']['action_id']}")
                # print(action_list)
                step_data['action_output'] = action_list
                step_data['action_raw_response'] = step_data['current_plan']['description']
                print(f"Action: {action_list}")
                # print('Reason: ' + reason)
                step_data['action_reason'] = reason
                # iteratly execute actions
                for action_output in action_list:
                    print(f"Executing: {action_output['action_output']}")
                    print('***************************************************************')
                    try:
                        action_reason, action = m3a_utils.parse_reason_action_output(action_output['action_output'])
                        action = json.loads(action)

                        if 'index' in action:
                            if action['action_type'] != 'scroll':
                                bbox_pixel = action_output['target_element']['bbox_pixels']
                                action['x'] = (bbox_pixel['x_min'] + bbox_pixel['x_max']) / 2.0
                                action['y'] = (bbox_pixel['y_min'] + bbox_pixel['y_max']) / 2.0
                                del action['index']
                            else:
                                bbox_pixel = action_output['target_element']['bbox_pixels']
                                action['x_min'] = bbox_pixel['x_min']
                                action['x_max'] = bbox_pixel['x_max']
                                action['y_min'] = bbox_pixel['y_min']
                                action['y_max'] = bbox_pixel['y_max']
                                del action['index']
                        # print(action)
                        converted_action = json_action.JSONAction(
                            **agent_utils.extract_json(json.dumps(action)),
                        )
                        step_data['action_output_json'] = converted_action
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print('Failed to convert the output to a valid action, falling back to exploration')
                        need_exploration = True
                        execution_failed_reason = "Can not parse valid action from KG"
                        break

                    if converted_action.action_type == 'status':
                        if converted_action.goal_status == 'infeasible':
                            print('Agent stopped since it thinks mission impossible.')
                        step_data['summary'] = 'Agent thinks the request has been completed.'
                        self.history.append(step_data)
                        return base_agent.AgentInteractionResult(
                            True,
                            step_data,
                        )

                    if converted_action.action_type == 'answer':
                        print('Agent answered with: ' + converted_action.text)

                    try:
                        self.env.execute_action(converted_action)
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print('Failed to execute action, falling back to exploration')
                        execution_failed_reason = "Failed to execute action."
                        need_exploration = True
                        break
                    time.sleep(self.wait_after_action_seconds)
                step_data['summary'] = 'Current Goal: ' + step_data['current_plan']['description'] + 'Current Action: ' + step_data['current_plan']['action_id']
                self.info['current_plan_idx'] += 1
        else:
            need_exploration = True


    elif step_data['current_plan']['operation_type'] == 'exploration':
        need_exploration = True
        execution_failed_reason = "Direct exploration request"


    if need_exploration:
        round_history = []
        exploration_step_data = {
            'raw_screenshot': None,
            'before_screenshot_with_som': None,
            'before_ui_elements': [],
            'after_screenshot_with_som': None,
            'action_prompt': None,
            'action_output': None,
            'action_output_json': None,
            'action_reason': None,
            'action_raw_response': None,
            'summary_prompt': None,
            'summary': None,
            'summary_raw_response': None,
        }
        print(f"Fall back to exploration: {execution_failed_reason}")
        # Fall back to elements augmented operation
        for exploration_step in range(5):
            #Extract UI elements
            state = self.get_post_transition_state()
            logical_screen_size = self.env.logical_screen_size
            orientation = self.env.orientation
            physical_frame_boundary = self.env.physical_frame_boundary

            before_ui_elements = state.ui_elements
            exploration_step_data['before_ui_elements'] = before_ui_elements
            before_ui_elements_list = _generate_ui_elements_description_list(
                before_ui_elements, logical_screen_size
            )
            exploration_step_data['raw_screenshot'] = state.pixels.copy()
            before_screenshot = state.pixels.copy()
            logical_screen_size = self.env.logical_screen_size
            for index, ui_element in enumerate(before_ui_elements):
                if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                    m3a_utils.add_ui_element_mark(
                        before_screenshot,
                        ui_element,
                        index,
                        logical_screen_size,
                        physical_frame_boundary,
                        orientation,
                    )
            exploration_step_data['before_screenshot_with_som'] = before_screenshot.copy()
            #Extract task related elements hints
            print("Navigating the current page in Knowledge Base...")
            save_screenshot_to_current_dir(exploration_step_data['raw_screenshot'], "screenshot/screenshot.jpg")
            query_feature = extract_features(
                image_inputs="screenshot/screenshot.jpg",
                model_name="resnet50")
            os.remove("screenshot/screenshot.jpg")
            vector_base = VectorStore(
                api_key=config.PINECONE_API_KEY,
                index_name=os.environ.get("DATABASE"),
                dimension=2048,
                batch_size=2,
            )
            recall_node = vector_base.query_similar(query_feature["features"], node_type='Page')
            print(recall_node)
            reranked_elements = None
            if recall_node[0] != []:
                # print(recall_node)
                if recall_node[1][0] > 0.95:
                    print(f"Page successfully found:{recall_node[0][0]}")
                    match_page_id = recall_node[0][0]
                    print("Looking for elements connected to the page...")
                    elements = self.graph_base.get_page_elements(match_page_id)
                    reranked_elements = get_relevant_elements_by_intent(step_data['current_plan']['description'], elements)
                    if len(reranked_elements) > 3:
                        reranked_elements = reranked_elements[:1]
                    print(reranked_elements)
            print(step_data['current_plan']['description'])
            if reranked_elements is not None:
                action_prompt = build_exploration_prompt(
                user_task=goal,
                completed_subgoal=json.dumps(['sub-goal ' + str(i+1) + '_ ' + subgoal['description'] for i, subgoal in enumerate(self.info['plan'][:self.info['current_plan_idx']])]) if self.info['current_plan_idx'] > 0 else [],
                remaining_subgoals=json.dumps(['sub-goal ' + str(i+1) + '_ ' + subgoal['description'] for i, subgoal in enumerate(self.info['plan'][(self.info['current_plan_idx']+1):])]) if self.info['current_plan_idx'] < len(self.history) - 1 else [],
                task_goal=step_data['current_plan']['description'],
                history=json.dumps([
                    'Step ' + str(i + 1) + '- ' + step_info['summary']
                    for i, step_info in enumerate(round_history)
                ]),
                ui_elements=before_ui_elements_list,
                knowledge=json.dumps(reranked_elements),
                expected_state=step_data['current_plan']['expected_state']
                )
            else:
                action_prompt = build_exploration_prompt(
                    user_task=goal,
                    completed_subgoal=json.dumps(
                        ['sub-goal ' + str(i + 1) + '_ ' + subgoal['description'] for i, subgoal in
                         enumerate(self.info['plan'][:self.info['current_plan_idx']])]) if self.info[
                                                                                               'current_plan_idx'] > 0 else [],
                    remaining_subgoals=json.dumps(
                        ['sub-goal ' + str(i + 1) + '_ ' + subgoal['description'] for i, subgoal in
                         enumerate(self.info['plan'][(self.info['current_plan_idx'] + 1):])]) if self.info[
                                                                                                     'current_plan_idx'] < len(
                        self.history) - 1 else [],
                    task_goal=step_data['current_plan']['description'],
                    history=json.dumps([
                        'Step ' + str(i + 1) + '- ' + step_info['summary']
                        for i, step_info in enumerate(round_history)
                    ]),
                    ui_elements=before_ui_elements_list,
                    expected_state=step_data['current_plan']['expected_state']
                )
            exploration_step_data['action_prompt'] = action_prompt
            action_output, is_safe, raw_response = self.local_llm.predict_mm(
            action_prompt,
                [
                    step_data['raw_screenshot'],
                    before_screenshot,
                ],
            )
            # print(action_output)
            # about to execute exploration action
            if is_safe == False:  # pylint: disable=singleton-comparison
                #  is_safe could be None
                action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
            Action: {{"action_type": "status", "goal_status": "infeasible"}}"""

            if not raw_response:
                raise RuntimeError('Error calling LLM in action selection phase.')
            exploration_step_data['action_output'] = action_output
            exploration_step_data['action_raw_response'] = raw_response
            reason, action = m3a_utils.parse_reason_action_output(action_output)

            # If the output is not in the right format, add it to step summary which
            # will be passed to next step and return.
            if (not reason) or (not action):
                print('Action prompt output is not in the correct format.')
                exploration_step_data['summary'] = (
                    'Output for action selection is not in the correct format, so no'
                    ' action is performed.'
                )
                round_history.append(exploration_step_data)
                continue

            print('Action: ' + action)
            print('Reason: ' + reason)
            exploration_step_data['action_reason'] = reason

            try:
                converted_action = json_action.JSONAction(
                    **agent_utils.extract_json(action),
                )
                exploration_step_data['action_output_json'] = converted_action
            except Exception as e:  # pylint: disable=broad-exception-caught
                print('Failed to convert the output to a valid action.')
                print(str(e))
                exploration_step_data['summary'] = (
                    'Can not parse the output to a valid action. Please make sure to pick'
                    ' the action from the list with required parameters (if any) in the'
                    ' correct JSON format!'
                )
                round_history.append(exploration_step_data)
                continue

            action_index = converted_action.index
            num_ui_elements = len(before_ui_elements)
            if (
                    converted_action.action_type
                    in ['click', 'long_press', 'input_text', 'scroll']
                    and action_index is not None
            ):
                if action_index >= num_ui_elements:
                    print(
                        f'Index out of range, prediction index is {action_index}, but the'
                        f' UI element list only has {num_ui_elements} elements.'
                    )
                    exploration_step_data['summary'] = (
                        'The parameter index is out of range. Remember the index must be in'
                        ' the UI element list!'
                    )
                    round_history.append(exploration_step_data)
                    continue

                # Add mark to the target element.
                m3a_utils.add_ui_element_mark(
                    exploration_step_data['raw_screenshot'],
                    before_ui_elements[action_index],
                    action_index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )

            if converted_action.action_type == 'status':
                if converted_action.goal_status == 'infeasible':
                    print('Agent stopped since it thinks mission impossible.')
                exploration_step_data['summary'] = 'Agent thinks the request has been completed.'
                round_history.append(exploration_step_data)
                self.info['current_plan_idx'] += 1
                exploration_done = True

                break

            if converted_action.action_type == 'answer':
                print('Agent answered with: ' + converted_action.text)

            try:
                self.env.execute_action(converted_action)
                # exploration_step_data['summary'] = 'Reason: ' + reason + '. Action: ' + action
                # exploration_step_data['summary'] = 'Action: ' + action
            except Exception as e:  # pylint: disable=broad-exception-caught
                print('Failed to execute action.')
                print(str(e))
                exploration_step_data['summary'] = (
                    'Can not execute the action, make sure to select the action with'
                    ' the required parameters (if any) in the correct JSON format!'
                )
                continue


            time.sleep(self.wait_after_action_seconds)

            state = self.env.get_state(wait_to_stabilize=False)
            logical_screen_size = self.env.logical_screen_size
            orientation = self.env.orientation
            physical_frame_boundary = self.env.physical_frame_boundary
            after_ui_elements = state.ui_elements
            after_ui_elements_list = _generate_ui_elements_description_list(
                after_ui_elements, logical_screen_size
            )
            after_screenshot = state.pixels.copy()
            for index, ui_element in enumerate(after_ui_elements):
                if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                    m3a_utils.add_ui_element_mark(
                        after_screenshot,
                        ui_element,
                        index,
                        logical_screen_size,
                        physical_frame_boundary,
                        orientation,
                    )
            m3a_utils.add_screenshot_label(
                exploration_step_data['before_screenshot_with_som'], 'before'
            )
            m3a_utils.add_screenshot_label(after_screenshot, 'after')
            exploration_step_data['after_screenshot_with_som'] = after_screenshot.copy()
            print(step_data['current_plan']['description'])
            summary_prompt = _summarize_prompt(
                action=action,
                reason=reason,
                goal=step_data['current_plan']['description'],
                before_elements=before_ui_elements_list,
                after_elements=after_ui_elements_list,
            )
            summary, is_safe, raw_response = self.local_llm.predict_mm(
                summary_prompt,
                [
                    before_screenshot,
                    after_screenshot,
                ],
            )

            if is_safe == False:  # pylint: disable=singleton-comparison
                #  is_safe could be None
                summary = """Summary triggered LLM safety classifier."""

            if not raw_response:
                print(
                    'Error calling LLM in summarization phase. This should not happen: '
                    f'{summary}'
                )
                step_data['summary'] = (
                        'Some error occurred calling LLM during summarization phase: %s'
                        % summary
                )
                round_history.append(exploration_step_data)
                continue

            exploration_step_data['summary_prompt'] = summary_prompt
            exploration_step_data['summary'] = f'Action selected: {action}. {summary}'
            print('Summary: ' + summary)
            exploration_step_data['summary_raw_response'] = raw_response
            round_history.append(exploration_step_data)
        step_data['summary'] = 'Current Goal: ' + step_data['current_plan']['description'] + 'Current Action and Summary: ' + exploration_step_data['summary']

    # if (need_exploration is False) or ((need_exploration is True) and (exploration_done is True) ):
    # self.info['current_plan_idx'] += 1

    self.history.append(step_data)
    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )

