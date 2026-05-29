import os
import sys
import json
import time
import re
from task_explorer.utils.utils import (
    get_apk,
    APK,
    str_to_md5,
    openai_request,
    resize_pil_image,
    pil_to_webp_base64,
    load_object_from_disk,
    save_object_to_disk,
)
from task_explorer.utils.device import Device, UIElement, _generate_ui_elements_description_list
from task_explorer.MLLM_Agent.GUI_explorer import GUI_explorer, execute_adb_action
from PIL import Image
from task_explorer.utils.prompt_templates import TASK_GOAL_GENERATOR, SUBTASK_GOAL_GENERATOR, TASK_COMPLETION_CHECKER, STATE_EVALUATOR
from datetime import datetime
from glob import glob
from typing import List, Dict, Optional

def parse_task(text: str) -> list[str]:
    pattern = r"\d+\.+(.*)"
    matches = re.findall(pattern, text)
    for i, match in enumerate(matches):
        matches[i] = match.strip()
    return matches


def parse_subgoals(text: str) -> List[Dict[str, any]]:
    """
    解析Sub-goal格式的文本，提取结构化信息

    Returns:
        List[Dict]: [{'id': 1, 'anchor': '...', 'directive': '...', 'confidence': 0.9}, ...]
    """
    pattern = r'Sub-goal\s+(\d+):\s*\nAnchor:\s*(.+?)\nDirective:\s*(.+?)\nConfidence:\s*([\d.]+)'
    matches = re.findall(pattern, text, re.MULTILINE)

    return [
        {
            'id': int(match[0]),
            'anchor': match[1].strip(),
            'directive': match[2].strip(),
            'confidence': float(match[3])
        }
        for match in matches
    ]


def parse_task_completion_result(text: str) -> Optional[bool]:
    # 匹配 Result: 后面的 True/False (大小写不敏感)
    pattern = r'Result:\s*(True|False)'
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        result_str = match.group(1).lower()
        return result_str == 'true'

    return None

def parse_state_evaluation(text: str) -> tuple[str, str]:
    """提取状态评估的reasoning和result部分

    Returns:
        tuple: (reasoning, result) 其中result为CONTINUE/BACKTRACK/COMPLETED
    """
    # 提取reasoning部分（从Reasoning:到Result:之间的内容）
    reasoning_match = re.search(r'Reasoning:\s*(.*?)\s*Result:', text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # 提取result部分
    result_match = re.search(r'Result:\s*(CONTINUE|BACKTRACK|COMPLETED)', text)
    result = result_match.group(1) if result_match else "CONTINUE"

    return reasoning, result

def _call_llm_with_screenshot(
    screenshot: Image.Image,
    prompt: str,
    usage: dict[str, int],
) -> str:
    low_resolution = os.getenv("LOW_RESOLUTION", "False").lower() == "true"
    if low_resolution:
        screenshot = resize_pil_image(screenshot, 1000)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{pil_to_webp_base64(screenshot)}",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    rsp_txt = openai_request(
        messages=messages, timeout=300, usage=usage, max_tokens=8192
    )
    print(rsp_txt)
    return rsp_txt


def task_goal_generator(
    screenshot: Image.Image,
    package_name: str = None,
    app_name: str = None,
    activity_list: str = None,
    action_history: list = None,
    subgoal_history: list = None,
    element_list: str = None,
    user_task: str = None,
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
    state_summary: str = None,
):
    if action_history:
        if len(action_history) > 5:
            action_history = action_history[-5:]
    p = SUBTASK_GOAL_GENERATOR.format(
        package_name=package_name if package_name else "Not Available",
        app_name=app_name if app_name else "Not Available",
        activity_list=activity_list if activity_list else "Not Available",
        element_list=element_list if element_list else "Not Available",
        action_history=action_history if action_history else "The task has just begun",
        subgoal_history=subgoal_history if subgoal_history else "The task has just begun",
        user_task=user_task if user_task else "Not Available",
        state_summary=state_summary if state_summary else "Not Available",
    )
    return parse_subgoals(_call_llm_with_screenshot(screenshot, p, usage))

def task_completion_checker(
    screenshot: Image.Image,
    package_name: str = None,
    app_name: str = None,
    activity_list: str = None,
    action_history: list = None,
    subgoal_history: list = None,
    element_list: str = None,
    user_task: str = None,
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
):
    if action_history:
        if len(action_history) > 5:
            action_history = action_history[-5:]
    p = TASK_COMPLETION_CHECKER.format(
        package_name=package_name if package_name else "Not Available",
        app_name=app_name if app_name else "Not Available",
        activity_list=activity_list if activity_list else "Not Available",
        element_list=element_list if element_list else "Not Available",
        action_history=action_history if action_history else "The task has just begun",
        subgoal_history=subgoal_history if subgoal_history else "The task has just begun",
        user_task=user_task if user_task else "Not Available",
    )
    return parse_task_completion_result(_call_llm_with_screenshot(screenshot, p, usage))


def state_evaluator(
    screenshot: Image.Image,
    package_name: str = None,
    app_name: str = None,
    activity_list: str = None,
    action_history: list = None,
    subgoal_history: list = None,
    element_list: str = None,
    user_task: str = None,
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
):
    if action_history:
        if len(action_history) > 5:
            action_history = action_history[-5:]
    p = STATE_EVALUATOR.format(
        package_name=package_name if package_name else "Not Available",
        app_name=app_name if app_name else "Not Available",
        activity_list=activity_list if activity_list else "Not Available",
        element_list=element_list if element_list else "Not Available",
        action_history=action_history if action_history else "The task has just begun",
        subgoal_history=subgoal_history if subgoal_history else "The task has just begun",
        user_task=user_task if user_task else "Not Available",
    )
    return parse_state_evaluation(_call_llm_with_screenshot(screenshot, p, usage))

class TaskCompletedException(Exception):
    """任务完成异常"""
    pass

def is_task_explored(root_dir: str, task: str) -> bool:
    t = str_to_md5(task)[:16]
    res = glob(os.path.join(root_dir, "**", f"*{t}*"), recursive=True)
    return len(res) > 0


def explore_dfs(
    current_task: str,
    current_depth: int,
    exploration_output_root_dir: str,
    max_exploration_tasks: int,
    max_exploration_steps: int,
    apk_object: APK,
    device_controller: Device,
    agent: GUI_explorer,
    previous_actions: list,
    previous_subgoals: list,
    package_name: str,
    is_first_task: bool = False,
    max_exploration_depth: int = 3,
    user_task: str = None,

    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
):
    if (
        not is_first_task
    ):  # 如果不是第一个任务，那么需要根据之前的动作序列来复现到父节点的结束状态
        print(f"{'*' * current_depth}Restore To Parent State")
        device_controller.stop_all_apps()
        device_controller.launch_app(package_name, front=True)
        for action in previous_actions:
            converted_action, before_ui_elements, logical_screen_size = action
            time.sleep(3)
            device_controller.wait_to_stabilize()
            print(converted_action)
            execute_adb_action(
                converted_action,
                device_controller,
                before_ui_elements,
                logical_screen_size,
            )
    # 运行任务
    exploration_output_root_dir = os.path.abspath(exploration_output_root_dir)
    if is_task_explored(exploration_output_root_dir, current_task):
        print(f"{'*' * current_depth}Task {current_task} already explored. Skip.")
        return

    exploration_output_root_dir = os.path.join(
        exploration_output_root_dir, str_to_md5(current_task)
    )

    records = agent.run(
        current_task,
        max_rounds=max_exploration_steps,
        step_data_output_dir=exploration_output_root_dir,
    )
    if current_depth >= max_exploration_depth:  # 结束探索
        return
    # 生成用于恢复到current_task结束后的状态的动作序列
    restore_actions = []
    for step_data in records:
        if isinstance(step_data["converted_action"], str):
            continue
        action = (
            step_data["converted_action"],
            [UIElement(**ui_element) for ui_element in step_data["ui_elements"]],
            step_data["logical_screen_size"],
        )
        restore_actions.append(action)
    restore_actions = previous_actions + restore_actions
    if previous_subgoals == None:
        previous_subgoals = []
    previous_subgoals = previous_subgoals + [current_task]
    # 生成任务列表
    element_list = device_controller.wait_to_stabilize()
    screenshot = device_controller.get_screenshot()
    screen_size = device_controller.get_screen_size()
    elements_list = _generate_ui_elements_description_list(element_list, screen_size)
    activity = [act for act in apk_object.get_activities() if "sdk" not in act.lower()]
    activity_str = "\n".join(activity)
    app_name = apk_object.get_app_name()
    #检查用户任务是否完成
    state_reason, if_completed = state_evaluator(
        screenshot=screenshot,
        package_name=package_name,
        app_name=app_name,
        activity_list=activity_str,
        user_task=user_task,
        element_list=elements_list,
        usage=usage,
        action_history=restore_actions,
        subgoal_history=previous_subgoals
    )
    if if_completed == 'COMPLETED':
        print(f"{'*' * current_depth}Task completed! Raising exception to exit all recursions.")
        raise TaskCompletedException("User task completed successfully")
    elif if_completed == 'BACKTRACK':
        print(f"{'*' * current_depth}Task misdirected, back to upper depth.")
        return

    task_list = task_goal_generator(
        screenshot=screenshot,
        package_name=package_name,
        app_name=app_name,
        activity_list=activity_str,
        user_task=user_task,
        element_list=elements_list,
        usage=usage,
        action_history=restore_actions,
        subgoal_history=previous_subgoals,
        state_summary=state_reason
    )
    print(f"{'*' * current_depth}Generated {len(task_list)} sub-tasks: {task_list}, previous goals: {previous_subgoals}")
    if len(task_list) == 0:
        return
    for i in range(max_exploration_tasks):
        if i < len(task_list):
            task = task_list[i]
            print(
                f"{'*' * current_depth}Exploring sub-task {i+1}/{min(len(task_list),max_exploration_tasks)}: {task}"
            )
            if is_task_explored(exploration_output_root_dir, task['directive']):
                print(f"{'*' * current_depth}Task {task} already explored. Skip.")
                continue
            explore_dfs(
                current_task=task['directive'],
                current_depth=current_depth + 1,
                exploration_output_root_dir=exploration_output_root_dir,
                max_exploration_tasks=max_exploration_tasks,
                max_exploration_steps=max_exploration_steps,
                agent=agent,
                apk_object=apk_object,
                device_controller=device_controller,
                previous_actions=restore_actions,
                previous_subgoals=previous_subgoals,
                package_name=package_name,
                is_first_task=bool(i == 0),
                max_exploration_depth=max_exploration_depth,
                user_task=user_task,
                usage=usage,
            )

def _setup_exploration_env(
    package_name: str,
    root_dir: str,
    device_serial: str,
) -> tuple:
    """Initialize exploration environment: create dirs, fetch APK, parse app info, create device/agent.

    Returns:
        (root_dir, apk_object, app_info, device, agent)
    """
    root_dir = os.path.abspath(root_dir)
    root_dir = os.path.join(root_dir, package_name)
    os.makedirs(root_dir, exist_ok=True)
    apk_path = os.path.join(root_dir, f"{package_name}.apk")
    if os.path.exists(apk_path):
        os.remove(apk_path)
    print(f"Fetching the APK file of {package_name}.")
    res = get_apk(package_name, apk_path, device_serial)
    if res == "ERROR":
        print(f"Failed to get the APK file of {package_name}.")
        sys.exit(1)
    print("Analyzing the APK file.")
    apk_object = APK(apk_path)
    app_info = {
        "app_name": apk_object.get_app_name(),
        "app_version": apk_object.get_androidversion_code(),
        "app_version_name": apk_object.get_androidversion_name(),
        "app_pkg": apk_object.get_package(),
        "app_main_activity": apk_object.get_main_activity(),
    }
    with open(
        os.path.join(root_dir, "app_info.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(app_info, file, indent=2, ensure_ascii=False)
    device = Device(device_serial)
    agent = GUI_explorer(device_serial=device_serial)
    return root_dir, apk_object, app_info, device, agent


def manually_exploration(
    package_name: str,
    exploration_output_root_dir: str = "./output",
    device_serial: str = None,
    user_task: str = None,
    task_dir: str = None,
):
    root_dir, apk_object, app_info, device, agent = _setup_exploration_env(
        package_name, exploration_output_root_dir, device_serial,
    )
    agent.manual(task_goal=user_task, max_rounds=60, step_data_output_dir=task_dir)

def auto_exploration(
    package_name: str,
    exploration_output_root_dir: str = "./output",
    device_serial: str = None,
    max_exploration_tasks: int = 10,
    max_exploration_steps: int = 30,
    max_exploration_depth: int = 5,  # 从首页开始的任务扩展深度（最多扩展 max_exploration_depth-1 代）
    user_task: str = None,
    task_dir: str = None,
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> dict:
    """Run auto exploration and return a result summary.

    Returns:
        dict with keys:
            exploration_completed (bool): True if TaskCompletedException fired.
            num_subtasks_generated (int): Number of subtasks generated.
            num_subtasks_explored (int): Number of subtasks actually explored.
            usage (dict): Token usage dict (prompt_tokens, completion_tokens).
    """
    exploration_output_root_dir, apk_object, app_info, device, agent = _setup_exploration_env(
        package_name, exploration_output_root_dir, device_serial,
    )
    task_list = []
    print("Generating exploration sub-goals.")
    exploration_output_root_dir = os.path.join(
        exploration_output_root_dir, task_dir
    )
    initial_elements = device.wait_to_stabilize()
    initial_screen_size = device.get_screen_size()
    initial_elements_list = _generate_ui_elements_description_list(initial_elements, initial_screen_size)
    screenshot = device.get_screenshot()
    activity = [act for act in apk_object.get_activities() if "sdk" not in act.lower()]
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
                current_task=task['directive'],
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
        print("Auto exploration finished - all tasks explored but user task not completed.")
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


import argparse

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

    # 设置ADB服务器地址为Windows主机
    # 获取Windows主机IP
    windows_ip = os.popen("ip route show | grep -i default | awk '{ print $3}'").read().strip()

    # 或者直接设置为localhost（如果端口转发配置正确）
    os.environ['ANDROID_ADB_SERVER_HOST'] = windows_ip
    os.environ['ANDROID_ADB_SERVER_PORT'] = '5037'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-package_name", help="The package name of the APK, e.g. com.android.settings", default="com.google.android.contacts"
    )
    parser.add_argument(
        "-device_serial", help="The serial number of the device, see `adb devices`", default='emulator-5554'
    )
    parser.add_argument(
        "-output_dir",
        help="The directory to save the task file",
        default="./exploration_output",
    )
    parser.add_argument(
        "-max_branching_factor",
        help="The max number of tasks to explore at each node",
        default=3,
    )
    parser.add_argument(
        "-max_exploration_steps",
        help="The max number of steps to explore for each task",
        default=10,
    )
    parser.add_argument(
        "-max_exploration_depth",
        help="The max depth of exploration",
        default=10,
    )
    parser.add_argument(
        "-user_task", help="The task to explore for", default="Create a new contact for Fatima Wang. Their number is +12783095137."
    )
    parser.add_argument(
        "-task_dir", help="The directory to save", default="ContactsAddContact")
    args = parser.parse_args()
    
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    
    print(args)
    print("Starting auto exploration.")
    auto_exploration(
        package_name=args.package_name,
        exploration_output_root_dir=args.output_dir,
        device_serial=args.device_serial,
        max_exploration_tasks=int(args.max_branching_factor),
        max_exploration_steps=int(args.max_exploration_steps),
        max_exploration_depth=int(args.max_exploration_depth),
        user_task=args.user_task,
        task_dir=args.task_dir,
        usage=usage,
    )
    print(f"Task goal generator token usage: {usage}")
