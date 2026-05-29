from platform import node
import random as random_module
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict, Set, Optional, Tuple
import math
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import re
# from KG_to_files import AppGraph, ElementNode, ActionNode, PageNode
# from reward_model import HfCausalModel as rm
import argparse
import sys
from sentence_transformers import SentenceTransformer
import torch
import time
import numpy as np
import torch.nn.functional as F
import os
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from rm_regression import *


def construct_score_prompt(instruction: str, page_caption: str, history: List[str], action: str) -> str:
    """
    构造 Prompt 部分。
    """
    # 将历史动作列表转换为字符串路径
    if not history:
        current_state_str = "Start of Task"
    else:
        current_state_str = " -> ".join(history)

    # 你的 Prompt 模板
    prompt_content = (
        f"Task: {instruction}\n\n"
        f"You are at: {page_caption}\n\n"
        f"Executed path: {current_state_str}\n\n"
        f"Proposed action:"
    )
    # 注意：这里把 action 拼进去了，说明训练的是完整的 (Input + Action) -> Score
    prompt_content = prompt_content + f"\n<|assistant|>: {action}<end_of_step>"
    
    full_prompt = f"<|user|>:\n{prompt_content}"
    return full_prompt

# ==========================================
# 2. 辅助函数：将 Transition 转为字符串
# ==========================================
def get_transition_desc(transition) -> str:
    """
    从 GUITransition 对象中提取可读的动作描述。
    你需要根据你的 GUITransition 类定义来修改这里。
    """
    # 假设 transition 有 action_content 属性，或者直接转 str
    if hasattr(transition, 'action_content'):
        return str(transition.action_content)
    # 或者如果不确定，直接转 str
    return str(transition)

# ==========================================
# 3. 核心：遍历树并生成符合格式的 JSONL
# ==========================================
def export_training_dataset(
        root_node: 'GUIMCTSNode', 
        instruction: str, 
        output_file: str = "gui_rm_train.jsonl"
):
    """
    Args:
        root_node: 已经计算完 q_target 的树根节点
        instruction:这一整棵树对应的总任务指令 (Task)
        output_file: 保存路径
    """
    
    dataset = []
    
    # 使用 BFS 遍历树
    queue = [root_node]
    
    while queue:
        current_node = queue.pop(0)
        
        # 将子节点加入队列以便后续遍历
        queue.extend(current_node.children)
        
        # === 准备构造 prompt 所需的公共参数 ===
        # 1. page_caption (对应 page_name 或其他描述)
        # current_page_caption = current_node.page_name
        
        # 2. history (对应 node.path)
        # 将路径里的 transition 对象转成字符串列表
        history_strs = extract_action_from_transitions(current_node.path)
        
        # === 遍历所有子节点 (代表可能的 Actions) ===
        for child in current_node.children:
            if not child.path:
                continue
            
            # 3. action (对应 child.path 的最后一步)
            last_transition = str(child.path[-1]) # 这是一个 GUITransition 对象
            action_str = re.search(r'\[action:\s*(.*?)\]', last_transition).group(1).strip()
            current_page_caption = re.search(r'<From:\s*(.*?)\s*--', last_transition, re.DOTALL).group(1).strip()

            # === 调用你的 Prompt 函数 ===
            full_input_text = construct_score_prompt(
                instruction=instruction,
                page_caption=current_page_caption,
                history=history_strs,
                action=action_str
            )
            
            # === 获取 Label (Q Target) ===
            # 确保这里用的是 child.q_target (动作执行后的长期价值)
            target_score = child.q_target
            
            # === 构造最终样本 ===
            sample = {
                "text": full_input_text,  # 模型输入
                "label": target_score,    # 回归目标 (可能 > 1.0)
                
                # 保留一些元数据方便 Debug，训练时不读这些即可
                "metadata": {
                    "page_id": current_node.page_id,
                    "action_raw": action_str,
                    "is_gold": child.if_gold if hasattr(child, 'if_gold') else False
                }
            }
            dataset.append(sample)

    # === 保存到 JSONL ===
    # print(f"正在保存 {len(dataset)} 条数据到 {output_file} ...")
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     for entry in dataset:
    #         f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    if dataset:
        # print(f"Appending {len(dataset)} samples to {output_file}")
        with open(output_file, 'a', encoding='utf-8') as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print("保存完成。")


@dataclass
class PageNode:
    page_id: str
    description: str
    task_steps: Dict[str, List[int]]  # {"task_name": [0, 1, 2]}
    element_ids: List[str]  # HAS_ELEMENT连接的所有element_id
    action_ids: List[str]
    function_summary: str


# Element节点结构
@dataclass
class ElementNode:
    element_id: str
    name: str
    reasoning: str
    task_steps: Dict[str, List[int]]  # {"task_name": [0, 1, 2]}
    leads_to_page_id: List[str]  # LEADS_TO的目标page_id
    function_summary: str

@dataclass
class ActionNode:
    action_id: str
    name: str
    function: str
    element_sequence: List[dict]
    leads_to_page_id: List[str]

# 整个应用的图结构
@dataclass
class AppGraph:
    app_name: str
    pages: Dict[str, PageNode]  # page_id -> PageNode
    elements: Dict[str, ElementNode]  # element_id -> ElementNode
    actions: Dict[str, ActionNode]

# ==========================================
# 1. 必须复制原训练代码中的模型定义
# ==========================================

class ValueHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        summary_dropout_prob = getattr(config, "summary_dropout_prob", 0.1)
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        hidden_size = getattr(config, "hidden_size", 4096)

        # 你的训练代码中可能是 "single"
        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        # 强制转换类型以保持稳定
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        output = self.summary(output)
        return output


class RewardModelWithValueHead(nn.Module):
    def __init__(self, pretrained_model, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        self.v_head = ValueHead(self.config, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 确保 output_hidden_states=True
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        last_hidden_state = outputs.hidden_states[-1]
        value = self.v_head(last_hidden_state).squeeze(-1)
        return value


# ==========================================
# 2. 加载与推理逻辑
# ==========================================


def load_model_and_tokenizer(model_path):
    print(f"Loading Base Model from {model_path}...")

    # 1. 加载 Config (防止之前的 model_type 报错)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(config, "model_type") or config.model_type is None:
        config.model_type = "qwen2"  # 这里假设你是Qwen，如果是Llama改成llama

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. 包装
    print("Wrapping model with Value Head...")
    model = RewardModelWithValueHead(base_model)

    # 3. 加载 Value Head 权重
    print("Loading Value Head weights...")
    state_dict = load_file(f"{model_path}/model.safetensors")
    v_head_state_dict = {k: v for k, v in state_dict.items() if "v_head" in k}

    if len(v_head_state_dict) > 0:
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded {len(v_head_state_dict)} keys for Value Head.")
    else:
        # 尝试非严格加载
        model.load_state_dict(state_dict, strict=False)

    # ========================================================
    # === 关键修复：把 v_head 也移动到 GPU 上，并且对齐精度 ===
    # ========================================================
    target_device = base_model.device  # 获取基座模型所在的设备 (通常是 cuda:0)
    target_dtype = base_model.dtype  # 获取基座模型的精度 (通常是 bfloat16)

    # 将 v_head 移动到该设备，并转换精度
    model.v_head.to(device=target_device, dtype=target_dtype)
    print(f"Moved v_head to {target_device} with dtype {target_dtype}")
    # ========================================================

    model.eval()
    return model, tokenizer

# def load_model_and_tokenizer(model_path):
#     print(f"Loading Base Model from {model_path}...")
#     config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#     if not hasattr(config, "model_type") or config.model_type is None:
#         config.model_type = "qwen2"

#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         config=config,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

#     print("Wrapping model with Value Head...")
#     model = RewardModelWithValueHead(base_model)
    
#     # === 增强的加载逻辑 ===
#     # 迭代训练后，checkpoints 可能是标准 HF 格式 (safetensors 里直接有 v_head.summary.weight)
#     # 也可能是你初始的那种分离格式。我们需要兼容两种情况。
    
#     try:
#         # 情况 1: 标准 HF Trainer 保存的 (迭代后的 checkpoints)
#         # 这种情况下，model.from_pretrained 其实已经加载大部分权重了，
#         # 但由于 RewardModelWithValueHead 是 wrapper，可能没加载 v_head
#         # 我们尝试直接 load_state_dict
#         if os.path.exists(os.path.join(model_path, "model.safetensors")):
#              state_dict = load_file(os.path.join(model_path, "model.safetensors"))
#              missing, unexpected = model.load_state_dict(state_dict, strict=False)
#              print(f"Loaded HFCheckpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
             
#     except Exception as e:
#         print(f"Standard load failed, trying custom v_head load: {e}")
#         # 情况 2: 你初始的特殊模型 (v_head 藏在某个 key 里)
#         state_dict = load_file(f"{model_path}/model.safetensors")
#         model.load_state_dict(state_dict, strict=False)

#     # 确保 v_head 在 GPU
#     target_device = base_model.device
#     target_dtype = base_model.dtype
#     model.v_head.to(device=target_device, dtype=target_dtype)
    
#     model.eval()
#     return model, tokenizer

def load_untrained_base_model(model_name_or_path="Qwen/Qwen2.5-3B-Instruct"):
    """
    加载未经 Reward Model 训练的原始基座模型，并包装上随机初始化的 Value Head。
    用于作为 Baseline 进行对比。
    """
    print(f"Loading Untrained Base Model from {model_name_or_path}...")

    # 1. 加载配置和 Tokenizer
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # 2. 加载原始的 Causal LM (不带 v_head)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,  # 保持和训练时一致的精度
        device_map="auto",
        trust_remote_code=True
    )

    # 3. 包装模型 (这一步会自动随机初始化 v_head)
    print("Wrapping base model with Random Value Head...")
    model = RewardModelWithValueHead(base_model)

    # !!! 关键点 !!!
    # RewardModelWithValueHead 初始化时，v_head 是 Linear 层，
    # 默认可能是 float32 并且在 CPU 上。我们需要把它移到和 base_model 一样的设备和精度上。

    target_device = base_model.device
    target_dtype = base_model.dtype

    model.v_head.to(device=target_device, dtype=target_dtype)
    print(f"Params of v_head initialized randomly. Moved to {target_device} as {target_dtype}.")

    # 开启评估模式
    model.eval()

    return model, tokenizer


def get_score(model, tokenizer, prompt, response):
    # 1. 拼接文本 (必须和训练时的 preprocess 逻辑一致)
    # 注意：你的训练代码里是 pos_ids = source_ids + pos_ids
    # 也就是 Prompt + Response
    input_text = prompt + response

    # 2. 编码
    inputs = tokenizer(input_text, return_tensors="pt").to(model.pretrained_model.device)

    # 3. 推理
    with torch.no_grad():
        # 输出 value shape: [1, Seq_Len]
        value_seq = model(**inputs)

    # 4. 提取最后一个有效 Token 的分数值
    # 对于 batch size = 1，最后一个 token 就是序列末尾
    # (如果你的 response 后面有 padding，需要用 attention_mask判断，但 inference 通常没有 padding)
    score = value_seq[0, -1].item()

    return score


def get_batch_scores(model, tokenizer, prompts, max_length=2048):
    """
    批量计算 Reward Score
    """
    model.eval()
    device = model.pretrained_model.device  # 自动获取模型所在设备

    # 结果列表
    all_scores = []

    # 0. 检查 Padding Token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # 这一点很重要，RM通常使用右侧填充，padding_side不影响计算结果，但影响索引位置
        tokenizer.padding_side = "right"

    # 2. 按 Batch 处理
    total_len = len(prompts)

    # 使用 torch.no_grad 节省显存
    with torch.no_grad():

        # 3. Tokenize (Padding 到 batch 内最长)
        inputs = tokenizer(
            prompts,
            padding=True,  # 必须 padding 才能组成 batch tensor
            truncation=True,  # 防止显存爆炸
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        # 4. 前向传播
        # value_seq shape: [Batch_Size, Max_Seq_Len]
        # 例如: [[0.1, 0.5, 0.9, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0, 0.0]] (0.0是padding位置的无效输出)
        value_seq = model(**inputs)

        # 5. 提取最后一个有效 Token 的分数值 (关键步骤)

        # 5.1 获取 Attention Mask (1为有效，0为padding)
        # shape: [Batch_Size, Max_Seq_Len]
        masks = inputs.attention_mask

        # 5.2 计算每个样本的实际长度
        # 假设是右侧填充(Right Padding)，长度就是 sum(mask)
        # 也就是最后一个有效 token 的索引是 sum(mask) - 1
        # shape: [Batch_Size]
        end_indices = masks.sum(dim=1) - 1

        # 5.3 使用 gather 提取分数
        # 我们需要把 end_indices 变成 [Batch_Size, 1] 才能给 gather 用
        # value_seq: [B, L]
        # gather_indices: [B, 1] -> 指示每一行取第几列的数据
        scores = torch.gather(value_seq, 1, end_indices.unsqueeze(1))

        # scores shape 现在是 [Batch_Size, 1]，转回 list float
        batch_scores = scores.squeeze(1).float().cpu().numpy().tolist()
        all_scores.extend(batch_scores)

        # 打印进度 (可选)
        # print(f"Processed {min(i + batch_size, total_len)}/{total_len} samples")

    return all_scores


TASK_PLANNER = """Given the user instruction: {question}

You are at: 
{page}

Executed path (history):
{path}

Based on the task goal, current page and execution history, generate the NEXT operation to continue the task.

Please provide your response as a brief operation summary:
"""

EVALUATION_PROMPT = """Task: {question}

You are at: {page}

Executed path: {current_state}

Proposed action: {action}

Question: Evaluate if the taking the proposed action in the current state is logically heading in the correct direction for completing the task. Provide an answer of helpful or unhelpful.
Answer: This action is"""

def construct_score_prompt(instruction: str, page_caption: str, history: List[str], action) -> str:
    """
    构造 Prompt 部分。
    保持 Prompt 不变，动作的具体内容留给 pos/neg 填充。
    """

    # 将历史动作列表转换为字符串路径
    if not history:
        current_state_str = "Start of Task"
    else:
        current_state_str = " -> ".join(history)

    # 你的 Prompt 模板
    prompt_content = (
        f"Task: {instruction}\n\n"
        f"You are at: {page_caption}\n\n"
        f"Executed path: {current_state_str}\n\n"
        f"Proposed action:"
    )
    prompt_content = prompt_content + f"\n<|assistant|>: {action}<end_of_step>"
    # 加上 <|user|> 前缀
    full_prompt = f"<|user|>:\n{prompt_content}"
    return full_prompt

# def construct_score_prompt(instruction: str, page_caption: str, history: List[str], action) -> str:
#     """
#     构造基于轨迹的 Prompt (Partial Solution)。
#     格式：
#     <|user|>: Task: ... Current Screen: ...
#     <|assistant|>: Let's think step by step. # action 1: ... <end_of_step> # action 2: ... <end_of_step>
#     """

#     # 1. 构造 User 部分
#     user_part = (
#         f"<|user|>: Task: {instruction}\n"
#         f"Screen Context Before Taking The Last Action: {page_caption}"
#     )

#     # 2. 构造 Assistant 的历史轨迹部分
#     assistant_prefix = "\n<|assistant|>: Let's think step by step."

#     history_traj = ""
#     # 遍历历史动作，添加序号
#     if history:
#         for idx, action in enumerate(history, 1):  # idx 从 1 开始
#             clean_action = action.strip()
#             # 格式: [空格] # action [i]: [内容] <end_of_step>
#             history_traj += f" # action {idx}: {clean_action} <end_of_step>"
#     action = f" # action {len(history) + 1}: {action} <end_of_step>"
#     # 3. 拼接
#     full_prompt = user_part + assistant_prefix + history_traj + action

#     return full_prompt


class ModuleRedirectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 将__main__模块的类重定向到graph_structures模块
        if module == '__main__':
            # 根据您的实际类名调整这个列表
            class_mapping = {
                'AppGraph': AppGraph,
                'ElementNode': ElementNode,
                'ActionNode': ActionNode,
                'PageNode': PageNode,
                # 添加其他可能的类名
            }
            if name in class_mapping:
                return class_mapping[name]

        return super().find_class(module, name)

TASK_APP_MAPPING = {
    # Audio Recorder
    'AudioRecorderRecordAudio': ('com.dimowner.audiorecorder', 'audio recorder'),
    'AudioRecorderRecordAudioWithFileName': ('com.dimowner.audiorecorder', 'audio recorder'),

    # Browser (Chrome)
    'BrowserDraw': ('com.google.android.documentsui', 'files'),
    'BrowserMaze': ('com.google.android.documentsui', 'files'),
    'BrowserMultiply': ('com.google.android.documentsui', 'files'),

    # Calendar (Simple Calendar Pro)
    'SimpleCalendarAddOneEvent': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarAddOneEventInTwoWeeks': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarAddOneEventRelativeDay': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarAddOneEventTomorrow': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarAddRepeatingEvent': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarDeleteEvents': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarDeleteEventsOnRelativeDay': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarDeleteOneEvent': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarEventsInNextWeek': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarEventsOnDate': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarNextEvent': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),

    # Camera
    'CameraTakePhoto': ('com.android.camera2', 'camera'),
    'CameraTakeVideo': ('com.android.camera2', 'camera'),

    # Clock
    'ClockStopWatchPausedVerify': ('com.google.android.deskclock', 'clock'),
    'ClockStopWatchRunning': ('com.google.android.deskclock', 'clock'),
    'ClockTimerEntry': ('com.google.android.deskclock', 'clock'),

    # Contacts
    'ContactsAddContact': ('com.google.android.contacts', 'contacts'),
    'ContactsNewContactDraft': ('com.google.android.contacts', 'contacts'),

    # Pro Expense
    'ExpenseAddMultiple': ('com.arduia.expense', 'pro expense'),
    'ExpenseAddMultipleFromGallery': ('com.arduia.expense', 'pro expense'),
    'ExpenseAddMultipleFromMarkor': ('com.arduia.expense', 'pro expense'),
    'ExpenseAddSingle': ('com.arduia.expense', 'pro expense'),
    'ExpenseDeleteDuplicates': ('com.arduia.expense', 'pro expense'),
    'ExpenseDeleteDuplicates2': ('com.arduia.expense', 'pro expense'),
    'ExpenseDeleteMultiple': ('com.arduia.expense', 'pro expense'),
    'ExpenseDeleteMultiple2': ('com.arduia.expense', 'pro expense'),
    'ExpenseDeleteSingle': ('com.arduia.expense', 'pro expense'),

    # 'ExpenseAddMultiple': ('com.arduia.expense', 'settings'),
    # 'ExpenseAddMultipleFromGallery': ('com.arduia.expense', 'settings'),
    # 'ExpenseAddMultipleFromMarkor': ('com.arduia.expense', 'settings'),
    # 'ExpenseAddSingle': ('com.arduia.expense', 'settings'),
    # 'ExpenseDeleteDuplicates': ('com.arduia.expense', 'settings'),
    # 'ExpenseDeleteDuplicates2': ('com.arduia.expense', 'settings'),
    # 'ExpenseDeleteMultiple': ('com.arduia.expense', 'settings'),
    # 'ExpenseDeleteMultiple2': ('com.arduia.expense', 'settings'),
    # 'ExpenseDeleteSingle': ('com.arduia.expense', 'settings'),

    # Files
    'FilesDeleteFile': ('com.google.android.documentsui', 'files'),
    'FilesMoveFile': ('com.google.android.documentsui', 'files'),

    # Markor
    'MarkorAddNoteHeader': ('net.gsantner.markor', 'markor'),
    'MarkorChangeNoteContent': ('net.gsantner.markor', 'markor'),
    'MarkorCreateFolder': ('net.gsantner.markor', 'markor'),
    'MarkorCreateNote': ('net.gsantner.markor', 'markor'),
    'MarkorCreateNoteFromClipboard': ('net.gsantner.markor', 'markor'),
    'MarkorDeleteAllNotes': ('net.gsantner.markor', 'markor'),
    'MarkorDeleteNewestNote': ('net.gsantner.markor', 'markor'),
    'MarkorDeleteNote': ('net.gsantner.markor', 'markor'),
    'MarkorEditNote': ('net.gsantner.markor', 'markor'),
    'MarkorMergeNotes': ('net.gsantner.markor', 'markor'),
    'MarkorMoveNote': ('net.gsantner.markor', 'markor'),
    'MarkorTranscribeReceipt': ('net.gsantner.markor', 'markor'),
    'MarkorTranscribeVideo': ('net.gsantner.markor', 'markor'),

    # Markor + SMS composite
    'MarkorCreateNoteAndSms': ('net.gsantner.markor', 'markor'),  # 复合任务，先用markor

    # information retrieval in joplin
    "NotesIsTodo" : ('net.cozic.joplin', 'joplin'),
    "NotesMeetingAttendeeCount": ('net.cozic.joplin', 'joplin'),
    "NotesRecipeIngredientCount": ('net.cozic.joplin', 'joplin'),
    "NotesTodoItemCount": ('net.cozic.joplin', 'joplin'),

    # OsmAnd
    'OsmAndFavorite': ('net.osmand', 'osmand'),
    'OsmAndMarker': ('net.osmand', 'osmand'),
    'OsmAndTrack': ('net.osmand', 'osmand'),

    # Recipe (Broccoli)
    'RecipeAddMultipleRecipes': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeAddMultipleRecipesFromImage': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeAddMultipleRecipesFromMarkor': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeAddMultipleRecipesFromMarkor2': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeAddSingleRecipe': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteDuplicateRecipes': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteDuplicateRecipes2': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteDuplicateRecipes3': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteMultipleRecipes': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteMultipleRecipesWithConstraint': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteMultipleRecipesWithNoise': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteSingleRecipe': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteSingleWithRecipeWithNoise': ('com.flauschcode.broccoli', 'broccoli'),

    # Retro Music
    'RetroCreatePlaylist': ('code.name.monkey.retromusic', 'retro music'),
    'RetroPlayingQueue': ('code.name.monkey.retromusic', 'retro music'),
    'RetroPlaylistDuration': ('code.name.monkey.retromusic', 'retro music'),
    'RetroSavePlaylist': ('code.name.monkey.retromusic', 'retro music'),

    # Simple Draw Pro
    'SimpleDrawProCreateDrawing': ('com.simplemobiletools.draw.pro', 'simple draw pro'),

    # Simple Gallery Pro
    'SaveCopyOfReceiptTaskEval': ('com.simplemobiletools.gallery.pro', 'simple gallery pro'),

    # SMS (Simple SMS Messenger)
    'SimpleSmsReply': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),
    'SimpleSmsReplyMostRecent': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),
    'SimpleSmsResend': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),
    'SimpleSmsSend': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),
    'SimpleSmsSendClipboardContent': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),
    'SimpleSmsSendReceivedAddress': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),

    #sport tracker
    "SportsTrackerActivitiesCountForWeek": ('de.dennisguse.opentracks', 'open tracks sports tracker'),
    "SportsTrackerActivitiesOnDate": ('de.dennisguse.opentracks', 'open tracks sports tracker'),
    "SportsTrackerActivityDuration": ('de.dennisguse.opentracks', 'open tracks sports tracker'),
    "SportsTrackerLongestDistanceActivity": ('de.dennisguse.opentracks', 'open tracks sports tracker'),
    "SportsTrackerTotalDistanceForCategoryOverInterval": ('de.dennisguse.opentracks', 'open tracks sports tracker'),
    "SportsTrackerTotalDurationForCategoryThisWeek": ('de.dennisguse.opentracks', 'open tracks sports tracker'),

    # System tasks (需要Settings应用)
    'OpenAppTaskEval': (None, None),  # 这个任务是打开其他应用，不需要settings
    'SystemBluetoothTurnOff': ('com.android.settings', 'settings'),
    'SystemBluetoothTurnOffVerify': ('com.android.settings', 'settings'),
    'SystemBluetoothTurnOn': ('com.android.settings', 'settings'),
    'SystemBluetoothTurnOnVerify': ('com.android.settings', 'settings'),
    'SystemBrightnessMax': ('com.android.settings', 'settings'),
    'SystemBrightnessMaxVerify': ('com.android.settings', 'settings'),
    'SystemBrightnessMin': ('com.android.settings', 'settings'),
    'SystemBrightnessMinVerify': ('com.android.settings', 'settings'),
    'SystemCopyToClipboard': (None, None),  # 剪贴板操作不需要特定应用
    'SystemWifiTurnOff': ('com.android.settings', 'settings'),
    'SystemWifiTurnOffVerify': ('com.android.settings', 'settings'),
    'SystemWifiTurnOn': ('com.android.settings', 'settings'),
    'SystemWifiTurnOnVerify': ('com.android.settings', 'settings'),

    # Task anwser
    "TasksCompletedTasksForDate": ('org.tasks', 'tasks'),
    "TasksDueNextWeek": ('org.tasks', 'tasks'),
    "TasksDueOnDate": ('org.tasks', 'tasks'),
    "TasksHighPriorityTasksDueOnDate": ('org.tasks', 'tasks'),
    "TasksIncompleteTasksOnDate": ('org.tasks', 'tasks'),

    # System composite tasks
    'TurnOffWifiAndTurnOnBluetooth': ('com.android.settings', 'settings'),
    'TurnOnWifiAndOpenApp': ('com.android.settings', 'settings'),

    # VLC
    'VlcCreatePlaylist': ('org.videolan.vlc', 'vlc'),
    'VlcCreateTwoPlaylists': ('org.videolan.vlc', 'vlc'),
}

PACKAGE_APP_MAPPING = {
    'code.name.monkey.retromusic': 'retro-music',
    'com.android.camera2': 'camera',
    'com.android.settings': 'settings',
    'com.arduia.expense': 'pro-expense',
    'com.dimowner.audiorecorder': 'audio-recorder',
    'com.flauschcode.broccoli': 'broccoli',
    'com.google.android.contacts': 'contacts',
    'com.google.android.deskclock': 'clock',
    'com.google.android.documentsui': 'files',
    'com.simplemobiletools.calendar.pro': 'simple-calendar-pro',
    'com.simplemobiletools.draw.pro': 'simple-draw-pro',
    'com.simplemobiletools.gallery.pro': 'simple-gallery-pro',
    'com.simplemobiletools.smsmessenger': 'simple-sms-messenger',
    'de.dennisguse.opentracks': 'open-tracks-sports-tracker',
    'net.cozic.joplin': 'joplin',
    'net.gsantner.markor': 'markor',
    'net.osmand': 'osmand',
    'org.tasks': 'tasks',
    'org.videolan.vlc': 'vlc',
}

def get_app_info(task_name: str):
    """获取任务对应的应用信息"""
    return TASK_APP_MAPPING.get(task_name, (None, None))


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_success_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "1", "success", "successful"}
    return False


def _normalize_app_key(app_name: Optional[str]) -> Optional[str]:
    if not app_name:
        return None
    return re.sub(r'(?<=\S) (?=\S)', '-', app_name.strip())


def _resolve_app_key(data_item: Dict[str, Any], task_name: str) -> str:
    package = data_item.get("app") or data_item.get("package")
    if package in PACKAGE_APP_MAPPING:
        return PACKAGE_APP_MAPPING[package]

    _, mapped_app = get_app_info(task_name)
    app_key = _normalize_app_key(mapped_app)
    if app_key:
        return app_key

    raise KeyError(
        f"Cannot resolve app graph for task={task_name!r}; "
        f"package/app field={package!r}"
    )


def _extract_start_page_id(data_item: Dict[str, Any]) -> str:
    try:
        start_page = data_item["path"][0][0][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Invalid or empty gold path for task={data_item.get('task')!r}") from exc

    match = re.search(r'_\$([^$]+)\$', start_page)
    if not match:
        raise ValueError(
            f"Cannot parse start page id for task={data_item.get('task')!r}: {start_page!r}"
        )
    return match.group(1)


def find_all_task_folders(root_path):
    """
    遍历根文件夹，找出所有任务文件夹

    Args:
        root_path: 根文件夹路径

    Returns:
        list: 任务文件夹路径列表
    """
    root_path = Path(root_path)
    task_folders = []

    if not root_path.exists() or not root_path.is_dir():
        print(f"路径不存在或不是文件夹: {root_path}")
        return task_folders

    # 遍历根目录下的所有子文件夹
    for item in root_path.iterdir():
        if item.is_dir():
            task_folders.append(item)
            print(f"发现任务文件夹: {item.name}")

    # 按名称排序
    task_folders.sort(key=lambda x: x.name)

    return task_folders


@dataclass
class GUITransition:
    """GUI操作转换"""
    from_page: str  # 起始页面
    action_type: str  # "element" or "action"
    action_id: str  # element_id 或 action_id
    action_name: str  # 可读名称
    to_page: str  # 目标页面
    to_page_id: str

    def __repr__(self):
        return f"<From:{self.from_page} --[action:{self.action_name}]--> To:{self.to_page}>"


class GUIMCTSNode:
    """简化的MCTS节点，专为GUI场景设计"""

    def __init__(
            self,
            page_id: str,
            page_name: str,
            path: List[GUITransition] = None,
            parent: 'GUIMCTSNode' = None,
            action_list = [],
    ):
        """
        Args:
            page_id: 当前页面ID
            page_name: 页面名称（用于显示）
            path: 从根节点到此的完整路径
            parent: 父节点
        """
        # === 状态信息 ===
        self.page_id = page_id
        self.page_name = page_name
        self.action_list = action_list or []
        # === 路径信息 ===
        self.path = path or []  # 完整的操作序列

        # === MCTS统计 ===
        self.visits = 1
        # self.total_reward = 0.0
        self.value = 0.0  # 由evaluate()设置
        self.policy_prior = 0.0
        # === 树结构 ===
        self.parent = parent
        self.children: List['GUIMCTSNode'] = []

        # === 扩展状态 ===
        self.is_fully_expanded = False
        self.expanded_actions: Set[str] = set()  # 已扩展的action_id
        self.if_end = False
        self.if_gold = False
        self.task_terminal = False
        self.gold_match_len = 0
        # === 缓存 ===
        # 格式: {page_id: {'elements': [...], 'actions': [...]}}
        self.cached_page_actions: Dict = {}
        self.q_target = 0.0
        self.r = 0.0

    def add_child(
            self,
            transition: GUITransition,
            next_page_name: str
    ) -> 'GUIMCTSNode':
        """
        创建子节点

        Args:
            transition: GUI操作转换
            next_page_name: 目标页面名称

        Returns:
            新创建的子节点
        """
        # 构造新路径
        new_path = self.path + [transition]

        # 创建子节点
        child = GUIMCTSNode(
            page_id=transition.to_page_id,
            page_name=next_page_name,
            path=new_path,
            parent=self
        )

        # 继承缓存
        child.cached_page_actions = self.cached_page_actions.copy()

        # 添加到children
        self.children.append(child)

        # 标记action已扩展
        self.expanded_actions.add(transition.action_id)

        return child

    def cache_page_actions(self, page_id: str, elements: List, actions: List):
        """缓存页面的可用操作"""
        self.cached_page_actions[page_id] = {
            'elements': elements,
            'actions': actions
        }

    def get_cached_actions(self, page_id: str) -> Optional[Dict]:
        """获取缓存的页面操作"""
        return self.cached_page_actions.get(page_id)

    def get_uct_value(self, exploration_constant: float = 1.4) -> float:
        """计算UCT值（用于Selection）"""
        if self.visits == 0:
            return float('inf')

        if self.parent is None:
            return self.value / self.visits

        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        # print(  f"q_value: {self.value}, exploration: {self.visits} ")
        return exploitation + exploration

    def get_puct_value(self, exploration_constant: float = 1.4) -> float:
        """计算UCT值（用于Selection）"""
        if self.visits == 0:
            return float('inf')

        if self.parent is None:
            return self.value / self.visits

        exploitation = self.value / self.visits
        exploration = exploration_constant * self.policy_prior * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def is_terminal(self, max_depth: int) -> bool:
        """
        判断是否为终止节点

        Args:
            max_depth: 最大深度
            visited_pages: 路径中已访问的页面（检测循环）
        """
        # 深度限制
        if len(self.path) >= max_depth:
            return True

        # 完全扩展且无子节点（死路）
        if self.is_fully_expanded and not self.children:
            return True

        return False

    def __repr__(self):
        return (f"GUIMCTSNode(page={self.page_name}, "
                f"depth={len(self.path)}, "
                f"value={self.value:.3f}, "
                f"visits={self.visits}, "
                f"children={len(self.children)}, "
                f"expanded={self.is_fully_expanded})")


class GUIPathFinder:
    """GUI场景的MCTS路径搜索"""

    def __init__(
            self,
            app_graph: 'AppGraph',
            task_description: str,
            task_name: str,
            initial_page_id: str,
            max_depth: int = 30,
            max_iterations: int = 100,
            exploration_constant: float = 5.0,
            early_stop_threshold: float = 1.95,
            semantic_model: Optional['SentenceTransformer'] = None,
            semantic_model_name: str = 'all-MiniLM-L6-v2',
            value_weight: float = 1.0,
            GT_path: List[dict] = [],
            GT_paths: List[List[dict]] = [],
    ):
        """
        Args:
            app_graph: 应用的知识图谱
            task_description: 任务描述
            initial_page_id: 起始页面ID
            max_depth: 最大搜索深度
            max_iterations: 最大迭代次数
            exploration_constant: UCT探索常数
            early_stop_threshold: 早停阈值（value超过此值立即返回）
        """
        self.app_graph = app_graph
        self.task_description = task_description
        self.task_name = task_name
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.early_stop_threshold = early_stop_threshold
        self.semantic_model = semantic_model
        self.semantic_model_name = semantic_model_name
        self.task_embedding = None  # 缓存任务 embedding
        self.value_weight = value_weight

        # 初始化根节点
        initial_page = app_graph.pages[initial_page_id]
        self.root = GUIMCTSNode(
            page_id=initial_page_id,
            page_name=initial_page.function_summary
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._emb_cache: Dict[str, torch.Tensor] = {}
        self._emb_cache_max = 4096  # 可按显存调整，缓存标准化后的向量（已在 self.device 上）
        self.GT_path = GT_path  # 任务的GT路径，用于评估搜索质量
        self.GT_paths = GT_paths  # 任务的多个GT路径，用于评估搜索质量

    




    def search(self, top_k: int, reward_model) -> List[List[GUITransition]]:
        """
        执行MCTS搜索
        """
        for iteration in range(self.max_iterations):
            # print(f"\n=== Iteration {iteration + 1}/{self.max_iterations} ===")

            # 1. Selection
            leaf = self._select(self.root)
            if leaf is None:
                # print("No more nodes to expand")
                break

            # print(f"Selected: {leaf}")
            # print(leaf.page_name)
            # print(leaf.page_id)

            # 2. Expansion (返回子节点 + 必经之路)
            children, mandatory_path, mandatory_end = self._expand(leaf)  # ⭐ 接收必经之路
            # print(mandatory_end)
            # print(children)
            if not children:
                # print("No children created (dead end or fully expanded)")
                continue

            # print(f"Created {len(children)} children")

            # 3. Evaluation
            # ⭐ 修改：prefix = parent path + mandatory path
            if mandatory_path:
                if mandatory_end is True:
                    eval_prefix_path = leaf.path
                else:
                    # 有必经之路：前缀 = 父节点路径 + 必经之路
                    eval_prefix_path = leaf.path + mandatory_path
            else:
                # 无必经之路：前缀 = 父节点路径
                eval_prefix_path = leaf.path

            # 转换为字符串

            prefix = path_to_str(eval_prefix_path)

            # 准备待评估的内容（只包含最后一步的差异）
            contents = []
            for child in children:
                # child.path = leaf.path + mandatory_path + [最后一步]
                # 我们只需要最后一步
                if mandatory_end is True:
                    last_transition = []
                    for path in child.path[-len(mandatory_path):]:
                        last_transition.append(str(path))
                else:
                    last_transition = str(child.path[-1])
                contents.append(last_transition)

            # 调用评估
            # reward_acc, reward_avg, sim_scores = self.evaluate(prefix, contents, reward_model, leaf.page_name)
            reward_acc, reward_avg, sim_scores = self.evaluate_score(eval_prefix_path, contents, reward_model, leaf.page_name)

            # 赋值结果
            for i, child in enumerate(children):
                child.value = self.value_weight * reward_avg[i] + (1 - self.value_weight) * sim_scores[i]
                child.policy_prior = reward_acc[i]
                # print(f"  {child.page_name[:150]}...: value={child.value:.3f}")

            # 4. Early stopping check
            best_child = max(children, key=lambda c: c.value)
            # if best_child.value >= self.early_stop_threshold:
            #     print(f"\n✓ Early stop: found high-value path (value={best_child.value:.3f})")
            #     return [best_child.path]

            # 5. Backpropagation
            self._backpropagate(leaf, leaf.value)

        # 搜索完成，返回top-k路径
        return self._get_top_k_paths(top_k)

    def _select(self, node: GUIMCTSNode) -> Optional[GUIMCTSNode]:
        """Selection阶段：选择到叶节点"""
        if node.if_end is True:
            return None

        while node.children and node.is_fully_expanded:

            non_end_children = [child for child in node.children if not child.if_end]
            non_end_gold = [child for child in non_end_children if child.if_gold is True]
            if len(non_end_gold) > 0:
                node = non_end_gold[0]
            else:       
                node = max(
                    non_end_children,   
                    key=lambda c: c.get_uct_value(self.exploration_constant)
                )
            # print(f"choose child: {node.page_name}, total value: {node.value} visited: {node.visits}, exploitation:{node.value/node.visits}, exploration:{math.sqrt(
            #     math.log(node.parent.visits) / node.visits)}")

        # 检查是否terminal
        if node.is_terminal(self.max_depth):
            return None

        return node



    def _if_end_backward(self, node: GUIMCTSNode):
        node.if_end = True
        while node.parent is not None:
            if not node.parent.is_fully_expanded:
                return node
            for child in node.parent.children:
                if child.if_end is False:
                    return node
            node.parent.if_end = True
            node = node.parent
        return node
    
    def _if_UTG_leaf(self, page_id):
        page = self.app_graph.pages[page_id]
        elements = [
            self.app_graph.elements[eid]
            for eid in page.element_ids
        ]
        actions = [
            self.app_graph.actions[aid]
            for aid in page.action_ids
        ]
        return len(elements) == 0 and len(actions) == 0

    def _expand(self, node: GUIMCTSNode) -> tuple[List[GUIMCTSNode], List[GUITransition], bool]:
        """
        Expansion阶段：智能扩展，自动跳过单分支路径

        Returns:
            (children, mandatory_path): 子节点列表 + 必经之路
        """
        if getattr(node, "task_terminal", False):
            node.if_end = True
            node.is_fully_expanded = True
            return [], [], True

        children = []
        page_id = node.page_id
        page_name = node.page_name
        # print(page_name)
        # ===== 自动跳过单分支路径 =====
        mandatory_path = []  # 存储必经之路
        mandatory_action = [] # 存储action id信息
        current_page_id = page_id
        current_page_name = page_name
        visited_in_chain = {page_id}  # 防止循环
        max_chain_length = 10  # 防止无限循环
        mandatory_end = False
        # print(f"node_state: {node.is_fully_expanded}")

        for _ in range(max_chain_length):
            # 1. 获取当前页面的可用操作
            cached = node.get_cached_actions(current_page_id)

            if cached:
                elements = cached['elements']
                actions = cached['actions']
            else:
                page = self.app_graph.pages[current_page_id]
                elements = [
                    self.app_graph.elements[eid]
                    for eid in page.element_ids
                ]
                actions = [
                    self.app_graph.actions[aid]
                    for aid in page.action_ids
                ]
                node.cache_page_actions(current_page_id, elements, actions)

            # if actions != []:
            #     print(actions)

            # 2. 过滤已扩展的操作（只在第一层需要）
            if not mandatory_path:
                elements = [
                    e for e in elements
                    if e.element_id not in node.expanded_actions
                ]
                actions = [
                    a for a in actions
                    if a.action_id not in node.expanded_actions
                ]

            total_actions = len(elements) + len(actions)

            # 3. 检查是否是单分支
            if total_actions == 0:
                if len(mandatory_path) == 0:
                    # print(f"⚠️  Dead end at {current_page_name[:max(len(current_page_name), 30)]}...")
                    node.is_fully_expanded = True
                    # node.if_end = True
                    break
                else:
                    # print(f"rollout to the end, need to construct evaluation...")
                    mandatory_end = True
                    break

            elif total_actions == 1:
                # 单分支，继续向前探索
                if elements:
                    action_obj = elements[0]
                    action_type = "element"
                    action_id = action_obj.element_id
                    action_name = action_obj.function_summary
                    target_page_ids = action_obj.leads_to_page_id
                else:
                    action_obj = actions[0]
                    action_type = "action"
                    action_id = action_obj.action_id
                    action_name = action_obj.function
                    target_page_ids = action_obj.leads_to_page_id

                if not target_page_ids:
                    break

                next_page_id = target_page_ids[0]

                # 循环检测
                if next_page_id in visited_in_chain:
                    # print(f"⚠️  Loop detected in mandatory chain, stopping")
                    if len(mandatory_path) <= 0:
                        break
                    else:
                        mandatory_end = True
                        break

                # 回退检测
                # if node.parent and next_page_id == node.parent.page_id:
                #     print(f"⚠️  Would return to parent, stopping chain")
                #     next_page = self.app_graph.pages[next_page_id]
                #     break

                next_page = self.app_graph.pages[next_page_id]
                next_page_name = next_page.function_summary

                # 创建transition
                transition = GUITransition(
                    from_page=current_page_name,
                    action_type=action_type,
                    action_id=action_id,
                    action_name=action_name,
                    to_page=next_page_name,
                    to_page_id=next_page_id
                )

                mandatory_path.append(transition)
                mandatory_action.append({'type': action_type, 'id': action_id})
                visited_in_chain.add(next_page_id)
                current_page_id = next_page_id
                current_page_name = next_page_name

                if self._is_task_terminal_actions(node.action_list + mandatory_action):
                    mandatory_end = True
                    break

                # print(f"⚡ Auto-forward: {action_name[:40]}... (chain: {len(mandatory_path)})")

            else:
                # 多分支，停止探索
                # print(f"🔀 Branch point: {total_actions} actions")
                break

        # ===== 输出摘要 =====
        # if mandatory_path:
        #     print(f"\n📍 Merged {len(mandatory_path)} mandatory steps:")
        #     for i, step in enumerate(mandatory_path, 1):
        #         print(f"   [{i}] {step.action_name[:50]}...")
        #     print(f"   Final: {current_page_name[:100]}...\n")
        # print(mandatory_path)
        if mandatory_path and mandatory_end is True:
            child = GUIMCTSNode(
                page_id=current_page_id,
                page_name=current_page_name,
                path=node.path + mandatory_path,
                action_list=node.action_list + mandatory_action,
                parent=node
            )
            child.if_end = self._if_UTG_leaf(current_page_id)
            # 判断是否存在一条 gold_path 包含这个动作
            self._apply_gold_state(child, parent=node)
            child.cached_page_actions = node.cached_page_actions.copy()
            node.children.append(child)
            node.expanded_actions.add(mandatory_path[0].action_id)
            children.append(child)
        else:
            # ===== 获取分支点的操作 =====
            cached = node.get_cached_actions(current_page_id)
            if cached:
                elements = cached['elements']
                actions = cached['actions']
            else:
                page = self.app_graph.pages[current_page_id]
                elements = [
                    self.app_graph.elements[eid]
                    for eid in page.element_ids
                ]
                actions = [
                    self.app_graph.actions[aid]
                    for aid in page.action_ids
                ]
                node.cache_page_actions(current_page_id, elements, actions)

            if not mandatory_path:
                elements = [e for e in elements if e.element_id not in node.expanded_actions]
                actions = [a for a in actions if a.action_id not in node.expanded_actions]

            # ===== 创建子节点 =====
            for element in elements:
                for target_page_id in element.leads_to_page_id:
                    if node.parent and target_page_id == node.parent.page_id:
                        continue

                    target_page = self.app_graph.pages[target_page_id]
                    if target_page.function_summary is None:
                        continue

                    if self.task_name in self.app_graph.pages[node.page_id].task_steps.keys() and self.task_name in element.task_steps.keys():
                        if self.task_name not in self.app_graph.pages[target_page_id].task_steps.keys():
                            continue

                    current_transition = GUITransition(
                        from_page=current_page_name,
                        action_type="element",
                        action_id=element.element_id,
                        action_name=element.function_summary,
                        to_page=target_page.function_summary,
                        to_page_id=target_page_id
                    )

                    full_path = node.path + mandatory_path + [current_transition]
                    # action_list_before = node.action_list.extend(mandatory_action)
                    # action_list_after = action_list_before.append({'type': 'element', 'id': element.element_id})
                    child = GUIMCTSNode(
                        page_id=target_page_id,
                        page_name=target_page.function_summary,
                        path=full_path,
                        action_list=node.action_list + mandatory_action + [({'type': 'element', 'id': element.element_id})],
                        parent=node
                    )
                    child.if_end = self._if_UTG_leaf(target_page_id)
                    self._apply_gold_state(child, parent=node)
                    child.cached_page_actions = node.cached_page_actions.copy()
                    node.children.append(child)

                    if mandatory_path:
                        node.expanded_actions.add(mandatory_path[0].action_id)
                    else:
                        node.expanded_actions.add(element.element_id)

                    children.append(child)


            for action in actions:
                for target_page_id in action.leads_to_page_id:
                    if node.parent and target_page_id == node.parent.page_id:
                        continue

                    target_page = self.app_graph.pages[target_page_id]
                    if target_page.function_summary is None:
                        continue

                    if self.task_name in self.app_graph.pages[node.page_id].task_steps.keys() and self.task_name in self.app_graph.elements[action.element_sequence[-1]['element_id']].task_steps.keys():
                        if self.task_name not in self.app_graph.pages[target_page_id].task_steps.keys():
                            continue

                    current_transition = GUITransition(
                        from_page=current_page_name,
                        action_type="action",
                        action_id=action.action_id,
                        action_name=action.function,
                        to_page=target_page.function_summary,
                        to_page_id=target_page_id
                    )
                       
                    full_path = node.path + mandatory_path + [current_transition]

                    child = GUIMCTSNode(
                        page_id=target_page_id,
                        page_name=target_page.function_summary,
                        path=full_path,
                        action_list=node.action_list + mandatory_action +
                            [{'type': 'action', 'id': action.action_id}],
                        parent=node
                    )
                    child.if_end = self._if_UTG_leaf(target_page_id)
                    self._apply_gold_state(child, parent=node)
                    child.cached_page_actions = node.cached_page_actions.copy()
                    node.children.append(child)

                    if mandatory_path:
                        node.expanded_actions.add(mandatory_path[0].action_id)
                    else:
                        node.expanded_actions.add(action.action_id)

                    children.append(child)

        # 标记完全扩展
        if mandatory_path or len(node.expanded_actions) >= len(elements) + len(actions):
        # if len(node.expanded_actions) >= len(elements) + len(actions):
        #     print(len(actions))
            node.is_fully_expanded = True

        if all(child.if_end for child in children):
            self._if_end_backward(node)
        # if node.if_end is True:
        #     self._if_end_backward(node)


        return children, mandatory_path, mandatory_end  # ⭐ 返回必经之路



    def evaluate(self, prefix, path, reward_model, page_name):
        """
        评估节点的价值（抽象方法，需要实现）

        Args:
            node: 要评估的节点

        Returns:
            0-1之间的分数

        实现建议：
        1. 使用LLM评估路径与任务的相关性
        2. 检查是否达到目标状态
        3. 考虑路径长度（越短越好）
        """
        # TODO: 实现你的评估逻辑


        # 1. 策略概率P(a|s)
        question = self.task_description
        policy_prefix = TASK_PLANNER.format(question=question, path=prefix, page=page_name)

        policy_contents = []
        evaluation_prefix = []
        action_text = []

        for i, content in enumerate(path):
            # print(content)
            if isinstance(content, str):
                action = re.search(r'\[action:\s*(.*?)\]', content).group(1).strip()
                action_text.append(action)
                policy_contents.append(policy_prefix + action)
                evaluation_prefix.append(EVALUATION_PROMPT.format(question=question, current_state=prefix, action=action, page=page_name))
            # for mandatory path, policy计算第一个action， evaluation计算最后一步的模拟结果
            elif isinstance(content, list):
                action_policy = re.search(r'\[action:\s*(.*?)\]', content[0]).group(1).strip()
                action_value = re.search(r'\[action:\s*(.*?)\]', content[-1]).group(1).strip()
                action_text.append(action_value)
                from_page = re.search(r'<From:\s*(.*?)\s*--', content[-1], re.DOTALL).group(1).strip()
                policy_contents.append(policy_prefix + action_policy)
                evaluation_prefix.append(
                    EVALUATION_PROMPT.format(question=question, current_state=prefix + path_to_str(content[:-1]), action=action_value, page=from_page))

        # 计算相似度项

        sim_scores = [0.0] * len(evaluation_prefix)
            



        # print(formatted_contents)
        policy_avg = [0.0] * len(evaluation_prefix)

        # policy_acc, policy_avg = reward_model.get_loglikelihood(policy_prefix, policy_contents)

        # 2. 评估概率p(if_help|s+a)
        if self.value_weight == 0.0:
            rewards = [0.0] * len(evaluation_prefix)
            return policy_avg, rewards, sim_scores

        # rewards = self._batched_helpfulness_ll(evaluation_prefix, reward_model)

        EVAL_TOKENS_BINARY = [
            " helpful",  # 有帮助
            " unhelpful",  # 没帮助
        ]
        rewards = []
        for i, evaluation_content in enumerate(evaluation_prefix):
            evaluation_contents = []
            for eval_token in EVAL_TOKENS_BINARY:
                evaluation_contents.append(evaluation_content + eval_token)
                # print(evaluation_content)
            # print(evaluation_content)
            eval_acc, eval_avg = reward_model.get_loglikelihood(evaluation_content, evaluation_contents)
            prob_y = np.exp(eval_avg[0]) / (np.exp(eval_avg[0]) + np.exp(eval_avg[-1]))
            rewards.append(prob_y)
            # rewards.append(eval_avg[0] - eval_avg[-1])



        if rewards is not None:
            return policy_avg, rewards, sim_scores
        # 示例：
        # - 调用LLM评估当前路径
        # - 检查当前页面是否是目标页面
        # - 计算路径效率
        raise NotImplementedError("需要实现evaluate方法")

    def evaluate_score(self, prefix, path, reward_model, page_name):
        """
        评估节点的价值（抽象方法，需要实现）

        Args:
            node: 要评估的节点

        Returns:
            0-1之间的分数

        实现建议：
        1. 使用LLM评估路径与任务的相关性
        2. 检查是否达到目标状态
        3. 考虑路径长度（越短越好）
        """
        # TODO: 实现你的评估逻辑


        # 1. 策略概率P(a|s)
        question = self.task_description
        evaluation_prefix = []
        action_text = []

        for i, content in enumerate(path):
            # print(content)
            if isinstance(content, str):
                action = re.search(r'\[action:\s*(.*?)\]', content).group(1).strip()
                action_text.append(action)
                history = extract_action_from_transitions(prefix)
                evaluation_prefix.append(construct_score_prompt(question, page_name, history, action))
            elif isinstance(content, list):
                action_value = re.search(r'\[action:\s*(.*?)\]', content[-1]).group(1).strip()
                action_text.append(action_value)
                from_page = re.search(r'<From:\s*(.*?)\s*--', content[-1], re.DOTALL).group(1).strip()
                history = extract_action_from_transitions(prefix+content[:-1])
                evaluation_prefix.append(construct_score_prompt(question, from_page, history, action_value))

        # 计算相似度项
        if self.value_weight == 1.0:
            sim_scores = [0.0] * len(evaluation_prefix)
        else:
            sim_scores = self.semantic_similarity_batch(action_text)



        # print(formatted_contents)
        policy_avg = [0.0] * len(evaluation_prefix)

        # policy_acc, policy_avg = reward_model.get_loglikelihood(policy_prefix, policy_contents)

        # 2. 评估概率p(if_help|s+a)
        if self.value_weight == 0.0:
            rewards = [0.0] * len(evaluation_prefix)
            return policy_avg, rewards, sim_scores

        # rewards = self._batched_helpfulness_ll(evaluation_prefix, reward_model)


        rewards = get_batch_scores(reward_model, tokenizer, evaluation_prefix)



        if rewards is not None:
            return policy_avg, rewards, sim_scores
        # 示例：
        # - 调用LLM评估当前路径
        # - 检查当前页面是否是目标页面
        # - 计算路径效率
        raise NotImplementedError("需要实现evaluate方法")

    def simulate(self, node: GUIMCTSNode, steps: int = 5) -> float:
        """
        Simulation阶段（可选，可用evaluate替代）

        Args:
            node: 起始节点
            steps: 模拟步数

        Returns:
            模拟得到的价值
        """
        # TODO: 如果需要simulation，在这里实现
        # 对于GUI场景，可能不需要random simulation
        # 直接用evaluate就够了
        raise NotImplementedError("可选：实现simulate方法")

    def _backpropagate(self, node: GUIMCTSNode, reward: float):
        """回传更新统计信息"""
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent


    def _get_top_k_paths(self, k: int):
        """提取 Top-K 路径（基于路径平均 value）"""

        all_leaves = []

        def collect_leaves(node):
            if not node.children:
                all_leaves.append(node)
            for child in node.children:
                collect_leaves(child)
        # def collect_leaves(node):
        #     # 情况 1: 真正的终止节点（走到底了）
        #     # if node.is_terminal(self.max_depth):
        #     #     all_leaves.append(node)
        #     # 情况 2: 虽然不是 terminal，但被高频访问（说明 MCTS 认为它重要）
        #     if not node.children and node.if_end is True:
        #         all_leaves.append(node)
        #     # 递归
        #     for child in node.children:
        #         collect_leaves(child)

        collect_leaves(self.root)

        # ✅ 修改这里：计算路径平均 value
        def get_path_score(leaf):
            total_value = 0
            total_visits = 0
            current = leaf
            
            while current.parent is not None:
                total_value += current.value
                total_visits += current.visits
                current = current.parent
            
            return total_value / total_visits if total_visits > 0 else 0

        # def get_path_score(leaf):
        #     total_value = 0
        #     total_visits = 0
        #     current = leaf

        #     while current.parent is not None:
        #         total_value += current.value
        #         total_visits += current.visits
        #         current = current.parent

        #     if total_visits == 0:
        #         return 0
            
        #     avg_value = total_value / total_visits
            
        #     # ⭐ 加权：高访问量的路径更可信
        #     confidence = min(1.0, total_visits / 10)  # visits=10 时达到满分
        #     return avg_value * confidence

        # 按路径平均 value 排序
        all_leaves.sort(key=get_path_score, reverse=True)

        length = min(k, len(all_leaves))
        return [leaf.path for leaf in all_leaves[:length]], \
            [leaf.action_list for leaf in all_leaves[:length]]
    
    
    # def _get_top_k_paths(self, k: int):
    #     """
    #     提取 Top-K 路径
    #     策略：混合排序 (Hybrid Sort)
    #     1. 优先按访问次数 (visits) 降序排列 -> 找出被探索最充分的路径
    #     2. 如果访问次数相同 (例如都为 1)，按平均价值 (Q-value) 降序排列 -> 找出潜力最大的
    #     """
    #
    #     all_leaves = []
    #
    #     def collect_leaves(node):
    #         if not node.children:
    #             all_leaves.append(node)
    #             return
    #         for child in node.children:
    #             collect_leaves(child)
    #
    #     collect_leaves(self.root)
    #
    #     # ✅ 修改这里：构造一个排序键 (Sort Key)
    #     def get_hybrid_score(leaf):
    #         # 1. 计算平均价值 (Q)
    #         # 假设 leaf.value 存储的是累积 Reward (Sum)，如果是平均值则直接用 leaf.value
    #         if leaf.visits > 0:
    #             mean_value = leaf.value / leaf.visits
    #         else:
    #             mean_value = -float('inf') # 没被访问过的排最后
    #
    #         # 2. 返回元组 (Visits, Mean_Value)
    #         # Python sort 在比较元组时，会先比第一个元素；如果相等，再比第二个
    #         return (leaf.visits, mean_value)
    #
    #     # 降序排列 (reverse=True)
    #     # 效果：Visits 大的排前面；Visits 一样时，Mean Value 大的排前面
    #     all_leaves.sort(key=get_hybrid_score, reverse=True)
    #
    #     length = min(k, len(all_leaves))
    #     return [leaf.path for leaf in all_leaves[:length]], \
    #            [leaf.action_list for leaf in all_leaves[:length]]


    def _prefix_subseq_len(self, path_ids, gold_ids):
        i = 0
        for a in path_ids:
            if i < len(gold_ids) and a == gold_ids[i]:
                i += 1
        return i

    def _gold_match_len(self, action_list, gold_path):
        """Return how many gold actions are covered as an ordered subsequence."""
        matched = 0
        gold_path = gold_path or []
        for action in action_list or []:
            if matched < len(gold_path) and self._action_key(action) == self._action_key(gold_path[matched]):
                matched += 1
        return matched

    def _best_gold_match_len(self, action_list):
        if not self.GT_paths:
            return 0
        return max(self._gold_match_len(action_list, gold_path) for gold_path in self.GT_paths if gold_path)

    def _is_task_terminal_actions(self, action_list):
        for gold_path in self.GT_paths or []:
            if gold_path and self._gold_match_len(action_list, gold_path) >= len(gold_path):
                return True
        return False

    def _apply_gold_state(self, child, parent=None):
        """Set gold progress and task-terminal reward for a newly created child."""
        parent_match_len = getattr(parent, "gold_match_len", 0) if parent is not None else 0
        child_match_len = self._best_gold_match_len(getattr(child, "action_list", None) or [])
        child.gold_match_len = child_match_len

        if child_match_len > parent_match_len:
            child.if_gold = True

        if self._is_task_terminal_actions(getattr(child, "action_list", None) or []):
            child.if_gold = True
            child.r = 1.0
            child.task_terminal = True
            child.if_end = True
            child.is_fully_expanded = True

    def check_gold_path_coverage(self, gold_action_list):
        """Coverage over all expanded nodes, not just leaves.

        A task can finish at an internal/common page. Counting only leaves would
        miss paths where the gold path is already contained before extra actions.
        """
        best = 0
        target = gold_action_list or []
        G = len(target)

        stack = [self.root]
        while stack:
            node = stack.pop()
            path = getattr(node, "action_list", None) or []
            k = self._gold_match_len(path, target)
            if k > best:
                best = k
                if best == G:
                    return best
            stack.extend(getattr(node, "children", None) or [])
        return best

    @staticmethod
    def _action_key(action):
        if not action:
            return None
        return (str(action.get("type", "")), str(action.get("id", "")))

    @staticmethod
    def _action_json(action):
        if not action:
            return None
        return {"type": action.get("type"), "id": action.get("id")}

    @staticmethod
    def _node_mean_value(node):
        visits = getattr(node, "visits", 0) or 0
        if visits <= 0:
            return None
        return float(getattr(node, "value", 0.0)) / float(visits)

    def _child_debug_record(self, parent, child):
        if child is None:
            return None
        parent_len = len(getattr(parent, "action_list", None) or [])
        child_actions = getattr(child, "action_list", None) or []
        step_actions = child_actions[parent_len:]
        return {
            "page_id": getattr(child, "page_id", None),
            "page_name": getattr(child, "page_name", None),
            "step_actions": [self._action_json(a) for a in step_actions],
            "first_step_action": self._action_json(step_actions[0]) if step_actions else None,
            "last_step_action": self._action_json(step_actions[-1]) if step_actions else None,
            "path_action_len": len(child_actions),
            "if_gold": bool(getattr(child, "if_gold", False)),
            "if_end": bool(getattr(child, "if_end", False)),
            "task_terminal": bool(getattr(child, "task_terminal", False)),
            "gold_match_len": int(getattr(child, "gold_match_len", 0) or 0),
            "r": float(getattr(child, "r", 0.0)),
            "visits": int(getattr(child, "visits", 0) or 0),
            "value_sum": float(getattr(child, "value", 0.0)),
            "mean_value_q": self._node_mean_value(child),
            "q_target": float(getattr(child, "q_target", 0.0)),
            "policy_prior": float(getattr(child, "policy_prior", 0.0)),
        }

    def _find_child_matching_prefix(self, node, target_actions, matched_len):
        target_keys = [self._action_key(a) for a in (target_actions or [])]
        candidates = []
        for child in getattr(node, "children", None) or []:
            child_actions = getattr(child, "action_list", None) or []
            child_keys = [self._action_key(a) for a in child_actions]
            if len(child_keys) <= matched_len:
                continue
            if child_keys[:matched_len] != target_keys[:matched_len]:
                continue
            if child_keys[matched_len:len(child_keys)] == target_keys[matched_len:len(child_keys)]:
                candidates.append(child)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda c: (len(getattr(c, "action_list", None) or []), self._node_mean_value(c) or -1e9),
        )

    def build_gold_failure_diagnostic(
            self,
            gold_action_list,
            top_action_lists,
            best_top_rank,
            best_top_covered_steps,
            graph_covered_steps,
            top_children=8):
        gold_actions = gold_action_list or []
        gold_keys = [self._action_key(a) for a in gold_actions]
        node = self.root
        matched_len = 0
        reason = "gold_path_fully_present_in_tree"

        while matched_len < len(gold_keys):
            children = getattr(node, "children", None) or []
            if not children:
                reason = "tree_leaf_before_gold_complete"
                break
            next_gold_child = self._find_child_matching_prefix(node, gold_actions, matched_len)
            if next_gold_child is None:
                reason = "gold_next_action_not_in_expanded_children"
                break
            node = next_gold_child
            matched_len = len(getattr(node, "action_list", None) or [])

        children = getattr(node, "children", None) or []
        missing_action = gold_actions[matched_len] if matched_len < len(gold_actions) else None
        missing_key = self._action_key(missing_action)

        correct_child = None
        if missing_key is not None:
            for child in children:
                parent_len = len(getattr(node, "action_list", None) or [])
                step_actions = (getattr(child, "action_list", None) or [])[parent_len:]
                step_keys = [self._action_key(a) for a in step_actions]
                if step_keys and step_keys[0] == missing_key:
                    correct_child = child
                    break

        selected_child = None
        if best_top_rank is not None and 0 <= best_top_rank < len(top_action_lists):
            selected_child = self._find_child_matching_prefix(
                node, top_action_lists[best_top_rank], len(getattr(node, "action_list", None) or []))

        best_by_mean = None
        best_by_target = None
        if children:
            best_by_mean = max(children, key=lambda c: self._node_mean_value(c) if self._node_mean_value(c) is not None else -1e9)
            best_by_target = max(children, key=lambda c: float(getattr(c, "q_target", 0.0)))

        selected_for_gap = selected_child or best_by_mean
        correct_mean = self._node_mean_value(correct_child) if correct_child else None
        selected_mean = self._node_mean_value(selected_for_gap) if selected_for_gap else None
        correct_target = float(getattr(correct_child, "q_target", 0.0)) if correct_child else None
        selected_target = float(getattr(selected_for_gap, "q_target", 0.0)) if selected_for_gap else None

        child_candidates = sorted(
            children,
            key=lambda c: self._node_mean_value(c) if self._node_mean_value(c) is not None else -1e9,
            reverse=True,
        )[:max(0, int(top_children))]

        return {
            "reason": reason,
            "matched_prefix_len_exact": matched_len,
            "first_missing_step_0based": matched_len if matched_len < len(gold_actions) else None,
            "first_missing_step_1based": matched_len + 1 if matched_len < len(gold_actions) else None,
            "missing_gold_action": self._action_json(missing_action),
            "divergence_page_id": getattr(node, "page_id", None),
            "divergence_page_name": getattr(node, "page_name", None),
            "divergence_task_terminal": bool(getattr(node, "task_terminal", False)),
            "divergence_gold_match_len": int(getattr(node, "gold_match_len", 0) or 0),
            "divergence_r": float(getattr(node, "r", 0.0)),
            "divergence_child_count": len(children),
            "best_top_rank_1based": best_top_rank + 1 if best_top_rank is not None else None,
            "best_top_covered_steps": best_top_covered_steps,
            "graph_covered_steps": graph_covered_steps,
            "gold_length": len(gold_actions),
            "correct_child": self._child_debug_record(node, correct_child),
            "selected_child": self._child_debug_record(node, selected_child),
            "best_child_by_mean_value_q": self._child_debug_record(node, best_by_mean),
            "best_child_by_q_target": self._child_debug_record(node, best_by_target),
            "mean_value_q_gap_correct_minus_selected": (
                correct_mean - selected_mean if correct_mean is not None and selected_mean is not None else None
            ),
            "q_target_gap_correct_minus_selected": (
                correct_target - selected_target if correct_target is not None and selected_target is not None else None
            ),
            "top_children_by_mean_value_q": [self._child_debug_record(node, c) for c in child_candidates],
        }

    def _actions_match(self, action1, action2):
        """判断两个 action 是否相同（可以用模糊匹配）"""
        # 简单版本：直接比较
        for i, act in enumerate(action1):
            if act['id'] == action2:
                if i == len(action1) - 1:
                    return True, -1
                else:
                    return True, 1

        return False, -1

    def print_path(self, path: List[GUITransition]):
        """打印路径（用于调试）"""
        print("\n路径:")
        for i, transition in enumerate(path, 1):
            print(f"{i}. {transition}")


    def calculate_and_save_targets(self, node: GUIMCTSNode, gamma: float = 0.95):
        """Post-order Q target computation with task-level terminal handling."""

        if getattr(node, "task_terminal", False) or getattr(node, "r", 0.0) >= 1.0:
            node.q_target = 1.0
            return node.q_target

        if not node.children:
            node.q_target = node.r
            return node.q_target

        child_q_values = []
        for child_node in node.children:
            q = self.calculate_and_save_targets(child_node, gamma)
            child_q_values.append(q)

        mean_future_value = sum(child_q_values) / len(child_q_values)
        node.q_target = max(node.r, gamma * mean_future_value)

        return node.q_target

def _load_graph_data(path=None):
    if path is None:
        raise ValueError("--graph_dir is required")

    graph = {}
    graph_root = Path(path)
    if not graph_root.exists() or not graph_root.is_dir():
        raise FileNotFoundError(f"Graph directory does not exist: {graph_root}")

    graphs_dir = find_all_task_folders(path)
    if graphs_dir == []:
        raise ValueError(f"No graph subdirectories found under {graph_root}")

    for graph_dir in graphs_dir:
        graph_file = graph_dir / f"{graph_dir.name}_graph.pkl"
        if not graph_file.exists():
            print(f"Skipping {graph_dir}: missing graph file {graph_file.name}")
            continue
        try:
            with open(graph_file, "rb") as f:
                unpickler = ModuleRedirectUnpickler(f)
                data = unpickler.load()
            graph[getattr(data, "app_name", graph_dir.name)] = data
            graph[graph_dir.name] = data
        except Exception as e:
            print(f"加载 {graph_dir} 失败: {e}")
            continue

    if not graph:
        raise ValueError(f"No valid graph files loaded from {graph_root}")

    return graph


# def path_to_str(path: List[GUITransition]) -> str:
#     """
#     将路径转换为字符串
#
#     Args:
#         path: GUITransition对象列表
#
#     Returns:
#         格式化的路径字符串
#     """
#     if not path:
#         return "Start: "
#
#     # 方式1: 简洁版
#     path_str = "Start: "
#     for i, transition in enumerate(path):
#         if i == len(path) - 1:
#             path_str += str(transition)
#         else:
#             path_str += str(transition) + " -> "
#
#     return path_str
#
#     # 方式2: 更详细版（可选）
#     # parts = ["Start"]
#     # for transition in path:
#     #     parts.append(f"[{transition.action_name}]")
#     #     parts.append(transition.to_page)
#     # return " -> ".join(parts)

def path_to_str(transitions_list, initial: bool = True):
    """
    将转换列表转换为简化的顺序执行字符串

    参数:
        transitions_list: 列表，每个元素是字符串格式 "<From:... --[action:...]--> To:...>"

    返回:
        str: 格式化的顺序执行字符串
    """
    import re

    if not transitions_list:
        return ""

    result_parts = []

    for idx, transition in enumerate(transitions_list):
        # 正则表达式匹配每个转换块
        transition = str(transition)
        pattern = r'<From:\s*(.*?)\s*--\[action:\s*(.*?)\]-->\s*To:\s*(.*?)>'
        match = re.search(pattern, transition, re.DOTALL)

        if not match:
            continue

        from_page = match.group(1).strip()
        action = match.group(2).strip()
        to_page = match.group(3).strip()

        if initial is True:
            # 第一个转换，包含完整信息
            result_parts.append(f"Page: {from_page} -- Action: {action} --> Page: {to_page}")
            initial = False
        else:
            # 后续转换，省略From页面
            result_parts.append(f" -- Action: {action} --> Page: {to_page}")

    return "".join(result_parts)

def extract_action_from_transitions(transitions_list):
    """
    将转换列表转换为简化的顺序执行字符串

    参数:
        transitions_list: 列表，每个元素是字符串格式 "<From:... --[action:...]--> To:...>"

    返回:
        str: 格式化的顺序执行字符串
    """
    import re

    if not transitions_list:
        return ""

    result_parts = []

    for idx, transition in enumerate(transitions_list):
        # 正则表达式匹配每个转换块
        transition = str(transition)
        pattern = r'<From:\s*(.*?)\s*--\[action:\s*(.*?)\]-->\s*To:\s*(.*?)>'
        match = re.search(pattern, transition, re.DOTALL)

        if not match:
            continue

        from_page = match.group(1).strip()
        action = match.group(2).strip()
        to_page = match.group(3).strip()

        result_parts.append(action)

    return result_parts

# def extract_action_list(transitions):
#     """
#     从转换列表中提取action_list格式的列表

#     参数:
#         transitions: 嵌套列表，每个子列表包含3个元素 [from_page, action, to_page]
#                     其中 action/element 的格式为 "..._$Element_id#n$" 或 "..._$Action_id$"

#     返回:
#         list: 格式为 [{'type': 'element'/'action', 'id': '...'}, ...]
#     """
#     action_list = []

#     for transition in transitions:
#         if len(transition) < 2:
#             continue

#         # 获取中间的action/element字符串
#         action_str = transition[1]

#         # 🔧 修复：允许匹配 # 和负号
#         # 原始：r'_\$((?:Element|Action))_([a-f0-9\-]+)\$'
#         # 修改：添加 #- 到字符集，或者用更精确的模式
#         pattern = r'_\$((?:Element|Action))_([a-f0-9]+(?:#-?\d+)?)\$'
#         #                                    ^^^^^^^^^ 十六进制哈希
#         #                                             ^^^^^^^^^^ 可选的 #数字 后缀

#         match = re.search(pattern, action_str, re.IGNORECASE)

#         if match:
#             action_type = match.group(1).lower()  # 'Element' 或 'Action' 转为小写
#             action_id = match.group(2)  # 完整 ID（包括 #n）

#             action_list.append({
#                 'type': action_type,
#                 'id': action_id
#             })
#         else:
#             # 调试：打印未匹配的部分
#             print(f"⚠️ 未匹配: {action_str}")

#     return action_list

def extract_action_list(transitions):
    """
    从转换列表中提取action_list格式的列表

    参数:
        transitions: 嵌套列表，每个子列表包含3个元素 [from_page, action, to_page]
                    其中 action/element 的格式为 "..._$Type_id$" 或 "..._$Action_id$"

    返回:
        list: 格式为 [{'type': 'element'/'action', 'id': '...'}, ...]
    """
    action_list = []

    for transition in transitions:
        if len(transition) < 2:
            continue

        # 获取中间的action/element字符串
        action_str = transition[1]

        # 正则表达式提取类型和ID
        # 匹配 _$Type_id$ 或 _$Action_id$ 格式
        pattern = r'_\$((?:Element|Action))_([a-f0-9\-]+)\$'
        match = re.search(pattern, action_str, re.IGNORECASE)

        if match:
            action_type = match.group(1).lower()  # 'Element' 或 'Action' 转为小写
            action_id = match.group(2)

            action_list.append({
                'type': action_type,
                'id': action_id
            })

    return action_list


def is_path_covered(mcts_actions, data_actions):
    """
    检查MCTS路径是否包含数据路径

    参数:
        mcts_actions: MCTS提取的action列表
        data_actions: 数据路径提取的action列表

    返回:
        bool: True表示包含，False表示不包含
    """
    # if not data_actions:
    #     return True
    #
    # if len(data_actions) > len(mcts_actions):
    #     return False

    # 检查data_actions是否为mcts_actions的子序列
    data_idx = 0
    # print(len(mcts_actions))
    # print(len(data_actions))
    last_action = None
    for mcts_action in mcts_actions:
        if data_idx < len(data_actions) and mcts_action == data_actions[data_idx]:
            data_idx += 1
            last_action = mcts_action
    if_covered = data_idx == len(data_actions)

    return if_covered, data_idx/len(data_actions), data_idx, last_action


PATH_METRIC_KEYS = ('gold_q', 'gold_u', 'gold_uct', 'avg_others_q', 'avg_others_uct')



def get_averaged_path_metrics(gold_path, node, expl_const=5):
    """Average MCTS values along children that advance gold-path coverage."""
    sums = {key: 0.0 for key in PATH_METRIC_KEYS}
    steps = 0
    matched_len = 0

    def action_key(action):
        if not action:
            return None
        return (str(action.get("type", "")), str(action.get("id", "")))

    gold_keys = [action_key(action) for action in (gold_path or [])]

    def match_len(action_list):
        i = 0
        for action in action_list or []:
            if i < len(gold_keys) and action_key(action) == gold_keys[i]:
                i += 1
        return i

    while node.children:
        p_vis = node.visits
        next_node = None
        next_match_len = matched_len
        step_others_q, step_others_uct = [], []
        candidate_records = []

        for child in node.children:
            q = child.value / child.visits if child.visits > 0 else 0.0
            u = expl_const * math.sqrt(math.log(p_vis) / child.visits) if child.visits > 0 and p_vis > 1 else 0.0
            uct = q + u
            child_match_len = match_len(child.action_list)

            if child_match_len > matched_len:
                candidate_records.append((child_match_len, q, u, uct, child))
            else:
                step_others_q.append(q)
                step_others_uct.append(uct)

        if not candidate_records:
            break

        candidate_records.sort(key=lambda item: (item[0], item[3]), reverse=True)
        next_match_len, q, u, uct, next_node = candidate_records[0]
        for _, other_q, _, other_uct, _ in candidate_records[1:]:
            step_others_q.append(other_q)
            step_others_uct.append(other_uct)

        sums['gold_q'] += q
        sums['gold_u'] += u
        sums['gold_uct'] += uct

        n_others = len(step_others_q)
        if n_others > 0:
            sums['avg_others_q'] += sum(step_others_q) / n_others
            sums['avg_others_uct'] += sum(step_others_uct) / n_others

        steps += 1
        matched_len = next_match_len
        node = next_node

    if steps == 0:
        return dict(sums)
    return {k: v / steps for k, v in sums.items()}


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w',encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()



# --- Main Entry ---
if __name__ == "__main__":
    # 1. 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--max_iterations", type=int, default=30)
    parser.add_argument("--max_depth", type=int, default=60)
    parser.add_argument("--exploration_constant", type=float, default=3.0)
    parser.add_argument("--test", type=_parse_bool, default=True)
    parser.add_argument('--initial_type', type=str, default="True")
    parser.add_argument('--iter', type=str, default='0')
    parser.add_argument('--log_name', type=str, default="default_log")
    parser.add_argument('--log_dir', type=str, default="logs/androidworld")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--coverage_debug_log', type=str, default=None)
    parser.add_argument('--coverage_debug_top_children', type=int, default=8)
    args = parser.parse_args()

    seed = args.seed
    random_module.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if os.path.exists(args.output_file):
        os.remove(args.output_file)
        print(f"Removed existing file: {args.output_file}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    coverage_debug_path = Path(args.coverage_debug_log) if args.coverage_debug_log else log_dir / f"{args.log_name}_failed_paths.jsonl"
    if args.test and coverage_debug_path.exists():
        coverage_debug_path.unlink()
    sys.stdout = Logger(str(log_dir / f"{args.log_name}.txt"))
    if args.test:
        print(f"Coverage debug log: {coverage_debug_path}")
    graph = _load_graph_data(path=args.graph_dir)
    # 2. 加载模型
    if _parse_bool(args.initial_type):
        # reward_model, tokenizer = load_untrained_base_model(args.model_path)
        print("Loading trained model...")
        reward_model, tokenizer = load_model_and_tokenizer(args.model_path)
    else:
        print("Loading untrained base model...")
        reward_model, tokenizer = load_untrained_base_model(args.model_path)

    covered_count = 0
    strict_covered_count = 0
    covered_rank = 0
    strict_covered_rank = 0
    covered_rate_list = []
    covered_length_list = []
    
    # 3. 加载任务 (从 launch 脚本切分好的 batch 文件读)
    with open(args.task_file, 'r', encoding='utf-8') as f:
        dataset_raw = json.load(f)

    dataset = []
    for data_dict in dataset_raw:
        if _is_success_value(data_dict.get("success")):
            dataset.append(data_dict)
    if not dataset:
        raise ValueError(f"No successful tasks found in {args.task_file}")
    # 初始化辅助模型 (Semantic sim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 4. 循环任务
    all_data_buffer = [] # 可以选择先存内存，最后一起写
    sums = {key: 0.0 for key in PATH_METRIC_KEYS}

    start_time = time.time()
    
    for idx in tqdm(range(len(dataset)), desc="MCTS Rollout"):
        data_item = dataset[idx]
        query = data_item['goal']
        task = data_item['task']
        start_page = _extract_start_page_id(data_item)

        max_gold_path = []
        gold_paths = []
        for gold_path in data_item.get('path', []):
            gold_path = extract_action_list(gold_path)
            if gold_path:
                gold_paths.append(gold_path)
            max_gold_path = gold_path if len(gold_path) > len(max_gold_path) else max_gold_path
        if not gold_paths:
            raise ValueError(f"No valid action sequence found for task={task!r}")
        
        # gold_action_list = extract_action_list(data_item['path'])
        gold_action_list = max_gold_path
        app = _resolve_app_key(data_item, task)
        if app not in graph:
            available = ", ".join(sorted(graph.keys())[:30])
            raise KeyError(
                f"Graph for app={app!r} task={task!r} is not loaded. "
                f"Available graph keys: {available}"
            )
        app_graph = graph[app]


        finder = GUIPathFinder(
            app_graph=app_graph,
            task_description=query,
            task_name=task,
            initial_page_id=start_page,
            # 使用参数控制超参
            max_depth=args.max_depth,
            max_iterations=args.max_iterations, 
            exploration_constant=args.exploration_constant,
            value_weight=1.0,
            GT_path=gold_action_list,
            GT_paths=gold_paths
        )

        # Run search. _get_top_k_paths ranks leaves by mean path value:
        # sum(node.value) / sum(node.visits), not by visits alone.
        top_paths, top_action_list = finder.search(5, reward_model)

        # Compute q_target before diagnostics so the debug log contains both
        # search-time mean value and training target value.
        finder.calculate_and_save_targets(finder.root, gamma=0.95)

        if args.test:
            max_covered_rate = -1
            path_max = None
            data_idx_max = -1
            last_action_max = None
            best_top_rank = None
            best_top_covered_steps = 0
            strict_task_covered = False
            metric_task_covered = False
            top_path_summaries = []
            path_metrics = get_averaged_path_metrics(gold_action_list, finder.root, 3)
            for key in sums.keys():
                sums[key] += path_metrics.get(key, 0.0)
                sums[key] /= (idx + 1)

            for i, path in enumerate(top_paths):
                strict_if_covered, covered_rate, data_idx, last_action = is_path_covered(top_action_list[i], gold_action_list)
                if covered_rate > max_covered_rate:
                    max_covered_rate = covered_rate
                    path_max = path
                    data_idx_max = data_idx
                    last_action_max = last_action
                    best_top_rank = i
                    best_top_covered_steps = data_idx

                terminal_hit = False
                for gp in gold_paths:
                    if not gp:
                        continue
                    if gp[-1] in top_action_list[i]:
                        terminal_hit = True
                        break

                top_path_summaries.append({
                    "rank": i + 1,
                    "strict_full_gold_covered": bool(strict_if_covered),
                    "terminal_action_hit": bool(terminal_hit),
                    "covered_rate": covered_rate,
                    "covered_steps": data_idx,
                    "last_matched_action": finder._action_json(last_action),
                    "path_action_len": len(top_action_list[i]),
                })

                if strict_if_covered is True:
                    strict_task_covered = True
                    strict_covered_rank += i + 1

                if strict_if_covered is True or terminal_hit is True:
                    metric_task_covered = True
                    covered_count += 1
                    covered_rank += i + 1
                    max_covered_rate = 1
                    break

            if strict_task_covered:
                strict_covered_count += 1

            covered_length = finder.check_gold_path_coverage(gold_action_list)
            covered_length_list.append(covered_length / len(gold_action_list))
            covered_rate_list.append(max_covered_rate)

            if not strict_task_covered:
                diagnostic = finder.build_gold_failure_diagnostic(
                    gold_action_list=gold_action_list,
                    top_action_lists=top_action_list,
                    best_top_rank=best_top_rank,
                    best_top_covered_steps=best_top_covered_steps,
                    graph_covered_steps=covered_length,
                    top_children=args.coverage_debug_top_children,
                )
                diagnostic.update({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "iter": args.iter,
                    "log_name": args.log_name,
                    "task_index": idx,
                    "task": task,
                    "app": app,
                    "goal": query,
                    "current_metric_covered": bool(metric_task_covered),
                    "strict_full_gold_covered": bool(strict_task_covered),
                    "best_top_covered_rate": max_covered_rate,
                    "graph_covered_rate": covered_length / len(gold_action_list),
                    "top_path_summaries": top_path_summaries,
                })
                coverage_debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(coverage_debug_path, 'a', encoding='utf-8') as debug_f:
                    debug_f.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")

        export_training_dataset(
            root_node=finder.root, 
            instruction=query, 
            output_file=args.output_file # 这里要支持 append 模式，或者我们在函数里处理
        )
        # ==========================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Rollout Time for {len(dataset)} tasks: {elapsed_time:.2f} seconds")
    if args.test:
        covered_rate = covered_count / len(dataset) if dataset else 0.0
        average_rank = covered_rank / covered_count if covered_count else 0.0
        inner_rate = sum(covered_rate_list) / len(covered_rate_list) if covered_rate_list else 0.0
        graph_covered = sum(covered_length_list) / len(covered_length_list) if covered_length_list else 0.0
        print(
        f"covered count: {covered_count}, total sample: {len(dataset)}, covered rate: {covered_rate}, average covered rank: {average_rank}, covered rate inner: {inner_rate}, graph_covered_length: {graph_covered}")

        strict_covered_rate = strict_covered_count / len(dataset) if dataset else 0.0
        strict_average_rank = strict_covered_rank / strict_covered_count if strict_covered_count else 0.0
        print(
        f"strict covered count: {strict_covered_count}, strict covered rate: {strict_covered_rate}, strict average covered rank: {strict_average_rank}")
        print(f"failed path debug log: {coverage_debug_path}")

        print(sums)
    print("Rollout logic finished.")
