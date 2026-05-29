import json
from typing import List
from rollout import *
import re

# ==========================================
# 1. 你的 Prompt 构造函数 (保持原样)
# ==========================================
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
        root_node: GUIMCTSNode, 
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
    print(f"正在保存 {len(dataset)} 条数据到 {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print("保存完成。")

# ==========================================
# 4. 调用示例
# ==========================================
# 假设:
# 1. tree_root 是之前跑完 MCTS 的根
# 2. task_instruction 是这棵树对应的任务，比如 "Create a new user named Bob"

# 先确保 Q 值算好了 (使用之前那个 calculate_and_save_targets)
# tree.calculate_and_save_targets(tree_root)

# 然后导出
# export_training_dataset(tree_root, "Create user Bob", "train_data.jsonl")