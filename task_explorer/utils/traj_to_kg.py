import copy
import os
import pickle
import zstandard as zstd
from neo4j import GraphDatabase
from pathlib import Path
import numpy as np
import imagehash
from PIL import Image
from uuid import uuid4
from typing import List, Dict, Tuple, Set, Any, Iterator, Optional
from task_explorer.utils.vector_db import VectorStore, VectorData, NodeType
from task_explorer.utils.img_tool import element_img, extract_features
import config
import json
from task_explorer.utils.prompt_templates import SCREENSHOT_COMPARISON
import re
import hashlib
from task_explorer.utils.chain_understand import process_and_update_chain
from datetime import datetime
import asyncio
import ast
from task_explorer.utils.utils import resize_pil_image, pil_to_webp_base64, openai_request, load_object_from_disk

def pil_image_to_phash(pil_image: Image.Image) -> str:
    """Convert a PIL Image to a perceptual hash.

    Args:
        pil_image (Image.Image): The PIL Image object.

    Returns:
        str: The perceptual hash.
    """

    return str(imagehash.phash(pil_image, hash_size=16, highfreq_factor=8)).upper()

def ndarray_image_to_phash(ndarray_image: np.ndarray) -> str:
    """Convert a NumPy ndarray image to a perceptual hash.

    Args:
        ndarray_image (np.ndarray): The NumPy ndarray image.

    Returns:
        str: The perceptual hash.
    """
    return pil_image_to_phash(Image.fromarray(ndarray_image))


def is_transition_valid(
        before_screenshot: np.ndarray, after_screenshot: np.ndarray
) -> bool:
    """判断两个截图之间的转换是否有效"""
    return ndarray_image_to_phash(before_screenshot) != ndarray_image_to_phash(
        after_screenshot
    )


def parse_comparison_result(text):
    """
    从模型输出中提取JSON并获取指定字段
    """
    try:
        # 方法1: 查找第一个完整的JSON对象
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in matches:
            json_str = match.group()
            try:
                # 尝试解析JSON
                data = json.loads(json_str)

                # 检查是否包含我们需要的字段
                if 'is_same_page' in data:
                    is_same_page = data.get('is_same_page', False)
                    matched_index = data.get('matched_candidate_index')
                    return is_same_page, matched_index
            except json.JSONDecodeError:
                continue

        # 如果找不到有效JSON，返回None
        return False, None

    except Exception as e:
        print(f"提取JSON时出错: {e}")
        return False, None


def screenshot_comparison_anthropic_format(screenshot, candidate_list, goal: str = None, action_output: str = None, usage: dict = None):
    """使用Anthropic推荐的格式"""

    # screenshot_b64 = pil_to_webp_base64(screenshot)

    # 构建描述文本
    description = SCREENSHOT_COMPARISON.format(goal=goal if goal else "Not Available", action_reasoning=action_output if action_output else "Not Available") + "\n\nI will show you multiple images:"
    description += "\n1. Current Screenshot"
    for i in range(len(candidate_list)):
        description += f"\n{i + 2}. Candidate Screenshot {i}"
    description += "\n\nPlease compare them and provide your analysis."

    # 构建content - 文本优先，然后是所有图像
    content = [{"type": "text", "text": description}]

    # 添加当前截图
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/webp;base64,{screenshot}"}
    })

    # 添加候选图像
    for candidate in candidate_list[:2]:
        candidate_b64 = candidate

        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/webp;base64,{candidate_b64}"}
        })

    messages = [{"role": "user", "content": content}]
    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
    rsp_txt = openai_request(
        messages=messages, timeout=300, max_tokens=15000, usage=usage
    )
    print(rsp_txt)

    return parse_comparison_result(rsp_txt)


def is_leaf_folder(folder_path):
    """
    判断当前文件夹是否为叶子文件夹（没有子文件夹）

    Args:
        folder_path (str or Path): 文件夹路径

    Returns:
        bool: True表示是叶子文件夹，False表示有子文件夹
    """
    folder_path = Path(folder_path)

    # 检查路径是否存在且是文件夹
    if not folder_path.exists() or not folder_path.is_dir():
        return False

    # 遍历文件夹内容，查找子文件夹
    for item in folder_path.iterdir():
        if item.is_dir():
            return False  # 找到子文件夹，不是叶子文件夹

    return True  # 没有找到子文件夹，是叶子文件夹

def parse_kwargs_from_call(s: str) -> Dict[str, Any]:
    """
    将类似 'JSONAction(action_type="input_text", index=9, text="Freelance Payment")'
    解析为 dict: {"action_type": "input_text", "index": 9, "text": "Freelance Payment"}
    非字面量（非常见基本类型）会被置为 None。
    """
    try:
        node = ast.parse(s, mode='eval').body
    except SyntaxError:
        return {}
    if not isinstance(node, ast.Call):
        return {}
    out: Dict[str, Any] = {}
    for kw in node.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            out[kw.arg] = None

    return out

def get_arg(s: str, key: str, default: Any = None) -> Any:
    """
    精确按键名获取值，若不存在则返回 default。
    """
    return parse_kwargs_from_call(s).get(key, default)


def _steps_of(task_value: Any, task_name: str) -> Set[int]:
        """
        将 node.task 解析为给定 task 的 step 集合（int）。
        兼容:
          - JSON 字符串: '{"Task":[0,1]}'
          - dict: {"Task":[0,1]}
          - 旧格式 list[str]: ['TaskA', ...] -> 仅视为 {0}
          - None/其他 -> 空集合
        """
        if task_value is None:
            return set()
        if isinstance(task_value, str):
            s = task_value.strip()
            if not s:
                return set()
            try:
                obj = json.loads(s)
            except Exception:
                return set()
        elif isinstance(task_value, dict):
            obj = task_value
        elif isinstance(task_value, list):
            return {0} if task_name in task_value else set()
        else:
            return set()

        steps = obj.get(task_name, [])
        try:
            return {int(x) for x in steps}
        except Exception:
            return set()


def _all_steps_zero_map(task: dict, require_nonempty: bool = True) -> bool:
    if not isinstance(task, dict):
        return False
    if require_nonempty and not task:
        return False
    for steps in task.values():
        # 必须是非空列表，且所有值都是 0
        if not isinstance(steps, list) or (require_nonempty and not steps):
            return False
        if any(int(s) != 0 for s in steps):
            return False
    return True

class TrajectoryToNeo4jImporter:
    def __init__(self, uri: str, auth: tuple, database: str, index: str, root_node: str=None, usage: dict=None):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.vector_store = VectorStore(
            api_key=config.PINECONE_API_KEY,
            index_name=index,
            dimension=2048,
            batch_size=2,
        )
        self.root_node = root_node
        self.is_first_import = True
        self.database = database
        self.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0}

    def manually_traj(self, root_path):
        task_name = root_path.name
    def dfs_traverse_and_import(self, root_path):
        """DFS遍历文件夹树并导入轨迹数据到Neo4j"""
        task_name = root_path.name
        def dfs_recursive(current_path, parent_last_node, task_name, depth):
            current_path = Path(current_path)

            # 1. 处理当前层的轨迹文件
            trajectory_file = self._find_trajectory_file(current_path)
            current_last_node = parent_last_node
            last_depth = depth

            if trajectory_file:
                print(f"precessing {trajectory_file}...")
                if_leaf = is_leaf_folder(current_path)
                current_last_node, last_depth = self._import_trajectory(
                    trajectory_file, parent_last_node, if_leaf, task_name, depth
                )
            else:
                print(f"跳过文件夹: {current_path.name} (无轨迹文件)")
            # 2. 获取所有子文件夹（按名称排序保证一致性）
            subdirs = [d for d in current_path.iterdir() if d.is_dir()]
            subdirs.sort()

            # 3. DFS递归遍历每个子文件夹
            # 每个子文件夹都从当前轨迹的最后一个节点开始
            for subdir in subdirs:
                print(f"进入分支: {subdir.name}, 起始节点: {current_last_node}")
                dfs_recursive(subdir, current_last_node, task_name, last_depth)
                print(f"完成分支: {subdir.name}")

        # 开始DFS遍历
        print(f"开始DFS遍历: {root_path}")
        init_depth = 0
        start_paths = find_all_task_folders(root_path)
        for start_path in start_paths:
            initial_parent = None if self.is_first_import else self.root_node
            print(initial_parent)
            print("*************************************")
            dfs_recursive(start_path, initial_parent, task_name, init_depth)
        print("DFS遍历完成")

    def _find_trajectory_file(self, folder_path):
        """在当前文件夹中找到轨迹文件"""
        pkl_files = sorted(folder_path.glob("*.pkl.zst"))
        return str(pkl_files[0]) if pkl_files else None

    def import_depth_zero_task(self, task_path, root_page_id=None):
        """Import a direct ``<package>/<task>/*.pkl.zst`` human trajectory."""
        task_path = Path(task_path)
        trajectory_file = self._find_trajectory_file(task_path)
        if trajectory_file is None:
            raise FileNotFoundError(f"No .pkl.zst trajectory found in {task_path}")

        if root_page_id is None:
            self.root_node = None
            self.is_first_import = True
            print(f"Importing depth-zero human task {task_path.name} and creating a new root")
        else:
            self.root_node = root_page_id
            self.is_first_import = False
            print(f"Importing depth-zero human task {task_path.name} from root {root_page_id}")
        return self._import_trajectory(
            trajectory_file=trajectory_file,
            parent_last_node=root_page_id,
            is_leaf=True,
            task_name=task_path.name,
            depth=0,
        )

    # def find_next_page_by_target_element(self, page_id, target_element):
    #     """根据目标元素查找下一个页面ID"""
    #     with self.driver.session() as session:
    #         result = session.run("""
    #             MATCH (p:Page {page_id: $page_id})-[:has_element]->(e:Element {target_element: $target_element})-[:lead_to]->(next_p:Page)
    #             RETURN next_p.page_id as next_page_id
    #             LIMIT 1
    #         """, page_id=page_id, target_element=target_element)
    #
    #         record = result.single()
    #         return record['next_page_id'] if record else None

    def find_next_page_by_target_element(self, page_id, target_image_uid, action):
        """根据目标元素的image_hash查找下一个页面ID"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p:Page {page_id: $page_id})-[:HAS_ELEMENT]->(e:Element)
                MATCH (e)-[:LEADS_TO]->(next_p:Page)
                RETURN next_p.page_id as next_page_id, e.element_id as element_id, e.target_element as target_element, e.converted_action as converted_action
            """, page_id=page_id)

            # 在Python中解析JSON并匹配image_hash
            for record in result:
                try:
                    target_element = json.loads(record['target_element'])
                    converted_action = str(record['converted_action'])

                    if target_element.get('uid') == target_image_uid['uid'] and converted_action == action:
                        print(f"找到匹配的uid: {target_image_uid}")
                        return record['next_page_id'], record['element_id']

                    elif target_element.get('element_id') == target_image_uid['element_id']:
                        action_type = get_arg('action_type', action, None)
                        converted_action_type = get_arg('action_type', converted_action, None)
                        bbox = target_element.get("bbox")
                        x_min = float(bbox["x_min"])
                        x_max = float(bbox["x_max"])
                        y_min = float(bbox["y_min"])
                        y_max = float(bbox["y_max"])

                        if x_min == target_image_uid['bbox']['x_min'] and y_min == target_image_uid['bbox']['y_min'] and x_max == target_image_uid['bbox']['x_max'] and y_max == target_image_uid['bbox']['y_max']:
                            if converted_action_type and action_type not in ['input_text', 'scroll']:
                                print(f"找到匹配的element_id和bbox: {target_image_uid['element_id']}")
                                return record['next_page_id'], record['element_id']
                            elif converted_action_type == 'scroll':
                                if get_arg('direction', converted_action) == get_arg('direction', action):
                                    print(f"找到匹配的element_id和bbox和action_type: {target_image_uid['element_id']}")
                                    return record['next_page_id'], record['element_id']
                            else:
                                if get_arg('text', converted_action) == get_arg('text', action):
                                    print(f"找到匹配的element_id和bbox和action_type: {target_image_uid['element_id']}")
                                    return record['next_page_id'], record['element_id']


                except (json.JSONDecodeError, TypeError) as e:
                    print(e)
                    continue

            return None, None

    def find_next_page_by_converted_action(self, page_id, action):
        """根据目标元素的image_hash查找下一个页面ID"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p:Page {page_id: $page_id})-[:HAS_ELEMENT]->(e:Element)
                MATCH (e)-[:LEADS_TO]->(next_p:Page)
                RETURN next_p.page_id as next_page_id, e.element_id as element_id, e.target_element as target_element, e.converted_action as converted_action
            """, page_id=page_id)

            # 在Python中解析JSON并匹配image_hash
            for record in result:
                if record['target_element'] is None:
                    converted_action = str(record['converted_action'])
                    if get_arg('action_type', converted_action) == get_arg('action_type', action) == 'scroll':
                        if get_arg('direction', converted_action) == get_arg('direction', action):
                            return record['next_page_id'], record['element_id']

                    elif get_arg('action_type', converted_action) == get_arg('action_type', action) == 'input_text':
                        if get_arg('text', converted_action) == get_arg('text', action):
                            return record['next_page_id'], record['element_id']
                else:
                    continue
            return None, None

    def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Generic node creation function"""
        query = f"CREATE (n:{label} $properties) " "RETURN elementId(n) as node_id"

        with self.driver.session(database=self.database) as session:
            result = session.run(query, properties=properties)
            record = result.single()
            return str(record["node_id"]) if record else None

    def add_element_to_page(self, page_id: str, element_id: str) -> bool:
        """Create Page-HAS_ELEMENT->Element relationship"""
        query = """
        MATCH (p:Page {page_id: $page_id})
        MATCH (e:Element {element_id: $element_id})
        MERGE (p)-[r:HAS_ELEMENT]->(e)
        RETURN type(r) as rel_type
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, page_id=page_id, element_id=element_id)
                record = result.single()
                success = record is not None
                if not success:
                    print(
                        f"Warning: Failed to create HAS_ELEMENT relationship between page {page_id} and element {element_id}"
                    )
                return success
        except Exception as e:
            print(f"Error creating HAS_ELEMENT relationship: {str(e)}")
            return False

    def add_element_leads_to(
        self,
        element_id: str,
        target_id: str,
        converted_action
    ) -> bool:
        """Create Element-LEADS_TO->Page relationship"""
        query = """
        MATCH (e:Element {element_id: $element_id})
        MATCH (t:Page {page_id: $target_id})
        MERGE (e)-[r:LEADS_TO {
            action: $converted_action
        }]->(t)
        RETURN type(r) as rel_type
        """

        try:
            # Serialize action_params to JSON string

            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    element_id=element_id,
                    target_id=target_id,
                    converted_action=converted_action,
                )
                record = result.single()
                success = record is not None
                if not success:
                    print(
                        f"Warning: Failed to create LEADS_TO relationship from element {element_id} to page {target_id}"
                    )
                return success
        except Exception as e:
            print(f"Error creating LEADS_TO relationship: {str(e)}")
            return False

    def update_no_transition_page(self, page_id, new_elements, new_logical_screen_size):
        """更新no transition页面的elements和logical_screen_size属性"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p:Page {page_id: $page_id})
                WHERE p.elements = 'no transition occurred' AND p.logical_screen_size = 'no transition occurred'
                SET p.elements = $new_elements,
                    p.logical_screen_size = $new_logical_screen_size
                RETURN p.page_id as updated_page_id
            """,
                                 page_id=page_id,
                                 new_elements=new_elements,
                                 new_logical_screen_size=new_logical_screen_size
                                 )

            record = result.single()
            return record['updated_page_id'] if record else None

    def get_task_by_page_id(self, page_id):
        """
        根据page_id列表获取对应的before_screenshot属性列表

        Args:
            page_id_list (list): page_id的列表

        Returns:
            list: before_screenshot属性的列表，顺序与输入的page_id_list对应
        """

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p:Page {page_id: $page_id})
                RETURN p.task as task
            """, page_id=page_id)

            # 创建page_id到screenshot的映射

            record = result.single()
            # 按照输入顺序返回截图列表
            return record['task'] if record else []

    def get_screenshots_by_page_ids(self, page_id_list):
        """
        根据page_id列表获取对应的before_screenshot属性列表

        Args:
            page_id_list (list): page_id的列表

        Returns:
            list: before_screenshot属性的列表，顺序与输入的page_id_list对应
        """
        if not page_id_list:
            return []

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                UNWIND $page_ids as page_id
                MATCH (p:Page {page_id: page_id})
                RETURN page_id, p.raw_page as screenshot
                ORDER BY page_id
            """, page_ids=page_id_list)

            # 创建page_id到screenshot的映射
            screenshot_map = {record['page_id']: json.loads(record['screenshot']) for record in result}

            # 按照输入顺序返回截图列表
            return [screenshot_map.get(page_id) for page_id in page_id_list]

    # def append_goal_to_element(self, element_id: str, new_goal: str) -> bool:
    #     """
    #     向Element节点的goal属性中追加新的goal（不允许重复）
    #
    #     Args:
    #         element_id (str): Element节点的ID
    #         new_goal (str): 要添加的新goal
    #
    #     Returns:
    #         bool: 操作是否成功
    #     """
    #     with self.driver.session(database="neo4j") as session:
    #         try:
    #             result = session.run("""
    #                 MATCH (e:Element {element_id: $element_id})
    #                 SET e.goal = CASE
    #                     WHEN e.goal IS NULL THEN [$new_goal]
    #                     WHEN NOT $new_goal IN e.goal THEN e.goal + [$new_goal]
    #                     ELSE e.goal
    #                 END
    #                 RETURN e.goal as updated_goal
    #             """, element_id=element_id, new_goal=new_goal)
    #
    #             record = result.single()
    #             if record:
    #                 print(f"Element {element_id} goals: {record['updated_goal']}")
    #                 return True
    #             else:
    #                 print(f"未找到element_id: {element_id}")
    #                 return False
    #
    #         except Exception as e:
    #             print(f"添加goal失败: {e}")
    #             return False

    # def append_task_to_node(self, node_type: str, node_id: str, new_task: str, step) -> bool:
    #     """
    #     向Element节点的goal属性中追加新的goal（不允许重复）
    #
    #     Args:
    #         element_id (str): Element节点的ID
    #         new_goal (str): 要添加的新goal
    #
    #     Returns:
    #         bool: 操作是否成功
    #     """
    #     with self.driver.session(database=self.database) as session:
    #         if node_type == "Element":
    #             try:
    #                 result = session.run("""
    #                     MATCH (e:Element {element_id: $element_id})
    #                     SET e.task = CASE
    #                         WHEN e.task IS NULL THEN [$new_task]
    #                         WHEN NOT $new_task IN e.task THEN e.task + [$new_task]
    #                         ELSE e.task
    #                     END
    #                 RETURN e.task as updated_task
    #                 """, element_id=node_id, new_task=new_task)
    #             except Exception as e:
    #                 print(f"添加task失败: {e}")
    #                 return False
    #         else:
    #             try:
    #                 result = session.run("""
    #                     MATCH (e:Page {page_id: $page_id})
    #                     SET e.task = CASE
    #                         WHEN e.task IS NULL THEN [$new_task]
    #                         WHEN NOT $new_task IN e.task THEN e.task + [$new_task]
    #                         ELSE e.task
    #                     END
    #                 RETURN e.task as updated_task
    #                 """, page_id=node_id, new_task=new_task)
    #             except Exception as e:
    #                 print(f"添加task失败: {e}")
    #                 return False
    #         record = result.single()
    #         if record:
    #             print(f"{node_type} {node_id} goals: {record['updated_task']}")
    #             return True
    #         else:
    #             print(f"未找到 node id: {node_id}")
    #             return False

    def append_task_to_node(self, node_type: str, node_id: str, task_name: str, step: int) -> bool:
        """
        将节点的 task 属性维护为: {task_name: [step1, step2, ...]}。
        - 陌生 task: 新建 key，值为包含当前 step 的列表；
        - 已有 task: 在其列表中追加 step（去重）。
        - 无 APOC；兼容旧格式 task=list[str]。

        Args:
            node_type: "Element" 或 "Page"
            node_id:   element_id 或 page_id
            task_name: 任务名
            step:      本次经过该节点时的 step（外部已按你的规则 depth+step 计算好）

        Returns:
            bool: 是否更新成功
        """
        label = "Element" if node_type == "Element" else "Page"
        id_key = "element_id" if label == "Element" else "page_id"

        try:
            with self.driver.session(database=self.database) as session:
                # 读现值
                rec = session.run(
                    f"""
                    MATCH (n:{label} {{{id_key}: $id}})
                    RETURN n.task AS task
                    """,
                    id=node_id
                ).single()

                if not rec:
                    print(f"未找到 node id: {node_id}")
                    return False
                if rec["task"] is not None:
                    cur = json.loads(rec["task"])
                else:
                    cur = {}
                # 规范化为 dict
                # if cur is None:
                #     task_map = {}
                # elif isinstance(cur, dict):
                #     task_map = dict(cur)  # 浅拷贝
                # elif isinstance(cur, list):
                #     # 兼容旧格式：把旧的字符串列表转成 {name: []}
                #     task_map = {str(name): [] for name in cur}
                # else:
                #     # 异常类型，重置为 dict，避免后续崩
                #     task_map = {}

                # 更新 steps（去重 + 可选排序）
                s = int(step)
                steps = cur.get(task_name, [])
                if s not in steps:
                    steps = steps + [s]
                    steps.sort()
                cur[task_name] = steps
                # 回写
                rec2 = session.run(
                    f"""
                    MATCH (n:{label} {{{id_key}: $id}})
                    SET n.task = $task_map
                    RETURN n.task AS updated_task
                    """,
                    id=node_id, task_map=json.dumps(cur)
                ).single()

                if rec2:
                    print(f"{node_type} {node_id} task: {rec2['updated_task']}")
                    return True
                print(f"未找到 node id: {node_id}")
                return False
        except Exception as e:
            print(f"添加task失败: {e}")
            return False


    def append_goal_to_node(self, node_type: str, node_id: str, new_task: str) -> bool:
        """
        向Element节点的goal属性中追加新的goal（不允许重复）

        Args:
            element_id (str): Element节点的ID
            new_goal (str): 要添加的新goal

        Returns:
            bool: 操作是否成功
        """
        with self.driver.session(database=self.database) as session:
            if node_type == "Element":
                try:
                    result = session.run("""
                        MATCH (e:Element {element_id: $element_id})
                        SET e.goal = CASE 
                            WHEN e.goal  IS NULL THEN [$new_task]
                            WHEN NOT $new_task IN e.goal THEN e.goal + [$new_task]
                            ELSE e.goal
                        END
                    RETURN e.goal as updated_goal
                    """, element_id=node_id, new_task=new_task)
                except Exception as e:
                    print(f"添加goal失败: {e}")
                    return False
            else:
                try:
                    result = session.run("""
                        MATCH (e:Page {page_id: $page_id})
                        SET e.goal = CASE 
                            WHEN e.goal IS NULL THEN [$new_task]
                            WHEN NOT $new_task IN e.goal THEN e.goal + [$new_task]
                            ELSE e.goal
                        END
                    RETURN e.goal as updated_goal
                    """, page_id=node_id, new_task=new_task)
                except Exception as e:
                    print(f"添加task失败: {e}")
                    return False
            record = result.single()
            if record:
                print(f"{node_type} {node_id} goals: {record['updated_goal']}")
                return True
            else:
                print(f"未找到 node id: {node_id}")
                return False



    def _import_trajectory(self, trajectory_file, parent_last_node, is_leaf, task_name, depth):
        """导入单个轨迹文件到Neo4j，返回任务ID和最后一个节点"""
        print(f"处理轨迹文件: {trajectory_file}")

        # 解压并加载轨迹数据

        trajectory_data = load_object_from_disk(trajectory_file)

        # 提取任务ID（从文件名或数据中）
        # task_id = self._extract_task_id(trajectory_file)
        parent_step = depth
        parent_node = parent_last_node
        for step, traj in enumerate(trajectory_data):
            print(f"处理第：{step} 个step")
            before_screenshot = traj["before_screenshot"]
            after_screenshot = traj["after_screenshot"]
            # 判断是否动作无效，无效则跳过
            if after_screenshot is None:
                continue
            if not is_transition_valid(before_screenshot, after_screenshot):
                continue
            # 判断是否是已有元素，已有则跳过
            duplicated_page = None
            if traj["target_element"] is not None:

                duplicated_page, matched_element_node = self.find_next_page_by_target_element(parent_node, traj["target_element"], str(traj['converted_action']))
            else:

                duplicated_page, matched_element_node = self.find_next_page_by_converted_action(parent_node, str(traj['converted_action']))

            if duplicated_page is not None:
                self.append_goal_to_node('Element', matched_element_node, traj['goal'])
                candidate_screenshots = self.get_screenshots_by_page_ids([duplicated_page])
                match, target_page = screenshot_comparison_anthropic_format(img_to_base64(traj['after_screenshot']),
                                                     candidate_screenshots, goal=traj['goal'], action_output=traj['action_output'], usage=self.usage)
                self.append_task_to_node('Page', parent_node, task_name, parent_step)
                self.append_task_to_node('Element', matched_element_node, task_name, parent_step)

                if match == True:
                    self.append_task_to_node('Page', duplicated_page, task_name, parent_step + 1)
                    self.append_goal_to_node('Page', duplicated_page, traj['goal'])
                    parent_node = duplicated_page
                    parent_step += 1
                    continue
                else:
                    target_page = str(uuid4())
                    target_page_properties = {
                        "page_id": target_page,
                        "description": "",
                        "raw_page": json.dumps(img_to_base64(traj["after_screenshot"])),
                        "elements": 'no transition occurred',
                        "logical_screen_size": 'no transition occurred',
                        "goal": [traj["goal"]],
                        "task": json.dumps({task_name: [parent_step + 1]}),
                    }
                    self.create_node('Page', target_page_properties)
                    save_screenshot_to_current_dir(traj["after_screenshot"])
                    page2vector(target_page, "screenshot.png", self.vector_store)
                    os.remove("screenshot.png")
                    print(f"已删除临时文件: screenshot.png")
                    self.add_element_leads_to(matched_element_node, target_page, str(traj["converted_action"]))
                    parent_node = target_page
                    parent_step += 1
                    continue
            # 是否是起始动作
            if parent_node is None:
                parent_node = str(uuid4())
                if self.root_node is None:
                    self.root_node = parent_node
                    self.is_first_import = False
                    print(f"捕捉到根节点: {self.root_node} (step {step})")
                page_properties = {
                    "page_id": parent_node,
                    "description": "",
                    "raw_page": json.dumps(img_to_base64(traj["before_screenshot"])),
                    "elements": json.dumps(traj["ui_elements"]),
                    "logical_screen_size": traj["logical_screen_size"],
                    "goal": [traj["goal"]],
                    "task": json.dumps({task_name: [parent_step]}),
                }
                self.create_node('Page', page_properties)
                save_screenshot_to_current_dir(traj["before_screenshot"])
                page2vector(parent_node, "screenshot.png", self.vector_store)
                _safe_remove("screenshot.png")
            else:
                self.update_no_transition_page(parent_node, json.dumps(traj["ui_elements"]), traj["logical_screen_size"])
                self.append_task_to_node('Page', parent_node, task_name, parent_step)
                self.append_goal_to_node('Page', parent_node, traj["goal"])
            # 创建element节点及关系
            element_node = str(uuid4())
            element_properties = {
                "element_id": element_node,
                "converted_action": str(traj["converted_action"]),
                "target_element": json.dumps(traj["target_element"]) if traj["target_element"] is not None else None,
                "description": "",
                "logical_screen_size": traj["logical_screen_size"],
                "goal": traj["goal"],
                "action_output": [traj["action_output"]],
                "task": json.dumps({task_name: [parent_step]})
            }
            self.create_node('Element', element_properties)
            self.add_element_to_page(parent_node, element_node)
            # 判断是否为叶子节点，叶子节点不考虑重复
            if is_leaf and (step == len(trajectory_data) - 1 or (step == len(trajectory_data) - 2 and trajectory_data[-1]['after_screenshot'] is None)):
                target_page = str(uuid4())
                target_page_properties = {
                    "page_id": target_page,
                    "description": "",
                    "raw_page": json.dumps(img_to_base64(traj["after_screenshot"])),
                    "elements": 'no transition occurred',
                    "logical_screen_size": 'no transition occurred',
                    "goal": [traj["goal"]],
                    "task": json.dumps({task_name: [parent_step + 1]})
                }
                self.create_node('Page', target_page_properties)
                save_screenshot_to_current_dir(traj["after_screenshot"])
                page2vector(target_page, "screenshot.png", self.vector_store)
                _safe_remove("screenshot.png")
                self.add_element_leads_to(element_node, target_page, str(traj["converted_action"]))
                parent_node = target_page
                parent_step += 1
                continue
            # 判断after screenshot是否为kg已有节点
            save_screenshot_to_current_dir(traj["after_screenshot"])
            query_feature = extract_features(image_inputs='screenshot.png', model_name="resnet50")
            recall_pages = self.vector_store.query_similar(query_feature["features"], node_type=NodeType.PAGE)
            match = False
            target_page = None
            similar_pages = []
            if recall_pages[0][0] and recall_pages[1][0] > 0.98:
                for i, score in enumerate(recall_pages[1]):
                    print(recall_pages[0][i])
                    print(self.get_task_by_page_id(recall_pages[0][i]))
                    if score > 0.98 and (task_name not in self.get_task_by_page_id(recall_pages[0][i])):
                        similar_pages.append(recall_pages[0][i])

                if len(similar_pages) > 3:
                    similar_pages = similar_pages[0:3]
                if similar_pages != []:
                    print(f"Found potential {len(similar_pages)} matched page node.")
                    candidate_screenshots = self.get_screenshots_by_page_ids(similar_pages)
                    match, target_page = screenshot_comparison_anthropic_format(img_to_base64(traj['after_screenshot']), candidate_screenshots, goal=traj['goal'], action_output=traj['action_output'], usage=self.usage)
                    print(match, target_page)
            # 建立target node及关系
            if match == True and target_page is not None:
                print(f"connecting the element node to the existed page node: {similar_pages[target_page]}")
                self.add_element_leads_to(element_node, similar_pages[target_page], str(traj["converted_action"]))
                self.append_task_to_node('Page', similar_pages[target_page], task_name, parent_step + 1)
                self.append_goal_to_node('Page', similar_pages[target_page], traj["goal"])
                parent_node = similar_pages[target_page]
                parent_step += 1
            else:
            # 创建新的target node
                target_page = str(uuid4())
                target_page_properties = {
                    "page_id": target_page,
                    "description": "",
                    "raw_page": json.dumps(img_to_base64(traj["after_screenshot"])),
                    "elements": 'no transition occurred',
                    "logical_screen_size": 'no transition occurred',
                    "goal": [traj["goal"]],
                    "task": json.dumps({task_name: [parent_step + 1]})
                }
                self.create_node('Page', target_page_properties)
                save_screenshot_to_current_dir(traj["after_screenshot"])
                page2vector(target_page, "screenshot.png", self.vector_store)
                _safe_remove("screenshot.png")
                self.add_element_leads_to(element_node, target_page, str(traj["converted_action"]))
                parent_node = target_page
                parent_step += 1

        return parent_node, parent_step

    def create_action(self, properties: Dict[str, Any]) -> str:
        """Create an Action node (high-level/composite action)

        Args:
            properties: Dictionary containing action_id, description, element_sequence, etc.
        """
        required_fields = ["action_id"]  # Only action_id is required
        if not all(field in properties for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        # Convert element_sequence to JSON string
        if "element_sequence" in properties and isinstance(
            properties["element_sequence"], list
        ):
            properties["element_sequence"] = json.dumps(properties["element_sequence"])

        return self.create_node("Action", properties)

    def add_element_to_action(
        self,
        action_id: str,
        element_id: str,
        order: int,
        atomic_action: str,
        action_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create Action-COMPOSED_OF->Element relationship"""
        query = """
        MATCH (a:Action {action_id: $action_id})
        MATCH (e:Element {element_id: $element_id})
        MERGE (a)-[r:COMPOSED_OF {
            order: $order,
            atomic_action: $atomic_action,
            action_params: $action_params
        }]->(e)
        RETURN type(r) as rel_type
        """

        # Serialize action_params to JSON string
        if action_params:
            action_params = json.dumps(action_params)

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                action_id=action_id,
                element_id=element_id,
                order=order,
                atomic_action=atomic_action,
                action_params=action_params or "",  # Use empty string instead of None
            )
            return result.single() is not None

    def find_start_page_ids(self, require_nonempty: bool = True) -> List[str]:
        """Return Page IDs whose task map contains only step 0 entries."""
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (p:Page) RETURN p.page_id AS pid, p.task AS task")
            candidates = []
            for rec in result:
                if rec["task"] is None:
                    continue
                try:
                    task_map = json.loads(rec["task"])
                except Exception:
                    continue
                if _all_steps_zero_map(task_map, require_nonempty=require_nonempty):
                    candidates.append(rec["pid"])
        return candidates

    def get_all_task_names(self) -> Set[str]:
        """Return every task name currently referenced by Page or Element nodes."""
        task_names: Set[str] = set()
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n)
                WHERE (n:Page OR n:Element) AND n.task IS NOT NULL
                RETURN n.task AS task
            """)
            for rec in result:
                task_value = rec["task"]
                if isinstance(task_value, str):
                    try:
                        task_value = json.loads(task_value)
                    except Exception:
                        continue
                if isinstance(task_value, dict):
                    task_names.update(str(name) for name in task_value.keys())
                elif isinstance(task_value, list):
                    task_names.update(str(name) for name in task_value)
        return task_names

    def find_unique_start_page_id(self, require_nonempty: bool = True) -> str:
        """
        返回唯一的起始 page_id（所有任务的 steps 均为 0）。
        若找不到或不唯一则抛出 ValueError。
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (p:Page) RETURN p.page_id AS pid, p.task AS task")
            candidates = []
            for rec in result:
                if rec["task"] is not None:
                    if _all_steps_zero_map(json.loads(rec["task"]), require_nonempty=require_nonempty):
                        candidates.append(rec["pid"])

        if len(candidates) != 1:
            raise ValueError(f"期望恰好 1 个起始节点，实际找到 {len(candidates)} 个: {candidates}")
        return candidates[0]

    def rewrite_action_ids(self, dry_run=False):
        with self.driver.session(database=self.database) as session:
            # 读取所有 Action 的内部 id 与旧 action_id
            recs = session.run("MATCH (a:Action) RETURN id(a) AS nid, a.action_id AS old_id").data()
            rows = [{"nid": r["nid"], "old": r["old_id"], "new": str(uuid4())} for r in recs]

            if dry_run:
                # 仅返回映射，不写库
                return rows

            # 批量写回（用内部 id 匹配最稳妥）
            session.run("""
                UNWIND $rows AS row
                MATCH (a) WHERE id(a) = row.nid
                SET a.action_id = row.new
            """, rows=rows)

            # 返回旧->新映射，便于你备份或更新下游缓存
            return rows

    def find_task_paths_lazy(self, start_page_id: str, target_task: str, start_step: Optional[int] = None) -> Iterator[
        Dict[str, Any]]:
        """
        从给定起始 Page 开始，按 step 约束懒加载遍历所有任务轨迹。
        终止条件：当前 Page 在 step=s 下，没有任一“符合条件”的 Element（见下）即结束。
        推进：Element 满足 s ∈ e.task[target_task] 且存在 p2=LEADS_TO(e)，并且 s+1 ∈ p2.task[target_task] 时，推进到 p2 且 step+=1。
        """
        seen_path_ids: Set[str] = set()

        with self.driver.session(database=self.database) as session:
            # 取得起始页的可用起步 step 集
            rec = session.run(
                "MATCH (p:Page {page_id: $pid}) RETURN p.task AS task",
                pid=start_page_id
            ).single()
            if not rec:
                return  # 起始页不存在，直接空
            start_steps = [start_step] if start_step is not None else sorted(_steps_of(rec["task"], target_task))
            # 若未提供 step 且解析不到任何 step，则不产生任何路径
            print(start_steps)
            print(rec)
            for s0 in start_steps:
                stack: List[Dict[str, Any]] = [{
                    "page_id": start_page_id,
                    "step": int(s0),
                    "triplets": [],  # [{source_page, element, target_page}, ...]
                    "nodes": [("P", start_page_id)]
                }]

                while stack:
                    st = stack.pop()
                    cur_pid = st["page_id"]
                    cur_step = st["step"]

                    # 拉取当前 Page 的 task + 一跳邻居 Element 及其 next Page
                    rows = session.run(
                        """
                        MATCH (p:Page {page_id: $pid})
                        OPTIONAL MATCH (p)-[:HAS_ELEMENT]->(e:Element)
                        OPTIONAL MATCH (e)-[:LEADS_TO]->(np:Page)
                        RETURN p.task AS ptask,
                               e.element_id AS eid,
                               e.task AS etask,
                               np.page_id AS npid,
                               np.task AS ntask
                        """,
                        pid=cur_pid
                    )

                    # 第一次行读取时校验 Page 在该 step 上是否有效
                    page_checked = False
                    candidates: List[Tuple[str, str]] = []  # (eid, npid)
                    for row in rows:
                        if not page_checked:
                            page_checked = True
                            if cur_step not in _steps_of(row["ptask"], target_task):
                                # 当前 Page 在该 step 上无效——视为无法继续，产出已构建路径（若有）
                                if st["triplets"]:
                                    node_ids = [f"P:{st['nodes'][0][1]}"]
                                    for t in st["triplets"]:
                                        node_ids.append(f"E:{t['element']}")
                                        node_ids.append(f"P:{t['target_page']}")
                                    path_id = "->".join(node_ids)
                                    if path_id not in seen_path_ids:
                                        seen_path_ids.add(path_id)
                                        yield {
                                            "triplets": st["triplets"],
                                            "leaf_node_id": cur_pid,
                                            "path_id": path_id
                                        }
                                candidates = []  # 强制终止
                                break

                        eid = row["eid"]
                        if not eid:
                            continue

                        # “符合条件”的 Element：s ∈ etask，并且存在 next page 且 s+1 ∈ next page 的 steps
                        if (cur_step in _steps_of(row["etask"], target_task)
                                and row["npid"]
                                and (cur_step + 1) in _steps_of(row["ntask"], target_task)):
                            candidates.append((eid, row["npid"]))

                    # 终止：没有任何符合条件的 Element
                    if not candidates:
                        if st["triplets"]:
                            node_ids = [f"P:{st['nodes'][0][1]}"]
                            for t in st["triplets"]:
                                node_ids.append(f"E:{t['element']}")
                                node_ids.append(f"P:{t['target_page']}")
                            path_id = "->".join(node_ids)
                            if path_id not in seen_path_ids:
                                seen_path_ids.add(path_id)
                                yield {
                                    "triplets": st["triplets"],
                                    "leaf_node_id": cur_pid,
                                    "path_id": path_id
                                }
                        continue

                    # 扩展到 step+1
                    for eid, npid in reversed(candidates):
                        new_triplets = st["triplets"] + [{
                            "source_page": cur_pid,
                            "element": eid,
                            "target_page": npid
                        }]
                        new_nodes = st["nodes"] + [("E", eid), ("P", npid)]
                        stack.append({
                            "page_id": npid,
                            "step": cur_step + 1,
                            "triplets": new_triplets,
                            "nodes": new_nodes
                        })

#     def find_task_paths_lazy(self, target_task: str) -> Iterator[Dict[str, Any]]:
#         """
#         懒加载方式迭代返回路径，内存友好
#
#         Args:
#             target_task (str): 目标任务名称
#
#         Yields:
#             Dict: 单个路径信息
#         """
#
#         query = """
#         WITH $targetTask AS targetTask
#
#         MATCH (root:Page)
#         WHERE NOT EXISTS((root)<-[:LEADS_TO]-())
#         AND targetTask IN root.task
#
#         WITH root, targetTask
#         MATCH path = (root)-[:HAS_ELEMENT|LEADS_TO*]->(leaf:Page)
#         WHERE NOT EXISTS {
#     MATCH (leaf)-[:HAS_ELEMENT]->(laterElement:Element)
#     WHERE targetTask IN laterElement.task
# }
#         AND ALL(node IN nodes(path) WHERE targetTask IN COALESCE(node.task, []))
#
#         WITH path, leaf, nodes(path) AS pathNodes
#
#         // 创建唯一路径标识 - 使用reduce替代apoc.text.join
#         WITH pathNodes, leaf,
#              [node IN pathNodes |
#               CASE WHEN 'Page' IN labels(node) THEN 'P:' + node.page_id
#                    ELSE 'E:' + node.element_id END
#              ] AS nodeIds
#
#         WITH pathNodes, leaf,
#              REDUCE(pathId = '', nodeId IN nodeIds |
#                  CASE WHEN pathId = '' THEN nodeId
#                       ELSE pathId + '->' + nodeId END
#              ) AS pathId
#
#         // 构建三元组
#         WITH [node IN pathNodes WHERE 'Page' IN labels(node)] AS pages,
#              [node IN pathNodes WHERE 'Element' IN labels(node)] AS elements,
#              leaf, pathId
#
#         WITH pages, elements, leaf, pathId,
#              [i IN range(0, size(elements)-1) | {
#                  source_page: pages[i].page_id,
#                  element: elements[i].element_id,
#                  target_page: CASE WHEN i+1 < size(pages) THEN pages[i+1].page_id ELSE null END
#              }] AS allTriplets
#
#         WITH [t IN allTriplets WHERE t.target_page IS NOT NULL] AS triplets,
#              leaf, pathId
#
#         RETURN DISTINCT triplets, leaf.page_id AS leafNodeId, pathId
#         ORDER BY leafNodeId, pathId
#         """
#
#         seen_path_ids = set()
#
#         with self.driver.session(database=self.database) as session:
#             # 使用流式查询避免一次性加载所有结果
#             result = session.run(query, targetTask=target_task)
#
#             try:
#                 # 正确的方式：直接迭代结果
#                 for record in result:
#                     path_id = record['pathId']
#
#                     if path_id not in seen_path_ids:
#                         seen_path_ids.add(path_id)
#                         yield {
#                             'triplets': record['triplets'],
#                             'leaf_node_id': record['leafNodeId'],
#                             'path_id': path_id
#                         }
#             except Exception as e:
#                 print(f"Error during path iteration: {e}")
#             finally:
#                 # 确保资源释放
#                 result.consume()

    def find_all_task_paths(self, target_task: str):
        """
        一次性返回所有包含指定task的从根节点到叶子节点的路径

        Args:
            target_task (str): 目标任务名称

        Returns:
            List[Dict]: 所有路径信息的列表，每个元素包含triplets、leaf_node_id、path_id
        """

        query = """
        WITH $targetTask AS targetTask

        MATCH (root:Page)
        WHERE NOT EXISTS((root)<-[:LEADS_TO]-()) 
        AND targetTask IN root.task

        WITH root, targetTask
        MATCH path = (root)-[:HAS_ELEMENT|LEADS_TO*]->(leaf:Page)
       WHERE NOT EXISTS {
    MATCH (leaf)-[:HAS_ELEMENT]->(laterElement:Element) 
    WHERE targetTask IN laterElement.task
}
        AND ALL(node IN nodes(path) WHERE targetTask IN COALESCE(node.task, []))

        WITH path, leaf, nodes(path) AS pathNodes

        // 创建唯一路径标识
        WITH pathNodes, leaf,
             [node IN pathNodes | 
              CASE WHEN 'Page' IN labels(node) THEN 'P:' + node.page_id 
                   ELSE 'E:' + node.element_id END
             ] AS nodeIds

        WITH pathNodes, leaf, apoc.text.join(nodeIds, '->') AS pathId

        // 构建三元组
        WITH [node IN pathNodes WHERE 'Page' IN labels(node)] AS pages,
             [node IN pathNodes WHERE 'Element' IN labels(node)] AS elements,
             leaf, pathId

        WITH pages, elements, leaf, pathId,
             [i IN range(0, size(elements)-1) | {
                 source_page: pages[i].page_id,
                 element: elements[i].element_id,
                 target_page: CASE WHEN i+1 < size(pages) THEN pages[i+1].page_id ELSE null END
             }] AS allTriplets

        WITH [t IN allTriplets WHERE t.target_page IS NOT NULL] AS triplets, 
             leaf, pathId

        RETURN DISTINCT triplets, leaf.page_id AS leafNodeId, pathId
        ORDER BY leafNodeId, pathId
        """

        all_paths = []
        seen_path_ids = set()

        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, targetTask=target_task)

                # 一次性获取所有记录
                for record in result:
                    path_id = record['pathId']
                    seen_chain = set()
                    if path_id not in seen_path_ids:
                        chain_key = tuple(
                            (
                                t["source_page"],
                                t["element"],
                                t["target_page"],
                            )
                            for t in record['triplets']
                        )
                        seen_chain.add(chain_key)
                        seen_path_ids.add(path_id)
                        all_paths.append(seen_chain)

                print(f"Found {len(all_paths)} unique paths for task: {target_task}")
                return all_paths

            except Exception as e:
                print(f"Error during path retrieval: {e}")
                return []


    def _extract_task_id(self, file_path):
        """从文件名提取任务ID"""
        filename = file_path.stem  # 移除.pkl.zst
        if '_' in filename:
            return filename.split('_', 1)[1]
        return filename

    def enrich_path_with_properties(self, triplets_list: List[Dict[str, str]]) -> List[Dict[str, Dict]]:
        """
        根据三元组中的ID获取节点的完整属性

        Args:
            triplets_list: 三元组列表，包含source_page, element, target_page的ID

        Returns:
            List[Dict]: 包含完整节点属性的三元组列表
        """
        if not triplets_list:
            return []

        # 收集所有需要查询的ID
        page_ids = set()
        element_ids = set()

        for triplet in triplets_list:
            page_ids.add(triplet['source_page'])
            page_ids.add(triplet['target_page'])
            element_ids.add(triplet['element'])

        query = """
        WITH $pageIds AS pageIds, $elementIds AS elementIds

        // 获取所有Page节点属性
        OPTIONAL MATCH (p:Page)
        WHERE p.page_id IN pageIds
        WITH collect({id: p.page_id, properties: properties(p)}) AS pageMap, elementIds

        // 获取所有Element节点属性  
        OPTIONAL MATCH (e:Element)
        WHERE e.element_id IN elementIds
        WITH pageMap, collect({id: e.element_id, properties: properties(e)}) AS elementMap

        RETURN pageMap, elementMap
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, pageIds=list(page_ids), elementIds=list(element_ids))
            record = result.single()

            if not record:
                return []

            # 构建ID到属性的映射
            page_map = {item['id']: item['properties'] for item in record['pageMap']}
            element_map = {item['id']: item['properties'] for item in record['elementMap']}

            # 构建enriched三元组列表
            enriched_triplets = []
            for triplet in triplets_list:
                enriched_triplet = {
                    'source_page': page_map.get(triplet['source_page'], {}),
                    'element': element_map.get(triplet['element'], {}),
                    'target_page': page_map.get(triplet['target_page'], {})
                }
                enriched_triplets.append(enriched_triplet)

            return enriched_triplets

    def is_action_duplicate(self, components_preview: list) -> bool:
        """
        检查是否存在具有相似组件预览的Action节点
        相似定义：除最后一个三元组的target_node外，其他完全相同

        Args:
            components_preview: 组件预览列表

        Returns:
            bool: 如果存在相似的动作则返回True，否则返回False
        """
        import json

        try:
            # 获取所有Action节点的components_preview
            query = """
            MATCH (a:Action)
            RETURN a.components_preview AS components
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(query)

                for record in result:
                    if record["components"] is not None:
                        try:
                            # 解析存储的components_preview
                            existing_components = json.loads(record["components"])

                            # 检查是否相似
                            if self._is_components_similar(components_preview, existing_components):
                                return True
                        except json.JSONDecodeError:
                            continue

                return False

        except Exception as e:
            print(f"Error checking Action duplicate: {str(e)}")
            return False

    def find_action_by_element_sequence(self, element_sequence: list) -> Optional[str]:
        """Return an existing Action ID with the same ordered element sequence."""
        try:
            ordered_items = sorted(
                element_sequence,
                key=lambda item: int(item.get("order", 0)),
            )
            element_ids = [
                item.get("element_id")
                for item in ordered_items
                if item.get("element_id")
            ]
            if not element_ids:
                return None

            query = """
            MATCH (a)
            WHERE 'Action' IN labels(a)
            MATCH (a)-[r:COMPOSED_OF]->(e:Element)
            WITH a, e.element_id AS eid, toInteger(r.order) AS ord
            ORDER BY a.action_id, ord
            WITH a, collect(eid) AS action_element_ids
            WHERE action_element_ids = $element_ids
            RETURN a.action_id AS action_id
            LIMIT 1
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(query, element_ids=element_ids)
                record = result.single()
                return record["action_id"] if record else None

        except Exception as e:
            print(f"Error checking Action element-sequence duplicate: {str(e)}")
            return None

    def is_action_duplicate_by_elements(self, element_sequence: list) -> bool:
        """Check whether an Action already has the same ordered element sequence."""
        return self.find_action_by_element_sequence(element_sequence) is not None

    def get_chain_by_chain_id(self, chain_id: List[str]):
        """根据三元组ID列表查找匹配的任务链

        直接使用三元组中的page/element ID作为查询条件，不依赖于起始点或路径完整性

        Args:
            chain_id: 三元组字符串列表，如 ['(b8a2df09..., f88160bf..., d86141b1...)']

        Returns:
            List[Dict]: 匹配的三元组链，每个三元组包含source_page、element、target_page和action
        """
        if not chain_id:
            return []

        try:
            # 构建每个三元组的查询条件
            triplet_conditions = []

            for triplet_str in chain_id:
                # 解析三元组字符串
                clean_str = triplet_str.strip("()").replace("...", "")
                parts = [part.strip() for part in clean_str.split(",")]

                if len(parts) >= 3:
                    triplet_conditions.append({
                        "source": parts[0],
                        "element": parts[1],
                        "target": parts[2]
                    })

            # 使用UNWIND处理多个三元组条件
            query = """
            UNWIND $triplets AS triplet
            MATCH (source:Page)-[:HAS_ELEMENT]->(element:Element)-[action:LEADS_TO]->(target:Page)
            WHERE source.page_id STARTS WITH triplet.source
            AND element.element_id STARTS WITH triplet.element
            AND target.page_id STARTS WITH triplet.target
            RETURN collect({source: source, element: element, target: target, action: action}) AS matches
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(query, triplets=triplet_conditions)
                record = result.single()

                if record and record["matches"]:
                    # 转换为所需的输出格式
                    additional_targets = []
                    chain = []
                    for match in record["matches"]:
                        triplet = {
                            "source_page": dict(match["source"]),
                            "element": dict(match["element"]),
                            "target_page": dict(match["target"]),
                            "action": dict(match["action"])
                        }
                        chain.append(triplet)

                    if chain and len(triplet_conditions) > 0:
                        last_triplet_condition = triplet_conditions[-1]
                        known_target_id = last_triplet_condition["target"]

                        # 查询最后一个element的所有LEADS_TO关系
                        additional_query = """
                                        MATCH (source:Page)-[:HAS_ELEMENT]->(element:Element)-[action:LEADS_TO]->(target:Page)
                                        WHERE source.page_id STARTS WITH $source
                                        AND element.element_id STARTS WITH $element
                                        AND NOT target.page_id STARTS WITH $known_target
                                        RETURN target
                                        """

                        additional_result = session.run(additional_query,
                                                        source=last_triplet_condition["source"],
                                                        element=last_triplet_condition["element"],
                                                        known_target=known_target_id)

                        for record in additional_result:
                            additional_targets.append(dict(record["target"]))


                    return chain, additional_targets

                return [], []

        except Exception as e:
            print(f"Error getting chain by chain_id: {str(e)}")
            return [], []

    def _is_components_similar(self, components1: list, components2: list) -> bool:
        """
        判断两个组件预览列表是否相似
        相似定义：除最后一个三元组的target_node外，其他完全相同

        Args:
            components1: 第一个组件预览列表
            components2: 第二个组件预览列表

        Returns:
            bool: 是否相似
        """
        if len(components1) != len(components2) or len(components1) == 0:
            return False

        # 检查除最后一个元素外的所有元素是否完全相同
        for i in range(len(components1) - 1):
            if components1[i] != components2[i]:
                return False

        # 对于最后一个元素，只比较前两个部分（source和element）
        # 三元组格式：(source, element, target)
        last1 = components1[-1]
        last2 = components2[-1]

        # 提取前两个逗号分隔的部分进行比较
        parts1 = last1.split(',')
        parts2 = last2.split(',')

        if len(parts1) >= 2 and len(parts2) >= 2:
            # 只比较source和element部分，忽略target
            return parts1[0] == parts2[0] and parts1[1] == parts2[1]
        else:
            # 如果格式不符合预期，直接比较整个字符串
            return last1 == last2

    async def chain_understand(self, task_name, db):
        all_results = []
        if self.root_node is None:
            print("No root page input")
            return None
        piece = 0
        for path_info in self.find_task_paths_lazy(start_page_id=self.root_node, target_task=task_name):
            piece += 1
            print(piece)
            print(path_info)
            print(self.root_node)
            print(f"处理路径到叶子节点: {path_info['leaf_node_id']}")
            triplets = path_info['triplets']
            triplets = self.enrich_path_with_properties(triplets)
            processed_triplets = await process_and_update_chain(task_name, triplets, db)
            if not processed_triplets:
                print("❌ No processable triplets found")
                continue

            print(f"✓ Successfully processed {len(processed_triplets)} triplets")
            all_results.append({
                'leaf_node_id': path_info['leaf_node_id'],
                'path_id': path_info['path_id'],
                'processed_triplets': processed_triplets
            })
        return all_results

    def optimize_paths_with_action_groups(self, root_path):
        tasks = find_all_task_folders(root_path)
        action_chains_list = []
        start_page = self.find_unique_start_page_id()
        for task in tasks:
            # if task.name != "ExpenseAddSingle1":
            #     continue
            for path_info in self.find_task_paths_lazy(start_page_id=start_page, target_task=task.name):
                triplets = path_info['triplets']
                triplets = self.enrich_path_with_properties(triplets)
                action_chain = self._optimize_single_path_with_action_groups(triplets)
                action_summary_chain = self._simplify_path_nodes(action_chain)
                action_chains_list.append(action_summary_chain)
        save_paths_to_json(action_chains_list, root_path)
        return action_chains_list

    # def find_gold_paths(self, root_path):
    #     tasks = find_all_task_folders(root_path)
    #     task_gold_path = []
    #     start_page = self.find_unique_start_page_id()
    #     for task in tasks:
    #         # if task.name != "ExpenseAddSingle1":
    #         #     continue
    #         action_summary_chain = []
    #         index = 0
    #         for path_info in self.find_task_paths_lazy(start_page_id=start_page, target_task=task.name):
    #             triplets = path_info['triplets']
    #             triplets = self.enrich_path_with_properties(triplets)
    #             index += 1
    #             if index == 1:
    #                 gold_path = triplets
    #             else:
    #                 if len(triplets) <= len(gold_path):
    #                     continue
    #                 else:
    #                     gold_path = triplets
    #             action_chain = self._optimize_single_path_with_action_groups(gold_path)
    #             action_summary_chain = self._simplify_path_nodes(action_chain)
    #         task_gold_path.append({'task': task.name, 'path': action_summary_chain})
    #     # save_paths_to_json(task_gold_path, root_path)
    #     return task_gold_path
    def find_gold_paths(
        self,
        root_path,
        task_names: Optional[List[str]] = None,
        use_action_groups: bool = True,
        app: Optional[str] = None,
        success: bool = True,
    ):
        if task_names is None:
            tasks = [task.name for task in find_all_task_folders(root_path)]
        else:
            tasks = []
            seen_tasks = set()
            for task_name in task_names:
                if task_name not in seen_tasks:
                    tasks.append(task_name)
                    seen_tasks.add(task_name)

        task_gold_path = []
        start_page = self.find_unique_start_page_id()
        for task_name in tasks:
            action_summary_chain = []
            task_goal = ""
            for path_info in self.find_task_paths_lazy(start_page_id=start_page, target_task=task_name):
                triplets = path_info['triplets']
                triplets = self.enrich_path_with_properties(triplets)
                if not task_goal:
                    task_goal = self._extract_goal_from_path(triplets)
                if use_action_groups:
                    triplets = self._optimize_single_path_with_action_groups(triplets)
                action_summary_chain.append(self._simplify_path_nodes(triplets))

            entry = {
                'task': task_name,
                'path': action_summary_chain,
                'goal': task_goal,
                'success': success,
            }
            if app is not None:
                entry['app'] = app
            task_gold_path.append(entry)
        return task_gold_path

    @staticmethod
    def _extract_goal_from_path(path: List[Dict]) -> str:
        for triplet in path:
            for node_key in ("source_page", "element", "target_page"):
                node = triplet.get(node_key, {})
                goal = node.get("goal")
                if isinstance(goal, list) and goal:
                    return str(goal[0])
                if isinstance(goal, str) and goal:
                    return goal
        return ""


    def _optimize_single_path_with_action_groups(self, path: List[Dict]) -> List[Dict]:
        """
        优化单条路径，检查并替换动作组
        """
        # 1. 获取路径中所有的element_id
        path_element_ids = set()
        for triplet in path:
            element_id = triplet["element"].get("element_id")
            if element_id:
                path_element_ids.add(element_id)

        # 2. 查找所有可能匹配的Action节点
        matching_actions = self._find_matching_action_groups(path_element_ids)

        # 3. 按element数量降序排序，优先处理包含更多element的Action
        matching_actions.sort(key=lambda x: len(x["elements"]), reverse=True)

        # 4. 依次替换匹配的Action节点
        optimized_path = path.copy()
        used_element_ids = set()

        for action_info in matching_actions:
            action_element_ids = set(action_info["elements"])

            # 检查是否有重叠（避免重复替换）
            if action_element_ids.intersection(used_element_ids):
                continue

            # 执行替换
            optimized_path = self._replace_elements_with_action(
                optimized_path,
                action_element_ids,
                action_info["action_node"]
            )
            used_element_ids.update(action_element_ids)

        return optimized_path

    def _find_matching_action_groups(self, path_element_ids: set) -> List[Dict]:
        """
        查找路径中element_ids完全匹配的Action节点
        """
        query = """
        MATCH (action:Action)-[:COMPOSED_OF]->(element:Element)
        WITH action, collect(element.element_id) AS action_elements
        WHERE ALL(elem_id IN action_elements WHERE elem_id IN $path_elements)
        RETURN action, action_elements
        """

        matching_actions = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, path_elements=list(path_element_ids))

            for record in result:
                matching_actions.append({
                    "action_node": dict(record["action"]),
                    "elements": record["action_elements"]
                })

        return matching_actions

    def _simplify_path_nodes(self, path: List[Dict]) -> List[Dict]:
        """
        简化路径中的节点，只保留function_summary属性
        """
        simplified_path = []

        for triplet in path:
            if 'element_id' in triplet['element']:
                simplified_triplet = {
                    "source_page": f"{triplet["source_page"].get("function_summary", "Unknown source")}_${triplet["source_page"].get('page_id', 'Unknown page')}$",
                    "element": f"{triplet["element"].get("function_summary", "Unknown element")}_$Element_{triplet['element'].get('element_id', 'Unknown element')}$",
                    "target_page": f"{triplet["target_page"].get("function_summary", "Unknown source")}_${triplet["target_page"].get('page_id', 'Unknown page')}$",
                }
            else:
                simplified_triplet = {
                    "source_page": f"{triplet["source_page"].get("function_summary", "Unknown source")}_${triplet["source_page"].get('page_id', 'Unknown page')}$",
                    "element": f"{triplet["element"].get("function", "Unknown element")}_$Action_{triplet['element'].get('action_id', 'Unknown element')}$",
                    "target_page": f"{triplet["target_page"].get("function_summary", "Unknown source")}_${triplet["target_page"].get('page_id', 'Unknown page')}$",
                }
            simplified_path.append(simplified_triplet)

        return simplified_path

    def _replace_elements_with_action(self, path: List[Dict], element_ids_to_replace: set, action_node: Dict) -> List[
        Dict]:
        """
        在路径中用Action节点替换指定的element节点序列
        """
        new_path = []
        i = 0

        while i < len(path):
            current_element_id = path[i]["element"].get("element_id")

            if current_element_id in element_ids_to_replace:
                # 找到要替换的element序列的开始
                sequence_start = i

                # 收集连续的要替换的element
                while i < len(path) and path[i]["element"].get("element_id") in element_ids_to_replace:
                    i += 1

                # 创建Action节点的三元组
                action_triplet = {
                    "source_page": path[sequence_start]["source_page"],
                    "element": action_node,  # 用Action节点替换element
                    "target_page": path[i - 1]["target_page"] if i > 0 else path[sequence_start]["target_page"],
                    "action": {
                        "action_name": "execute_action_group",
                        "action_type": "action_group"
                    }
                }

                new_path.append(action_triplet)
            else:
                # 保持原有的element三元组
                new_path.append(path[i])
                i += 1

        return new_path


    def close(self):
        self.driver.close()





def save_paths_to_json(paths: List[List[Dict]], root_path, filename: str = None) -> str:
    """
    将路径数据保存为JSON文件

    Args:
        paths: 路径列表，每条路径包含三元组字典列表
        filename: 保存的文件名，如果为None则自动生成

    Returns:
        保存的文件名
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"ui_automation_paths_{timestamp}_{os.path.basename(root_path)}.json"
        filename = f"ui_automation_paths_{os.path.basename(root_path)}_denoise.json"

    # 转换格式
    converted_chains = []
    for path in paths:
        converted_path = []
        for triplet in path:
            # 将字典格式转换为列表格式 [source_page, element, target_page]
            converted_triplet = [
                triplet.get("source_page", "Unknown source page"),
                triplet.get("element", "Unknown element"),
                triplet.get("target_page", "Unknown target page")
            ]
            converted_path.append(converted_triplet)
        converted_chains.append(converted_path)

    # 构建JSON结构
    json_data = {
        "total_chains": len(converted_chains),
        "chains": converted_chains
    }

    file_path = os.path.join("path_denoise", filename)

    # 保存到文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(converted_chains)} paths to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        raise


# def save_paths_to_json(paths: List[List[Dict]], root_path, filename: str = None) -> str:
#     """
#     将路径数据保存为JSON文件
#
#     Args:
#         paths: 路径列表，每条路径包含三元组字典列表
#         filename: 保存的文件名，如果为None则自动生成
#
#     Returns:
#         保存的文件名
#     """
#     if filename is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         # filename = f"ui_automation_paths_{timestamp}_{os.path.basename(root_path)}.json"
#         filename = f"Gold_paths_{os.path.basename(root_path)}.json"
#
#     # 转换格式
#     converted_chains = []
#     for path in paths:
#         converted_path = []
#         for triplet in path:
#             # 将字典格式转换为列表格式 [source_page, element, target_page]
#             converted_triplet = [
#                 triplet.get("source_page", "Unknown source page"),
#                 triplet.get("element", "Unknown element"),
#                 triplet.get("target_page", "Unknown target page")
#             ]
#             converted_path.append(converted_triplet)
#         converted_chains.append(converted_path)
#
#     # 构建JSON结构
#     json_data = {
#         "total_chains": len(converted_chains),
#         "chains": converted_chains
#     }
#
#     # 保存到文件
#     try:
#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump(json_data, f, indent=2, ensure_ascii=False)
#         print(f"Successfully saved {len(converted_chains)} paths to {filename}")
#         return filename
#     except Exception as e:
#         print(f"Error saving to JSON: {e}")
#         raise


def img_to_base64(img):
    img_base64 = pil_to_webp_base64(Image.fromarray(img))
    return img_base64


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

def page2vector(
    page_id: str,
    page_path: str,
    vector_store: VectorStore,
) -> bool:
    """
    Process the visual features of the element and store them in the vector database

    Parameters:
        ID: str, the ID of the element in the JSON
        element_id: str, the unique ID of the element in the graph database
        elements_json: str, the element JSON string
        page_path: str, the page image path
        vector_store: VectorStore, the vector database instance

    Returns:
        bool: Whether the storage was successful
    """
    try:
        # 1. Extract visual features
        features = extract_features(page_path, "resnet50")
        # 2. Prepare vector data
        vector_data = VectorData(
            id=page_id,
            values=features["features"][0],
            metadata={
                "id": page_id,
            },
            node_type=NodeType.PAGE,
        )

        # 5. Store in vector database
        return vector_store.upsert_batch([vector_data])

    except Exception as e:
        print(f"Error processing page vector: {str(e)}")
        return False

def save_screenshot_to_current_dir(before_screenshot, filename="screenshot.png"):
    """将NumPy数组截图保存到当前目录"""
    Image.fromarray(before_screenshot).save(filename)
    return filename

def _safe_remove(path):
    """Remove file, ignore errors (Windows file lock, etc.)."""
    try:
        os.remove(path)
    except OSError:
        pass


if __name__ == "__main__":
    main()

