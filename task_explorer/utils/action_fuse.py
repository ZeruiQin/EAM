import config
import os
import time
from datetime import datetime
from pathlib import Path
from graph_db import Neo4jDatabase
import asyncio
from traj_to_kg import TrajectoryToNeo4jImporter, find_all_task_folders, _all_steps_zero_map
import json
import numpy as np
from graph_db import ActionMergeAnalyzer
from node_summary import LLMFunctionSummarizer, LLMElementName
import logging
from typing import Dict, Any, List, Optional, Union, Set
from chain_evolve import format_chain_operations, extract_element_details, extract_reasoning_results, create_action_generation_chain, create_action_node_in_db, create_action_element_relations
from langchain_community.callbacks import get_openai_callback




def demonstrate_BPE(task_chains_input):
    """
    演示如何使用ActionMergeAnalyzer

    Args:
        task_chains_input: 任务链输入，可以是列表、NumPy数组等
    """
    try:
        # 创建分析器并运行
        analyzer = ActionMergeAnalyzer(verbose=True)
        results = analyzer.analyze(task_chains_input, num_merges=20, min_freq=2)


        # 打印结果
        print("\n=== High Level Actions Found ===")
        print(f"Total high-level actions identified: {results['total_found']}")

        print("\nTop High-Level Actions by Frequency:")
        for name, action in list(results['high_level_actions'].items())[:10]:
            print(f"\n{name}:")
            print(f"  Frequency: {action['frequency']}")
            print(f"  Level: {action['level']} (higher means more complex)")
            print(f"  Components: {action['num_components']} actions")
            print(f"  Preview: {action['components_preview']}")
        return results

    except Exception as e:
        import traceback
        print(f"Error during demonstration: {str(e)}")
        print(traceback.format_exc())





def Action_envolving(root_path, database, index):
    """演示如何使用该功能"""

    # 配置Neo4j连接（请替换为实际连接信息）
    db = TrajectoryToNeo4jImporter(
        uri=config.Neo4j_URI,
        auth=config.Neo4j_AUTH,
        database=database,
        index=index
    )



    # 获取所有other_info中step为0的Page节点
    tasks = find_all_task_folders(root_path)
    start_page = db.find_unique_start_page_id()
    print(start_page)
    task_chains = []
    for task in tasks:
        # if i > 1:
        #     break
        all_paths = []
        seen_path_ids = set()
        print(f"开始提取任务：{task}")

        for task_chain in db.find_task_paths_lazy(start_page_id=start_page, target_task=task.name):
            path_id = task_chain['path_id']
            # print(task_chain)
            seen_chain = set()
            if path_id not in seen_path_ids:
                chain_key = tuple(
                    (
                        t["source_page"],
                        t["element"],
                        t["target_page"],
                    )
                    for t in task_chain['triplets']
                )
                seen_chain.add(chain_key)
                seen_path_ids.add(path_id)
                all_paths.append(seen_chain)

        print(f"Found {len(all_paths)} unique paths for task: {task.name}")
        task_chains.append(all_paths)
        # chains = db.find_all_task_paths(task.name)
        # chain_list.append(chains)
    task_chains = np.concatenate(task_chains, axis=0)

    results = demonstrate_BPE(task_chains)
    print(results)
    for name, result in list(results['high_level_actions'].items()):

        chain = result['components_preview']

        if not db.is_action_duplicate(chain):
            print("generating new action node ... ")
            print(f"Chain length: {len(chain)}")
            print(chain[-1])
            action_chain, additional_targets = db.get_chain_by_chain_id(chain)
            # print(additional_targets)
            print(f"Retrieved action_chain length: {len(action_chain)}")
            action_data = generate_action_node(action_chain, additional_targets)
            if not action_data:
                print(f"Skipping action {name}: failed to generate action node data")
                continue
            if db.is_action_duplicate_by_elements(action_data.get("element_sequence", [])):
                print("pass existing action node by COMPOSED_OF element sequence")
                continue
            action_data = create_action_node_in_db(action_data, chain, db)
            if not action_data:
                print("Skipping action relation creation because no new Action node was created")
                continue
            relations_success = create_action_element_relations(action_data, db)

            if not relations_success:
                print("Some element relations creation failed")

            print(
                f"Successfully completed chain evolution, created high-level action node: {action_data['name']} (ID: {action_data['action_id']})"
            )
        else:
            print('pass existing action node')
            continue





    # 关闭连接
    db.close()


def Action_envolving_from_tasks(database, index, task_names, num_merges=20, min_freq=2):
    """Run action fuse using task names read from the current KG."""
    db = TrajectoryToNeo4jImporter(
        uri=config.Neo4j_URI,
        auth=config.Neo4j_AUTH,
        database=database,
        index=index
    )

    try:
        task_names = sorted(set(task_names))
        if not task_names:
            print(f"No KG tasks found for {database}, skipping action evolution.")
            return {"total_found": 0, "high_level_actions": {}}

        start_page = db.find_unique_start_page_id()
        print(f"Action evolution start page for {database}: {start_page}")
        task_chains = []

        for task_name in task_names:
            all_paths = []
            seen_path_ids = set()
            print(f"Start extracting KG task: {task_name}")

            for task_chain in db.find_task_paths_lazy(start_page_id=start_page, target_task=task_name):
                path_id = task_chain["path_id"]
                if path_id in seen_path_ids:
                    continue
                chain_key = tuple(
                    (
                        t["source_page"],
                        t["element"],
                        t["target_page"],
                    )
                    for t in task_chain["triplets"]
                )
                all_paths.append({chain_key})
                seen_path_ids.add(path_id)

            print(f"Found {len(all_paths)} unique paths for task: {task_name}")
            task_chains.extend(all_paths)

        if not task_chains:
            print(f"No task paths found for {database}, skipping BPE action evolution.")
            return {"total_found": 0, "high_level_actions": {}}

        analyzer = ActionMergeAnalyzer(verbose=True)
        results = analyzer.analyze(task_chains, num_merges=num_merges, min_freq=min_freq)
        print("\n=== High Level Actions Found ===")
        print(f"Total high-level actions identified: {results['total_found']}")

        for name, result in list(results["high_level_actions"].items()):
            chain = result["components_preview"]
            if db.is_action_duplicate(chain):
                print("pass existing action node")
                continue

            print("generating new action node ... ")
            print(f"Chain length: {len(chain)}")
            action_chain, additional_targets = db.get_chain_by_chain_id(chain)
            print(f"Retrieved action_chain length: {len(action_chain)}")
            action_data = generate_action_node(action_chain, additional_targets)
            if not action_data:
                print(f"Skipping action {name}: failed to generate action node data")
                continue
            if db.is_action_duplicate_by_elements(action_data.get("element_sequence", [])):
                print("pass existing action node by COMPOSED_OF element sequence")
                continue

            action_data = create_action_node_in_db(action_data, chain, db)
            if not action_data:
                print("Skipping action relation creation because no new Action node was created")
                continue
            relations_success = create_action_element_relations(action_data, db)
            if not relations_success:
                print("Some element relations creation failed")

            print(
                f"Successfully completed chain evolution, created high-level action node: {action_data['name']} (ID: {action_data['action_id']})"
            )

        return results

    finally:
        db.close()


def Postprocess_KG(database, index, task_names,
                   mining_usage=None, mining_duration=None,
                   token_log_path=None, package=None,
                   dry_run=False):
    """Run action fuse and function-summary generation for current KG tasks."""
    if mining_usage is None:
        mining_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if mining_duration is None:
        mining_duration = [0.0]

    task_names = sorted(set(task_names))
    print("*****************************************************************************************")
    print(f"Postprocessing {database}: {len(task_names)} KG tasks")
    print(f"Tasks: {task_names}")
    if dry_run:
        print("Dry run enabled; no action fuse or node summary will be executed.")
        return

    t0 = time.time()
    with get_openai_callback() as cb:
        Action_envolving_from_tasks(database=database, index=index, task_names=task_names)
    elapsed = time.time() - t0
    mining_usage["prompt_tokens"] += cb.prompt_tokens
    mining_usage["completion_tokens"] += cb.completion_tokens
    mining_duration[0] += elapsed
    _append_token_log(token_log_path, package or database, "ALL", "action_evolution",
                      cb.prompt_tokens, cb.completion_tokens, elapsed,
                      {"prompt_tokens": 0, "completion_tokens": 0},
                      mining_usage, 0.0, mining_duration[0])

    t0 = time.time()
    summarizer = LLMFunctionSummarizer(
        config.Neo4j_URI, config.Neo4j_AUTH, database, config.LLM_API_KEY, False
    )
    try:
        summarizer.get_node_statistics()
        summarizer.process_all_nodes()
        summarizer.get_node_statistics()
    finally:
        summarizer.close()
    elapsed = time.time() - t0
    mining_duration[0] += elapsed
    _append_token_log(token_log_path, package or database, "ALL", "node_summary",
                      0, 0, elapsed,
                      {"prompt_tokens": 0, "completion_tokens": 0},
                      mining_usage, 0.0, mining_duration[0])


def get_common_descriptions(chain: List[Dict[str, Any]]) -> List[str]:
    """
    比较任务链中第一页与最后一页的任务描述，返回共有的描述

    Args:
        chain: 任务链

    Returns:
        第一页和最后一页共有的任务描述列表
    """

    if not chain:
        return []

    # 找出第一个三元组的source_page和最后一个三元组的target_page
    first_page = chain[0].get("source_page") if chain else None
    last_page = chain[-1].get("target_page") if chain else None

    # 确保两个页面都存在
    if not first_page or not last_page:
        return []

    # 辅助函数：从页面中提取任务描述
    def extract_descriptions(page):
        descriptions = set()
        if "other_info" in page:
            other_info = page["other_info"]

            # 解析JSON字符串
            if isinstance(other_info, str):
                try:
                    other_info = json.loads(other_info)
                except:
                    return set()

            # 提取描述
            if isinstance(other_info, list):
                for item in other_info:
                    if isinstance(item, dict) and "task_info" in item:
                        desc = item.get("task_info", {}).get("description")
                        if desc:
                            descriptions.add(desc)
        return descriptions

    # 提取两个页面的描述并计算交集
    first_descriptions = extract_descriptions(first_page)
    last_descriptions = extract_descriptions(last_page)

    common_descriptions = first_descriptions.intersection(last_descriptions)

    return list(common_descriptions)



def generate_action_node(chain, additional_targets) -> Optional[Dict[str, Any]]:
    """Generate high-level action node content.

    Args:
        chain: Triplet chain

    Returns:
        Generated high-level action node content (dictionary)
    """
    # Create generation chain
    generation_chain = create_action_generation_chain()

    # Prepare generation input
    # task_description = get_common_descriptions(chain)
    task_description = "unknown_task"
    chain_operations = format_chain_operations(chain, additional_targets)
    element_details = extract_element_details(chain)
    reasoning_results = extract_reasoning_results(chain)

    generation_input = {
        "task_description": task_description,
        "chain_operations": chain_operations,
        "element_details": element_details,
        "reasoning_results": reasoning_results,
    }

    try:
        # Execute generation - note that this returns a dictionary rather than a Pydantic object
        generation_result = generation_chain.invoke(generation_input)

        # Check if the returned result is a valid dictionary
        if isinstance(generation_result, dict) and "action_id" in generation_result:
            return generation_result
        else:
            print(
                f"Warning: The format of the generation result returned by LLM is incorrect: {generation_result}"
            )
            return None
    except Exception as e:
        print(f"Error generating high-level action node: {str(e)}")
        return None


def _append_token_log(log_path, package, task, stage,
                      prompt_tokens, completion_tokens, duration,
                      kg_usage, mining_usage, kg_duration, mining_duration):
    """Append a single token/duration log entry to the JSONL file."""
    if log_path is None:
        return
    entry = {
        "timestamp": datetime.now().isoformat(),
        "package": package,
        "task": task,
        "stage": stage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "duration_seconds": round(duration, 1),
        "cumulative_kg_prompt": kg_usage["prompt_tokens"],
        "cumulative_kg_completion": kg_usage["completion_tokens"],
        "cumulative_mining_prompt": mining_usage["prompt_tokens"],
        "cumulative_mining_completion": mining_usage["completion_tokens"],
        "cumulative_kg_duration": round(kg_duration, 1),
        "cumulative_mining_duration": round(mining_duration, 1),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _task_names_from_value(task_value) -> Set[str]:
    """Extract task names from the JSON/list task property used in KG nodes."""
    if task_value is None:
        return set()
    if isinstance(task_value, str):
        try:
            task_value = json.loads(task_value)
        except Exception:
            return set()
    if isinstance(task_value, dict):
        return {str(name) for name in task_value.keys()}
    if isinstance(task_value, list):
        return {str(name) for name in task_value}
    return set()


def get_existing_task_names(database, index=None) -> Set[str]:
    """Read every task name currently referenced by Page or Element nodes."""
    db = Neo4jDatabase(uri=config.Neo4j_URI, auth=config.Neo4j_AUTH, database=database, index=index or database)
    try:
        task_names: Set[str] = set()
        with db.driver.session(database=database) as session:
            result = session.run("""
                MATCH (n)
                WHERE (n:Page OR n:Element) AND n.task IS NOT NULL
                RETURN n.task AS task
            """)
            for rec in result:
                task_names.update(_task_names_from_value(rec["task"]))
        return task_names
    finally:
        db.close()


def get_start_page_ids(database, index=None) -> List[str]:
    """Read start Page IDs whose task map contains only step 0 entries."""
    db = Neo4jDatabase(uri=config.Neo4j_URI, auth=config.Neo4j_AUTH, database=database, index=index or database)
    try:
        candidates = []
        with db.driver.session(database=database) as session:
            result = session.run("MATCH (p:Page) RETURN p.page_id AS pid, p.task AS task")
            for rec in result:
                task_value = rec["task"]
                if task_value is None:
                    continue
                if isinstance(task_value, str):
                    try:
                        task_value = json.loads(task_value)
                    except Exception:
                        continue
                if _all_steps_zero_map(task_value):
                    candidates.append(rec["pid"])
        return candidates
    finally:
        db.close()


def KG_Construction(database, index, package, base_path,
                    successful_tasks=None,
                    kg_usage=None, mining_usage=None,
                    kg_duration=None, mining_duration=None,
                    token_log_path=None):
    if kg_usage is None:
        kg_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if mining_usage is None:
        mining_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if kg_duration is None:
        kg_duration = [0.0]
    if mining_duration is None:
        mining_duration = [0.0]

    # Neo4j连接配置
    importer = TrajectoryToNeo4jImporter(
        uri=config.Neo4j_URI,
        auth=config.Neo4j_AUTH,
        database=database,
        index=index,
        usage=kg_usage
    )
    db = Neo4jDatabase(uri=config.Neo4j_URI, auth=config.Neo4j_AUTH, database=database, index=index)

    try:
        root_path = os.path.join(base_path, package)
        tasks = find_all_task_folders(root_path)
        if successful_tasks is not None:
            tasks = [t for t in tasks if t.name in successful_tasks]
        if not tasks:
            print(f"No automatic tasks found for {package}, skipping.")
            return

        # Stage 1: KG Construction (DFS traverse and import)
        print("*****************************************************************************************")
        print(f"Start building Knowledge Graph for {database} ({len(tasks)} tasks).... ")
        for task in tasks:
            print(f"开始遍历任务：{task}")
            prev_prompt = kg_usage["prompt_tokens"]
            prev_completion = kg_usage["completion_tokens"]
            t0 = time.time()
            importer.dfs_traverse_and_import(task)
            elapsed = time.time() - t0
            kg_duration[0] += elapsed
            delta_prompt = kg_usage["prompt_tokens"] - prev_prompt
            delta_completion = kg_usage["completion_tokens"] - prev_completion
            _append_token_log(token_log_path, package, task.name, "kg_construction",
                              delta_prompt, delta_completion, elapsed,
                              kg_usage, mining_usage, kg_duration[0], mining_duration[0])
        print("Building Successful ! ")

        # Stage 2: Chain Understanding (knowledge mining)
        print("*****************************************************************************************")
        print("Start chain understanding .... ")
        for task in tasks:
            print(f"开始遍历任务：{task}")
            t0 = time.time()
            with get_openai_callback() as cb:
                result = asyncio.run(importer.chain_understand(task.name, db))
            elapsed = time.time() - t0
            mining_usage["prompt_tokens"] += cb.prompt_tokens
            mining_usage["completion_tokens"] += cb.completion_tokens
            mining_duration[0] += elapsed
            print(f"任务 {task.name} 处理结果: {len(result) if result else 0} 个chains")
            _append_token_log(token_log_path, package, task.name, "chain_understanding",
                              cb.prompt_tokens, cb.completion_tokens, elapsed,
                              kg_usage, mining_usage, kg_duration[0], mining_duration[0])
        print("Understanding Successful ! ")

    finally:
        importer.close()


