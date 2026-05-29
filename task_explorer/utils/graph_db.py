from neo4j import GraphDatabase
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict, Counter
import re
import copy
import json
from itertools import chain
import numpy as np
import pinecone
from pinecone import Pinecone
from task_explorer.utils.vector_db import VectorStore, VectorData, NodeType
import config


class Neo4jDatabase:
    def __init__(self, uri: str, auth: tuple, database: str, index: str):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.verify_connectivity()
        self.database = database

    def verify_connectivity(self):
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def update_node_property(
        self,
        node_id: str,
        property_name: str,
        property_value: Any,
        node_type: Optional[str] = None,  # "Page", "Element", "Action" or None
    ) -> bool:
        """Update a node property by node ID

        Args:
            node_id: Node ID value
            property_name: Name of the property to modify
            property_value: New value for the property
            node_type: Node type, such as "Page", "Element", "Action".
                      If None, will try to find any node with that ID

        Returns:
            bool: Whether the operation was successful
        """
        # Handle special property value types
        if isinstance(property_value, dict) or isinstance(property_value, list):
            property_value = json.dumps(property_value)

        if node_type:
            # Determine ID field name
            id_field = node_type.lower() + "_id"
            query = f"""
            MATCH (n:{node_type}) 
            WHERE n.{id_field} = $node_id
            SET n.{property_name} = $property_value
            RETURN n
            """
        else:
            # Try to find any node with that ID
            query = """
            MATCH (n) 
            WHERE n.page_id = $node_id OR n.element_id = $node_id OR n.action_id = $node_id
            SET n[$property_name] = $property_value
            RETURN n
            """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    node_id=node_id,
                    property_value=property_value,
                    property_name=property_name,
                )
                record = result.single()
                success = record is not None
                if not success:
                    print(
                        f"Warning: Failed to update property {property_name} for node with ID {node_id}"
                    )
                return success
        except Exception as e:
            print(f"Error updating node property: {str(e)}")
            return False

    def create_page(self, properties: Dict[str, Any]) -> str:
        """Create a Page node

        Args:
            properties: Dictionary containing page_id and other attributes
        """
        required_fields = ["page_id"]
        if not all(field in properties for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        properties["timestamp"] = properties.get(
            "timestamp", int(datetime.now().timestamp())
        )

        # Ensure visual_embedding_id is a string
        if "visual_embedding_id" in properties:
            properties["visual_embedding_id"] = str(properties["visual_embedding_id"])

        # Convert other_info to JSON string
        if "other_info" in properties:
            if isinstance(properties["other_info"], dict):
                properties["other_info"] = json.dumps(properties["other_info"])
            elif not isinstance(properties["other_info"], str):
                raise ValueError("other_info must be a dict or JSON string")

        return self.create_node("Page", properties)

    def create_element(self, properties: Dict[str, Any]) -> str:
        """Create an Element node

        Args:
            properties: Dictionary containing element_id, element_type and other attributes
            Visual features related:
                visual_embedding_id: Element screenshot embedding ID in vector database
                visual_features: Text description of element visual features
                visual_similarity_ids: List of visually similar elements' IDs in vector index
                bounding_box: Element bounding box coordinates on the page
                ocr_info: Text information recognized by OCR
                icon_type: Icon type recognition result
        """
        required_fields = ["element_id"]
        if not all(field in properties for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        # Ensure visual_embedding_id is a string
        if "visual_embedding_id" in properties:
            properties["visual_embedding_id"] = str(properties["visual_embedding_id"])

        # Convert other_info to JSON string
        if "other_info" in properties:
            if isinstance(properties["other_info"], dict):
                properties["other_info"] = json.dumps(properties["other_info"])
            elif not isinstance(properties["other_info"], str):
                raise ValueError("other_info must be a dict or JSON string")

        return self.create_node("Element", properties)

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

    # def get_page_elements(self, page_id: str) -> List[Dict[str, Any]]:
    #     """Get all elements on a page and their executable atomic actions"""
    #     query = """
    #     MATCH (p:Page {page_id: $page_id})-[:HAS_ELEMENT]->(e:Element)
    #     RETURN e
    #     """
    #
    #     with self.driver.session(database="neo4j") as session:
    #         result = session.run(query, page_id=page_id)
    #         elements = []
    #         for record in result:
    #             element = dict(record["e"])
    #             if "possible_actions" in element:
    #                 element["possible_actions"] = json.loads(
    #                     element["possible_actions"]
    #                 )
    #             elements.append(element)
    #         return elements

    def delete_nodes(self, node_type: str, node_ids: List[str]) -> int:
        """
        删除指定类型和ID的节点

        参数:
            node_type: 节点类型，'Page'或'Element'
            node_ids: 要删除的节点ID列表

        返回:
            成功删除的节点数量
        """
        # 确定ID属性名称
        if node_type == "Page":
            id_property = "page_id"
        elif node_type == "Element":
            id_property = "element_id"
        else:
            id_property = "action_id"


        with self.driver.session(database=self.database) as session:
            # 执行删除操作
            query = f"""
            MATCH (n:{node_type})
            WHERE n.{id_property} IN $ids
            DETACH DELETE n
            RETURN count(n) AS deleted_count
            """

            result = session.run(query, ids=node_ids)
            deleted = result.single()["deleted_count"]

            print(f"已删除 {deleted} 个 {node_type} 节点")
            return deleted

    def check_element_points_to_specific_page(self, element_id, page_id):

        with self.driver.session(database=self.database) as session:
            # 执行Cypher查询来检查元素是否指向特定页面
            result = session.run(
                """
                MATCH (e:Element)-[r]->(p:Page)
                WHERE e.element_id = $element_id AND p.page_id = $page_id
                RETURN COUNT(r) > 0 AS has_relationship
                """,
                element_id=element_id,
                page_id=page_id
            )

            record = result.single()
            return record["has_relationship"] if record else False

    def add_to_other_info_list(self, page_id, step, task_id=None, task_description=None):

        with self.driver.session(database=self.database) as session:
                # 获取当前页面数据
            result = session.run(
                "MATCH (p:Page {page_id: $id}) RETURN p.other_info",
                id=page_id
            )

            record = result.single()
            if not record:
                return {"error": "Page not found"}

            # 解析当前JSON列表
            other_info_list = json.loads(record["p.other_info"]) if record["p.other_info"] else []

            for other_info in other_info_list:
                if "task_info" in other_info:
                    if other_info["task_info"]["task_id"] == task_id and other_info["step"] == step["step"]:
                        print(f"No need to add '{task_description}' to {page_id}")
                        return None
            # 准备新的字典项
            new_entry = {"step": step["step"]}

            # 如果是第0步且有任务信息，则添加task_info

            new_entry["task_info"] = {
                "task_id": task_id,
                "description": task_description
            }

            # 将新字典添加到列表
            other_info_list.append(new_entry)

            # 更新节点
            session.run(
                "MATCH (p:Page {page_id: $id}) SET p.other_info = $info",
                id=page_id,
                info=json.dumps(other_info_list)
            )

            return {
                "page_id": page_id,
                "updated_other_info": other_info_list
            }

    # def add_to_task_id_list(self, id, task_id=None, node_type=None):
    #
    #     with self.driver.session(database="neo4j") as session:
    #             # 获取当前页面数据
    #         if node_type == "Page":
    #             result = session.run(
    #                 "MATCH (p:Page {page_id: $id}) RETURN p.task_id",
    #                 id=id
    #             )
    #             record = result.single()
    #             if record and "p.task_id" in record:
    #                 # 检查当前值的类型并适当处理
    #                 current_value = record["p.task_id"]
    #                 if current_value is None:
    #                     # 如果是None，初始化为空列表
    #                     task_id_list = []
    #                 elif isinstance(current_value, list):
    #                     # 如果已经是列表，直接使用
    #                     task_id_list = current_value
    #                 elif isinstance(current_value, str):
    #                     # 如果是字符串，尝试解析JSON
    #                     try:
    #                         task_id_list = json.loads(current_value)
    #                         if not isinstance(task_id_list, list):
    #                             task_id_list = [task_id_list]  # 确保结果是列表
    #                     except json.JSONDecodeError:
    #                         # 如果不是有效的JSON，将其作为单个元素的列表
    #                         task_id_list = [current_value]
    #                 else:
    #                     # 处理其他类型
    #                     task_id_list = [current_value]
    #             else:
    #                 task_id_list = []
    #             # # 解析当前JSON列表
    #             # task_id_list = json.loads(record["p.task_id"]) if record["p.task_id"] else []
    #             # new_entry = task_id
    #             task_id_list.append(task_id)
    #             # 更新节点
    #             session.run(
    #                 "MATCH (p:Page {page_id: $id}) SET p.task_id = $task_id_new",
    #                 id=id,
    #                 task_id_new=json.dumps(task_id_list)
    #             )
    #         else:
    #             result = session.run(
    #                 "MATCH (e:Element {element_id: $id}) RETURN e.task_id",
    #                 id=id
    #             )
    #             record = result.single()
    #             # 解析当前JSON列表
    #             if record and "e.task_id" in record:
    #                 # 检查当前值的类型并适当处理
    #                 current_value = record["e.task_id"]
    #                 if current_value is None:
    #                     # 如果是None，初始化为空列表
    #                     task_id_list = []
    #                 elif isinstance(current_value, list):
    #                     # 如果已经是列表，直接使用
    #                     task_id_list = current_value
    #                 elif isinstance(current_value, str):
    #                     # 如果是字符串，尝试解析JSON
    #                     try:
    #                         task_id_list = json.loads(current_value)
    #                         if not isinstance(task_id_list, list):
    #                             task_id_list = [task_id_list]  # 确保结果是列表
    #                     except json.JSONDecodeError:
    #                         # 如果不是有效的JSON，将其作为单个元素的列表
    #                         task_id_list = [current_value]
    #                 else:
    #                     # 处理其他类型
    #                     task_id_list = [current_value]
    #             else:
    #                 task_id_list = []
    #             # task_id_list = json.loads(record["e.task_id"]) if record["e.task_id"] else []
    #             # new_entry = task_id
    #             task_id_list.append(task_id)
    #             # 更新节点
    #             session.run(
    #                 "MATCH (e:Element {element_id: $id}) SET e.task_id = $task_id_new",
    #                 id=id,
    #                 task_id_new=json.dumps(task_id_list)
    #             )
    #
    #         return {
    #             "page_id": id,
    #             "updated_task_id": task_id_list
    #         }

    def add_to_task_id_list(self, id, task_id=None, node_type=None):
        """向节点的task_id列表中添加新的task_id，确保保留已有的task_id"""

        if not task_id:
            return {"error": "No task_id provided"}

        with self.driver.session(database=self.database) as session:
            # 确定节点类型和查询
            if node_type == "Page":
                node_label = "Page"
                id_property = "page_id"
                task_id_property = "p.task_id"
                query_pattern = "MATCH (p:Page {page_id: $id})"
            else:  # 默认为Element
                node_label = "Element"
                id_property = "element_id"
                task_id_property = "e.task_id"
                query_pattern = "MATCH (e:Element {element_id: $id})"

            # 获取当前task_id列表
            get_query = f"{query_pattern} RETURN {task_id_property}"
            result = session.run(get_query, id=id)
            record = result.single()

            if record is None:
                # 节点不存在
                return {
                    "error": f"{node_label} with {id_property}={id} not found",
                    "updated": False
                }

            # 初始化task_id列表
            task_id_list = []

            # 从记录中提取task_id
            # 注意：需要使用正确的键名访问记录
            property_name = task_id_property.split('.')[-1]  # 提取属性名(task_id)
            current_value = record[task_id_property]

            # 处理现有值
            if current_value is not None:
                if isinstance(current_value, list):
                    # 已经是列表类型，直接使用
                    task_id_list = list(current_value)  # 创建副本避免引用问题
                elif isinstance(current_value, str):
                    # 如果是字符串，尝试解析为JSON
                    try:
                        parsed = json.loads(current_value)
                        if isinstance(parsed, list):
                            task_id_list = parsed
                        else:
                            task_id_list = [parsed]
                    except json.JSONDecodeError:
                        # 不是有效的JSON，作为单个值处理
                        task_id_list = [current_value]
                else:
                    # 其他类型，作为单个值处理
                    task_id_list = [str(current_value)]

            # 检查要添加的task_id是否已存在
            if task_id not in task_id_list:
                task_id_list.append(task_id)

            # 直接使用Python列表而不是JSON字符串
            # Neo4j 4.0+支持直接存储列表类型
            update_query = f"{query_pattern} SET {task_id_property.split('.')[0]}.task_id = $task_id_list"
            session.run(update_query, id=id, task_id_list=task_id_list)

            return {
                f"{id_property}": id,
                "updated_task_id": task_id_list,
                "updated": True
            }

    def get_page_properties(self, page_id):
        query = """
        MATCH (p:Page) WHERE p.page_id = $pageId 
        RETURN properties(p) AS page_properties
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, pageId=page_id)
            return result.single()["page_properties"] if result.peek() else None

    def get_element_properties(self, element_id):
        query = """
        MATCH (e:Element) WHERE e.element_id = $elementId 
        RETURN properties(e) AS element_properties
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, elementId=element_id)
            return result.single()["element_properties"] if result.peek() else None

    def get_page_elements(self, page_id: str) -> List[Dict[str, Any]]:
        """Get all elements on a page and their executable atomic actions"""
        query = """
        MATCH (p:Page {page_id: $page_id})-[:HAS_ELEMENT]->(e:Element)
        RETURN e
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, page_id=page_id)
            elements = []
            for record in result:
                element = dict(record["e"])
                if "element_id" in element:
                    elements.append(element["element_id"])
            return elements

    def get_action_sequence(self, action_id: str) -> List[Dict[str, Any]]:
        """Get the execution sequence of a composite action"""
        query = """
        MATCH (a:Action {action_id: $action_id})-[r:COMPOSED_OF]->(e:Element)
        RETURN e.element_id as element_id, 
               e.element_type as element_type, 
               r.order as order,
               r.atomic_action as atomic_action, 
               r.action_params as action_params
        ORDER BY r.order
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, action_id=action_id)
            sequences = []
            for record in result:
                record_dict = {
                    "element_id": record["element_id"],
                    "element_type": record["element_type"],
                    "order": record["order"],  # Ensure this field exists
                    "atomic_action": record["atomic_action"],
                    "action_params": record["action_params"],
                }
                # Deserialize action_params
                if record_dict.get("action_params"):
                    try:
                        record_dict["action_params"] = json.loads(
                            record_dict["action_params"]
                        )
                    except json.JSONDecodeError:
                        pass  # Keep as is if not valid JSON
                sequences.append(record_dict)
            return sequences

    def add_element_leads_to(
        self,
        element_id: str,
        target_id: str,
        action_name: str,
        action_params: Optional[Dict[str, Any]] = None,
        confidence_score: float = 0.0,
    ) -> bool:
        """Create Element-LEADS_TO->Page relationship"""
        query = """
        MATCH (e:Element {element_id: $element_id})
        MATCH (t:Page {page_id: $target_id})
        MERGE (e)-[r:LEADS_TO {
            action_name: $action_name,
            action_params: $action_params,
            confidence_score: $confidence_score
        }]->(t)
        RETURN type(r) as rel_type
        """

        try:
            # Serialize action_params to JSON string
            if action_params:
                action_params = json.dumps(action_params)

            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    element_id=element_id,
                    target_id=target_id,
                    action_name=action_name,
                    action_params=action_params or "",
                    confidence_score=confidence_score,
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

    def get_chain_start_nodes(self) -> List[Dict[str, Any]]:
        """Get all chain starting nodes (Page nodes with no incoming edges)"""
        query = """
        MATCH (n:Page)
        WHERE NOT EXISTS { MATCH ()-[:LEADS_TO]->(n) }
        RETURN n
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                start_nodes = []
                for record in result:
                    node = dict(record["n"])
                    start_nodes.append(node)
                return start_nodes
        except Exception as e:
            print(f"Error getting chain start nodes: {str(e)}")
            return []

    def get_node_by_page_id(self, page_id):
        """Get a Page node by page_id"""
        query = """
        MATCH (n:Page) WHERE n.page_id = $page_id
        RETURN n
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, page_id=page_id)
                record = result.single()
                if record:
                    return dict(record["n"])
                return None
        except Exception as e:
            print(f"Error getting Page node: {str(e)}")
            return None

    def get_chain_from_start(self, start_page_id: str, task_id: str) -> List[List[Dict[str, Any]]]:
        """Get complete operation chain from starting node with specific task ID, returning triplet chain structure

        Args:
            start_page_id: ID of the starting page
            task_id: ID of the task to filter chains

        Returns:
            List[List[Dict]]: List containing complete chain information, each chain consists of multiple triplets
            Each triplet contains:
                - source_page: Source page node information
                - element: Element node information
                - target_page: Target page node information
                - action: Action information
        """
        query = """
        MATCH path = (start:Page {page_id: $start_page_id})-[:HAS_ELEMENT|LEADS_TO*]->(end:Page)
        WHERE NOT EXISTS { (end)-[:HAS_ELEMENT]->() }  // Ensure it's an endpoint page
        AND $task_id IN start.task_id  // Ensure the starting page's task_id list contains our target task_id
        WITH path, relationships(path) as rels, nodes(path) as nodes
        WHERE all(n IN nodes WHERE $task_id IN n.task_id)  // Ensure all nodes have our task_id in their task_id list
        WITH DISTINCT [n in nodes | n{.*}] as node_props,
             [r in rels | r{.*}] as rel_props
        RETURN node_props, rel_props
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, start_page_id=start_page_id, task_id=task_id)
                chains = []
                seen_chains = set()  # For deduplication

                for record in result:

                    nodes = record["node_props"]
                    rels = record["rel_props"]


                    # Build triplet chain
                    chain = []
                    current_page = nodes[0]
                    i = 0

                    while i < len(rels):
                        # Handle HAS_ELEMENT relationship
                        if "element_id" in nodes[i + 1]:  # Found element node
                            element = nodes[i + 1]
                            # Continue looking for LEADS_TO relationship
                            if i + 1 < len(rels) and "action_name" in rels[i + 1]:
                                target_page = nodes[i + 2]

                                # Verify that the task_id is in the task_id list of each component
                                current_page_task_ids = current_page.get("task_id", [])
                                element_task_ids = element.get("task_id", [])
                                target_page_task_ids = target_page.get("task_id", [])

                                # Check if all components contain the target task_id
                                if (task_id in current_page_task_ids and
                                        task_id in element_task_ids and
                                        task_id in target_page_task_ids):
                                    # Build triplet
                                    triplet = {
                                        "source_page": current_page,
                                        "element": element,
                                        "target_page": target_page,
                                        "action": rels[i + 1],
                                    }
                                    chain.append(triplet)

                                current_page = target_page  # Update current page
                                i += 2  # Skip two processed relationships
                            else:
                                i += 1
                        else:
                            i += 1

                    if chain:  # Only add non-empty chains
                        # Create unique identifier for the chain
                        chain_key = tuple(
                            (
                                t["source_page"]["page_id"],
                                t["element"]["element_id"],
                                t["target_page"]["page_id"],
                            )
                            for t in chain
                        )
                        if chain_key not in seen_chains:
                            seen_chains.add(chain_key)
                            chains.append(chain)

                # Return the first chain that matches the criteria
                return chains[0] if chains else []


        except Exception as e:
            print(f"Error getting chain from start node with task ID {task_id}: {str(e)}")
            return []

    def get_chain_id_from_start(self, start_page_id: str, task_id: str) -> List[List[Dict[str, Any]]]:
        """Get complete operation chain from starting node with specific task ID, returning triplet chain structure

        Args:
            start_page_id: ID of the starting page
            task_id: ID of the task to filter chains

        Returns:
            List[List[Dict]]: List containing complete chain information, each chain consists of multiple triplets
            Each triplet contains:
                - source_page: Source page node information
                - element: Element node information
                - target_page: Target page node information
                - action: Action information
        """
        query = """
        MATCH path = (start:Page {page_id: $start_page_id})-[:HAS_ELEMENT|LEADS_TO*]->(end:Page)
        WHERE NOT EXISTS { (end)-[:HAS_ELEMENT]->() }  // Ensure it's an endpoint page
        AND $task_id IN start.task_id  // Ensure the starting page's task_id list contains our target task_id
        WITH path, relationships(path) as rels, nodes(path) as nodes
        WHERE all(n IN nodes WHERE $task_id IN n.task_id)  // Ensure all nodes have our task_id in their task_id list
        WITH DISTINCT [n in nodes | n{.*}] as node_props,
             [r in rels | r{.*}] as rel_props
        RETURN node_props, rel_props
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, start_page_id=start_page_id, task_id=task_id)
                chains = []
                seen_chains = set()  # For deduplication

                for record in result:

                    nodes = record["node_props"]
                    rels = record["rel_props"]


                    # Build triplet chain
                    chain = []
                    current_page = nodes[0]
                    i = 0

                    while i < len(rels):
                        # Handle HAS_ELEMENT relationship
                        if "element_id" in nodes[i + 1]:  # Found element node
                            element = nodes[i + 1]
                            # Continue looking for LEADS_TO relationship
                            if i + 1 < len(rels) and "action_name" in rels[i + 1]:
                                target_page = nodes[i + 2]

                                # Verify that the task_id is in the task_id list of each component
                                current_page_task_ids = current_page.get("task_id", [])
                                element_task_ids = element.get("task_id", [])
                                target_page_task_ids = target_page.get("task_id", [])

                                # Check if all components contain the target task_id
                                if (task_id in current_page_task_ids and
                                        task_id in element_task_ids and
                                        task_id in target_page_task_ids):
                                    # Build triplet
                                    triplet = {
                                        "source_page": current_page,
                                        "element": element,
                                        "target_page": target_page,
                                        "action": rels[i + 1],
                                    }
                                    chain.append(triplet)

                                current_page = target_page  # Update current page
                                i += 2  # Skip two processed relationships
                            else:
                                i += 1
                        else:
                            i += 1

                    if chain:  # Only add non-empty chains
                        # Create unique identifier for the chain
                        chain_key = tuple(
                            (
                                t["source_page"]["page_id"],
                                t["element"]["element_id"],
                                t["target_page"]["page_id"],
                            )
                            for t in chain
                        )
                        if chain_key not in seen_chains:
                            seen_chains.add(chain_key)
                            chains.append(chain)

                # Return the first chain that matches the criteria
                # return chains[0] if chains else []
                return seen_chains

        except Exception as e:
            print(f"Error getting chain from start node with task ID {task_id}: {str(e)}")
            return []

    def get_all_start_chain(self, start_page_id: str) -> List[List[Dict[str, Any]]]:
        """Get all operation chains starting from a specific page node

        This function extracts all task_ids with step=0 from the starting page's other_info,
        then finds and returns all chains for these tasks.

        Args:
            start_page_id: ID of the starting page

        Returns:
            List[List[Dict]]: List containing all chains, where each chain consists of multiple triplets
            Each triplet contains:
                - source_page: Source page node information
                - element: Element node information
                - target_page: Target page node information
                - action: Action information
        """
        # First, get the starting page node to extract task_ids from other_info
        get_node_query = """
        MATCH (p:Page {page_id: $start_page_id})
        RETURN p.other_info as other_info, p.task_id as task_id
        """

        try:
            all_chains = []
            task_ids = []

            # Extract task_ids from other_info
            with self.driver.session(database=self.database) as session:
                result = session.run(get_node_query, start_page_id=start_page_id)
                record = result.single()

                if not record:
                    print(f"No page found with page_id: {start_page_id}")
                    return []

                # Extract the task_ids from other_info that have step=0
                if record["other_info"]:
                    try:
                        other_info = json.loads(record["other_info"])
                        if isinstance(other_info, list):
                            for item in other_info:
                                if isinstance(item, dict) and item.get("step") == 0 and "task_info" in item:
                                    task_info = item["task_info"]
                                    if isinstance(task_info, dict) and "task_id" in task_info:
                                        task_ids.append(task_info["task_id"])
                    except json.JSONDecodeError:
                        print(f"Error parsing other_info JSON for page_id {start_page_id}")


            # If no task_ids found, return empty list
            if not task_ids:
                print(f"No start task_ids found for page_id: {start_page_id}")
                return []

            print(f"Found {len(task_ids)} start task_ids to process: {task_ids}")

            # For each task_id, find the corresponding chain
            for task_id in task_ids:
                chain = self.get_chain_id_from_start(start_page_id, task_id)
                if chain:
                    all_chains.append(chain)

            return all_chains

        except Exception as e:
            print(f"Error getting chains from start node {start_page_id}: {str(e)}")
            return []

    def get_all_actions(self) -> List[Dict[str, Any]]:
        """Get all Action nodes from the database

        Returns:
            List[Dict[str, Any]]: List containing all Action node information
        """
        query = """
        MATCH (a:Action)
        RETURN a
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                actions = []
                for record in result:
                    action = dict(record["a"])
                    # Deserialize JSON string fields
                    if "element_sequence" in action and isinstance(
                        action["element_sequence"], str
                    ):
                        try:
                            action["element_sequence"] = json.loads(
                                action["element_sequence"]
                            )
                        except json.JSONDecodeError:
                            pass  # Keep as is if not valid JSON
                    actions.append(action)
                return actions
        except Exception as e:
            print(f"Error getting all actions: {str(e)}")
            return []

    def get_action_by_id(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get Action node by ID

        Args:
            action_id: ID of the Action node

        Returns:
            Dict[str, Any] or None: Action node information, or None if not found
        """
        query = """
        MATCH (a:Action)
        WHERE a.action_id = $action_id
        RETURN a
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, action_id=action_id)
                record = result.single()
                if record:
                    action = dict(record["a"])
                    # Deserialize JSON string fields
                    if "element_sequence" in action and isinstance(
                        action["element_sequence"], str
                    ):
                        try:
                            action["element_sequence"] = json.loads(
                                action["element_sequence"]
                            )
                        except json.JSONDecodeError:
                            pass  # Keep as is if not valid JSON
                    return action
                return None
        except Exception as e:
            print(f"Error getting action by ID {action_id}: {str(e)}")
            return None

    def get_element_by_id(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get Element node by ID

        Args:
            element_id: ID of the Element node

        Returns:
            Dict[str, Any] or None: Element node information, or None if not found
        """
        query = """
        MATCH (e:Element)
        WHERE e.element_id = $element_id
        RETURN e
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, element_id=element_id)
                record = result.single()
                if record:
                    element = dict(record["e"])
                    # Deserialize JSON string fields, if any
                    for field in ["possible_actions", "other_info"]:
                        if field in element and isinstance(element[field], str):
                            try:
                                element[field] = json.loads(element[field])
                            except json.JSONDecodeError:
                                pass  # Keep as is if not valid JSON
                    return element
                return None
        except Exception as e:
            print(f"Error getting element by ID {element_id}: {str(e)}")
            return None

    def get_all_high_level_actions(self) -> List[Dict[str, Any]]:
        """Get all high-level action nodes from the database

        Returns:
            List[Dict[str, Any]]: List containing all high-level action node information
        """
        query = """
        MATCH (a:Action)
        WHERE a.is_high_level = true OR a.high_level = true OR a.type = 'high_level'
        RETURN a
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                actions = []
                for record in result:
                    action = dict(record["a"])
                    # Deserialize JSON string fields
                    if "element_sequence" in action and isinstance(
                        action["element_sequence"], str
                    ):
                        try:
                            action["element_sequence"] = json.loads(
                                action["element_sequence"]
                            )
                        except json.JSONDecodeError:
                            pass  # Keep as is if not valid JSON
                    actions.append(action)
                return actions
        except Exception as e:
            print(f"Error getting all high level actions: {str(e)}")
            return []

    def get_high_level_actions_for_task(self, task: str) -> List[Dict[str, Any]]:
        """Get high-level action nodes related to a specific task

        Args:
            task: User task description

        Returns:
            List[Dict[str, Any]]: List containing relevant high-level action node information
        """
        # First try fuzzy matching by task description
        query = """
        MATCH (a:Action)
        WHERE (a.is_high_level = true OR a.high_level = true OR a.type = 'high_level')
        AND (a.name CONTAINS $task OR a.description CONTAINS $task)
        RETURN a
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, task=task)
                actions = []
                for record in result:
                    action = dict(record["a"])
                    # Deserialize JSON string fields
                    if "element_sequence" in action and isinstance(
                        action["element_sequence"], str
                    ):
                        try:
                            action["element_sequence"] = json.loads(
                                action["element_sequence"]
                            )
                        except json.JSONDecodeError:
                            pass  # Keep as is if not valid JSON
                    actions.append(action)

                # If no results found, return all high-level actions for further processing
                if not actions:
                    return self.get_all_high_level_actions()

                return actions
        except Exception as e:
            print(f"Error getting high level actions for task '{task}': {str(e)}")
            return []

    def get_shortcuts_for_action(self, action_id: str) -> List[Dict[str, Any]]:
        """Get shortcuts associated with a specific high-level action

        Args:
            action_id: High-level action ID

        Returns:
            List[Dict[str, Any]]: List containing associated shortcut information
        """
        query = """
        MATCH (s:Shortcut)-[:REFERS_TO]->(a:Action {action_id: $action_id})
        RETURN s
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, action_id=action_id)
                shortcuts = []
                for record in result:
                    shortcut = dict(record["s"])
                    # Deserialize JSON string fields
                    for field in ["conditions", "page_flow"]:
                        if field in shortcut and isinstance(shortcut[field], str):
                            try:
                                shortcut[field] = json.loads(shortcut[field])
                            except json.JSONDecodeError:
                                pass  # Keep as is if not valid JSON
                    shortcuts.append(shortcut)
                return shortcuts
        except Exception as e:
            print(f"Error getting shortcuts for action '{action_id}': {str(e)}")
            return []

    def get_page_by_visual_embedding(
        self, embedding_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get page node by visual embedding ID

        Args:
            embedding_id: Visual embedding ID

        Returns:
            Dict[str, Any] or None: Page node information, or None if not found
        """
        query = """
        MATCH (p:Page)
        WHERE p.visual_embedding_id = $embedding_id
        RETURN p
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, embedding_id=embedding_id)
                record = result.single()
                if record:
                    page = dict(record["p"])
                    # Handle possible JSON fields
                    for field in ["elements_data", "metadata"]:
                        if field in page and isinstance(page[field], str):
                            try:
                                page[field] = json.loads(page[field])
                            except json.JSONDecodeError:
                                pass  # Keep as is if not valid JSON
                    return page
                return None
        except Exception as e:
            print(f"Error getting page by visual embedding ID {embedding_id}: {str(e)}")
            return None

    def get_pages_with_step_zero(self) -> List[Dict[str, Any]]:
        """获取所有other_info中包含step为0的Page节点

        手动解析other_info JSON字符串来查找step=0的节点

        Returns:
            包含符合条件的Page节点的列表
        """
        # 简单查询获取所有Page节点和它们的other_info
        query = """
        MATCH (p:Page)
        WHERE p.other_info IS NOT NULL
        RETURN p
        """

        try:
            with self.driver.session(database=self.database) as session:
                # 执行查询获取所有节点
                result = session.run(query)

                # 手动过滤包含step=0的节点
                filtered_nodes = []
                for record in result:
                    node = dict(record["p"])

                    if "other_info" in node:
                        try:
                            # 尝试解析JSON字符串
                            info = json.loads(node["other_info"])

                            # 处理列表情况 (如 [{"step": 0, ...}])
                            if isinstance(info, list):
                                if any(isinstance(item, dict) and item.get("step") == 0 for item in info):
                                    filtered_nodes.append(node)

                            # 处理单个对象情况 (如 {"step": 0, ...})
                            elif isinstance(info, dict):
                                if info.get("step") == 0:
                                    filtered_nodes.append(node)

                        except json.JSONDecodeError:
                            # 如果JSON解析失败，尝试简单的字符串匹配
                            if '"step": 0' in node["other_info"] or '"step":0' in node["other_info"]:
                                filtered_nodes.append(node)

                return filtered_nodes

        except Exception as e:
            print(f"Error retrieving nodes: {str(e)}")
            return []

    def get_page_ids_with_step_zero(self) -> List[str]:
        """获取所有other_info中包含step为0的Page节点的page_id

        Returns:
            页面ID的列表
        """
        nodes = self.get_pages_with_step_zero()
        return [node.get("page_id") for node in nodes if "page_id" in node]

    def get_chain_by_chain_id(self, chain_id: List[str]) -> List[Dict[str, Any]]:
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
                    chain = []
                    for match in record["matches"]:
                        triplet = {
                            "source_page": dict(match["source"]),
                            "element": dict(match["element"]),
                            "target_page": dict(match["target"]),
                            "action": dict(match["action"])
                        }
                        chain.append(triplet)

                    return chain

                return []

        except Exception as e:
            print(f"Error getting chain by chain_id: {str(e)}")
            return []

    def is_action_duplicate(self, components_preview: list) -> bool:
        """
        检查是否存在具有相同组件预览的Action节点

        Args:
            components_preview: 组件预览列表

        Returns:
            bool: 如果存在重复则返回True，否则返回False
        """
        import json

        try:
            # 将组件预览列表转换为JSON字符串
            components_json = json.dumps(components_preview)

            # 查询具有相同components_preview的Action节点
            query = """
            MATCH (a:Action)
            WHERE a.components_preview = $components
            RETURN count(a) > 0 AS exists
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(query, components=components_json)
                record = result.single()
                return record is not None and record["exists"]

        except Exception as e:
            print(f"Error checking Action duplicate: {str(e)}")
            return False

    def find_action_by_element_sequence(self, element_sequence: list) -> Optional[str]:
        """Return an existing Action ID with the same ordered COMPOSED_OF elements."""
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
            MATCH (a:Action)
            MATCH (a)-[r:COMPOSED_OF]->(e:Element)
            WITH a, e.element_id AS eid, r.order AS ord
            ORDER BY ord ASC
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

    # def delete_task_nodes(self, target_task_id):
    #     """
    #     根据指定task_id删除相关节点或更新节点信息
    #
    #     Args:
    #         driver: Neo4j数据库驱动
    #         target_task_id: 要删除的task_id
    #     """
    #     with self.driver.session(database="neo4j") as session:
    #         # 第一步：删除只包含目标task_id的节点
    #         delete_query = """
    #         MATCH (n)
    #         WHERE $target_task_id IN n.task_id AND size(n.task_id) = 1
    #         DETACH DELETE n
    #         RETURN count(n) as deleted_count
    #         """
    #         delete_result = session.run(delete_query, target_task_id=target_task_id)
    #         deleted_count = delete_result.single()['deleted_count']
    #
    #         # 第二步：更新包含多个task_id的节点
    #         update_query = """
    #         MATCH (n)
    #         WHERE $target_task_id IN n.task_id AND size(n.task_id) > 1
    #         SET n.task_id = [task_id IN n.task_id WHERE task_id <> $target_task_id]
    #         RETURN elementId(n) as node_id, n.other_info as other_info, labels(n) as labels
    #         """
    #         update_result = session.run(update_query, target_task_id=target_task_id)
    #
    #         # 第三步：处理Page节点的other_info
    #         updated_count = 0
    #         for record in update_result:
    #             updated_count += 1
    #             node_id = record['node_id']
    #             other_info = record['other_info']
    #             labels = record['labels']
    #
    #             # 如果是Page节点且有other_info，需要移除相关task信息
    #             if 'Page' in labels and other_info:
    #                 try:
    #                     other_info_list = json.loads(other_info)
    #                     filtered_info = [
    #                         item for item in other_info_list
    #                         if not (isinstance(item, dict) and
    #                                 item.get('task_info', {}).get('task_id') == target_task_id)
    #                     ]
    #                     updated_other_info = json.dumps(filtered_info)
    #
    #                     # 更新other_info
    #                     update_other_info_query = """
    #                     MATCH (n)
    #                     WHERE elementId(n) = $node_id
    #                     SET n.other_info = $updated_other_info
    #                     """
    #                     session.run(update_other_info_query,
    #                                 node_id=node_id,
    #                                 updated_other_info=updated_other_info)
    #                 except:
    #                     pass  # 如果JSON解析失败，跳过other_info更新
    #
    #         print(f"删除了 {deleted_count} 个节点，更新了 {updated_count} 个节点")
    def delete_task_nodes(self, target_task_id):
        """
        根据指定task_id删除相关节点或更新节点信息，同时删除Pinecone中的向量

        Args:
            target_task_id: 要删除的task_id
        """
        with self.driver.session(database=self.database) as session:
            # 第一步：获取只包含目标task_id的节点信息（用于删除Pinecone向量）
            get_exclusive_nodes_query = """
            MATCH (n)
            WHERE $target_task_id IN n.task_id AND size(n.task_id) = 1
            RETURN elementId(n) as node_id, labels(n) as labels, 
                   CASE 
                       WHEN 'Element' IN labels(n) THEN n.element_id
                       WHEN 'Page' IN labels(n) THEN n.page_id
                       ELSE null
                   END as vector_id
            """
            exclusive_nodes_result = session.run(get_exclusive_nodes_query, target_task_id=target_task_id)

            # 按节点类型分组收集向量ID（只处理Element和Page）
            vectors_by_type = {
                'Element': [],
                'Page': []
            }

            for record in exclusive_nodes_result:
                vector_id = record['vector_id']
                labels = record['labels']

                if vector_id:
                    if 'Element' in labels:
                        vectors_by_type['Element'].append(vector_id)
                    elif 'Page' in labels:
                        vectors_by_type['Page'].append(vector_id)

            # 按节点类型删除Pinecone中的向量
            vector_store = VectorStore(
                    api_key=config.PINECONE_API_KEY,
                    index_name=index,
                    dimension=2048,
                    batch_size=2,
                )
            total_deleted = 0
            for node_type, vector_ids in vectors_by_type.items():
                if vector_ids:
                    try:
                        # 根据你的NodeType枚举调整namespace
                        namespace = node_type.lower()  # 或者使用你的NodeType枚举值
                        vector_store.delete_vectors(ids=vector_ids, node_type=namespace)
                        total_deleted += len(vector_ids)
                        print(f"从Pinecone namespace '{namespace}' 删除了 {len(vector_ids)} 个向量")
                    except Exception as e:
                        print(f"删除 {node_type} 类型的Pinecone向量时出错: {e}")

            if total_deleted > 0:
                    print(f"总共从Pinecone删除了 {total_deleted} 个向量")

            # 第二步：删除只包含目标task_id的节点（原有功能保持不变）
            delete_query = """
            MATCH (n)
            WHERE $target_task_id IN n.task_id AND size(n.task_id) = 1
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
            delete_result = session.run(delete_query, target_task_id=target_task_id)
            deleted_count = delete_result.single()['deleted_count']

            # 第三步：更新包含多个task_id的节点（原有功能保持不变）
            update_query = """
            MATCH (n)
            WHERE $target_task_id IN n.task_id AND size(n.task_id) > 1
            SET n.task_id = [task_id IN n.task_id WHERE task_id <> $target_task_id]
            RETURN elementId(n) as node_id, n.other_info as other_info, labels(n) as labels
            """
            update_result = session.run(update_query, target_task_id=target_task_id)

            # 第四步：处理Page节点的other_info（原有功能保持不变）
            updated_count = 0
            for record in update_result:
                updated_count += 1
                node_id = record['node_id']
                other_info = record['other_info']
                labels = record['labels']

                # 如果是Page节点且有other_info，需要移除相关task信息
                if 'Page' in labels and other_info:
                    try:
                        other_info_list = json.loads(other_info)
                        filtered_info = [
                            item for item in other_info_list
                            if not (isinstance(item, dict) and
                                    item.get('task_info', {}).get('task_id') == target_task_id)
                        ]
                        updated_other_info = json.dumps(filtered_info)

                        # 更新other_info
                        update_other_info_query = """
                        MATCH (n)
                        WHERE elementId(n) = $node_id
                        SET n.other_info = $updated_other_info
                        """
                        session.run(update_other_info_query,
                                    node_id=node_id,
                                    updated_other_info=updated_other_info)
                    except:
                        pass  # 如果JSON解析失败，跳过other_info更新

            print(f"删除了 {deleted_count} 个节点，更新了 {updated_count} 个节点")


class ActionMergeAnalyzer:
    """
    使用BPE算法思想识别高频动作组合
    """

    def __init__(self, verbose: bool = False):
        """
        初始化ActionMergeAnalyzer

        Args:
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
        self.vocab = {}  # 存储生成的高频动作组合词表
        self.vocab_id = 0  # 为词表中的每个条目分配唯一ID

    def analyze(self,
                task_chains: List[Set[Tuple[Tuple[str, str, str], ...]]],
                num_merges: int = 200,
                min_freq: int = 2) -> Dict[str, Dict]:
        """
        分析任务链，识别高频动作组合

        Args:
            task_chains: 所有任务链列表，每个任务链是一组三元组序列
            num_merges: 要执行的合并次数
            min_freq: 考虑合并的最小频率

        Returns:
            包含生成的高级行为词表的字典
        """
        # 第1步：准备数据 - 将复杂的嵌套结构展平为线性链
        flat_chains = self._flatten_chains(task_chains)
        if self.verbose:
            print(f"Flattened {len(task_chains)} task chains into {len(flat_chains)} linear chains")

        # 第2步：计算初始的单个动作频率
        action_freqs = self._count_initial_freqs(flat_chains)
        if self.verbose:
            print(f"Found {len(action_freqs)} unique atomic actions")
            top_actions = sorted(action_freqs.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Top 5 atomic actions: {top_actions}")

        # 第3步：运行BPE算法，找到最常见的连续动作对并合并
        merged_vocab = self._run_bpe(flat_chains, num_merges, min_freq)

        # 第4步：格式化结果
        return self._format_results(merged_vocab)

    def _flatten_chains(self, task_chains: List[Set[Tuple[Tuple[str, str, str], ...]]]) -> List[
        List[Tuple[str, str, str]]]:
        """
        将嵌套的任务链结构展平为线性链列表

        Args:
            task_chains: 嵌套的任务链结构

        Returns:
            线性链列表，每个链是一个三元组列表
        """
        flat_chains = []

        for chain_set in task_chains:
            for chain_tuple in chain_set:
                # 将单个链(元组)转换为列表
                flat_chain = list(chain_tuple)
                flat_chains.append(flat_chain)

        return flat_chains

    def _count_initial_freqs(self, chains: List[List[Tuple[str, str, str]]]) -> Dict[Tuple[str, str, str], int]:
        """
        计算单个动作的初始频率

        Args:
            chains: 线性化的任务链列表

        Returns:
            单个动作及其频率的字典
        """
        freqs = Counter()
        for chain in chains:
            for action in chain:
                freqs[action] += 1
        return freqs

    # def _get_action_pairs(self, chains: List[List[Any]]) -> Dict[Tuple[Any, Any], int]:
    #     """
    #     计算所有连续动作对的频率
    #
    #     Args:
    #         chains: 当前状态下的任务链
    #
    #     Returns:
    #         动作对及其频率的字典
    #     """
    #     pair_freqs = Counter()
    #     for chain in chains:
    #         if len(chain) < 2:  # 跳过长度小于2的链
    #             continue
    #
    #         # 计算所有连续的动作对
    #         for i in range(len(chain) - 1):
    #             pair = (chain[i], chain[i + 1])
    #             pair_freqs[pair] += 1
    #
    #     return pair_freqs

    # def _merge_pair(self,
    #                 chains: List[List[Any]],
    #                 pair: Tuple[Any, Any],
    #                 merged_id: str) -> List[List[Any]]:
    #     """
    #     在所有链中将指定的动作对替换为合并后的单个动作
    #
    #     Args:
    #         chains: 当前的任务链
    #         pair: 要合并的动作对
    #         merged_id: 合并后的新动作ID
    #
    #     Returns:
    #         更新后的任务链
    #     """
    #     new_chains = []
    #     for chain in chains:
    #         new_chain = []
    #         i = 0
    #         while i < len(chain):
    #             # 检查当前位置是否是目标对的开始
    #             if i < len(chain) - 1 and chain[i] == pair[0] and chain[i + 1] == pair[1]:
    #                 new_chain.append(merged_id)
    #                 i += 2  # 跳过已合并的两个动作
    #             else:
    #                 new_chain.append(chain[i])
    #                 i += 1
    #         new_chains.append(new_chain)
    #     return new_chains

    def _run_bpe(self,
                 chains: List[List[Tuple[str, str, str]]],
                 num_merges: int,
                 min_freq: int) -> Dict[str, Dict]:
        """
        运行BPE算法来识别和合并高频动作组合

        Args:
            chains: 初始的任务链
            num_merges: 要执行的合并次数
            min_freq: 考虑合并的最小频率

        Returns:
            合并后得到的词表
        """
        # 复制链以避免修改原始数据
        current_chains = copy.deepcopy(chains)

        # 初始化词表，首先添加所有原子动作
        action_vocab = {}
        for chain in current_chains:
            for action in chain:
                if action not in action_vocab:
                    action_id = str(self.vocab_id)
                    self.vocab_id += 1
                    action_vocab[action] = {
                        "id": action_id,
                        "components": [action],
                        "frequency": 1,
                        "level": 1  # 原子动作的级别为1
                    }
                else:
                    action_vocab[action]["frequency"] += 1

        # 记录合并历史
        merge_history = []

        # 执行指定次数的合并
        for i in range(num_merges):
            # 获取所有动作对及其频率
            pair_freqs = self._get_action_pairs(current_chains)

            # 如果没有更多可合并的对，则退出
            if not pair_freqs:
                if self.verbose:
                    print(f"No more pairs to merge after {i} merges")
                break

            # 找出频率最高的动作对
            most_freq_pair, freq = max(pair_freqs.items(), key=lambda x: x[1])

            # 如果频率低于阈值，则停止合并
            if freq < min_freq:
                if self.verbose:
                    print(f"Stopped at merge {i}: highest frequency {freq} < minimum {min_freq}")
                break

            if self.verbose:
                print(f"Merge {i + 1}: {most_freq_pair} with frequency {freq}")

            # 创建新的合并动作ID和记录
            merged_action = (most_freq_pair[0], most_freq_pair[1])
            merged_id = f"M{self.vocab_id}"
            self.vocab_id += 1

            # 计算新动作的组成部分和级别
            components = []
            level = 1

            # 对于pair中的每个动作，获取其组件
            for part in most_freq_pair:
                if isinstance(part, str) and part.startswith('M'):
                    # 这是之前合并的动作
                    part_key = None
                    for k, v in action_vocab.items():
                        if v["id"] == part:
                            part_key = k
                            break

                    if part_key:
                        components.extend(action_vocab[part_key]["components"])
                        level = max(level, action_vocab[part_key]["level"] + 1)
                else:
                    # 这是原子动作
                    components.append(part)
                    level = max(level, 2)  # 至少是2级动作

            # 添加到词表
            action_vocab[merged_action] = {
                "id": merged_id,
                "components": components,
                "frequency": freq,
                "level": level
            }

            # 记录这次合并
            merge_history.append({
                "merged_id": merged_id,
                "pair": most_freq_pair,
                "frequency": freq,
                "components": components
            })

            # 在所有链中执行合并
            current_chains = self._merge_pair(current_chains, most_freq_pair, merged_id)

        # 过滤出高级别的动作组合(level >= 2)
        high_level_vocab = {k: v for k, v in action_vocab.items() if v["level"] >= 2}

        return {
            "high_level_actions": high_level_vocab,
            "merge_history": merge_history,
            "all_actions": action_vocab
        }

    def _format_results(self, merged_vocab: Dict) -> Dict:
        """
        格式化结果为更易读的形式

        Args:
            merged_vocab: BPE算法生成的原始词表

        Returns:
            格式化后的结果
        """
        high_level_actions = merged_vocab["high_level_actions"]
        formatted_results = {}

        # 按频率排序
        sorted_actions = sorted(
            high_level_actions.items(),
            key=lambda x: (x[1]["frequency"], x[1]["level"]),
            reverse=True
        )

        for i, (action_key, action_data) in enumerate(sorted_actions):
            # 为复合动作创建可读的名称
            action_name = f"HighLevelAction_{i + 1}"

            # 格式化动作的组成部分
            components = []
            for comp in action_data["components"]:
                if isinstance(comp, tuple) and len(comp) == 3:
                    # 这是一个三元组动作
                    src, elem, tgt = comp
                    components.append(f"({src[:8]}..., {elem[:8]}..., {tgt[:8]}...)")
                    # components.append(comp)
                else:
                    components.append(str(comp))
                    # components.append(comp)

            formatted_results[action_name] = {
                "id": action_data["id"],
                "frequency": action_data["frequency"],
                "level": action_data["level"],
                "num_components": len(action_data["components"]),
                "components_preview": components
                # "components_preview": components[:3] + (["..."] if len(components) > 3 else [])
            }

        return {
            "high_level_actions": formatted_results,
            "total_found": len(formatted_results),
            "merge_history_length": len(merged_vocab["merge_history"])
        }

    def _get_action_pairs(self, chains: List[List[Any]]) -> Dict[Tuple[Any, Any], int]:
        """
        计算所有连续动作对的频率
        对于仅最后一个三元组的target_node不同的动作对，归为同一类统计频率
        但保留原始动作对信息
        """
        # 用于存储标准化key到原始pairs的映射
        canonical_to_originals = {}
        pair_freqs = Counter()

        for chain in chains:
            if len(chain) < 2:
                continue

            for i in range(len(chain) - 1):
                pair = (chain[i], chain[i + 1])

                # 找到标准化的key
                canonical_key = self._get_canonical_key(pair)

                # 记录这个标准化key对应的原始pairs
                if canonical_key not in canonical_to_originals:
                    canonical_to_originals[canonical_key] = []
                if pair not in canonical_to_originals[canonical_key]:
                    canonical_to_originals[canonical_key].append(pair)

                # 使用标准化key统计频率
                pair_freqs[canonical_key] += 1

        # 为每个标准化key选择一个代表性的原始pair作为合并时使用的pair
        final_pair_freqs = {}
        for canonical_key, freq in pair_freqs.items():
            # 选择第一个原始pair作为代表
            representative_pair = canonical_to_originals[canonical_key][0]
            final_pair_freqs[representative_pair] = freq
            # 保存这个映射关系，供merge时使用
            setattr(self, f'_canonical_group_{id(representative_pair)}', canonical_to_originals[canonical_key])

        return final_pair_freqs

    def _get_canonical_key(self, pair: Tuple[Any, Any]) -> str:
        """
        为动作对生成标准化的key，仅用于频率统计
        """
        action1, action2 = pair
        components = self._get_all_components_from_pair(action1, action2)

        if not components:
            return str(pair)

        # 创建标准化的组件序列用于生成key
        canonical_components = []
        for i, comp in enumerate(components):
            if i == len(components) - 1:  # 最后一个组件
                canonical_comp = (comp[0], comp[1], "*")  # target用通配符
            else:
                canonical_comp = comp
            canonical_components.append(canonical_comp)

        return str(tuple(canonical_components))

    def _merge_pair(self,
                    chains: List[List[Any]],
                    pair: Tuple[Any, Any],
                    merged_id: str) -> List[List[Any]]:
        """
        在所有链中将指定的动作对替换为合并后的单个动作
        需要将所有属于同一标准化组的动作对都进行合并
        """
        # 获取这个pair对应的所有原始pairs（如果存在标准化组的话）
        canonical_group_attr = f'_canonical_group_{id(pair)}'
        if hasattr(self, canonical_group_attr):
            pairs_to_merge = getattr(self, canonical_group_attr)
        else:
            pairs_to_merge = [pair]

        new_chains = []
        for chain in chains:
            new_chain = []
            i = 0
            while i < len(chain):
                if i < len(chain) - 1:
                    current_pair = (chain[i], chain[i + 1])

                    # 检查当前pair是否在要合并的pairs列表中
                    if current_pair in pairs_to_merge:
                        new_chain.append(merged_id)
                        i += 2
                    else:
                        new_chain.append(chain[i])
                        i += 1
                else:
                    new_chain.append(chain[i])
                    i += 1
            new_chains.append(new_chain)

        return new_chains

    def _get_all_components_from_pair(self, action1: Any, action2: Any) -> List[Tuple[str, str, str]]:
        """
        获取动作对中所有动作的组件序列
        """
        components = []

        # 获取第一个动作的组件
        components.extend(self._get_action_components(action1))
        # 获取第二个动作的组件
        components.extend(self._get_action_components(action2))

        return components

    def _get_action_components(self, action: Any) -> List[Tuple[str, str, str]]:
        """
        获取单个动作的组件
        """
        if isinstance(action, tuple) and len(action) == 3:
            return [action]
        elif hasattr(self, '_current_vocab'):
            # 查找复合动作的组件
            for k, v in self._current_vocab.get("all_actions", {}).items():
                if (isinstance(action, str) and v.get("id") == action) or k == action:
                    return v.get("components", [])

        return []


