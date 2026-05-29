import json
import os
import time
import requests
from neo4j import GraphDatabase
# from traj_to_kg import TrajectoryToNeo4jImporter
from typing import Dict, List, Optional
import logging
import config
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _chat_completions_url() -> str:
    base_url = getattr(config, "LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _request_proxies():
    proxy = os.environ.get("LLM_PROXY")
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


class LLMFunctionSummarizer:
    def __init__(self, neo4j_uri: str, auth: tuple, database: str,
                 openrouter_api_key: str, force_update: bool = False):
        """
        初始化Neo4j连接和OpenRouter配置

        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_username: Neo4j用户名
            neo4j_password: Neo4j密码
            openrouter_api_key: OpenRouter API密钥
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.database = database
        self.verify_connectivity()
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_url = _chat_completions_url()
        self.model_name = getattr(config, "LLM_MODEL", "openai/gpt-4o")
        self.force_update = force_update

        # API调用延迟（避免频率限制）
        self.api_delay = 1.0  # 秒

    def verify_connectivity(self):
        self.driver.verify_connectivity()

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def call_llm_api(self, prompt: str, max_tokens: int = 200) -> str:
        """
        调用OpenRouter API获取Claude 3.7的回复

        Args:
            prompt: 输入prompt
            max_tokens: 最大输出token数

        Returns:
            LLM生成的总结文本
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app-domain.com",  # 可选：你的应用域名
            "X-Title": "GUI Function Summarizer"  # 可选：应用名称
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  # 低温度确保一致性
        }

        max_retries = max(1, int(getattr(config, "LLM_MAX_RETRIES", 3)))
        timeout = getattr(config, "LLM_REQUEST_TIMEOUT", 500)
        proxies = _request_proxies()
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.openrouter_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    proxies=proxies,
                )
                response.raise_for_status()

                result = response.json()
                return result['choices'][0]['message']['content'].strip()

            except requests.exceptions.RequestException as e:
                logger.error(f"API call failed ({attempt + 1}/{max_retries}) at {self.openrouter_url}: {e}")
            except (KeyError, IndexError) as e:
                logger.error(f"API response parse failed ({attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 8))
        return ""

        max_retries = max(1, int(getattr(config, "LLM_MAX_RETRIES", 3)))
        timeout = getattr(config, "LLM_REQUEST_TIMEOUT", 500)
        proxies = _request_proxies()
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.openrouter_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    proxies=proxies,
                )
                response.raise_for_status()

                result = response.json()
                return result['choices'][0]['message']['content'].strip()

            except requests.exceptions.RequestException as e:
                logger.error(f"API call failed ({attempt + 1}/{max_retries}) at {self.openrouter_url}: {e}")
            except (KeyError, IndexError) as e:
                logger.error(f"API response parse failed ({attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 8))
        return ""

        try:
            response = requests.post(self.openrouter_url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"API调用失败: {e}")
            return "Error: Failed to generate summary"
        except (KeyError, IndexError) as e:
            logger.error(f"API响应解析失败: {e}")
            return "Error: Invalid API response"

    def generate_page_summary_prompt(self, description: str, page_id: str = "") -> str:
        """
        生成Page节点总结的prompt
        """
        prompt = f"""You are a GUI analysis expert. Please analyze the following mobile app page description and provide a concise functional summary.

**Task**: Summarize the main function/purpose of this GUI page in 1 sentence.

**Page Description**:
{description}

**Requirements**:
- Focus on the primary function and purpose of this page
- Use clear, professional language
- Keep it concise (max 2 sentences)
- Avoid repeating the exact description text
- Highlight what users can accomplish on this page

**Output Format**: Just provide the functional summary without any additional explanation.

**Example Output**: "Settings configuration page that allows users to modify system preferences and application settings."

**Your Summary**:"""
        return prompt

    def generate_action_summary_prompt(self, function_text: str, action_id: str = "") -> str:
        """
        生成Action节点总结的prompt
        """
        prompt = f"""You are a GUI automation expert. Please analyze the following action description and provide a concise functional summary.

**Task**: Summarize what this action accomplishes in 1 sentence.

**Action Function Description**:
{function_text}

**Requirements**:
- Focus on the end goal/outcome of this action
- Use active voice and clear language
- Keep it concise (max 2 sentences)
- Avoid technical jargon
- Highlight the user intent behind this action

**Output Format**: Just provide the functional summary without any additional explanation.

**Example Output**: "Opens the device settings menu to access system configuration options."

**Your Summary**:"""
        return prompt

    def generate_element_summary_prompt(self, reasoning_function: str, element_id: str = "") -> str:
        """
        生成Element节点总结的prompt
        """
        prompt = f"""You are a GUI element analysis expert. Please analyze the following element operation and provide a concise functional summary.

**Task**: Summarize what this GUI element operation does in 1 sentence.

**Element Operation Description**:
{reasoning_function}

**Requirements**:
- Focus on the direct interaction and its immediate effect
- Use clear, actionable language
- Keep it concise (max 2 sentences)
- Avoid repeating exact technical details
- Highlight the specific interaction type and target

**Output Format**: Just provide the functional summary without any additional explanation.

**Example Output**: "Taps the settings icon to launch the system configuration application."

**Your Summary**:"""
        return prompt

    def extract_page_description(self, page_node: Dict) -> str:
        """从Page节点提取description字段"""
        return page_node.get('description', '')

    def extract_action_function(self, action_node: Dict) -> str:
        """从Action节点提取function字段"""
        return action_node.get('function', '')

    def extract_element_reasoning(self, element_node: Dict) -> str:
        """从Element节点提取reasoning.function字段"""
        reasoning_str = element_node.get('reasoning', '{}')
        try:
            if isinstance(reasoning_str, str):
                reasoning_dict = json.loads(reasoning_str)
            else:
                reasoning_dict = reasoning_str
            return reasoning_dict.get('function', '')
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse reasoning JSON: {reasoning_str}")
            return ''

    def summarize_page_function(self, description: str, page_id: str = "") -> str:
        """
        使用LLM总结Page节点的function
        """
        if not description.strip():
            return "Unknown page function"

        prompt = self.generate_page_summary_prompt(description, page_id)
        summary = self.call_llm_api(prompt, max_tokens=150)
        if not summary or summary.startswith("Error:"):
            return ""

        # 添加延迟避免API频率限制
        time.sleep(self.api_delay)

        return summary

    def summarize_action_function(self, function_text: str, action_id: str = "") -> str:
        """
        使用LLM总结Action节点的function
        """
        if not function_text.strip():
            return "Unknown action function"

        prompt = self.generate_action_summary_prompt(function_text, action_id)
        summary = self.call_llm_api(prompt, max_tokens=150)
        if not summary or summary.startswith("Error:"):
            return ""

        time.sleep(self.api_delay)

        return summary

    def summarize_element_function(self, reasoning_function: str, element_id: str = "") -> str:
        """
        使用LLM总结Element节点的function
        """
        if not reasoning_function.strip():
            return "Unknown element function"

        prompt = self.generate_element_summary_prompt(reasoning_function, element_id)
        summary = self.call_llm_api(prompt, max_tokens=150)
        if not summary or summary.startswith("Error:"):
            return ""

        time.sleep(self.api_delay)

        return summary

    def process_page_nodes(self):
        """
        处理Page节点 - 增量更新
        """
        with self.driver.session(database=self.database) as session:
            # 只获取没有function_summary的Page节点
            if self.force_update:
                result = session.run(
                    "MATCH (p:Page) RETURN p"
                )

            else:
                result = session.run(
                    "MATCH (p:Page) WHERE p.function_summary IS NULL OR p.function_summary = '' OR p.function_summary STARTS WITH 'Error:' "
                    "RETURN p"
                )


            nodes_to_process = list(result)
            logger.info(f"Found {len(nodes_to_process)} Page nodes to process")

            for i, record in enumerate(nodes_to_process):
                page_node = dict(record['p'])
                node_id = page_node.get('page_id')

                logger.info(f"Processing Page node {i + 1}/{len(nodes_to_process)}: {node_id}")

                # 提取description
                description = self.extract_page_description(page_node)

                if not description.strip():
                    logger.warning(f"Page node {node_id} has empty description, skipping")
                    continue

                # 生成function总结
                try:
                    function_summary = self.summarize_page_function(description, node_id)
                    if not function_summary:
                        logger.warning(f"Skipping Page node {node_id}: summary generation failed")
                        continue

                    # 更新节点属性
                    session.run(
                        "MATCH (p:Page) WHERE p.page_id = $node_id "
                        "SET p.function_summary = $function_summary",
                        node_id=node_id,
                        function_summary=function_summary
                    )

                    logger.info(f"✅ Updated Page node {node_id}")

                except Exception as e:
                    logger.error(f"❌ Failed to process Page node {node_id}: {e}")
                    continue

    def process_action_nodes(self):
        """
        处理Action节点 - 增量更新
        """
        with self.driver.session(database=self.database) as session:
            # 只获取没有function_summary的Action节点
            if self.force_update:
                result = session.run("MATCH (a:Action) RETURN a")
            else:
                result = session.run(
                    "MATCH (a:Action) WHERE a.function_summary IS NULL OR a.function_summary = '' OR a.function_summary STARTS WITH 'Error:' "
                    "RETURN a"
                )



            nodes_to_process = list(result)
            logger.info(f"Found {len(nodes_to_process)} Action nodes to process")

            for i, record in enumerate(nodes_to_process):
                action_node = dict(record['a'])
                node_id = action_node.get('action_id')

                logger.info(f"Processing Action node {i + 1}/{len(nodes_to_process)}: {node_id}")

                # 提取function
                function_text = self.extract_action_function(action_node)

                if not function_text.strip():
                    logger.warning(f"Action node {node_id} has empty function, skipping")
                    continue

                # 生成function总结
                try:
                    function_summary = self.summarize_action_function(function_text, node_id)
                    if not function_summary:
                        logger.warning(f"Skipping Action node {node_id}: summary generation failed")
                        continue

                    # 更新节点属性
                    session.run(
                        "MATCH (a:Action) WHERE a.action_id = $node_id "
                        "SET a.function_summary = $function_summary",
                        node_id=node_id,
                        function_summary=function_summary
                    )

                    logger.info(f"✅ Updated Action node {node_id}")

                except Exception as e:
                    logger.error(f"❌ Failed to process Action node {node_id}: {e}")
                    continue

    def process_element_nodes(self):
        """
        处理Element节点 - 增量更新
        """
        with self.driver.session(database=self.database) as session:

            if self.force_update:
                result = session.run("MATCH (e:Element) RETURN e")
            else:
                result = session.run(
                    "MATCH (e:Element) WHERE e.function_summary IS NULL OR e.function_summary = '' OR e.function_summary STARTS WITH 'Error:' "
                    "RETURN e"
                )

            nodes_to_process = list(result)
            logger.info(f"Found {len(nodes_to_process)} Element nodes to process")

            for i, record in enumerate(nodes_to_process):
                element_node = dict(record['e'])
                node_id = element_node.get('element_id')

                logger.info(f"Processing Element node {i + 1}/{len(nodes_to_process)}: {node_id}")

                # 提取reasoning.function
                reasoning_function = self.extract_element_reasoning(element_node)

                if not reasoning_function.strip():
                    logger.warning(f"Element node {node_id} has empty reasoning function, skipping")
                    continue

                # 生成function总结
                try:
                    function_summary = self.summarize_element_function(reasoning_function, node_id)
                    if not function_summary:
                        logger.warning(f"Skipping Element node {node_id}: summary generation failed")
                        continue

                    # 更新节点属性
                    session.run(
                        "MATCH (e:Element) WHERE e.element_id = $node_id "
                        "SET e.function_summary = $function_summary",
                        node_id=node_id,
                        function_summary=function_summary
                    )

                    logger.info(f"✅ Updated Element node {node_id}")

                except Exception as e:
                    logger.error(f"❌ Failed to process Element node {node_id}: {e}")
                    continue

    def process_all_nodes(self):
        """
        增量处理所有类型的节点
        """
        logger.info("🚀 Starting incremental node processing...")

        logger.info("📄 Processing Page nodes...")
        self.process_page_nodes()

        logger.info("⚡ Processing Action nodes...")
        self.process_action_nodes()

        logger.info("🔧 Processing Element nodes...")
        self.process_element_nodes()

        logger.info("✅ All nodes processed successfully!")

    def get_node_statistics(self):
        """
        获取处理统计信息
        """
        with self.driver.session(database=self.database) as session:
            # 统计各类型节点数量
            page_total = session.run("MATCH (p:Page) RETURN count(p) as count").single()['count']
            action_total = session.run("MATCH (a:Action) RETURN count(a) as count").single()['count']
            element_total = session.run("MATCH (e:Element) RETURN count(e) as count").single()['count']

            # 统计已处理节点数量
            page_processed = session.run(
                "MATCH (p:Page) WHERE p.function_summary IS NOT NULL AND p.function_summary <> '' "
                "RETURN count(p) as count"
            ).single()['count']

            action_processed = session.run(
                "MATCH (a:Action) WHERE a.function_summary IS NOT NULL AND a.function_summary <> '' "
                "RETURN count(a) as count"
            ).single()['count']

            element_processed = session.run(
                "MATCH (e:Element) WHERE e.function_summary IS NOT NULL AND e.function_summary <> '' "
                "RETURN count(e) as count"
            ).single()['count']

            # 统计待处理节点数量
            page_pending = page_total - page_processed
            action_pending = action_total - action_processed
            element_pending = element_total - element_processed

            print(f"\n📊 Processing Statistics:")
            print(f"Page nodes: {page_processed}/{page_total} processed ({page_pending} pending)")
            print(f"Action nodes: {action_processed}/{action_total} processed ({action_pending} pending)")
            print(f"Element nodes: {element_processed}/{element_total} processed ({element_pending} pending)")
            print(
                f"Total: {page_processed + action_processed + element_processed}/{page_total + action_total + element_total} processed")



class LLMElementName:
    def __init__(self, neo4j_uri: str, auth: tuple, database: str,
                 openrouter_api_key: str, force_update: bool = False):
        """
        初始化Neo4j连接和OpenRouter配置

        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_username: Neo4j用户名
            neo4j_password: Neo4j密码
            openrouter_api_key: OpenRouter API密钥
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.database = database
        self.verify_connectivity()
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_url = _chat_completions_url()
        self.model_name = getattr(config, "LLM_MODEL", "openai/gpt-4o")
        self.force_update = force_update

        # API调用延迟（避免频率限制）
        self.api_delay = 1.0  # 秒

    def verify_connectivity(self):
        self.driver.verify_connectivity()

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def call_llm_api(self, prompt: str, max_tokens: int = 200) -> str:
        """
        调用OpenRouter API获取Claude 3.7的回复

        Args:
            prompt: 输入prompt
            max_tokens: 最大输出token数

        Returns:
            LLM生成的总结文本
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app-domain.com",  # 可选：你的应用域名
            "X-Title": "GUI Function Summarizer"  # 可选：应用名称
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  # 低温度确保一致性
        }

        try:
            response = requests.post(self.openrouter_url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"API调用失败: {e}")
            return "Error: Failed to generate summary"
        except (KeyError, IndexError) as e:
            logger.error(f"API响应解析失败: {e}")
            return "Error: Invalid API response"

    def generate_element_name_prompt(self, reasoning_function: str, element_id: str = "") -> str:
        """
        生成Element节点名称的prompt
        """
        prompt = f"""You are a GUI element naming expert. Please analyze the following element operation and generate a concise, functional name.

    **Task**: Create a short, descriptive name for this GUI element operation that reflects its functional purpose.

    **Element Operation Description**:
    {reasoning_function}

    **Naming Requirements**:
    - Use snake_case format (lowercase with underscores)
    - Keep it concise (2-4 words maximum)
    - Focus on the action + target/context
    - Make it intuitive and self-explanatory
    - Follow the pattern: action_object or action_context_object

    **Reference Examples**:
    - "add_new_entry" - for creating a new expense entry
    - "input_entry_price" - for entering amount in an entry field
    - "save_entry" - for saving an edited entry
    - "scroll_brightness_slider" - for adjusting brightness via scrolling
    - "tap_settings_icon" - for opening settings
    - "toggle_dark_mode" - for switching theme

    **Output Format**: Provide only the name in snake_case format, no explanations.

    **Element Name**:"""
        return prompt


    def extract_element_reasoning(self, element_node: Dict) -> str:
        """从Element节点提取reasoning.function字段"""
        reasoning_str = element_node.get('reasoning', '{}')
        try:
            if isinstance(reasoning_str, str):
                reasoning_dict = json.loads(reasoning_str)
            else:
                reasoning_dict = reasoning_str
            return reasoning_dict.get('function', '')
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse reasoning JSON: {reasoning_str}")
            return ''


    def summarize_element_name(self, reasoning_function: str, element_id: str = "") -> str:
        """
        使用LLM总结Element节点的function
        """
        if not reasoning_function.strip():
            return "Unknown element function"

        prompt = self.generate_element_name_prompt(reasoning_function, element_id)
        summary = self.call_llm_api(prompt, max_tokens=150)
        if not summary or summary.startswith("Error:"):
            return ""

        time.sleep(self.api_delay)

        return summary


    def process_element_nodes(self):
        """
        处理Element节点 - 增量更新
        """
        with self.driver.session(database=self.database) as session:

            if self.force_update:
                result = session.run("MATCH (e:Element) RETURN e")
            else:
                result = session.run(
                    "MATCH (e:Element) WHERE e.name IS NULL OR e.name = '' OR e.name STARTS WITH 'Error:' "
                    "RETURN e"
                )

            nodes_to_process = list(result)
            logger.info(f"Found {len(nodes_to_process)} Element nodes to process")

            for i, record in enumerate(nodes_to_process):
                element_node = dict(record['e'])
                node_id = element_node.get('element_id')

                logger.info(f"Processing Element node {i + 1}/{len(nodes_to_process)}: {node_id}")

                # 提取reasoning.function
                reasoning_function = self.extract_element_reasoning(element_node)

                if not reasoning_function.strip():
                    logger.warning(f"Element node {node_id} has empty reasoning function, skipping")
                    continue

                # 生成function总结
                try:
                    element_name = self.summarize_element_name(reasoning_function, node_id)
                    if not element_name:
                        logger.warning(f"Skipping Element node {node_id}: name generation failed")
                        continue

                    # 更新节点属性
                    session.run(
                        "MATCH (e:Element) WHERE e.element_id = $node_id "
                        "SET e.name = $element_name",
                        node_id=node_id,
                        element_name=element_name
                    )

                    logger.info(f"✅ Updated Element node {node_id}")

                except Exception as e:
                    logger.error(f"❌ Failed to process Element node {node_id}: {e}")
                    continue

    def process_all_nodes(self):
        """
        增量处理所有类型的节点
        """
        logger.info("🚀 Starting incremental node processing...")

        logger.info("🔧 Processing Element nodes...")
        self.process_element_nodes()

        logger.info("✅ All nodes processed successfully!")

    def get_node_statistics(self):
        """
        获取处理统计信息
        """
        with self.driver.session(database=self.database) as session:
            # 统计各类型节点数量
            element_total = session.run("MATCH (e:Element) RETURN count(e) as count").single()['count']

            # 统计已处理节点数量

            element_processed = session.run(
                "MATCH (e:Element) WHERE e.name IS NOT NULL AND e.name <> '' "
                "RETURN count(e) as count"
            ).single()['count']

            # 统计待处理节点数量

            element_pending = element_total - element_processed

            print(f"\n📊 Processing Statistics:")
            print(f"Element nodes: {element_processed}/{element_total} processed ({element_pending} pending)")
            print(
                f"Total: {element_processed}/{element_total} processed")


# 使用示例
def main():

    # 创建处理器实例
    summarizer = LLMFunctionSummarizer(
        config.Neo4j_URI, config.Neo4j_AUTH, config.Neo4j_DATABASE, config.LLM_API_KEY, True
    )

    try:
        # 获取处理前的统计信息
        logger.info("📊 Getting current statistics...")
        summarizer.get_node_statistics()

        # 增量处理所有节点
        summarizer.process_all_nodes()

        # 获取处理后的统计信息
        logger.info("📊 Getting final statistics...")
        summarizer.get_node_statistics()

    except KeyboardInterrupt:
        logger.info("⏹️  Processing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
    finally:
        # 关闭数据库连接
        summarizer.close()
        logger.info("🔌 Database connection closed")


if __name__ == "__main__":
    main()
