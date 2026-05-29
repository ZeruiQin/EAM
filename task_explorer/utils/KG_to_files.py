from task_explorer.utils.traj_to_kg import TrajectoryToNeo4jImporter, find_all_task_folders
import argparse
import config
from typing import Dict, List, Optional, Set
import pickle
import json
import os
from dataclasses import dataclass


PACKAGE_APP_MAPPING = {
    # Audio Recorder
    'com.dimowner.audiorecorder': 'audio-recorder',

    # Browser (Chrome)
    'com.google.android.documentsui': 'files',

    # Calendar (Simple Calendar Pro)
    'com.simplemobiletools.calendar.pro': 'simple-calendar-pro',

    # Camera
    'com.android.camera2': 'camera',

    # Clock
    'com.google.android.deskclock': 'clock',

    # Contacts
    'com.google.android.contacts': 'contacts',

    # Pro Expense
    'com.arduia.expense': 'pro-expense',


    # Markor
    'net.gsantner.markor': 'markor',


    # information retrieval in joplin
    'net.cozic.joplin': 'joplin',

    # OsmAnd
    'net.osmand': 'osmand',

    # Recipe (Broccoli)
    'com.flauschcode.broccoli': 'broccoli',

    # Retro Music
    'code.name.monkey.retromusic': 'retro-music',


    # Simple Draw Pro
    'com.simplemobiletools.draw.pro': 'simple-draw-pro',

    # Simple Gallery Pro
    'com.simplemobiletools.gallery.pro': 'simple-gallery-pro',

    # SMS (Simple SMS Messenger)
    'com.simplemobiletools.smsmessenger': 'simple-sms-messenger',


    #sport tracker
    'de.dennisguse.opentracks': 'open-tracks-sports-tracker',

    # System tasks (需要Settings应用)

    'com.android.settings': 'settings',


    # Task anwser
    'org.tasks': 'tasks',


    # VLC
    'org.videolan.vlc': 'vlc',
}

# Page节点结构
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



def export_kg_to_pkl(database: str, index, output_dir: str ):
    """从Neo4j导出到pkl文件"""
    importer = TrajectoryToNeo4jImporter(
        uri=config.Neo4j_URI,
        auth=config.Neo4j_AUTH,
        database=database,
        index=index
    )
    # 1. 查询所有Page
    pages = {}
    # rows = importer.rewrite_action_ids()
    with importer.driver.session(database=database) as session:
        page_result = session.run("""
            MATCH (p:Page)
            OPTIONAL MATCH (p)-[:HAS_ELEMENT]->(e:Element)
            WITH p, collect(DISTINCT e.element_id) AS element_ids
            RETURN p.page_id AS pid, p.description AS desc, p.task AS ptask, p.function_summary AS function_summary, element_ids
        """)
        query = """
            MATCH (e:Element {element_id: $element_id})
            OPTIONAL MATCH (a:Action)-[:COMPOSED_OF]->(e)
            RETURN a.action_id as action_id, a.element_sequence as element_sequence
        """
        for record in page_result:
            action_list = []
            for element in record['element_ids']:
                action_results = session.run(query, {'element_id': element})
                for action in action_results:
                    action_id = action['action_id']
                    if action_id is None:
                        continue
                    element_sequence = json.loads(action['element_sequence'])
                    if element == element_sequence[0]['element_id']:
                        action_list.append(action_id)
            pages[record['pid']] = PageNode(
                page_id=record['pid'],
                description=record['desc'] or "",
                task_steps=json.loads(record['ptask']) if record['ptask'] else "",
                element_ids=[eid for eid in record['element_ids'] if eid],
                action_ids=[aid for aid in action_list if aid],
                function_summary=record['function_summary']
            )

    # 2. 查询所有Element
        elements = {}

        element_result = session.run("""
            MATCH (e:Element)
            WITH e
            RETURN e.element_id AS eid, e.name AS name, e.reasoning AS reasoning, e.function_summary AS function_summary, e.task AS etask
        """)

        query_leads_to = """
            MATCH (e:Element {element_id: $element_id})
            OPTIONAL MATCH (e)-[:LEADS_TO]->(np:Page)
            RETURN np.page_id AS next_page_id
        """

        for record in element_result:
            next_pages = []
            query_results = session.run(query_leads_to, element_id=record['eid'])
            for next_page in query_results:
                next_pages.append(next_page['next_page_id'])
            elements[record['eid']] = ElementNode(
                element_id=record['eid'],
                name=record['name'] or "",
                reasoning=record['reasoning'] or "",
                task_steps=json.loads(record['etask']) or "",
                leads_to_page_id=next_pages,
                function_summary=record['function_summary']
            )
        # 3. 查询所有Action
        actions = {}
        action_result = session.run("""
                MATCH (a: Action)
                RETURN a.action_id AS aid, a.name AS name, a.function AS function, a.element_sequence AS element_sequence
            """)
        i = 0
        for record in action_result:
            i += 1
            element_sequence = json.loads(record['element_sequence'])
            print(record['aid'])
            print(element_sequence)
            last_element = element_sequence[-1]['element_id']
            query = """
                    MATCH (ae:Element {element_id: $element_id})
                    OPTIONAL MATCH (ae)-[:LEADS_TO]->(np:Page)
                    RETURN np.page_id AS next_page_id
                """
            target_result = session.run(query=query, element_id=last_element)
            target_page = []
            for next_page in target_result:
                target_page.append(next_page['next_page_id'])
            # target_page = target_result.single()['next_page_id']
            print(record['aid'])
            actions[record['aid']] = ActionNode(
                action_id=record['aid'],
                name=record['name'] or "",
                function=record['function'] or "",
                element_sequence=json.loads(record['element_sequence']) or "",
                leads_to_page_id=target_page,
            )
        print(i)


    # 4. 构建完整图
    app_graph = AppGraph(
        app_name=database,
        pages=pages,
        elements=elements,
        actions=actions
    )

    # 5. 保存
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{database}_graph.pkl", "wb") as f:
        pickle.dump(app_graph, f)

    print(f"✓ 导出完成: {len(pages)} pages, {len(elements)} elements, {len(actions)} actions")



def main():
    parser = argparse.ArgumentParser(description="Export Neo4j app KGs to graph_env pickle files.")
    parser.add_argument(
        "--base_path",
        default="docker_exploration_output",
        help="Automatic trajectory output directory used to discover app packages.",
    )
    parser.add_argument(
        "--output_dir",
        default="data_AW/graph_env",
        help="Directory where exported app graph folders are written.",
    )
    args = parser.parse_args()

    base_path = args.base_path
    packages = find_all_task_folders(base_path)

    for package in packages:
        package_name = package.name
        # if package_name in ['net.gsantner.markor', ]:
        #     continue
        # try:
        app_name = PACKAGE_APP_MAPPING[package_name]
        print(f"processing {app_name}")
        export_kg_to_pkl(
            database=app_name,
            index=app_name,
            output_dir=os.path.join(args.output_dir, app_name),
        )

if __name__ == "__main__":
    main()
