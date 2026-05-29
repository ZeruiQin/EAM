from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field, SecretStr
import json
import os
from langchain_openai import ChatOpenAI
import task_explorer.utils.config as config
from task_explorer.utils.graph_db import Neo4jDatabase


os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "LearnTriplet"  # Keep specific project name

model = ChatOpenAI(
    openai_api_base=config.LLM_BASE_URL,
    openai_api_key=SecretStr(config.LLM_API_KEY),
    model_name=config.LLM_MODEL,
    request_timeout=config.LLM_REQUEST_TIMEOUT,
    max_retries=config.LLM_MAX_RETRIES,
    max_tokens=config.LLM_MAX_TOKEN,
    openai_proxy=os.environ.get("LLM_PROXY"),
)

URI = config.Neo4j_URI
AUTH = config.Neo4j_AUTH
# db = Neo4jDatabase(URI, AUTH, database=config.Neo4j_DATABASE,
#         index=config.PINECONE_INDEX)

class TripletReasoning(BaseModel):
    function: str = Field(description="Operation function description")
    user_intent: str = Field(description="User intention analysis")
    source_page_enhanced_desc: str = Field(
        description="Enhanced description of the source page"
    )
    element_enhanced_desc: str = Field(
        description="Enhanced description of the element"
    )
    target_page_enhanced_desc: str = Field(
        description="Enhanced description of the target page"
    )


async def process_and_update_chain(match_id:str, triplets = None, db = None) -> List[Dict[str, Any]]:
    """Process triplet chain and update database

    Args:
        start_page_id: Starting page ID

    Returns:
        List of processed triplets
    """
    # Create necessary reasoning chains
    reasoning_chain = create_triplet_reasoning_chain()
    merge_chain = create_merge_descriptions_chain()

    # Extract chain data
    if triplets is None:
        print(f"Warning: No triplets found")
        return []

    # Directly process returned triplet list
    processed_chain = await process_single_chain(triplets, reasoning_chain, merge_chain, match_id, db)

    return processed_chain




def create_triplet_reasoning_chain():
    """Create LCEL chain for triplet reasoning"""
    # Define reasoning prompt template
    triplet_reasoning_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant specialized in understanding and reasoning about UI operation chains. You need to analyze the given page-element-page triplet information and perform deep understanding and reasoning. You will receive textual descriptions and screenshots of pages, please analyze both.",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": """Please analyze the following UI operation triplet information:
        Source Page: {source_page_desc}
        Element: {element_desc}
        Target Page: {target_page_desc}
        Action: {action_desc}

        Please reason and expand from the following aspects:
        1. What is the purpose and function of this operation?
        2. What might be the user's intention when performing this operation?
        3. Based on your understanding, generate richer and more accurate descriptions for the source page, element, and target page.

        Please return your reasoning results in a structured way, including the following fields:
        - function: Operation function description
        - user_intent: User intention analysis
        - source_page_enhanced_desc: Enhanced description of source page
        - element_enhanced_desc: Enhanced description of element
        - target_page_enhanced_desc: Enhanced description of target page

        {format_instructions}""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{source_page_image}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{target_page_image}"},
                    },
                ],
            ),
        ]
    )

    # Use JsonOutputParser instead of StrOutputParser
    parser = JsonOutputParser(pydantic_object=TripletReasoning)

    # Inject format instructions into prompt template
    prompt = triplet_reasoning_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # Build LCEL chain
    reasoning_chain = RunnablePassthrough() | prompt | model | parser

    return reasoning_chain


def create_merge_descriptions_chain():
    """Create LCEL chain for merging page descriptions

    Returns:
        Description merging chain
    """
    # Define merge description prompt template
    merge_descriptions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant specialized in merging and optimizing page descriptions. You need to analyze descriptions of shared pages between two adjacent triplets and merge them into a more complete description.",
            ),
            (
                "human",
                """Please analyze the following two descriptions that describe the same page but from different contexts:

        Current Task: {task_info}

        Description 1 (as target page of previous triplet): {desc1}
        Description 2 (as source page of next triplet): {desc2}

        Please merge these two descriptions to generate a more complete and coherent description, requirements:
        1. Consider the context and goals of the current task
        2. Preserve all important information
        3. Eliminate redundant content
        4. Ensure descripnalitytion coherence
        5. Highlight core functio and features of the page
        6. Emphasize relevance to the current task

        Please return the merged description.""",
            ),
        ]
    )

    # Build LCEL chain
    merge_chain = (
            RunnablePassthrough() | merge_descriptions_prompt | model | StrOutputParser()
    )

    return merge_chain

async def merge_node_descriptions(
    chain: List[Dict[str, Any]], merge_chain, task_info: str
):
    """Merge description information of overlapping nodes in the chain

    Args:
        chain: Triplet chain
        merge_chain: Description merging chain
        task_info: Task information

    Returns:
        Triplet chain with updated descriptions
    """
    for i in range(len(chain) - 1):
        current_triplet = chain[i]
        next_triplet = chain[i + 1]

        # Check for overlapping nodes
        if (
            current_triplet["target_page"]["page_id"]
            == next_triplet["source_page"]["page_id"]
        ):
            # Prepare merge input
            merge_input = {
                "desc1": current_triplet["target_page"].get("description", ""),
                "desc2": next_triplet["source_page"].get("description", ""),
                "task_info": task_info,  # Add task information
            }

            try:
                # Execute description merge
                merged_desc = await merge_chain.ainvoke(merge_input)

                # Update descriptions in both triplets
                current_triplet["target_page"]["description"] = merged_desc
                next_triplet["source_page"]["description"] = merged_desc

                # Update node description in database
                # TODO: Implement database node description update
                # Can use neo4j_db.update_node_property method to update node properties
            except Exception as e:
                print(f"Error merging descriptions: {str(e)}")

    return chain


async def process_single_chain(
    chain: List[Dict[str, Any]], reasoning_chain, merge_chain, task_id, db
) -> List[Dict[str, Any]]:
    """Process all triplets in a single chain and merge descriptions

    Args:
        chain: Single triplet chain
        reasoning_chain: Triplet reasoning chain
        merge_chain: Description merging chain

    Returns:
        Processed chain
    """
    # Extract task information from first node in chain
    task_info = task_id

    # Process each triplet
    processed_triplets = []
    for triplet in chain:
        if triplet["element"]["description"] == "":
            processed_triplet = await process_triplet(triplet, reasoning_chain)
        else:
            processed_triplet = triplet
        processed_triplets.append(processed_triplet)

    # Merge node descriptions, pass task information
    merged_chain = await merge_node_descriptions(
        processed_triplets, merge_chain, task_info
    )

    # Update node information in database
    for triplet in merged_chain:
        # Update source page description
        update_node_in_db(
            triplet["source_page"]["page_id"],
            "description",
            triplet["source_page"].get("description", ""),
            "Page",
            db
        )

        # Update target page description
        update_node_in_db(
            triplet["target_page"]["page_id"],
            "description",
            triplet["target_page"].get("description", ""),
            "Page",
            db
        )

        # Update element description
        update_node_in_db(
            triplet["element"]["element_id"],
            "description",
            triplet["element"].get("description", ""),
            "Element",
            db
        )
        # print(triplet["reasoning"])
        # Save reasoning results (if exists)
        if "reasoning" in triplet:
            # Save complete reasoning results to element node
            update_node_in_db(
                triplet["element"]["element_id"],
                "reasoning",
                json.dumps(triplet["reasoning"]),
                "Element",
                db
            )

    return merged_chain


def update_node_in_db(
    node_id: str,
    property_name: str,
    property_value: Any,
    node_type: Optional[str] = None,
    db = None
) -> bool:
    """Update node properties in database

    Args:
        node_id: Node ID
        property_name: Property name
        property_value: Property value
        node_type: Node type (optional)

    Returns:
        Whether update was successful
    """
    try:
        return db.update_node_property(
            node_id=node_id,
            property_name=property_name,
            property_value=property_value,
            node_type=node_type,
        )
    except Exception as e:
        print(f"Error updating node property: {str(e)}")
        return False

async def process_triplet(triplet: Dict[str, Any], reasoning_chain):
    """Process reasoning for a single triplet

    Args:
        triplet: Triplet containing source page, element, target page and action information
        reasoning_chain: Triplet reasoning chain

    Returns:
        Triplet with added reasoning results
    """
    # Prepare reasoning input
    reasoning_input = {
        "source_page_desc": triplet["source_page"].get("description", ""),
        "element_desc": triplet["element"].get("description", ""),
        "target_page_desc": triplet["target_page"].get("description", ""),
        "action_desc": triplet["element"].get("action_output", ""),
    }

    # Load source and target page images
    try:

        source_page_image = triplet["source_page"]["raw_page"]
        if isinstance(source_page_image, str):
            source_page_image = json.loads(source_page_image)
        target_page_image = triplet["target_page"]["raw_page"]
        if isinstance(target_page_image, str):
            target_page_image = json.loads(target_page_image)

        # Add images to reasoning input
        reasoning_input["source_page_image"] = f"data:image/webp;base64,{source_page_image}"
        reasoning_input["target_page_image"] = f"data:image/webp;base64,{target_page_image}"
    except Exception as e:
        print(f"Error loading page images: {str(e)}")
        # Use empty base64 image if loading fails
        reasoning_input["source_page_image"] = "data:image/webp;base64,"
        reasoning_input["target_page_image"] = "data:image/webp;base64,"
    try:
        # Execute reasoning - now returns dictionary object
        reasoning_result = await reasoning_chain.ainvoke(reasoning_input)

        # Store result directly as reasoning field
        triplet["reasoning"] = reasoning_result

        # Update descriptions in triplet - update enhanced descriptions directly to description field
        triplet["source_page"]["description"] = reasoning_result[
            "source_page_enhanced_desc"
        ]
        triplet["element"]["description"] = reasoning_result["element_enhanced_desc"]
        triplet["target_page"]["description"] = reasoning_result[
            "target_page_enhanced_desc"
        ]

    except Exception as e:
        print(f"Error during triplet reasoning: {str(e)}")
        # Record detailed error information for debugging
        triplet["reasoning_error"] = str(e)

    return triplet