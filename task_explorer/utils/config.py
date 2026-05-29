"""Configuration for task_explorer utilities."""

from __future__ import annotations

import os


LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
LLM_MAX_TOKEN = int(os.environ.get("LLM_MAX_TOKEN", "1500"))
LLM_REQUEST_TIMEOUT = int(os.environ.get("LLM_REQUEST_TIMEOUT", "500"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))

LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_ENDPOINT = os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "eam")

Neo4j_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
Neo4j_AUTH = (
    os.environ.get("NEO4J_USER", "neo4j"),
    os.environ.get("NEO4J_PASSWORD", "password"),
)
Neo4j_DATABASE = os.environ.get("NEO4J_DATABASE", "")

Feature_URI = os.environ.get("FEATURE_URI", "http://127.0.0.1:8001")
Omni_URI = os.environ.get("OMNI_URI", "http://127.0.0.1:8000")

PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
