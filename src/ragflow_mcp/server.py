"""FastMCP server for RAGFlow.

Provides MCP tools for semantic retrieval and knowledge management
through the RAGFlow API.
"""
import argparse
import asyncio
import atexit
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP

from ragflow_mcp.config import Settings, get_settings
from ragflow_mcp.connector import RAGFlowConnector
from ragflow_mcp.tools.retrieval import register_retrieval_tools
from ragflow_mcp.tools.datasets import register_dataset_tools
from ragflow_mcp.tools.documents import register_document_tools
from ragflow_mcp.tools.chunks import register_chunk_tools
from ragflow_mcp.tools.chat import register_chat_tools
from ragflow_mcp.tools.graph import register_graph_tools


# Global connector instance
_connector: RAGFlowConnector | None = None
_initialized: bool = False


def _initialize_connector() -> None:
    """Initialize the global connector instance (lazy initialization)."""
    global _connector, _initialized

    if _initialized:
        return

    settings = get_settings()

    # Configure logging
    log_level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting RAGFlow MCP Server")
    logger.info("Connecting to RAGFlow at %s", settings.ragflow_url)

    # Initialize connector (sync init, async connect on first use)
    _connector = RAGFlowConnector(
        api_key=settings.ragflow_api_key,
        base_url=settings.ragflow_url,
    )
    # Initialize the httpx client synchronously
    import httpx
    _connector.client = httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {settings.ragflow_api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(_connector.timeout),
    )

    _initialized = True
    logger.info("RAGFlow connector initialized")


def get_connector() -> RAGFlowConnector:
    """Get the global connector instance.

    Returns:
        The initialized RAGFlowConnector.

    Raises:
        RuntimeError: If connector is not initialized.
    """
    if not _initialized:
        _initialize_connector()
    if _connector is None:
        raise RuntimeError("Connector not initialized")
    return _connector


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Manage server lifecycle.

    Initializes and closes the RAGFlowConnector during server lifecycle.
    """
    # Initialize if not already done
    _initialize_connector()

    try:
        yield
    finally:
        global _connector, _initialized
        if _connector is not None and _connector.client is not None:
            logger = logging.getLogger(__name__)
            logger.info("Shutting down RAGFlow MCP Server")
            await _connector.client.aclose()
            _connector = None
            _initialized = False


# Create FastMCP server
mcp = FastMCP(
    "ragflow-mcp",
    instructions="""RAGFlow MCP Server - Semantic retrieval and knowledge management.

Available capabilities:
- Semantic search across datasets using ragflow_retrieval_tool
- Dataset management (create, list, update, delete)
- Document management (upload, parse, download, delete)
- Chunk management (add, list, update, delete)
- Chat assistant and session management
- GraphRAG and RAPTOR construction

Use the status resource to check server health.""",
    lifespan=lifespan,
    host="0.0.0.0"
)

# Register all tools
register_retrieval_tools(mcp)
register_dataset_tools(mcp)
register_document_tools(mcp)
register_chunk_tools(mcp)
register_chat_tools(mcp)
register_graph_tools(mcp)


@mcp.resource("http://ragflow/status")
async def get_status() -> str:
    """Get server health status.

    Returns:
        JSON string with server status information.
    """
    connector = get_connector()
    return f"""{{
    "status": "healthy",
    "connected": {str(connector.client is not None).lower()},
    "base_url": "{connector.base_url}"
}}"""



def main() -> None:
    """Main entry point for the server."""

    try:
        # Validate configuration early
        get_settings()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
