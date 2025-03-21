#!/bin/bash
# Wrapper script to run the Langfuse MCP server from the local installation using uv
cd "$(dirname "$0")"
/opt/homebrew/bin/uv run python -m langfuse_mcp "$@" 