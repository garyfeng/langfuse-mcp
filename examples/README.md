# Langfuse MCP Examples

This directory contains example scripts and demo clients that demonstrate how to use the Langfuse MCP integration.

## Available Examples

### langfuse_client_demo.py

A demonstration client that connects to the Langfuse MCP server and shows key functionality by executing various tool calls. It demonstrates:

- Connecting to the Langfuse MCP server
- Listing available tools
- Getting error counts
- Retrieving traces
- Finding exceptions

To run the demo client:

```bash
uv run examples/langfuse_client_demo.py --public-key YOUR_PUBLIC_KEY --secret-key YOUR_SECRET_KEY
```


The wrapper will use environment variables (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`) if available. 
