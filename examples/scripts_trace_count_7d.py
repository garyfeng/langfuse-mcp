"""Print the number of Langfuse traces recorded over the last seven days."""

import asyncio
import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pk = os.environ["LANGFUSE_PUBLIC_KEY"]
sk = os.environ["LANGFUSE_SECRET_KEY"]
host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")


async def main():
    """Query Langfuse via MCP and emit the aggregate trace count metric."""
    params = StdioServerParameters(
        command="uv", args=["run", "-m", "langfuse_mcp", "--public-key", pk, "--secret-key", sk, "--host", host, "--log-level", "INFO"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("fetch_traces", {"age": 10080, "limit": 1, "page": 1, "output_mode": "compact"})
            payload = None
            if hasattr(res, "content") and res.content and res.content[0].text:
                try:
                    payload = json.loads(res.content[0].text)
                except Exception as e:
                    print("ERROR: unable to parse MCP response:", e)
                    return
            if isinstance(payload, dict) and "metadata" in payload:
                meta = payload.get("metadata") or {}
                total = meta.get("total")
                if total is None:
                    data = payload.get("data")
                    total = (
                        meta.get("item_count")
                        if meta.get("item_count") is not None
                        else (len(data) if isinstance(data, list) else (1 if data else 0))
                    )
                print(f"traces_count_7d={total}")
            else:
                print("ERROR: unexpected MCP response format:", payload)


asyncio.run(main())
