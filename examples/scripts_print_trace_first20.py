"""Print the first 20 lines of the latest Langfuse trace via MCP."""

import asyncio
import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pk = os.environ["LANGFUSE_PUBLIC_KEY"]
sk = os.environ["LANGFUSE_SECRET_KEY"]
host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")


async def main():
    """Request full trace JSON and output the initial lines."""
    params = StdioServerParameters(
        command="uv", args=["run", "-m", "langfuse_mcp", "--public-key", pk, "--secret-key", sk, "--host", host, "--log-level", "INFO"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Request full JSON to avoid compact truncation
            res = await session.call_tool(
                "fetch_traces", {"age": 10080, "limit": 1, "page": 1, "include_observations": True, "output_mode": "full_json_string"}
            )
            if not hasattr(res, "content") or not res.content or not res.content[0].text:
                print("ERROR: empty MCP response")
                return
            try:
                traces = json.loads(res.content[0].text)
            except Exception as e:
                print("ERROR: unable to parse JSON:", e)
                return
            if not isinstance(traces, list) or not traces:
                print("No traces found in the last 7 days.")
                return
            trace = traces[0]
            pretty = json.dumps(trace, indent=2, default=str)
            lines = pretty.splitlines()
            to_print = "\n".join(lines[:20])
            print(to_print)
            if len(lines) > 20:
                print("\n... (truncated, total lines:", len(lines), ")")


asyncio.run(main())
