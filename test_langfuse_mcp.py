#!/usr/bin/env python
"""
Simple script to test the langfuse-mcp implementation.
Requires the MCP client package.
"""

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent


async def main():
    print("Testing langfuse-mcp server...")
    
    # Set up environment variables for the MCP server
    env = {
        "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY"),
        "LANGFUSE_HOST": os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    }
    
    # Create server parameters
    server_params = StdioServerParameters(
        command="./run-langfuse-mcp.sh",
        args=[],
        env=env
    )
    
    # Initialize the client session
    async with asyncio.TaskGroup() as tg:
        try:
            async with stdio_client(server_params) as stdio_transport:
                read_stream, write_stream = stdio_transport
                
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # List the available tools
                    tools_response = await session.list_tools()
                    print("\nAvailable tools:")
                    
                    # The tools_response is a ListToolsResult object with a tools attribute
                    if hasattr(tools_response, 'tools'):
                        tools = tools_response.tools
                        for tool in tools:
                            print(f"- {tool.name}: {tool.description.split('\n')[0]}")
                        
                        # Get the schema of spans and events tables
                        try:
                            schema_result = await session.call_tool("get_langfuse_schema", {})
                            print("\nSchema of Langfuse data:")
                            
                            # Extract content from the CallToolResult
                            if isinstance(schema_result, CallToolResult) and schema_result.content:
                                for content_item in schema_result.content:
                                    if isinstance(content_item, TextContent):
                                        print(content_item.text)
                                    else:
                                        print(f"Content type: {type(content_item)}")
                            else:
                                print(f"Unexpected schema result format: {type(schema_result)}")
                        except Exception as e:
                            print(f"\nError getting schema: {e}")
                            traceback.print_exc()
                        
                        # Try finding exceptions from the last 60 minutes
                        try:
                            exceptions_result = await session.call_tool("find_exceptions", {"age": 60})
                            print("\nExceptions in the last hour:")
                            
                            # Extract content from the CallToolResult
                            if isinstance(exceptions_result, CallToolResult) and exceptions_result.content:
                                for content_item in exceptions_result.content:
                                    if isinstance(content_item, TextContent):
                                        print(content_item.text)
                                    else:
                                        print(f"Content type: {type(content_item)}")
                            else:
                                print(f"Unexpected exceptions result format: {type(exceptions_result)}")
                        except Exception as e:
                            print(f"\nError finding exceptions: {e}")
                            traceback.print_exc()
                            
                        # Try finding exceptions in a specific file
                        try:
                            file_exceptions_result = await session.call_tool(
                                "find_exceptions_in_file", 
                                {"filepath": "langfuse_mcp/__main__.py", "age": 60}
                            )
                            print("\nExceptions in langfuse_mcp/__main__.py:")
                            
                            # Extract content from the CallToolResult
                            if isinstance(file_exceptions_result, CallToolResult) and file_exceptions_result.content:
                                for content_item in file_exceptions_result.content:
                                    if isinstance(content_item, TextContent):
                                        print(content_item.text)
                                    else:
                                        print(f"Content type: {type(content_item)}")
                            else:
                                print(f"Unexpected file exceptions result format: {type(file_exceptions_result)}")
                        except Exception as e:
                            print(f"\nError finding file exceptions: {e}")
                            traceback.print_exc()
                            
                        # Try running a custom SQL query
                        try:
                            query_result = await session.call_tool(
                                "arbitrary_query", 
                                {
                                    "query": "SELECT COUNT(*) as total_spans FROM spans", 
                                    "age": 60
                                }
                            )
                            print("\nCustom SQL query results:")
                            
                            # Extract content from the CallToolResult
                            if isinstance(query_result, CallToolResult) and query_result.content:
                                for content_item in query_result.content:
                                    if isinstance(content_item, TextContent):
                                        print(content_item.text)
                                    else:
                                        print(f"Content type: {type(content_item)}")
                            else:
                                print(f"Unexpected query result format: {type(query_result)}")
                        except Exception as e:
                            print(f"\nError running custom query: {e}")
                            traceback.print_exc()
                    else:
                        print("Unexpected tools response format")
                        print(f"Raw tools response: {tools_response}")
        except Exception as e:
            print(f"Error connecting to the server: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 