#!/usr/bin/env python
"""
Simple script to test the langfuse-mcp implementation.
Requires the MCP client package.
"""

import asyncio
import os
import traceback
from datetime import datetime, timedelta, timezone
import json
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent


async def main():
    print("Testing langfuse-mcp server...")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Set up environment variables for the MCP server
    env = {
        "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY"),
        "LANGFUSE_HOST": os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    }
    
    # Print the environment variables (without secret key)
    print(f"Using Langfuse host: {env['LANGFUSE_HOST']}")
    print(f"Using public key: {env['LANGFUSE_PUBLIC_KEY']}")
    
    # Create server parameters - add no-auth-check for testing
    server_params = StdioServerParameters(
        command="./run-langfuse-mcp.sh",
        args=["--no-auth-check"],  # Skip authentication check for testing
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
                            print(f"- {tool.name}: {tool.description.split('.')[0]}")
                        
                        # Get the data schema
                        try:
                            schema_result = await session.call_tool("get_data_schema", {})
                            print("\nSchema of Langfuse data:")
                            
                            # Extract content using our helper function
                            schema_data = extract_content(schema_result)
                            if schema_data:
                                print(schema_data)
                            else:
                                print("No schema data found or unexpected format")
                        except Exception as e:
                            print(f"\nError getting schema: {e}")
                            traceback.print_exc()
                        
                        # Try finding traces from the last hour
                        try:
                            # Use the current date (which is in 2025) and look back one hour
                            now = datetime.now(timezone.utc)
                            one_hour_ago = now - timedelta(hours=1)
                            
                            print(f"\nQuerying traces from {one_hour_ago} to {now}")
                            
                            traces_result = await session.call_tool(
                                "find_traces", 
                                {
                                    "from_timestamp": one_hour_ago.isoformat(),
                                    "to_timestamp": now.isoformat(),
                                    "limit": 5
                                }
                            )
                            print("\nTraces from the last hour:")
                            
                            # Extract content using our helper function
                            traces_data = extract_content(traces_result)
                            if traces_data:
                                if isinstance(traces_data, list):
                                    print(json.dumps(traces_data, indent=2))
                                else:
                                    print(traces_data)  # Likely an error message
                            else:
                                print("No traces found or unexpected format")
                        except Exception as e:
                            print(f"\nError finding traces: {e}")
                            traceback.print_exc()
                        
                        # Try finding exceptions with group_by
                        try:
                            exceptions_result = await session.call_tool(
                                "find_exceptions", 
                                {
                                    "age": 60,  # Last hour
                                    "group_by": "file"
                                }
                            )
                            print("\nExceptions in the last hour grouped by file:")
                            
                            # Extract content using our helper function
                            exceptions_data = extract_content(exceptions_result)
                            if exceptions_data:
                                print(json.dumps(exceptions_data, indent=2))
                            else:
                                print("No exceptions found or unexpected format")
                        except Exception as e:
                            print(f"\nError finding exceptions: {e}")
                            traceback.print_exc()
                        
                        # Get the error count
                        try:
                            error_count_result = await session.call_tool(
                                "get_error_count", 
                                {
                                    "age": 1440  # Last 24 hours
                                }
                            )
                            print("\nError count in the last 24 hours:")
                            
                            # Extract content using our helper function
                            error_count_data = extract_content(error_count_result)
                            if error_count_data:
                                print(json.dumps(error_count_data, indent=2))
                            else:
                                print("No error count data found or unexpected format")
                        except Exception as e:
                            print(f"\nError getting error count: {e}")
                            traceback.print_exc()
                        
                        # If we found any traces, try getting exception details for a trace
                        if traces_data and isinstance(traces_data, list) and len(traces_data) > 0:
                            # Get the first trace ID 
                            trace_id = traces_data[0].get("id")
                            if trace_id:
                                print(f"\nGetting exception details for trace {trace_id}:")
                                exception_details_result = await session.call_tool(
                                    "get_exception_details", 
                                    {
                                        "trace_id": trace_id
                                    }
                                )
                                
                                # Extract content using our helper function
                                exception_details_data = extract_content(exception_details_result)
                                if exception_details_data:
                                    print(json.dumps(exception_details_data, indent=2))
                                else:
                                    print("No exception details found or unexpected format")
                            
                            # If we still have the file-based function, test it too
                            for trace in traces_data:
                                if "metadata" in trace and "code.filepath" in trace["metadata"]:
                                    filepath = trace["metadata"]["code.filepath"]
                                    print(f"\nGetting exception details for file {filepath}:")
                                    exceptions_in_file_result = await session.call_tool(
                                        "find_exceptions_in_file", 
                                        {
                                            "filepath": filepath,
                                            "age": 1440  # Last 24 hours
                                        }
                                    )
                                    
                                    # Extract content using our helper function
                                    exceptions_in_file_data = extract_content(exceptions_in_file_result)
                                    if exceptions_in_file_data:
                                        print(json.dumps(exceptions_in_file_data, indent=2))
                                    else:
                                        print("No file exception details found or unexpected format")
                                        
                                    # Break after the first trace with a filepath
                                    break
                        
                        # Try the get_user_sessions tool if user_id is available
                        try:
                            # Try to get a user_id from traces data if available
                            user_id = "test_user"  # Default fallback
                            if traces_data and isinstance(traces_data, list):
                                for trace in traces_data:
                                    if "user_id" in trace and trace["user_id"]:
                                        user_id = trace["user_id"]
                                        break
                                    
                            print(f"\nGetting sessions for user {user_id}:")
                            user_sessions_result = await session.call_tool(
                                "get_user_sessions", 
                                {
                                    "user_id": user_id
                                }
                            )
                            
                            # Extract content using our helper function
                            user_sessions_data = extract_content(user_sessions_result)
                            if user_sessions_data:
                                print(json.dumps(user_sessions_data, indent=2))
                            else:
                                print("No user sessions found or unexpected format")
                        except Exception as e:
                            print(f"\nError getting user sessions: {e}")
                            traceback.print_exc()
                            
                    else:
                        print("Unexpected tools response format")
                        print(f"Raw tools response: {tools_response}")
        except Exception as e:
            print(f"Error connecting to the server: {e}")
            traceback.print_exc()


def extract_content(result):
    """Helper function to extract content from a CallToolResult"""
    if isinstance(result, CallToolResult) and result.content:
        for content_item in result.content:
            if isinstance(content_item, TextContent):
                try:
                    return json.loads(content_item.text)
                except json.JSONDecodeError:
                    return content_item.text
    return None


if __name__ == "__main__":
    asyncio.run(main()) 