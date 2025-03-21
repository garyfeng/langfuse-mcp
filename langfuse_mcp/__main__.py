import argparse
import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, List, Union

import duckdb
import pandas as pd
from langfuse import Langfuse
from mcp.server.fastmcp import Context, FastMCP
from pydantic import AfterValidator, BaseModel

HOUR = 60  # minutes
DAY = 24 * HOUR


class MCPState:
    def __init__(self, langfuse_client: Langfuse):
        self.langfuse_client = langfuse_client


class ExceptionCount(BaseModel):
    filepath: str | None
    count: int


def validate_age(age: int) -> int:
    """Validate that age is positive and ≤ 7 days."""
    if age <= 0:
        raise ValueError("Age must be positive")
    if age > 7 * DAY:
        raise ValueError("Age cannot be more than 7 days")
    return age


ValidatedAge = Annotated[int, AfterValidator(validate_age)]


async def load_data_into_duckdb(langfuse_client: Langfuse, age: int):
    """Fetch spans from Langfuse and load them into DuckDB."""
    min_timestamp = datetime.now(UTC) - timedelta(minutes=age)
    # Fetch spans synchronously (no async fetch available in this SDK version)
    spans_response = await asyncio.to_thread(
        langfuse_client.get_observations,
        from_start_time=min_timestamp,
        type="SPAN",
    )
    spans = spans_response.data

    # Convert spans to a list of dictionaries with safe attribute access
    span_dicts = []
    for span in spans:
        # Convert span to dict safely
        span_dict = {}
        for attr in ['id', 'trace_id', 'name', 'start_time', 'end_time', 'metadata']:
            span_dict[attr] = getattr(span, attr, None)
        span_dicts.append(span_dict)
    
    # Create spans DataFrame
    spans_df = pd.DataFrame(span_dicts)
    if spans_df.empty:
        # Create empty DataFrame with required columns
        spans_df = pd.DataFrame(columns=['id', 'trace_id', 'name', 'start_time', 'end_time', 'metadata'])

    # Extract events from spans
    events = []
    for span in spans:
        # Check if span has events attribute
        span_events = getattr(span, 'events', None)
        if span_events:
            for event in span_events:
                events.append(
                    {
                        "span_id": span.id,
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": getattr(event, 'metadata', {}),  # Safely get metadata
                    }
                )
    
    # Create events DataFrame
    events_df = pd.DataFrame(events)
    if events_df.empty:
        # Create empty DataFrame with required columns
        events_df = pd.DataFrame(columns=['span_id', 'name', 'timestamp', 'attributes'])

    # Create DuckDB connection and register tables
    con = duckdb.connect()
    con.register("spans", spans_df)
    con.register("events", events_df)
    return con


async def find_exceptions(ctx: Context, age: ValidatedAge) -> List[ExceptionCount]:
    """Get exception counts grouped by file path from spans.

    Args:
        ctx: MCP context providing access to server state.
        age: Number of minutes to look back (max 7 days).

    Returns:
        List of ExceptionCount objects with filepath and count.
    """
    state = ctx.request_context.lifespan_context
    langfuse_client = state.langfuse_client
    con = await load_data_into_duckdb(langfuse_client, age)
    
    # First check if we have any events with exceptions
    check_query = """
    SELECT COUNT(*) as count
    FROM events 
    WHERE attributes->>'exception.type' IS NOT NULL
    """
    count_df = con.sql(check_query).df()
    count = count_df['count'].iloc[0] if not count_df.empty else 0
    
    if count == 0:
        # No exceptions found
        return [ExceptionCount(filepath="No exceptions found in the last " + str(age) + " minutes", count=0)]
    
    # If we have exceptions, run the original query
    query = """
    SELECT metadata->>'code.filepath' as filepath, COUNT(*) as count
    FROM spans
    WHERE id IN (
        SELECT span_id 
        FROM events 
        WHERE attributes->>'exception.type' IS NOT NULL
    )
    GROUP BY filepath
    """
    result_df = con.sql(query).df()
    
    # If we got no results (could happen with empty or malformed data)
    if result_df.empty:
        return [ExceptionCount(filepath="No exceptions with valid filepath found", count=0)]
        
    return [ExceptionCount(**row) for row in result_df.to_dict(orient="records")]


async def find_exceptions_in_file(
    ctx: Context, filepath: str, age: ValidatedAge
) -> List[dict]:
    """Get detailed info about exceptions in a specific file.

    Args:
        ctx: MCP context providing access to server state.
        filepath: Path to the file to analyze.
        age: Number of minutes to look back (max 7 days).

    Returns:
        List of dictionaries with exception details.
    """
    state = ctx.request_context.lifespan_context
    langfuse_client = state.langfuse_client
    con = await load_data_into_duckdb(langfuse_client, age)
    
    # First check if we have any exceptions for this file
    check_query = """
    SELECT COUNT(*) as count
    FROM spans s
    JOIN events e ON s.id = e.span_id
    WHERE s.metadata->>'code.filepath' = ?
    AND e.attributes->>'exception.type' IS NOT NULL
    """
    count_df = con.sql(check_query, params=[filepath]).df()
    count = count_df['count'].iloc[0] if not count_df.empty else 0
    
    if count == 0:
        # No exceptions found for this file
        return [{"message": f"No exceptions found for file '{filepath}' in the last {age} minutes"}]
    
    # If we have exceptions, run the detailed query
    query = """
    SELECT 
        e.timestamp, 
        e.attributes->>'exception.message' as message, 
        e.attributes->>'exception.type' as exception_type,
        s.metadata->>'code.function' as function_name, 
        s.metadata->>'code.lineno' as line_number,
        s.trace_id, 
        s.id as span_id
    FROM spans s
    JOIN events e ON s.id = e.span_id
    WHERE s.metadata->>'code.filepath' = ?
    AND e.attributes->>'exception.type' IS NOT NULL
    ORDER BY e.timestamp DESC
    LIMIT 10
    """
    result_df = con.sql(query, params=[filepath]).df()
    
    # If we got no results (could happen with empty or malformed data)
    if result_df.empty:
        return [{"message": f"No valid exception data found for file '{filepath}'"}]
        
    return result_df.to_dict(orient="records")


async def arbitrary_query(ctx: Context, query: str, age: ValidatedAge) -> Union[List[dict], dict]:
    """Run a custom SQL query on spans and events.

    Args:
        ctx: MCP context providing access to server state.
        query: SQL query to run on the data.
        age: Number of minutes to look back (max 7 days).

    Returns:
        Either list of dictionaries with query results or error message.
    """
    state = ctx.request_context.lifespan_context
    langfuse_client = state.langfuse_client
    
    try:
        con = await load_data_into_duckdb(langfuse_client, age)
        result_df = con.sql(query).df()
        
        if result_df.empty:
            return {"message": "Query executed successfully but returned no results"}
            
        # Convert result to records
        return result_df.to_dict(orient="records")
    except Exception as e:
        # Return a helpful error message
        return {"error": f"Error executing query: {str(e)}"}


async def get_langfuse_schema(ctx: Context) -> str:
    """Get the schema of the spans and events tables.

    Args:
        ctx: MCP context (unused here).

    Returns:
        String describing the schema.
    """
    return """
The Langfuse MCP provides access to span data through two tables: `spans` and `events`.

### Table `spans`:
- **id**: VARCHAR - Unique identifier of the span
- **trace_id**: VARCHAR - Identifier of the trace this span belongs to
- **name**: VARCHAR - Name of the span
- **start_time**: TIMESTAMP - Start time of the span
- **end_time**: TIMESTAMP - End time of the span (nullable)
- **metadata**: JSON - Additional span attributes (e.g., 'code.filepath', 'code.function', 'code.lineno')
- **events**: JSON - Array of event objects (nested, but flattened into `events` table)

### Table `events`:
- **span_id**: VARCHAR - Identifier of the span this event belongs to
- **name**: VARCHAR - Name of the event
- **timestamp**: TIMESTAMP - Time the event occurred
- **attributes**: JSON - Event metadata (e.g., 'exception.type', 'exception.message')

You can query these tables using SQL, including JSON functions to access nested fields.

#### Example Queries:
- **Find exceptions:**
  ```sql
  SELECT span_id, attributes->>'exception.type' as exception_type
  FROM events
  WHERE attributes->>'exception.type' IS NOT NULL
  ```

- **Access metadata:**
  ```sql
  SELECT id, metadata->>'code.filepath' as filepath
  FROM spans
  WHERE metadata->>'code.filepath' IS NOT NULL
  ```
"""


def app_factory(public_key: str, secret_key: str, host: str) -> FastMCP:
    """Create and configure the FastMCP server."""

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[MCPState]:
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        yield MCPState(langfuse_client=client)
        client.shutdown()  # Ensure proper cleanup

    mcp = FastMCP("Langfuse", lifespan=lifespan)
    mcp.tool()(find_exceptions)
    mcp.tool()(find_exceptions_in_file)
    mcp.tool()(arbitrary_query)
    mcp.tool()(get_langfuse_schema)
    return mcp


def main():
    """Parse arguments and run the MCP server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-key",
        type=str,
        required=False,
        help="Langfuse public key. Can also be set via LANGFUSE_PUBLIC_KEY environment variable.",
    )
    parser.add_argument(
        "--secret-key",
        type=str,
        required=False,
        help="Langfuse secret key. Can also be set via LANGFUSE_SECRET_KEY environment variable.",
    )
    parser.add_argument(
        "--host",
        type=str,
        required=False,
        help="Langfuse host URL. Can also be set via LANGFUSE_HOST environment variable. Defaults to https://cloud.langfuse.com",
    )
    args = parser.parse_args()

    public_key = args.public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = args.secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = args.host or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"

    if not public_key or not secret_key:
        parser.error(
            "Langfuse public key and secret key must be provided via --public-key and --secret-key arguments "
            "or LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables"
        )

    app = app_factory(public_key, secret_key, host)
    app.run(transport="stdio")


if __name__ == "__main__":
    main() 