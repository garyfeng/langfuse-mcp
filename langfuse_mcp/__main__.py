import argparse
import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, List, TypedDict

import duckdb
import pandas as pd
from langfuse import Langfuse
from mcp.server.fastmcp import Context, FastMCP
from pydantic import AfterValidator, BaseModel

HOUR = 60  # minutes
DAY = 24 * HOUR

class MCPState:
    langfuse_client: Langfuse

class ExceptionCount(BaseModel):
    filepath: str | None
    count: int

def validate_age(age: int) -> int:
    """Validate that the age is within acceptable bounds (positive and <= 7 days)."""
    if age <= 0:
        raise ValueError("Age must be positive")
    if age > 7 * DAY:
        raise ValueError("Age cannot be more than 7 days")
    return age

ValidatedAge = Annotated[int, AfterValidator(validate_age)]

async def find_exceptions(ctx: Context, age: ValidatedAge) -> List[ExceptionCount]:
    """Get the exceptions on a file.

    Args:
        age: Number of minutes to look back, e.g. 30 for last 30 minutes. Maximum allowed value is 7 days.
    """
    state = ctx.request_context.lifespan_context
    min_timestamp = datetime.now(UTC) - timedelta(minutes=age)
    spans = await asyncio.to_thread(
        state.langfuse_client.get_spans,
        from_timestamp=min_timestamp,
    )
    exception_counts = {}
    for span in spans:
        if hasattr(span, 'events'):
            for event in span.events:
                if hasattr(event, 'attributes') and 'exception.type' in event.attributes:
                    filepath = span.metadata.get('code.filepath') if hasattr(span, 'metadata') else None
                    if filepath:
                        exception_counts[filepath] = exception_counts.get(filepath, 0) + 1
    return [ExceptionCount(filepath=fp, count=count) for fp, count in exception_counts.items()]

async def find_exceptions_in_file(ctx: Context, filepath: str, age: ValidatedAge) -> List[dict]:
    """Get the details about the 10 most recent exceptions on the file.

    Args:
        filepath: The path to the file to find exceptions in.
        age: Number of minutes to look back, e.g. 30 for last 30 minutes. Maximum allowed value is 7 days.
    """
    state = ctx.request_context.lifespan_context
    min_timestamp = datetime.now(UTC) - timedelta(minutes=age)
    spans = await asyncio.to_thread(
        state.langfuse_client.get_spans,
        from_timestamp=min_timestamp,
    )
    exceptions = []
    for span in spans:
        if hasattr(span, 'metadata') and span.metadata.get('code.filepath') == filepath:
            if hasattr(span, 'events'):
                for event in span.events:
                    if hasattr(event, 'attributes') and 'exception.type' in event.attributes:
                        exceptions.append({
                            'created_at': event.timestamp,
                            'message': event.attributes.get('exception.message'),
                            'exception_type': event.attributes.get('exception.type'),
                            'function_name': span.metadata.get('code.function'),
                            'line_number': span.metadata.get('code.lineno'),
                            'trace_id': span.trace_id,
                            'span_id': span.id,
                        })
    exceptions.sort(key=lambda x: x['created_at'], reverse=True)
    return exceptions[:10]

async def arbitrary_query(ctx: Context, query: str, age: ValidatedAge) -> List[dict]:
    """Run an arbitrary query on the Langfuse spans and events.

    The schema is available via the `get_langfuse_schema` tool.

    Args:
        query: The query to run, as a SQL string.
        age: Number of minutes to look back, e.g. 30 for last 30 minutes. Maximum allowed value is 7 days.
    """
    state = ctx.request_context.lifespan_context
    min_timestamp = datetime.now(UTC) - timedelta(minutes=age)
    spans = await asyncio.to_thread(
        state.langfuse_client.get_spans,
        from_timestamp=min_timestamp,
    )
    span_dicts = [span.dict() for span in spans]
    df = pd.DataFrame(span_dicts)
    con = duckdb.connect()
    con.register("spans", df)
    con.execute("""
    CREATE TABLE events AS
    SELECT id AS span_id, unnest(events) AS event
    FROM spans
    """)
    result_df = con.sql(query).df()
    return result_df.to_dict(orient='records')

async def get_langfuse_schema(ctx: Context) -> str:
    """Get the schema of the spans and events tables.

    To perform the `arbitrary_query` tool, you can use this schema to understand the structure of the data.
    """
    return """
The Langfuse MCP provides access to span data through two tables: `spans` and `events`.

Table `spans`:
- id: VARCHAR
- trace_id: VARCHAR
- name: VARCHAR
- start_time: TIMESTAMP
- end_time: TIMESTAMP
- metadata: JSON (contains span attributes, e.g., 'code.filepath', 'code.function', 'code.lineno')
- events: JSON (array of event objects)

Table `events`:
- span_id: VARCHAR
- event: JSON (contains event fields: 'name', 'timestamp', 'attributes')

You can query these tables using SQL, including JSON functions to access nested fields.
For example, to access the exception type from an event:
SELECT span_id, event->>'name', event->'attributes'->>'exception.type' as exception_type
FROM events
WHERE event->'attributes'->>'exception.type' IS NOT NULL

Similarly, to access metadata from spans:
SELECT id, metadata->>'code.filepath' as filepath
FROM spans
WHERE metadata->>'code.filepath' IS NOT NULL
"""

def app_factory(public_key: str, secret_key: str, base_url: str) -> FastMCP:
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[MCPState]:
        client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)
        yield MCPState(langfuse_client=client)
        # Cleanup if necessary

    mcp = FastMCP("Langfuse", lifespan=lifespan)
    mcp.tool()(find_exceptions)
    mcp.tool()(find_exceptions_in_file)
    mcp.tool()(arbitrary_query)
    mcp.tool()(get_langfuse_schema)
    return mcp

def main():
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
        "--base-url",
        type=str,
        required=False,
        help="Langfuse base URL. Can also be set via LANGFUSE_BASE_URL environment variable. "
        "Defaults to https://cloud.langfuse.com",
    )
    args = parser.parse_args()

    public_key = args.public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = args.secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    base_url = args.base_url or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"

    if not public_key or not secret_key:
        parser.error(
            "Langfuse public key and secret key must be provided either via --public-key and --secret-key arguments "
            "or LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables"
        )

    app = app_factory(public_key, secret_key, base_url)
    app.run(transport="stdio")

if __name__ == "__main__":
    main() 