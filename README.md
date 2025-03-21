# Langfuse MCP Server

This repository contains a Model Context Protocol (MCP) server with tools that can access traces and metrics stored in Langfuse, mirroring the functionality of `logfire-mcp`.

This MCP server enables LLMs to retrieve your application's telemetry data, analyze distributed traces, and execute arbitrary SQL queries on the fetched data.

## Available Tools

- **`find_exceptions`** - Get exception counts from spans grouped by file
  - **Required arguments:**
    - `age` (int): Number of minutes to look back (e.g., 30 for last 30 minutes, max 7 days)

- **`find_exceptions_in_file`** - Get detailed span information about exceptions in a specific file
  - **Required arguments:**
    - `filepath` (string): Path to the file to analyze
    - `age` (int): Number of minutes to look back (max 7 days)

- **`arbitrary_query`** - Run custom SQL queries on your spans and events
  - **Required arguments:**
    - `query` (string): SQL query to execute on the in-memory database
    - `age` (int): Number of minutes to look back (max 7 days)

- **`get_langfuse_schema`** - Get the schema of the spans and events tables
  - **No required arguments**

## Setup

### Install `uv`

Ensure `uv` is installed to manage dependencies and run the server. See the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

### Obtain Langfuse Credentials

You need your Langfuse public key and secret key from your project settings (cloud or self-hosted).

### Manually Run the Server

Use environment variables:

```bash
LANGFUSE_PUBLIC_KEY=YOUR_PUBLIC_KEY LANGFUSE_SECRET_KEY=YOUR_SECRET_KEY uvx langfuse-mcp
```

Or command-line flags:

```bash
uvx langfuse-mcp --public-key=YOUR_PUBLIC_KEY --secret-key=YOUR_SECRET_KEY
```

## Configuration with MCP Clients

### Cursor

Create a `.cursor/mcp.json` file in your project root:

```json
{
  "mcpServers": {
    "langfuse": {
      "command": "uvx",
      "args": ["langfuse-mcp", "--public-key=YOUR_PUBLIC_KEY", "--secret-key=YOUR_SECRET_KEY"]
    }
  }
}
```

### Claude Desktop

Add to your Claude settings:

```json
{
  "command": ["uvx"],
  "args": ["langfuse-mcp"],
  "type": "stdio",
  "env": {
    "LANGFUSE_PUBLIC_KEY": "YOUR_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY": "YOUR_SECRET_KEY"
  }
}
```

### Cline

Add to your Cline settings in `cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "langfuse": {
      "command": "uvx",
      "args": ["langfuse-mcp"],
      "env": {
        "LANGFUSE_PUBLIC_KEY": "YOUR_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY": "YOUR_SECRET_KEY"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### Customization - Host URL

Default API endpoint is `https://cloud.langfuse.com`. Override it with:

1. Command-line argument:
   ```bash
   uvx langfuse-mcp --host=https://your-langfuse-instance.com
   ```

2. Environment variable:
   ```bash
   LANGFUSE_HOST=https://your-langfuse-instance.com uvx langfuse-mcp
   ```

## Example Interactions

1. **Find exceptions in the last hour:**
   ```json
   {
     "name": "find_exceptions",
     "arguments": {
       "age": 60
     }
   }
   ```
   **Response:**
   ```json
   [
     {"filepath": "app/main.py", "count": 5},
     {"filepath": "utils/helper.py", "count": 3}
   ]
   ```

2. **Get exception details in a file:**
   ```json
   {
     "name": "find_exceptions_in_file",
     "arguments": {
       "filepath": "app/main.py",
       "age": 1440
     }
   }
   ```
   **Response:**
   ```json
   [
     {
       "timestamp": "2024-10-15T10:00:00Z",
       "message": "Division by zero",
       "exception_type": "ZeroDivisionError",
       "function_name": "divide",
       "line_number": "45",
       "trace_id": "abc123",
       "span_id": "def456"
     }
   ]
   ```

3. **Run a custom query:**
   ```json
   {
     "name": "arbitrary_query",
     "arguments": {
       "query": "SELECT id, trace_id, name FROM spans WHERE name = 'process_data' LIMIT 10",
       "age": 1440
     }
   }
   ```

## Examples of Questions for Claude

1. "What exceptions occurred in spans from the last hour?"
2. "Show recent errors in 'app/main.py' with their span context."
3. "How many errors were there in the last 24 hours per file?"
4. "What are the most common exception types in my spans?"
5. "Get me the schema for spans and events."
6. "Find all errors from yesterday and show their contexts."

## Getting Started

1. Obtain your Langfuse public and secret keys from your project settings.
2. Run the MCP server:
   ```bash
   uvx langfuse-mcp --public-key=YOUR_PUBLIC_KEY --secret-key=YOUR_SECRET_KEY
   ```
3. Configure your preferred client (Cursor, Claude Desktop, or Cline).
4. Start analyzing your Langfuse data!

## Contributing

Contributions are welcome! Add new tools, enhance querying, or improve docs. See the [Model Context Protocol servers repository](https://github.com/modelcontextprotocol/servers) for examples.

## License

Licensed under the MIT License.
