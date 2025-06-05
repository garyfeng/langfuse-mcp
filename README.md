# Langfuse MCP (Model Context Protocol)

[![Test](https://github.com/avivsinai/langfuse-mcp/actions/workflows/test.yml/badge.svg)](https://github.com/avivsinai/langfuse-mcp/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/langfuse-mcp.svg)](https://badge.fury.io/py/langfuse-mcp)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a Model Context Protocol (MCP) server for Langfuse, allowing AI agents to query Langfuse trace data for better debugging and observability.

## Quick Start with Cursor

[![Add to Cursor](https://img.shields.io/badge/Add%20to-Cursor-blue?style=for-the-badge&logo=cursor)](cursor://mcp/install?name=langfuse-mcp&command=uvx&args=%5B%22langfuse-mcp%22%2C%22--public-key%22%2C%22YOUR_PUBLIC_KEY%22%2C%22--secret-key%22%2C%22YOUR_SECRET_KEY%22%2C%22--host%22%2C%22https%3A//cloud.langfuse.com%22%5D)

*Click the button above to automatically add this MCP server to Cursor IDE (requires Cursor 1.0+)*

**Note:** Remember to replace `YOUR_PUBLIC_KEY` and `YOUR_SECRET_KEY` with your actual Langfuse credentials after installation.

## Features

- Integration with Langfuse for trace and observation data
- Tool suite for AI agents to query trace data
- Exception and error tracking capabilities
- Session and user activity monitoring

## Available Tools

The MCP server provides the following tools for AI agents:

- `fetch_traces` - Find traces based on criteria like user ID, session ID, etc.
- `fetch_trace` - Get a specific trace by ID
- `fetch_observations` - Get observations filtered by type
- `fetch_observation` - Get a specific observation by ID
- `fetch_sessions` - List sessions in the current project
- `get_session_details` - Get detailed information about a session
- `get_user_sessions` - Get all sessions for a user
- `find_exceptions` - Find exceptions and errors in traces
- `find_exceptions_in_file` - Find exceptions in a specific file
- `get_exception_details` - Get detailed information about an exception
- `get_error_count` - Get the count of errors
- `get_data_schema` - Get schema information for the data structures

## Setup

### Install `uv`

First, make sure `uv` is installed. For installation instructions, see the [`uv` installation docs](https://docs.astral.sh/uv/getting-started/installation/).

If you already have an older version of `uv` installed, you might need to update it with `uv self update`.

### Installation

```bash
uv pip install langfuse-mcp
```

### Obtain Langfuse credentials

You'll need your Langfuse credentials:
- Public key
- Secret key
- Host URL (usually https://cloud.langfuse.com or your self-hosted URL)

## Running the Server

Run the server using `uvx`:

```bash
uvx langfuse-mcp --public-key YOUR_KEY --secret-key YOUR_SECRET --host https://cloud.langfuse.com
```

## Configuration with MCP clients

### Configure for Cursor

Create a `.cursor/mcp.json` file in your project root:

```json
{
  "mcpServers": {
    "langfuse": {
      "command": "uvx",
      "args": ["langfuse-mcp", "--public-key", "YOUR_KEY", "--secret-key", "YOUR_SECRET", "--host", "https://cloud.langfuse.com"]
    }
  }
}
```

### Configure for Claude Desktop

Add to your Claude settings:

```json
{
  "command": ["uvx"],
  "args": ["langfuse-mcp"],
  "type": "stdio",
  "env": {
    "LANGFUSE_PUBLIC_KEY": "YOUR_KEY",
    "LANGFUSE_SECRET_KEY": "YOUR_SECRET",
    "LANGFUSE_HOST": "https://cloud.langfuse.com"
  }
}
```

## Output Modes

Each tool supports different output modes to control the level of detail in responses:

- `compact` (default): Returns a summary with large values truncated
- `full_json_string`: Returns the complete data as a JSON string
- `full_json_file`: Saves the complete data to a file and returns a summary with file information

## Development

### Clone the repository

```bash
git clone https://github.com/yourusername/langfuse-mcp.git
cd langfuse-mcp
```

### Create a virtual environment and install dependencies

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Set up environment variables

```bash
export LANGFUSE_SECRET_KEY="your-secret-key"
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_HOST="https://cloud.langfuse.com"  # Or your self-hosted URL
```

### Testing

To run the demo client:

```bash
uv run examples/langfuse_client_demo.py --public-key YOUR_PUBLIC_KEY --secret-key YOUR_SECRET_KEY
```

Or use the convenience wrapper:

```bash
uv run run_mcp.py
```

## Version Management

This project uses dynamic versioning based on Git tags:

1. The version is automatically determined from git tags using `uv-dynamic-versioning`
2. To create a new release:
   - Tag your commit with `git tag v0.1.2` (following semantic versioning)
   - Push the tag with `git push --tags`
   - Create a GitHub release from the tag
3. The GitHub workflow will automatically build and publish the package with the correct version to PyPI

For a detailed history of changes, please see the [CHANGELOG.md](CHANGELOG.md) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Cache Management

We use the `cachetools` library to implement efficient caching with proper size limits:

- Uses `cachetools.LRUCache` for better reliability
- Configurable cache size via the `CACHE_SIZE` constant
- Automatically evicts the least recently used items when caches exceed their size limits
