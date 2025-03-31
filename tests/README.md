# Tests for Langfuse MCP

This directory contains tests for the Langfuse MCP package.

## Test Structure

- `test_basic.py`: Basic tests that verify the package can be imported and works with the current Python version.
- `test_mcp_integration.py`: Tests that verify the MCP client can be used with mocked responses.

## Running Tests

To run tests:

```bash
# Run all tests
uv run -m pytest

# Run with verbose output
uv run -m pytest -v

# Run specific test file
uv run -m pytest tests/test_basic.py

# Run a specific test
uv run -m pytest tests/test_basic.py::test_package_importable
```

## GitHub Actions

Tests are run automatically via GitHub Actions when code is pushed to the main branch or when a pull request is created against the main branch.

## Adding New Tests

When adding new tests:

1. Place test files in this directory
2. Name files `test_*.py`
3. Name test functions `test_*`
4. For tests requiring credentials, use mocks to avoid needing real credentials in CI

For integration tests with real Langfuse, use the `pytest.mark.integration` marker and run them manually. 