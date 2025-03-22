#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Check if required environment variables are set
if [ -z "$LANGFUSE_PUBLIC_KEY" ]; then
    echo "Error: LANGFUSE_PUBLIC_KEY environment variable is not set"
    exit 1
fi

if [ -z "$LANGFUSE_SECRET_KEY" ]; then
    echo "Error: LANGFUSE_SECRET_KEY environment variable is not set"
    exit 1
fi

# Set default host if not provided
if [ -z "$LANGFUSE_HOST" ]; then
    LANGFUSE_HOST="https://cloud.langfuse.com"
fi

# Pass through all arguments to the Python script
/opt/homebrew/bin/uv run -m langfuse_mcp \
    --public-key "$LANGFUSE_PUBLIC_KEY" \
    --secret-key "$LANGFUSE_SECRET_KEY" \
    --host "$LANGFUSE_HOST" \
    $([[ "$DEBUG" == "true" ]] && echo "--debug") \
    "$@" 