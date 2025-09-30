# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Migrated Langfuse MCP server to use the Langfuse 3.x Python SDK, replacing deprecated v2 fetch_* helpers with the `langfuse.api.*` resource clients.
- Updated test doubles and unit tests to model the v3 API surface and ensure compatibility going forward.
- MCP CLI now reads Langfuse credentials (`public_key`, `secret_key`, `host`) from a `.env` file or environment variables by default, keeping CLI flags optional.
- Normalized output mode handling, tool envelopes, and logging configuration; added CLI options for log level/console output and standardized responses across all tools.
- Docker image installs the local repository (`pip install .`) so containers run the same code under development instead of the last PyPI release, and now bundles `git` so dynamic versioning works during image builds.
- README now documents how to execute the working tree with `uv run --from /path/to/langfuse-mcp` and clarifies why Docker builds should come from the local checkout.

### Removed
- Dropped the temporary Langfuse v2 pin in favour of depending on `langfuse>=3.0.0`.

## [0.1.8] - 2025-01-05

### Added
- Comprehensive test suite with 10 tests covering all functionality
- Enhanced CI workflow with improved logging and verbose output
- Complete documentation with proper docstrings for all test files

### Changed
- Improved GitHub Actions workflow for better visibility and debugging
- Enhanced repository structure with complete test coverage

### Fixed
- All linting and formatting issues resolved
- Proper dependency management and build process

## [0.1.7] - 2025-01-05

### Changed
- Pinned Langfuse dependency to stable v2 branch (`>=2.60,<3.0.0`) for compatibility
- Enhanced CI matrix to test both Langfuse v2 and v3 (v3 allowed to fail)
- Removed uv.lock from version control as recommended for libraries

### Added
- Optional dev-v3 dependency group for future Langfuse v3 migration testing

## [0.1.6] - 2025-04-01

## [0.1.5] - 2025-03-31

### Added
- Enhanced README with detailed output processing information
- Added publish guidelines in project documentation

### Changed
- Improved data processing and output handling
- Increased get_error_count max age limit from 100 minutes to 7 days
- Updated documentation to include README reference in Cursor rules

## [0.1.4] - 2025-03-25

### Added
- Enhanced response processing with truncation for large fields
- Added more robust date parsing
- Improved exception handling

### Changed
- Refactored MCP runner and updated logging 
- Removed Optional type hints from function signatures for better compatibility
- Updated project metadata and build configuration

## [0.1.2] - 2025-03-20

### Added
- Added dynamic versioning with uv-dynamic-versioning
- Version history documentation
- Recommended GitHub Actions for CI/CD

### Changed
- Removed mcp.json from git history and added to gitignore
- Improved test configuration

## [0.1.1] - 2025-03-15

### Added
- Initial release with basic MCP server functionality
- Tool for retrieving traces based on filters
- Tool for finding exceptions grouped by file, function, or type
- Tool for getting detailed exception information
- Tool for retrieving sessions
- Tool for getting error counts
- Tool for fetching data schema

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2025-XX-XX
- Initial release 
