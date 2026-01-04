# Contributing to agent-observe

Thank you for your interest in contributing to agent-observe! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all backgrounds and experience levels.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/agent-observe/agent-observe.git
   cd agent-observe
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent_observe --cov-report=html

# Run specific test file
pytest tests/test_policy.py
```

### Code Quality

We use the following tools to maintain code quality:

```bash
# Linting
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Type checking
mypy agent_observe
```

### Code Style

- We follow PEP 8 with a line length of 100 characters
- Use type hints for all function signatures
- Write docstrings for public functions and classes
- Keep functions focused and concise

## Making Changes

### Branching

1. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-fix-name
   ```

### Commits

- Write clear, concise commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issues when applicable

### Pull Requests

1. Ensure all tests pass
2. Ensure linting and type checking pass
3. Update documentation if needed
4. Update CHANGELOG.md for notable changes
5. Submit a pull request with a clear description

## Project Structure

```
agent_observe/
├── __init__.py      # Public API exports
├── observe.py       # Core runtime
├── decorators.py    # @tool, @model_call decorators
├── policy.py        # Policy engine
├── metrics.py       # Risk scoring and evaluation
├── replay.py        # Tool result caching
├── config.py        # Configuration management
├── context.py       # Run and span context
├── hashing.py       # Content hashing utilities
├── cli.py           # Command-line interface
├── sinks/           # Storage backends
│   ├── base.py
│   ├── sqlite_sink.py
│   ├── postgres_sink.py
│   ├── jsonl_sink.py
│   └── otel_sink.py
└── viewer/          # Web UI
    ├── app.py
    └── templates/
```

## Testing Guidelines

- Write tests for all new functionality
- Maintain or improve code coverage
- Use descriptive test names
- Group related tests in classes

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

## Questions?

Feel free to open an issue for questions or discussions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
