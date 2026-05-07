# Contributing to LLM Orchestrator

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We're committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

### Fork and Clone

```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/llm-orchestrator.git
cd llm-orchestrator

# Add upstream remote for staying in sync
git remote add upstream https://github.com/your-org/llm-orchestrator.git
```

### Set Up Development Environment

```bash
# Create virtual environment (optional but recommended)
python3.10 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
llm-orchestrate --help
pytest -q  # Should show test count
```

## Development Workflow

### 1. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git rebase upstream/master

# Create feature branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` — New features
- `fix/` — Bug fixes
- `refactor/` — Code refactoring
- `docs/` — Documentation updates
- `test/` — Test improvements

### 2. Make Changes

Follow these principles:

#### Code Style
- **Formatting:** Run `ruff format` before committing
- **Linting:** `ruff check` should pass (E, F, I, W rules)
- **Type checking:** `mypy` should pass with no errors
- **Line length:** 100 characters (enforced by black)

```bash
# Auto-format and check
ruff format llm_orchestrator tests
ruff check llm_orchestrator tests
mypy llm_orchestrator
```

#### Writing Code
- Write async-first code; use `asyncio` for all I/O
- Use `asyncio.to_thread()` for CPU-bound work
- No global mutable state; pass dependencies explicitly
- Type hints required (`mypy --strict` must pass)
- Docstrings for public API; keep internal code self-documenting

#### Good Example

```python
async def decide_next_step(
    self,
    model: str,
    failure_reason: str,
    options: list[tuple[str, Optional[str]]],
) -> dict[str, Any]:
    """Ask advisor what to try next given failure and available options.
    
    Args:
        model: Current model ID
        failure_reason: Why the launch failed (e.g., "CUDA out of memory")
        options: List of (model_id, quantization_variant) tuples
        
    Returns:
        Dict with keys: recommendation, reasoning, confidence, alternatives.
    """
    pipe = self.pipeline
    if pipe is None:
        raise RuntimeError("Model failed to load")
    
    response = await asyncio.to_thread(
        lambda: pipe(prompt, max_new_tokens=500, temperature=0.3)
    )
    # ...
```

### 3. Write Tests

**All new code must include tests.** Target ≥90% coverage.

```bash
# Run tests with coverage
pytest --cov=llm_orchestrator --cov-report=term-missing

# Run specific test
pytest tests/test_advisor.py::TestAdvisorDecisions::test_decide_next_step_success -v
```

#### Test Structure

```python
class TestFeatureName:
    """Tests for FeatureName."""

    def test_basic_behavior(self):
        """Test the main success case."""
        result = some_function()
        assert result is not None

    def test_error_handling(self):
        """Test error case."""
        with pytest.raises(ValueError):
            some_function_that_fails()

    @pytest.mark.asyncio
    async def test_async_behavior(self):
        """Test async function."""
        result = await async_function()
        assert result == expected
```

#### Fixtures

Use fixtures for common test setup:

```python
@pytest.fixture
def mock_pipeline():
    """Mock transformers pipeline."""
    return MagicMock()

@pytest.mark.asyncio
async def test_with_fixture(self, mock_pipeline):
    """Test using fixture."""
    advisor.pipeline = mock_pipeline
    # ...
```

### 4. Commit Changes

Write clear commit messages:

```bash
# Good commit message format:
# <type>(<scope>): <subject>
#
# <body>

git add llm_orchestrator/advisor.py tests/test_advisor.py
git commit -m "refactor(advisor): optimize model loading

- Cache pipeline in instance variable
- Reduce redundant model downloads
- All 14 tests pass"
```

Commit message types:
- `feat` — New feature
- `fix` — Bug fix
- `refactor` — Code restructuring (no behavior change)
- `test` — Test additions/improvements
- `docs` — Documentation updates
- `chore` — Build, dependencies, tooling

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create PR on GitHub
# - Title: short summary (under 70 chars)
# - Description: explain the change, why it's needed, test results
```

#### PR Description Template

```markdown
## Summary
- Brief explanation of what this PR does

## Motivation
- Why is this change needed?
- What problem does it solve?

## Changes
- Detailed list of changes
- One bullet per logical change

## Test Coverage
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] All tests pass: `pytest -q`
- [ ] Coverage maintained/improved: `pytest --cov`
- [ ] Type checking passes: `mypy`
- [ ] Linting passes: `ruff check`

## Verification
- How to verify this works (manual test steps if applicable)
```

## Architecture Guidelines

### Module Responsibilities

| Module | Responsibility |
|--------|-----------------|
| `advisor.py` | Query Qwen model for retry decisions; validate/normalize responses |
| `cli.py` | Parse CLI arguments; delegate to Orchestrator |
| `config.py` | Load/save YAML config; Pydantic models for validation |
| `model_discovery.py` | Query HuggingFace API; discover variants; build fallback chains |
| `monitor.py` | Tail vLLM logs; detect success/failure patterns |
| `orchestrator.py` | Main state machine; coordinate retry loop |

### Adding a New Module

1. Create `llm_orchestrator/new_module.py`
2. Add public API to `llm_orchestrator/__init__.py`
3. Create `tests/test_new_module.py` with comprehensive tests
4. Update `README.md` if user-facing
5. Commit with `feat(new_module): description`

### Extending Advisor

The advisor is designed to be swappable. Currently it uses transformers, but you could:

1. Extend `Advisor` class with new `decide_next_step()` implementation
2. Keep public interface identical for backward compatibility
3. Test with same test suite (mock the underlying model)

## Testing Guidelines

### Unit Tests

```python
# Test one thing per test
def test_validate_decision_clamps_confidence(self):
    """Test that confidence is clamped to 0-1."""
    options = [("M1", None)]
    decision = {"recommendation": "1", "confidence": 1.5}
    
    result = advisor._validate_decision(decision, options)
    
    assert result["confidence"] == 1.0
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_ensure_loaded_loads_model(self):
    """Test lazy loading on first access."""
    advisor.pipeline = None
    
    await advisor._ensure_loaded()
    
    assert advisor.pipeline is not None
```

### Mocking External Services

```python
# Mock HuggingFace API
def test_discover_variants_handles_network_error(self):
    """Test fallback when network fails."""
    with patch("huggingface_hub.model_info", side_effect=Exception("Network error")):
        result = ModelDiscovery.find_variants("Qwen/Qwen3.6-27B")
    
    assert result == {}

# Mock transformers pipeline
def test_advisor_fallback_when_model_unavailable(self):
    """Test graceful fallback."""
    with patch("llm_orchestrator.advisor.pipeline", side_effect=Exception("Load failed")):
        available = await advisor.check_available()
    
    assert available is False
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
async def decide_next_step(
    self,
    model: str,
    failure_reason: str,
    options: list[tuple[str, Optional[str]]],
    hardware_info: Optional[str] = None,
) -> dict[str, Any]:
    """Ask advisor what to try next given failure and available options.

    Args:
        model: Current model ID
        failure_reason: Why the launch failed (e.g., "CUDA out of memory")
        options: List of (model_id, quantization_variant) tuples to choose from
        hardware_info: Hardware description (e.g., "RTX Pro 6000 Blackwell (95GB)")

    Returns:
        Dict with keys: recommendation, reasoning, confidence, alternatives.
        
    Raises:
        RuntimeError: If model fails to load after retry attempts
    """
```

### README Updates

If you add a feature that's user-facing, update README.md:
- Add to features list
- Add CLI reference if new command
- Add usage example
- Update troubleshooting if applicable

## Pull Request Review

PRs require:
- ✅ All tests passing
- ✅ mypy and ruff passing
- ✅ Docstrings for public API
- ✅ Tests for new code (≥90% coverage)
- ✅ Clear commit messages
- ✅ No merge conflicts

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` (if it exists)
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions (if configured) handles PyPI release

## Reporting Issues

### Bug Reports

Include:
- Python version: `python --version`
- OS: `uname -a`
- How to reproduce (minimal example)
- Expected vs actual behavior
- Full error traceback
- `pip freeze` output

### Feature Requests

Include:
- What problem does it solve?
- How should it work (usage example)?
- Any design considerations

## Questions?

- Open a GitHub Issue for public discussion
- Questions in issues are welcome; we help!

---

**Thank you for contributing!** 🎉
