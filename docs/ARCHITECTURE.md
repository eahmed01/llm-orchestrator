# Architecture Overview

This document describes the system architecture, design decisions, and how components interact.

## System Design

### High-Level Flow

```
User Request:
  "llm-orchestrate start vllm --model Qwen/Qwen3.6-27B-FP8"
              ↓
    ┌─────────────────────┐
    │  CLI (typer app)    │  Parse args, validate input
    └──────────┬──────────┘
               ↓
    ┌─────────────────────────────────────────┐
    │  Orchestrator.start()                   │
    │  ├─ Load/validate model                 │
    │  ├─ Viability check (ask advisor)       │
    │  └─ Retry loop (MAX_RETRIES=5)         │
    │      ├─ Spawn vLLM subprocess          │
    │      ├─ Monitor logs (5 min timeout)   │
    │      ├─ Success? → Save config & exit  │
    │      ├─ Failure? → Ask advisor        │
    │      └─ Retry with new model/args      │
    └──────────┬──────────────────────────────┘
               ↓
    ┌──────────────────────┐
    │  Config saved to:    │
    │  ~/.config/          │
    │  llm-orchestrator/   │
    │  working-models.yaml │
    └──────────────────────┘
```

## Core Components

### 1. Orchestrator (`orchestrator.py`)

**Responsibility:** Retry loop state machine

**Key Methods:**
- `start()` — Main entry point; orchestrates the entire startup flow
- `_attempt_startup()` — Spawn vLLM subprocess and monitor
- `_build_vllm_command()` — Construct shell command with args

**Design Decisions:**
- **State machine pattern** — Clear, predictable flow
- **Max retries hardcoded** — Prevents infinite loops
- **Async/await** — Non-blocking I/O for monitoring
- **Dependency injection** — Advisor, Config, Monitor passed to constructor

**Example Flow:**
```python
orchestrator = Orchestrator(service="vllm", model="Qwen/Qwen3.6-27B-FP8")
success = await orchestrator.start()

# Internally:
# 1. Check if model valid
# 2. Loop (max 5 retries):
#    a. Spawn process
#    b. Monitor logs
#    c. If success: save config & break
#    d. If fail: ask advisor, update model, continue
```

### 2. Advisor (`advisor.py`)

**Responsibility:** Intelligent retry decision-making

**Key Methods:**
- `check_available()` — Verify model can load (pre-flight check)
- `decide_next_step()` — Given failure + options, recommend best next step
- `estimate_viability()` — Estimate if model will fit on hardware
- `_ensure_loaded()` — Lazy load transformers model on first use

**Design Decisions:**
- **In-process model loading** — No external service; uses `transformers.pipeline()`
- **Lazy loading** — Model downloads + caches only when first needed
- **JSON response parsing** — Extract structured decision from model output
- **Graceful fallback** — If advisor fails, return heuristic (first option, low confidence)

**Technical Details:**

```python
# Model: Qwen/Qwen2.5-Coder-1.5B
# Runs on: CPU (via asyncio.to_thread to avoid blocking)
# Cache location: ~/.cache/huggingface/
# Size: ~3GB (quantized)

# Public interface stays identical:
# - Takes same args regardless of implementation
# - Returns same dict structure
# - Tests mock transformers.pipeline, not HTTP calls
```

### 3. Monitor (`monitor.py`)

**Responsibility:** Real-time log monitoring for startup success/failure

**Key Methods:**
- `monitor_startup()` — Tail logs, detect patterns, return (success, reason)
- `reset()` — Reset file position for new attempt

**Design Decisions:**
- **Regex pattern matching** — Cheap, no ML overhead
- **Incremental file reading** — Don't re-read entire log each iteration
- **Timeout handling** — Return failure after N seconds
- **Error extraction** — Return raw error line to advisor

**Patterns Detected:**

Success:
```
Application startup complete
Uvicorn running on
Started server process
```

Failure:
```
CUDA out of memory
OutOfMemoryError
RuntimeError.*cuda
Failed to load model
```

### 4. Model Discovery (`model_discovery.py`)

**Responsibility:** Find quantized variants on HuggingFace; build fallback chains

**Key Methods:**
- `find_variants()` — Query HuggingFace API, extract file sizes
- `build_fallback_chain()` — Construct ordered list of models to try
- `get_model_info()` — Fetch model metadata (downloads, tags, etc.)
- `validate_model_exists()` — Pre-flight check that model exists

**Design Decisions:**
- **Size-based heuristics** — Extract "27B", "7B", "1.5B" from model name
- **Quantization priority** — Q4 > Q5 > FP8 > AWQ (size reduction)
- **Graceful degradation** — Fallback to smaller models only if necessary
- **Deterministic ordering** — Same input always produces same chain

**Example Fallback Chain:**

Starting with: `Qwen/Qwen3.6-27B-FP8`

```
[
  ("Qwen/Qwen3.6-27B-FP8", None),           # Original (try first)
  ("Qwen/Qwen3.6-27B-Q4", "q4_k_m"),       # Quantized same size
  ("Qwen/Qwen3.6-27B-Q5", "q5_k_m"),       # Quantized same size
  ("Qwen/Qwen3.6-7B-FP8", None),           # Smaller, full precision
  ("Qwen/Qwen3.6-7B-Q4", "q4_k_m"),        # Smaller, quantized
  ("Qwen/Qwen2.5-Coder-1.5B-Instruct", None), # Tiny fallback
  ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "q4_k_m"), # Tiny, quantized
]
```

### 5. Config (`config.py`)

**Responsibility:** Load/save working configurations to YAML

**Key Classes:**
- `ServiceConfig` — Pydantic model for one service
- `OrchestratorConfig` — Top-level config (can have multiple services)

**Design Decisions:**
- **Pydantic validation** — Type-safe defaults, coercion
- **YAML persistence** — Human-readable (can edit by hand)
- **No encryption** — Config is not sensitive (just model IDs + args)
- **Single file** — `~/.config/llm-orchestrator/working-models.yaml`

**Config Example:**
```yaml
vllm:
  model: Qwen/Qwen3.6-27B-Q4
  variant: q4_k_m
  args:
    --max-num-seqs: 848
    --gpu-memory-utilization: 0.92
  last_successful: "2026-05-07T12:35:00Z"
```

### 6. CLI (`cli.py`)

**Responsibility:** Command-line interface; argument parsing and delegation

**Commands:**
- `start` — Orchestrate startup
- `status` — Show current config
- `stop` — Stop service (placeholder)
- `config` — Manage config file
- `discover` — Find variants on HuggingFace

**Design Decisions:**
- **Typer framework** — Type hints + automatic help
- **Async support** — `typer` runs async functions seamlessly
- **Simple delegation** — CLI just parses args and calls Orchestrator

## Data Flow

### Retry Cycle (Detailed)

```
Attempt N:
│
├─ Build command: ["python", "-m", "vllm", "serve", model, "--port", "7999", ...]
│
├─ Spawn subprocess (asyncio.create_subprocess_exec)
│  └─ vLLM process starts in background
│     ├─ Loads model from disk/cache
│     ├─ Initializes CUDA
│     ├─ Starts inference engine
│     └─ Writes logs to /work/fast3/.../vllm.log
│
├─ Monitor logs (tail file, check patterns)
│  ├─ Read new lines since last position
│  ├─ Match against SUCCESS_PATTERNS
│  │  └─ If match: return (True, None) → DONE ✓
│  ├─ Match against ERROR_PATTERNS
│  │  └─ If match: return (False, error_line)
│  └─ Loop until success, error, or timeout (300s)
│
├─ On success: Save config, exit
│
└─ On failure:
   │
   ├─ Get failure_reason from monitor ("CUDA out of memory", etc.)
   │
   ├─ Build fallback chain (ModelDiscovery.build_fallback_chain)
   │  └─ [next_model_1, next_model_2, ..., tiny_fallback]
   │
   ├─ Query advisor:
   │  └─ Prompt: "Model X failed with Y. Pick from options Z."
   │  └─ Response: {"recommendation": "2", "confidence": 0.85, ...}
   │
   ├─ Validate advisor response (clamp bounds, ensure dict)
   │
   ├─ Extract recommendation: fallback_chain[1]
   │  └─ Update self.model = next_model
   │
   └─ Increment retry_count, continue loop (goto Attempt N+1)
```

## Decision: In-Process vs External Advisor

### Original Design (Ollama)
```
User → CLI → Orchestrator ─┐
                            ├→ Ollama HTTP API → Qwen2.5-Coder (served)
                            └─ vLLM (main model)
```

**Pros:**
- Ollama binary installed separately
- Reuse same Ollama for multiple apps

**Cons:**
- Requires Ollama binary + service running
- Extra setup step for users
- HTTP latency (small but unnecessary)

### New Design (In-Process)
```
User → CLI → Orchestrator
              ├─ transformers.pipeline("text-generation", model=..., device="cpu")
              │  └─ Auto-download & cache from HuggingFace (first run only)
              └─ vLLM (main model)
```

**Pros:**
- One-click setup; no external service
- Smaller latency
- Easier deployment

**Cons:**
- First run downloads ~3GB (transformers + model)
- Memory overhead (both models in memory)
- Can't reuse advisor across processes

**Trade-off Accepted:** Simplicity > efficiency for small advisor model

## Concurrency Model

### Design Principles

1. **Async-first** — All I/O operations use `asyncio`
2. **Non-blocking CPU** — Use `asyncio.to_thread()` for transformers
3. **Single event loop** — No threading except for blocking I/O
4. **Lazy loading** — Model loads only when first used

### Example: Advisor Loading

```python
# First call to advisor.decide_next_step():
# 1. Call await self._ensure_loaded()
# 2. Check if self.pipeline is already loaded (fast path)
# 3. If not, acquire lock (self._loading = True)
# 4. Load model in thread: await asyncio.to_thread(lambda: pipeline(...))
# 5. Future calls skip to step 2 (cached)

async def _ensure_loaded(self) -> None:
    if self.pipeline is not None or self._loading:
        return  # Already loaded or loading
    
    self._loading = True  # Prevent duplicate loads
    try:
        self.pipeline = await asyncio.to_thread(
            lambda: pipeline("text-generation", model=self.MODEL, device="cpu")
        )
    except Exception as e:
        self._loading = False
        raise
    finally:
        if self.pipeline is None:
            self._loading = False  # Failed; allow retry
```

## Testing Strategy

### Unit Tests

Each module has dedicated tests:

| Module | Tests | Coverage |
|--------|-------|----------|
| advisor.py | 17 | Mocks transformers.pipeline |
| config.py | 10 | Mocks file system |
| model_discovery.py | 8 | Tests chain building |
| monitor.py | 8 | Mocks log files |
| orchestrator.py | 10 | Mocks subprocess |

### Mocking Strategy

```python
# Mock transformers.pipeline (don't actually load model)
with patch("llm_orchestrator.advisor.pipeline") as mock_pipeline:
    mock_pipeline.return_value = MagicMock()
    # Test advisor logic without loading real model

# Mock subprocess
with patch("asyncio.create_subprocess_exec") as mock_subprocess:
    mock_subprocess.return_value = AsyncMock()
    # Test orchestrator without spawning vLLM
```

### Coverage Target

- **Overall:** ≥90%
- **Public API:** 100% (every public method tested)
- **Error paths:** All exception handlers tested

## Error Handling

### Three-Tier Fallback

1. **Advisor-level** — If advisor query fails → use heuristic (first option, low confidence)
2. **Monitor-level** — If log monitoring fails → assume timeout
3. **Orchestrator-level** — If all retries fail → return False, user must debug

### Example

```python
# Advisor fails to load model
try:
    await self._ensure_loaded()
except Exception as e:
    logger.error(f"Failed to load advisor model: {e}")
    # Fall back to heuristic
    return self._fallback_decision(options)

# Monitor fails to read logs
try:
    with open(self.log_file, "r") as f:
        lines = f.readlines()
except Exception as e:
    logger.warning(f"Error reading log: {e}")
    # Continue anyway; might succeed on next iteration

# Orchestrator hits max retries
if self.retry_count >= self.MAX_RETRIES:
    return self._exit_failure(f"Max retries reached ({self.MAX_RETRIES})")
```

## Performance Considerations

### Latency Budget

```
Total time to startup (success case):
├─ Model validation (100ms) — HF API call
├─ Viability check (500ms) — Advisor inference on CPU
├─ vLLM startup (2-5min) — Actual model loading & warmup
├─ Log monitoring (5min) — Wait for startup signal
└─ Config save (50ms) — Write YAML

Total: ~7-10 minutes (first run, includes model download)
       ~2-5 minutes (subsequent runs, model cached)
```

### Memory Usage

```
Before startup:
├─ Advisor model: 3GB (Q4 quantized)
└─ Python/transformers overhead: 1-2GB
Total: ~4-5GB

After vLLM startup:
├─ Advisor model: 3GB (still in memory)
├─ vLLM main model: depends on model size
│  ├─ 27B-FP8: 54GB
│  ├─ 27B-Q4: 13GB
│  └─ 7B-FP8: 14GB
└─ System overhead: 2-3GB
Total: 20GB-60GB depending on main model
```

## Future Extensions

### Possible Improvements

1. **Multiple advisor models** — Switch between Qwen, LLaMA, Phi based on hardware
2. **Learned preferences** — Learn which fallback sequence works best for each user
3. **Multi-model coordination** — Manage both main model + reranker simultaneously
4. **API endpoints** — Expose orchestrator logic as HTTP service
5. **Telemetry** — Track success rates, timing, fallback patterns
6. **Hardware profiling** — Auto-detect GPU model and suggest optimal settings
7. **Custom prompts** — Allow users to provide custom advisor prompts

---

**Version:** 1.0  
**Last Updated:** 2026-05-07
