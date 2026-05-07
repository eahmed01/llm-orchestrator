# API Reference

Complete reference for all public classes, methods, and functions.

## `orchestrator.Orchestrator`

Main retry loop state machine for LLM startup.

### Constructor

```python
Orchestrator(
    service: str = "vllm",
    model: Optional[str] = None,
    args: Optional[dict[str, Any]] = None,
    advisor_endpoint: Optional[str] = None,
    vllm_log_file: Optional[str] = None,
)
```

**Parameters:**
- `service` — Service name (default: "vllm")
- `model` — Model ID to load (e.g., "Qwen/Qwen3.6-27B-FP8")
- `args` — Custom vLLM arguments (e.g., {"--max-num-seqs": 1024})
- `advisor_endpoint` — (Deprecated) Ollama endpoint (kept for compatibility)
- `vllm_log_file` — Path to vLLM log file (default: "/work/fast3/xeio/logs/vllm.log")

**Example:**
```python
orchestrator = Orchestrator(
    service="vllm",
    model="Qwen/Qwen3.6-27B-FP8",
    args={"--max-num-seqs": 848},
)
```

### Methods

#### `async start() → bool`

Start the service with advisor-guided retry logic.

**Returns:**
- `True` if startup succeeded
- `False` if max retries exceeded

**Raises:**
- Exception if model validation fails

**Example:**
```python
try:
    success = await orchestrator.start()
    if success:
        print("✓ Service started successfully!")
    else:
        print("✗ Failed to start after max retries")
finally:
    await orchestrator.close()
```

**Flow:**
1. Load model (from arg or saved config)
2. Validate model exists on HuggingFace
3. Check viability (ask advisor: will this fit?)
4. Loop (max 5 retries):
   - Spawn vLLM subprocess
   - Monitor logs (5 min timeout)
   - If success → save config & return True
   - If failure → ask advisor, try next model
5. If max retries exceeded → return False

#### `async close() → None`

Clean up resources (advisor model).

**Example:**
```python
async with Orchestrator(...) as orchestrator:
    success = await orchestrator.start()
# close() called automatically
```

---

## `advisor.Advisor`

Query in-process Qwen2.5-Coder model for intelligent retry decisions.

### Constructor

```python
Advisor(endpoint: Optional[str] = None)
```

**Parameters:**
- `endpoint` — (Deprecated) Ollama endpoint (kept for compatibility)

**Example:**
```python
advisor = Advisor()
```

### Methods

#### `async check_available() → bool`

Check if advisor model can be loaded.

**Returns:**
- `True` if model loads successfully
- `False` if load fails (network error, disk space, etc.)

**Note:** First call may take 2-3 minutes to download model (~3GB)

**Example:**
```python
available = await advisor.check_available()
if not available:
    print("Advisor unavailable; will use heuristic fallback")
```

#### `async decide_next_step(model: str, failure_reason: str, options: list[tuple[str, Optional[str]]], hardware_info: Optional[str] = None) → dict[str, Any]`

Ask advisor what to try next given failure and available options.

**Parameters:**
- `model` — Current model ID (e.g., "Qwen/Qwen3.6-27B-FP8")
- `failure_reason` — Why launch failed (e.g., "CUDA out of memory")
- `options` — List of (model_id, quantization_variant) tuples to choose from
- `hardware_info` — Hardware description (optional, e.g., "RTX 6000 Blackwell (95GB)")

**Returns:**
```python
{
    "recommendation": "2",           # Number of recommended option (1-indexed)
    "reasoning": "...",              # Brief explanation of choice
    "confidence": 0.82,              # 0.0-1.0, how confident in this choice
    "alternatives": ["3", "1"],      # Ordered backup options
    "next_action": "retry"           # "retry", "investigate", or "escalate"
}
```

**Example:**
```python
decision = await advisor.decide_next_step(
    model="Qwen/Qwen3.6-27B-FP8",
    failure_reason="CUDA out of memory",
    options=[
        ("Qwen/Qwen3.6-27B-Q4", "q4_k_m"),
        ("Qwen/Qwen3.6-7B-FP8", None),
        ("Qwen/Qwen2.5-Coder-1.5B", None),
    ],
    hardware_info="RTX Pro 6000 Blackwell (95GB)",
)

# advisor recommends: options[1] = ("Qwen/Qwen3.6-7B-FP8", None)
print(f"Try: {options[int(decision['recommendation']) - 1]}")
```

**Fallback Behavior:**
- If advisor fails to load → returns heuristic (first option, confidence=0.3)
- If advisor returns invalid JSON → returns heuristic
- If JSON not valid dict → returns heuristic

#### `async estimate_viability(model: str, hardware_vram_gb: int, batch_size: Optional[int] = None) → dict[str, Any]`

Estimate whether model will fit on given hardware.

**Parameters:**
- `model` — Model ID (e.g., "Qwen/Qwen3.6-27B-FP8")
- `hardware_vram_gb` — Available VRAM in GB (e.g., 95)
- `batch_size` — Batch size (optional; affects memory calculation)

**Returns:**
```python
{
    "viable": True,                  # Will model fit?
    "confidence": 0.85,              # 0.0-1.0
    "reasoning": "27B-FP8 = ~54GB... # Explanation
    "suggestions": ["..."],          # Optimization suggestions
}
```

**Example:**
```python
viability = await advisor.estimate_viability(
    model="Qwen/Qwen3.6-27B-FP8",
    hardware_vram_gb=95,
    batch_size=32,
)

if viability["viable"]:
    print("Model should fit! Confidence: {:.0%}".format(viability["confidence"]))
else:
    print("Model won't fit. Try: " + ", ".join(viability["suggestions"]))
```

#### `async close() → None`

Clean up model from memory.

**Example:**
```python
advisor = Advisor()
try:
    available = await advisor.check_available()
    # ... use advisor
finally:
    await advisor.close()
```

---

## `config.OrchestratorConfig`

Load/save service configurations to YAML.

### Class Methods

#### `@classmethod config_dir() → Path`

Get orchestrator config directory (creates if not exists).

**Returns:** `~/.config/llm-orchestrator/`

**Example:**
```python
config_dir = OrchestratorConfig.config_dir()
print(f"Config directory: {config_dir}")
```

#### `@classmethod config_file() → Path`

Get path to working-models.yaml config file.

**Returns:** `~/.config/llm-orchestrator/working-models.yaml`

#### `@classmethod load_from_disk() → OrchestratorConfig`

Load config from disk (or return empty if file doesn't exist).

**Returns:** OrchestratorConfig instance

**Example:**
```python
config = OrchestratorConfig.load_from_disk()
if config.vllm:
    print(f"Saved model: {config.vllm.model}")
else:
    print("No saved config yet")
```

### Instance Methods

#### `save_to_disk() → None`

Save config to `~/.config/llm-orchestrator/working-models.yaml`.

**Raises:** RuntimeError if write fails

**Example:**
```python
config = OrchestratorConfig(vllm=ServiceConfig(model="Qwen/Qwen3.6-27B-FP8"))
config.save_to_disk()
```

#### `get_service_config(service: str) → Optional[ServiceConfig]`

Get config for specific service.

**Parameters:**
- `service` — Service name (e.g., "vllm")

**Returns:** ServiceConfig or None if not found

**Raises:** ValueError if service unknown

#### `set_service_config(service: str, config: ServiceConfig) → None`

Set config for specific service.

**Parameters:**
- `service` — Service name (e.g., "vllm")
- `config` — ServiceConfig instance

**Raises:** ValueError if service unknown

#### `record_success(service: str, model: str, variant: Optional[str] = None, args: Optional[dict[str, Any]] = None) → None`

Record a successful startup (saves timestamp + args).

**Parameters:**
- `service` — Service name
- `model` — Model that succeeded
- `variant` — Quantization variant (optional)
- `args` — vLLM arguments used

**Example:**
```python
config = OrchestratorConfig.load_from_disk()
config.record_success(
    "vllm",
    "Qwen/Qwen3.6-27B-Q4",
    variant="q4_k_m",
    args={"--max-num-seqs": 848},
)
# Automatically saves to disk
```

---

## `model_discovery.ModelDiscovery`

Discover model variants and build fallback chains.

### Static Methods

#### `@staticmethod async find_variants(model_id: str) → dict[str, dict[str, Any]]`

Find quantized variants of model on HuggingFace.

**Parameters:**
- `model_id` — Model ID (e.g., "Qwen/Qwen3.6-27B")

**Returns:**
```python
{
    "Qwen/Qwen3.6-27B-Q4": {
        "size_gb": 13.89,
        "format": "q4_k_m",
        "filename": "model-q4_k_m.gguf",
    },
    # ... more variants
}
```

**Raises:** No exceptions; returns empty dict on error

**Example:**
```python
variants = await ModelDiscovery.find_variants("Qwen/Qwen3.6-27B")
for name, info in variants.items():
    print(f"{name}: {info['size_gb']:.2f}GB")
```

#### `@staticmethod async get_model_info(model_id: str) → dict[str, Any]`

Fetch model metadata from HuggingFace.

**Parameters:**
- `model_id` — Model ID

**Returns:**
```python
{
    "model_id": "Qwen/Qwen3.6-27B",
    "author": "Qwen",
    "tags": ["text-generation", ...],
    "downloads": 125000,
    "likes": 3500,
    "pipeline_tag": "text-generation",
    "config": {...},
}
```

#### `@staticmethod def build_fallback_chain(base_model: str, available_vram_gb: int = 95, prefer_quantized: bool = True) → list[tuple[str, Optional[str]]]`

Build intelligent fallback chain (size-aware, quality-aware).

**Parameters:**
- `base_model` — Starting model (e.g., "Qwen/Qwen3.6-27B-FP8")
- `available_vram_gb` — GPU VRAM (used for heuristics)
- `prefer_quantized` — Whether to prefer quantized variants

**Returns:**
```python
[
    ("Qwen/Qwen3.6-27B-FP8", None),      # Try original first
    ("Qwen/Qwen3.6-27B-Q4", "q4_k_m"),   # Quantized same size
    ("Qwen/Qwen3.6-7B-FP8", None),       # Smaller
    ("Qwen/Qwen3.6-7B-Q4", "q4_k_m"),    # Smaller + quantized
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", None),  # Tiny fallback
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "q4_k_m"),
]
```

**Example:**
```python
chain = ModelDiscovery.build_fallback_chain("Qwen/Qwen3.6-27B-FP8")
for i, (model, variant) in enumerate(chain, 1):
    print(f"{i}. {model} ({variant or 'original'})")
```

#### `@staticmethod async validate_model_exists(model_id: str) → bool`

Check if model exists on HuggingFace (pre-flight check).

**Parameters:**
- `model_id` — Model ID

**Returns:** True if exists, False otherwise

---

## `monitor.Monitor`

Real-time log monitoring for startup success/failure detection.

### Constructor

```python
Monitor(log_file: str)
```

**Parameters:**
- `log_file` — Path to vLLM log file

**Example:**
```python
monitor = Monitor("/work/fast3/xeio/logs/vllm.log")
```

### Methods

#### `reset() → None`

Reset file position to start (for retry attempts).

**Example:**
```python
monitor.reset()  # After vLLM restart, read from beginning
```

#### `async monitor_startup(timeout_seconds: int = 300) → tuple[bool, Optional[str]]`

Monitor logs until startup complete or failure detected.

**Parameters:**
- `timeout_seconds` — Max seconds to wait (default: 300 = 5 min)

**Returns:**
- `(True, None)` — Startup successful
- `(False, "error message")` — Startup failed (reason extracted from log)
- `(False, "startup_timeout")` — Exceeded timeout

**Example:**
```python
success, reason = await monitor.monitor_startup(timeout_seconds=300)

if success:
    print("✓ Service started!")
else:
    print(f"✗ Startup failed: {reason}")
```

**Success Patterns Detected:**
- "Application startup complete"
- "Uvicorn running on"
- "Started server process"

**Failure Patterns Detected:**
- "CUDA out of memory"
- "OutOfMemoryError"
- "RuntimeError.*cuda"
- "Failed to load model"

---

## Data Models

### `ServiceConfig` (Pydantic)

Configuration for one LLM service.

**Fields:**
```python
class ServiceConfig(BaseModel):
    model: str                              # HuggingFace model ID
    variant: Optional[str] = None           # Quantization variant
    args: dict[str, Any] = {}               # vLLM startup arguments
    last_successful: Optional[str] = None   # ISO timestamp
```

**Example:**
```python
config = ServiceConfig(
    model="Qwen/Qwen3.6-27B-FP8",
    variant=None,
    args={"--max-num-seqs": 848},
)
```

### `OrchestratorConfig` (Pydantic)

Top-level orchestrator configuration.

**Fields:**
```python
class OrchestratorConfig(BaseModel):
    vllm: Optional[ServiceConfig] = None
```

---

## Exceptions

All modules gracefully handle errors and return sensible defaults:

| Scenario | Behavior |
|----------|----------|
| Advisor model fails to load | Return heuristic (first option, low confidence) |
| HuggingFace API unreachable | Return empty variants dict |
| Log file missing | Continue; check on next iteration |
| vLLM subprocess spawn fails | Return False, try next retry |

**No custom exceptions defined** — all failures caught and logged.

---

## Type Hints

All code is fully typed for mypy compliance:

```python
# Function signatures always include return types
async def start(self) -> bool: ...
async def decide_next_step(...) -> dict[str, Any]: ...

# Container types always parameterized
options: list[tuple[str, Optional[str]]]
result: dict[str, Any]
config: Optional[ServiceConfig]
```

---

**API Version:** 1.0  
**Last Updated:** 2026-05-07
