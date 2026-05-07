# LLM Orchestrator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Run massive LLMs locally. Instantly. No trial-and-error.

**Stop wasting hours manually testing models.** LLM Orchestrator automatically finds the right model variant for your GPU, instantly adapting when your first choice runs out of memory.

### The Problem

You want to run a 70B model locally. You try it. **Out of memory.** 

Now what? Manually tweak batch sizes? Quantize it? Try a smaller model? Try again. And again. And save the config somewhere. Maybe next time you remember what worked.

**LLM Orchestrator automates all of this.** Give it one model, and it will:
- ✓ Auto-detect if it fits your GPU
- ✓ Detect failures in real-time (OOM, crashes, timeouts)
- ✓ Automatically retry with optimized versions (Q4, smaller sizes, etc.)
- ✓ Save the working config for next time
- ✓ Do it all without external services or complex setup

### The Solution (In 30 Seconds)

```bash
# Install (one-time)
git clone https://github.com/eahmed01/llm-orchestrator.git
cd llm-orchestrator && bash setup.sh

# Run (and forget)
source venv/bin/activate
./llm-orchestrate start vllm --model Qwen/Qwen3.6-70B-Instruct

# If OOM → Automatically retries with Q4 variant ✓ Success → Config saved
```

### Real Example: What Changed

**Before (Manual Hell):**
```
❌ Qwen3.6-70B-FP8 → Out of memory
→ Manually edit args, find quantized versions, download 
→ ❌ Try Qwen3.6-70B-Q4 → Crashes
→ Manually research alternatives
→ Try Qwen3.6-7B → Success
→ Manually save config
→ NEXT TIME: Remember where you saved it
```

**After (Automated):**
```
✓ ./llm-orchestrate start vllm --model Qwen/Qwen3.6-70B-Instruct
→ Detects OOM after 2min
→ Advisor recommends Q4 variant
→ Retries automatically → SUCCESS
→ Config auto-saved
→ NEXT TIME: Just run the same command
```

## Why You Need This

- **You have a 24GB-96GB GPU** and want to use it for the best possible models
- **You're tired of manual model tweaking** and want something that Just Works™
- **You want to push your hardware limits** without spending hours on trial-and-error
- **You need it running locally** (no cloud, no API dependencies)
- **You want it fast** (setup should take <5 minutes)

## Key Features

- **Zero external services** — Runs on your machine with no Ollama, API endpoints, or web UIs
- **Intelligent advisor** — Built-in AI model decides what to try next based on failure reason
- **Automatic model discovery** — Finds quantized variants (Q4, Q5, FP8, AWQ) on HuggingFace
- **Persistent memory** — Saves working configs, reuses them next time
- **Real-time monitoring** — Detects failures as they happen, no polling
- **Hardware-aware** — Knows your GPU VRAM, makes smart retry decisions
- **Multiple models supported** — Browse trending models by ambition level and hardware fit

## Quick Start

```bash
# Clone & install (one time)
git clone https://github.com/eahmed01/llm-orchestrator.git
cd llm-orchestrator
bash setup.sh
source venv/bin/activate

# Run a model (with auto-retry on failure)
./llm-orchestrate start vllm --model Qwen/Qwen3.6-70B-Instruct

# Or explore what models you can run
./llm-orchestrate models

# Or check what's currently configured
./llm-orchestrate status vllm
```

**That's it.** No complex config files, no environment variables to set, no external services to run.

---

## 📚 Full Documentation Below

## Quick Start

### Installation

#### Option 1: Quick Start (No pip knowledge required)

```bash
# Clone the repo
git clone https://github.com/eahmed01/llm-orchestrator.git
cd llm-orchestrator

# Run setup script (creates virtual environment + installs dependencies)
bash setup.sh

# Activate and use
source venv/bin/activate
./llm-orchestrate start vllm --model Qwen/Qwen3.6-27B-FP8
```

#### Option 2: Using pip (Standard Python approach)

```bash
git clone https://github.com/eahmed01/llm-orchestrator.git
cd llm-orchestrator
pip install -e ".[dev]"  # Includes dev dependencies for testing

# Then use directly
llm-orchestrate start vllm --model Qwen/Qwen3.6-27B-FP8
```

**Requirements:**
- Python 3.10+
- GPU with at least 24GB VRAM (for 7B models; larger models need more)
- vLLM installed: `pip install vllm`

### Usage

#### Start a model with automatic retry

```bash
# If you used setup.sh:
source venv/bin/activate
./llm-orchestrate start vllm --model Qwen/Qwen3.6-27B-FP8

# If you used pip install:
llm-orchestrate start vllm --model Qwen/Qwen3.6-27B-FP8

# Or use saved config (no model arg needed)
./llm-orchestrate start vllm
```

#### Check what's currently configured

```bash
llm-orchestrate status vllm
# Output:
#   Service: vllm
#   Model: Qwen/Qwen3.6-27B-FP8
#   Variant: None
#   Last successful: 2026-05-07T12:35:00Z
```

#### Discover model variants available on HuggingFace

```bash
llm-orchestrate discover Qwen/Qwen3.6-27B

# Output:
#   Discovering variants for Qwen/Qwen3.6-27B...
#   Qwen/Qwen3.6-27B-FP8: 54.23GB (fp16)
#   Qwen/Qwen3.6-27B-Q5: 18.45GB (q5_k_m)
#   Qwen/Qwen3.6-27B-Q4: 13.89GB (q4_k_m)
```

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  llm-orchestrate start vllm --model Qwen/Qwen3.6-27B-FP8   │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────▼─────────┐
        │  Orchestrator    │
        │  (State machine) │
        └────────┬─────────┘
                 │
        ┌────────▼──────────────────┐
        │ 1. Validate model exists  │
        │ 2. Check viability (ask   │
        │    advisor: fits GPU?)    │
        └────────┬──────────────────┘
                 │
        ┌────────▼─────────────────────────┐
        │ 3. Spawn vLLM subprocess         │
        │    sys.executable -m vllm serve \│
        │      Qwen/Qwen3.6-27B-FP8        │
        └────────┬──────────────────────────┘
                 │
        ┌────────▼──────────────────────────┐
        │ 4. Monitor logs for 5 minutes     │
        │    (tail /work/fast3/.../vllm.log)│
        └────────┬──────────────────────────┘
                 │
        ┌────────▼───────────────────┐
        │  Success?                  │
        └─┬──────────────────────┬───┘
          │ YES                  │ NO (OOM/crash/timeout)
          │                      │
    ┌─────▼────┐         ┌───────▼──────────────────────┐
    │  Save     │         │ 5. Ask advisor:              │
    │  config   │         │    "What should we try next?"│
    │           │         │    Given: failure_reason,    │
    │  ✓ Done   │         │    options: [                │
    │           │         │      Q4 variant,             │
    └───────────┘         │      7B model,               │
                          │      1.5B model              │
                          │    ]                         │
                          └───────┬──────────────────────┘
                                  │
                          ┌───────▼──────────────────────┐
                          │ 6. Advisor (Qwen2.5-Coder)   │
                          │    loads locally via         │
                          │    transformers.pipeline()   │
                          │    Runs inference:           │
                          │    Recommendation: #2 (Q4)   │
                          │    Confidence: 0.85          │
                          └───────┬──────────────────────┘
                                  │
                          ┌───────▼──────────────────────┐
                          │ 7. Update model to Q4        │
                          │    variant, retry loop       │
                          └───────┬──────────────────────┘
                                  │
                    ┌─────────────┴────────────────────┐
                    │                                  │
                    ▼                                  ▼
            [Back to step 3]                    [Max retries exceeded]
                    ✓ Success                          ✗ Failed
```

### Retry Loop

Each retry attempt:
1. **Build fallback chain** from current model (e.g., "Qwen3.6-27B-FP8")
   - Try with lower batch size
   - Try quantized variants (Q4, Q5, etc.)
   - Fall back to smaller models (7B, 1.5B)
2. **Ask advisor** which option is best, given hardware and failure reason
3. **Update model/args** and attempt startup again
4. **Repeat** until success or max retries (default: 5)

### Advisor Model

The advisor is **Qwen2.5-Coder-1.5B**, a 1.5 billion parameter model that runs on CPU:
- **Auto-downloaded** on first use: `~3GB with quantization`
- **Cached** in `~/.cache/huggingface/` for reuse
- **Prompts advisor** with:
  - Current model ID
  - Failure reason (OOM, crash, timeout, etc.)
  - Available fallback options
  - Hardware specs (VRAM, etc.)
- **Returns** JSON with: recommendation, reasoning, confidence, alternatives

Example advisor decision:
```json
{
  "recommendation": "2",
  "reasoning": "OOM on 27B-FP8 suggests memory pressure. Q4 quantization reduces size ~50%. Should fit with margin.",
  "confidence": 0.82,
  "alternatives": ["3", "1"],
  "next_action": "retry"
}
```

## Configuration

### Config File Location

```
~/.config/llm-orchestrator/working-models.yaml
```

### Config File Format

```yaml
vllm:
  model: Qwen/Qwen3.6-27B-FP8
  variant: null
  args:
    --max-num-seqs: 848
    --gpu-memory-utilization: 0.92
  last_successful: "2026-05-07T12:35:00Z"
```

### Log File Location

vLLM startup logs are monitored at:
```
/work/fast3/xeio/logs/vllm.log
```

(Configurable via `--vllm-log-file` when initializing Orchestrator)

## Examples

### Example 1: Deploy Qwen 27B with automatic fallback

```bash
$ llm-orchestrate start vllm --model Qwen/Qwen3.6-27B-FP8

[INFO] Starting vllm with model Qwen/Qwen3.6-27B-FP8
[INFO] Validating model Qwen/Qwen3.6-27B-FP8...
[INFO] Viability check: {'viable': True, 'confidence': 0.72, ...}
[INFO] Launching: /usr/bin/python -m vllm serve Qwen/Qwen3.6-27B-FP8 ...
[INFO] vLLM started (PID: 12345)
[ERROR] Startup failed: CUDA out of memory (attempt 1/5)
[INFO] Considering 6 fallback options...
[INFO] Loading advisor model (first run only)...
[INFO] ✓ Advisor model loaded
[INFO] Advisor recommendation: Try Q4 quantized variant (confidence: 0.85)
[INFO] Retrying with: Qwen/Qwen3.6-27B-Q4 (variant: q4_k_m)
[INFO] vLLM started (PID: 12346)
[INFO] ✓ Startup successful!
[INFO] Configuration saved to ~/.config/llm-orchestrator/working-models.yaml
```

### Example 2: Reuse saved config on next startup

```bash
$ llm-orchestrate start vllm
[INFO] Using saved config: Qwen/Qwen3.6-27B-Q4
[INFO] Launching: /usr/bin/python -m vllm serve Qwen/Qwen3.6-27B-Q4 ...
[INFO] vLLM started (PID: 54321)
[INFO] ✓ Startup successful!
```

### Example 3: Discover available variants

```bash
$ llm-orchestrate discover Qwen/Qwen3.6-27B
[INFO] Discovering variants for Qwen/Qwen3.6-27B...
  Qwen/Qwen3.6-27B: 54.23GB (fp16)
  Qwen/Qwen3.6-27B-Q5: 18.45GB (q5_k_m)
  Qwen/Qwen3.6-27B-Q4: 13.89GB (q4_k_m)
```

## CLI Reference

### Commands

#### `start`
Start an LLM service with intelligent retry.

```bash
llm-orchestrate start [SERVICE] [OPTIONS]

Arguments:
  SERVICE         Service to start (default: vllm)

Options:
  --model TEXT    Model ID to load (e.g., Qwen/Qwen3.6-27B-FP8)
  --max-retries INTEGER  Max retry attempts (default: 5)
```

#### `status`
Check the current configuration and last successful startup.

```bash
llm-orchestrate status [SERVICE]

Arguments:
  SERVICE         Service to check (default: vllm)
```

#### `stop`
Stop a running service.

```bash
llm-orchestrate stop [SERVICE]

Arguments:
  SERVICE         Service to stop (default: vllm)
```

#### `config`
Manage service configuration.

```bash
llm-orchestrate config [SERVICE] [OPTIONS]

Arguments:
  SERVICE         Service name (default: vllm)

Options:
  --action [show|set|delete]  Action to perform (default: show)
  --model TEXT                Model ID (for set action)
```

#### `discover`
Discover model variants on HuggingFace.

```bash
llm-orchestrate discover MODEL

Arguments:
  MODEL           Model ID to search (e.g., Qwen/Qwen3.6-27B)
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest -q

# Run with coverage
pytest --cov=llm_orchestrator --cov-report=html

# Run specific test file
pytest tests/test_advisor.py -v

# Run with type checking
mypy llm_orchestrator
ruff check llm_orchestrator
```

### Project Structure

```
llm-orchestrator/
├── llm_orchestrator/
│   ├── __init__.py
│   ├── advisor.py           # In-process Qwen advisor model
│   ├── cli.py               # Typer CLI interface
│   ├── config.py            # Pydantic config + YAML persistence
│   ├── model_discovery.py   # HuggingFace API integration
│   ├── monitor.py           # Real-time log monitoring
│   └── orchestrator.py      # Main retry loop state machine
├── tests/
│   ├── test_advisor.py      # Advisor tests (17 tests)
│   ├── test_config.py       # Config persistence tests
│   ├── test_model_discovery.py  # Variant discovery tests
│   ├── test_monitor.py      # Log monitoring tests
│   └── test_orchestrator.py # Orchestrator state machine tests
├── docs/                    # Documentation
├── pyproject.toml
├── README.md
└── CONTRIBUTING.md
```

### Key Design Decisions

1. **In-process advisor** — No external service dependency; transformers library handles model loading
2. **Async-first** — All I/O operations use asyncio; CPU-bound work via `asyncio.to_thread()`
3. **Pydantic + YAML** — Type-safe config with human-readable persistence
4. **Lazy model loading** — Advisor model downloads and caches only on first use
5. **Graceful fallback** — If advisor fails, uses heuristic fallback (first option, low confidence)

## Troubleshooting

### Advisor model fails to load

**Symptom:** `Failed to load advisor model: ...`

**Solution:** 
- First run may take 2-3 minutes to download and cache the model (~3GB)
- Ensure you have internet access
- Check `~/.cache/huggingface/` has ~3GB free space

### vLLM startup takes forever

**Symptom:** Advisor waits >5 minutes, then times out

**Cause:** Model is genuinely taking a long time to load (cold start, model compilation)

**Solution:**
- Increase timeout: `orchestrator.start()` → adjust `monitor.monitor_startup(timeout_seconds=600)`
- Try a smaller model first
- Check VRAM utilization: `nvidia-smi`

### "CUDA out of memory" on first attempt

**Expected behavior** — Orchestrator detects this and retries with a smaller model

**If it keeps retrying:** Check your GPU VRAM:
```bash
nvidia-smi
```
If < 24GB, you'll need a quantized or smaller model

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [HuggingFace Hub](https://huggingface.co/) for model discovery
- [vLLM](https://github.com/lm-sys/vllm) for high-performance LLM serving
- [Qwen](https://huggingface.co/Qwen) for the 1.5B code model
- [transformers](https://github.com/huggingface/transformers) for model loading
