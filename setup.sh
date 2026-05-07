#!/bin/bash
# setup.sh - One-time setup script

set -e

echo "🔧 LLM Orchestrator Setup"
echo "========================="
echo ""

# Check if Python 3.10+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo "📚 Installing dependencies..."
pip install -q --upgrade pip
pip install -q \
    typer[all]>=0.9 \
    transformers>=4.35 \
    torch>=2.0 \
    safetensors>=0.3 \
    pydantic>=2.0 \
    pyyaml>=6.0 \
    httpx>=0.24 \
    huggingface-hub>=0.17

echo "✓ Dependencies installed"

# Install dev dependencies if requested
if [ "$1" = "dev" ]; then
    echo "📝 Installing dev dependencies..."
    pip install -q \
        pytest>=7.0 \
        pytest-asyncio>=0.21 \
        pytest-cov>=4.0 \
        mypy>=1.0 \
        ruff>=0.1 \
        black>=23.0
    echo "✓ Dev dependencies installed"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use the tool, activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "Then run:"
echo "  ./llm-orchestrate --help"
echo ""
