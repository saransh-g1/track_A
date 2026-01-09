#!/bin/bash
# Setup script for Track A Pipeline

set -e

echo "=========================================="
echo "Track A Pipeline Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies with progress
echo "Installing dependencies (this may take 5-10 minutes)..."
echo "Using optimized requirements (reduced from ~8GB to ~3GB)"
pip install --progress-bar on -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)" || echo "NLTK download skipped (may already be installed)"

# Create data directories
echo "Creating data directories..."
mkdir -p data
mkdir -p pathway_vector_store

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python run_pipeline.py novel.txt backstory.txt --output results.json"
echo ""
echo "Note: You'll need Hugging Face access token for LLaMA-3.1-8B"
echo "Set it with: export HF_TOKEN=your_token_here"
echo ""

