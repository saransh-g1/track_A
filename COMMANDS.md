# Quick Command Reference

## Installation Commands

### Option 1: Fast Installation (CPU-only PyTorch - Recommended)
```bash
cd track_a

# Install CPU-only PyTorch first (much faster, ~2-3 minutes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### Option 2: Full Installation (with GPU support)
```bash
cd track_a

# Install PyTorch with CUDA (if you have GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Option 3: Step-by-step (if PyTorch is taking too long)
```bash
cd track_a

# Install everything except PyTorch first
pip install pathway transformers sentence-transformers accelerate faiss-cpu nltk numpy tqdm python-dotenv bitsandbytes

# Then install PyTorch separately (CPU version - faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Setup Commands

```bash
# Set Hugging Face token (required for LLaMA-3.1-8B)
export HF_TOKEN=your_huggingface_token_here

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt', quiet=True)"
```

## Running Commands

### Test with Small Files (Recommended First)
```bash
python test_pipeline.py 1
```

### Run with Large Dataset
```bash
python test_pipeline.py 2
```

### Direct Pipeline Usage
```bash
# Small files
python run_pipeline.py files/novel1.txt files/backstory1.txt --output results_small.json

# Large files
python run_pipeline.py files/novel.txt files/backstory.txt --output results_large.json
```

### Run Specific Example
```bash
# Example 1: Basic usage
python example_usage.py 1

# Example 2: Custom config
python example_usage.py 2

# Example 3: Batch processing
python example_usage.py 3

# Example 4: Detailed analysis
python example_usage.py 4
```

## Complete Setup from Scratch

```bash
# 1. Navigate to track_a directory
cd track_a

# 2. Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install CPU-only PyTorch (fastest option)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Set Hugging Face token
export HF_TOKEN=your_token_here

# 6. Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True)"

# 7. Test with small files
python test_pipeline.py 1
```

## Troubleshooting Commands

### If PyTorch installation is stuck:
```bash
# Cancel and install CPU-only version (much faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### If you get "File not found" error:
```bash
# Check if files exist
ls files/

# Should show:
# - backstory.txt
# - novel.txt
# - backstory1.txt
# - novel1.txt
```

### If you get "HF_TOKEN" error:
```bash
# Set the token
export HF_TOKEN=your_token_here

# Or create .env file
echo "HF_TOKEN=your_token_here" > .env
```

### Check installation:
```bash
# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Verify other packages
python -c "import pathway, transformers, sentence_transformers; print('All packages installed')"
```

## Quick Start (Copy-Paste)

```bash
cd track_a
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export HF_TOKEN=your_token_here
python test_pipeline.py 1
```


