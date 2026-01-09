# Quick Start Guide - Using Your Dataset

## Your Dataset Files

You have provided:
- **Large files**: `files/novel.txt` and `files/backstory.txt` (full Project Gutenberg books)
- **Small test files**: `files/novel1.txt` and `files/backstory1.txt` (simple examples)

## Step 1: Setup Environment

```bash
cd track_a

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt', quiet=True)"
```

## Step 2: Set Hugging Face Token

You need a Hugging Face token to access LLaMA-3.1-8B:

```bash
export HF_TOKEN=your_huggingface_token_here
```

Or create a `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

## Step 3: Test with Small Files (Recommended First)

```bash
python test_pipeline.py 1
```

This will:
- Use `files/novel1.txt` and `files/backstory1.txt`
- Run faster (fewer claims, smaller text)
- Generate `results_small.json`

## Step 4: Run with Large Dataset

```bash
python test_pipeline.py 2
```

This will:
- Use `files/novel.txt` and `files/backstory.txt`
- Take longer (large files, more claims)
- Generate `results_large.json`

## Step 5: Direct Pipeline Usage

You can also run the pipeline directly:

```bash
# Small files
python run_pipeline.py files/novel1.txt files/backstory1.txt --output results_small.json

# Large files
python run_pipeline.py files/novel.txt files/backstory.txt --output results_large.json
```

## Expected Output

The pipeline will:
1. ✅ Ingest both files using Pathway
2. ✅ Chunk and index the novel (with temporal metadata)
3. ✅ Extract atomic claims from backstory using LLaMA-3.1-8B
4. ✅ Retrieve evidence for each claim (multi-hop RAG)
5. ✅ Verify claims using hybrid reasoning (LLM + symbolic rules)
6. ✅ Aggregate results and make final decision

Output format:
```json
{
  "label": 1,  // 1 = coherent, 0 = not coherent
  "confidence": 0.85,
  "rationale": "Analyzed 15 claims...",
  "satisfied_claims": 12,
  "violated_claims": 3,
  "total_claims": 15,
  "claim_details": [...],
  "conflict_summary": {...}
}
```

## Troubleshooting

### Issue: "File not found"
- Make sure you're in the `track_a` directory
- Check that files exist: `ls files/`

### Issue: "CUDA out of memory"
- Set `use_quantization=True` in config (default)
- Or use CPU: `device="cpu"` in config

### Issue: "Hugging Face authentication"
- Make sure `HF_TOKEN` is set
- Request access to LLaMA-3.1-8B on Hugging Face if needed

### Issue: "Pathway import error"
- Make sure Pathway is installed: `pip install pathway>=0.7.0`
- Check Python version: `python --version` (needs 3.8+)

## Performance Notes

- **Small files**: ~2-5 minutes (depending on GPU)
- **Large files**: ~15-30 minutes (depending on GPU and file size)
- First run is slower (model downloads, indexing)
- Subsequent runs are faster (cached models, saved vector store)

## Next Steps

After running the pipeline:
1. Check the output JSON file for detailed results
2. Review `claim_details` to see which claims were satisfied/violated
3. Use the `rationale` field to understand the decision
4. Adjust `consistency_threshold` in config if needed

