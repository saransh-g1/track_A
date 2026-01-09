# Using Your Dataset Files

## Your Dataset

You have provided the following files in `track_a/files/`:

### Large Dataset (Full Books)
- **`novel.txt`** - The Count of Monte Cristo (Project Gutenberg)
- **`backstory.txt`** - A Voyage Round the World / In Search of the Castaways (Project Gutenberg)

### Small Test Dataset
- **`novel1.txt`** - Short narrative about Ravi
- **`backstory1.txt`** - Short backstory about Ravi

## Quick Start

### 1. Setup (One Time)

```bash
cd track_a

# Install dependencies
pip install -r requirements.txt

# Set Hugging Face token (required for LLaMA-3.1-8B)
export HF_TOKEN=your_token_here
```

### 2. Test with Small Files (Recommended First)

This will test the pipeline quickly with the small example files:

```bash
python test_pipeline.py 1
```

Expected output: `results_small.json`

### 3. Run with Large Dataset

This will process the full Project Gutenberg books:

```bash
python test_pipeline.py 2
```

**Note**: This will take 15-30 minutes depending on your hardware.

Expected output: `results_large.json`

### 4. Direct Usage

You can also run directly:

```bash
# Small files
python run_pipeline.py files/novel1.txt files/backstory1.txt --output results_small.json

# Large files  
python run_pipeline.py files/novel.txt files/backstory.txt --output results_large.json
```

## What the Pipeline Does

1. **Ingests** both files using Pathway
2. **Chunks** the novel with temporal metadata (position tracking)
3. **Extracts claims** from backstory using LLaMA-3.1-8B
4. **Retrieves evidence** for each claim (multi-hop RAG)
5. **Verifies claims** using hybrid reasoning (LLM + symbolic rules)
6. **Aggregates** results and makes final decision

## Output Format

The pipeline generates a JSON file with:

```json
{
  "label": 1,  // 1 = coherent, 0 = not coherent
  "confidence": 0.85,
  "rationale": "Analyzed 15 claims...",
  "satisfied_claims": 12,
  "violated_claims": 3,
  "total_claims": 15,
  "claim_details": [
    {
      "claim": "Character X has trait Y",
      "type": "character_trait",
      "satisfied": true,
      "confidence": 0.9,
      "violations": [],
      "contradictions": []
    }
  ],
  "conflict_summary": {
    "total_conflicts": 2,
    "symbolic_violations": [...],
    "contradiction_types": {...}
  }
}
```

## Expected Results

### Small Files (novel1.txt / backstory1.txt)
- **Backstory**: Ravi vowed never to raise his hand in anger, prefers silence over confrontation
- **Novel**: Ravi intervenes calmly, never resorts to force
- **Expected**: Should be **COHERENT** (label=1) - the novel is consistent with the backstory

### Large Files (novel.txt / backstory.txt)
- These are different books (Count of Monte Cristo vs Voyage Round the World)
- **Expected**: Likely **NOT COHERENT** (label=0) - they are unrelated narratives
- This is a good test case to verify the pipeline detects inconsistencies

## Troubleshooting

### "File not found"
- Make sure you're in the `track_a` directory
- Check: `ls files/` should show your files

### "Hugging Face authentication error"
- Set your token: `export HF_TOKEN=your_token`
- Request access to LLaMA-3.1-8B on Hugging Face if needed

### "CUDA out of memory"
- The pipeline uses 4-bit quantization by default (reduces memory)
- If still issues, set `device="cpu"` in config (will be slower)

### "Pathway import error"
- Install: `pip install pathway>=0.7.0`
- Check Python version: `python --version` (needs 3.8+)

## Performance

- **Small files**: ~2-5 minutes
- **Large files**: ~15-30 minutes
- First run slower (model downloads)
- Subsequent runs faster (cached models)

## Next Steps

After running:
1. Check the JSON output file
2. Review `claim_details` to see which claims were satisfied/violated
3. Read the `rationale` to understand the decision
4. Adjust `consistency_threshold` in config if needed

## Files Generated

- `results_small.json` - Results from small test files
- `results_large.json` - Results from large dataset
- `pathway_vector_store/` - Cached vector index (can be reused)

