# Track A – End-to-End RAG + Reasoning Pipeline

**Model:** LLaMA-3.1-8B  
**Framework:** Pathway (mandatory backbone)

## System Philosophy

This is **Constraint-Verification RAG**, not QA-RAG.

The system answers one question:
> **Can this hypothetical backstory causally and logically produce the observed full narrative?**

The pipeline is **multi-stage**, not single-prompt.

## Architecture Overview

```
┌───────────────────────────────────────────┐
│ External Data Sources                     │
│ (Google Drive / Local / Cloud)            │
└───────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────┐
│ Pathway Ingestion Layer                   │
│ - full novel (.txt)                       │
│ - backstory text                          │
│ - metadata (chapter, timeline, etc.)      │
└───────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────┐
│ Pathway Document Store + Vector Index     │
│ - temporal chunking                       │
│ - embeddings                              │
│ - metadata-aware retrieval                │
└───────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────┐
│ Backstory Claim Decomposer (LLM)          │
│ - atomic constraints extraction           │
└───────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────┐
│ Claim-wise Evidence Retrieval (RAG)       │
│ - multi-hop retrieval                     │
│ - time-distributed evidence               │
└───────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────┐
│ Hybrid Reasoning Layer                    │
│ - LLM causal checks                       │
│ - symbolic contradiction rules            │
│ - reranking + scoring                     │
└───────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────┐
│ Global Consistency Aggregator             │
│ - evidence accumulation                   │
│ - conflict resolution                     │
│ - final decision                          │
└───────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────┐
│ Output                                    │
│ - Binary label (1 / 0)                    │
│ - Optional short rationale                │
└───────────────────────────────────────────┘
```

## Pathway Usage (Requirements Satisfaction)

✅ **Ingesting and managing long-context data**
- Uses Pathway connectors to load full novels (100k+ words, no truncation)
- Handles backstory text and optional metadata

✅ **Storing and indexing full novels and metadata**
- Sequential chunking (not random)
- Metadata attached: `chunk_id`, `chapter`, `position_ratio` (0→start, 1→end)

✅ **Enabling retrieval over long documents**
- Pathway Vector Store with embeddings
- Multi-hop retrieval to avoid single-passage bias

✅ **Connecting to external data sources**
- Supports Google Drive, local storage, cloud buckets
- Extensible for various data sources

✅ **Serving as orchestration layer**
- Pathway coordinates: ingestion → indexing → retrieval → downstream processing

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### Basic Usage

```bash
python main_pipeline.py novel.txt backstory.txt --output results.json
```

### With Custom Config

```python
from config import TrackAConfig
from main_pipeline import TrackAPipeline

# Create custom config
config = TrackAConfig(
    llm_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    top_k_retrieval=10,
    consistency_threshold=0.7
)

# Initialize pipeline
pipeline = TrackAPipeline(config)

# Process
decision = pipeline.process(
    novel_path="data/novel.txt",
    backstory_path="data/backstory.txt",
    output_path="results.json"
)

print(f"Label: {decision.label} (1=coherent, 0=not coherent)")
print(f"Confidence: {decision.confidence:.3f}")
print(f"Rationale: {decision.rationale}")
```

## Components

### 1. Pathway Ingestion Layer (`pathway_ingestion.py`)
- Ingests novels and backstory from local files, Google Drive, or cloud storage
- Creates Pathway tables with metadata

### 2. Document Store (`document_store.py`)
- **TemporalChunker**: Chunks documents sequentially with sentence boundaries
- **PathwayVectorStore**: FAISS-based vector store with embeddings
- Supports multi-hop retrieval and position-weighted scoring

### 3. Claim Decomposer (`claim_decomposer.py`)
- Uses LLaMA-3.1-8B to extract atomic constraints from backstory
- Outputs structured claims with types: `character_trait`, `temporal`, `causal`, `moral`, `world_rule`

### 4. Reasoning Layer (`reasoning_layer.py`)
- **SymbolicRuleEngine**: Pattern-based contradiction detection
- **LLMReasoningEngine**: Causal reasoning with LLaMA-3.1-8B
- **HybridReasoningLayer**: Combines both approaches

### 5. Consistency Aggregator (`consistency_aggregator.py`)
- Aggregates verification results from all claims
- Resolves conflicts and makes final binary decision
- Generates human-readable rationale

### 6. Main Pipeline (`main_pipeline.py`)
- Orchestrates all components end-to-end
- Provides CLI interface and programmatic API

## Configuration

Edit `config.py` or set environment variables:

```bash
export LLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export USE_QUANTIZATION="true"
export DEVICE="cuda"
export TOP_K_RETRIEVAL=5
```

## Output Format

```json
{
  "label": 1,
  "confidence": 0.85,
  "rationale": "Analyzed 15 claims from the backstory. 12 claims are satisfied (80.0%), 3 claims are violated...",
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
    "symbolic_violations": ["moral_code_violation: ..."],
    "contradiction_types": {"moral_code_violation": 1}
  }
}
```

## Modeling Choices

### A. Transformer-based LLM (LLaMA-3.1-8B)
- Used for: Backstory → Claim decomposition, Causal reasoning, Local contradiction detection
- Not used end-to-end blindly

### B. Classical NLP Pipeline
- Named entity tracking (character consistency)
- Temporal cues (before/after, age, time jumps)
- Coreference sanity checks

### C. Hybrid Symbolic–Neural Approach
- **Neural**: Extract signals from text
- **Symbolic**: Apply rules (e.g., "if backstory claims fear of violence, later repeated violent leadership = conflict")

### D. Rerankers / Classifiers / Heuristics
- Cross-encoder reranker (optional)
- Heuristic weighting: later-story evidence > early hints, repeated evidence > single occurrence

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for LLaMA-3.1-8B)
- 16GB+ RAM (for full novels)
- Hugging Face account (for LLaMA model access)

## License

See LICENSE file.
