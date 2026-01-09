"""
Configuration for Track A Pipeline
"""
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class TrackAConfig:
    """Configuration for Track A RAG + Reasoning Pipeline"""
    
    # Model Configuration
    llm_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_quantization: bool = True  # Use 4-bit quantization for LLaMA
    device: str = "cuda" if os.getenv("CUDA_AVAILABLE") == "true" else "cpu"
    
    # Pathway Configuration
    pathway_data_dir: str = "./data"
    pathway_vector_store_path: str = "./pathway_vector_store"
    
    # Hugging Face Token (set via environment variable HF_TOKEN)
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    chunk_size: int = 512  # tokens per chunk
    chunk_overlap: int = 50  # overlap between chunks
    
    # Retrieval Configuration
    top_k_retrieval: int = 5  # chunks per claim
    max_evidence_chunks: int = 20  # total chunks for all claims
    rerank_top_k: int = 10  # top k to rerank
    
    # Claim Decomposition
    max_claims_per_backstory: int = 20
    claim_min_length: int = 10
    claim_max_length: int = 200
    
    # Reasoning Configuration
    use_symbolic_rules: bool = True
    use_reranker: bool = True
    contradiction_threshold: float = 0.7
    consistency_threshold: float = 0.6
    
    # Temporal Weighting
    later_story_weight: float = 1.5  # Weight for later story evidence
    repeated_evidence_weight: float = 1.3  # Weight for repeated evidence
    
    # Output Configuration
    generate_rationale: bool = True
    max_rationale_length: int = 500
    
    # External Data Sources
    google_drive_enabled: bool = False
    google_drive_credentials: Optional[str] = None
    cloud_storage_enabled: bool = False
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            llm_model_name=os.getenv("LLM_MODEL", cls.llm_model_name),
            embedding_model_name=os.getenv("EMBEDDING_MODEL", cls.embedding_model_name),
            use_quantization=os.getenv("USE_QUANTIZATION", "true").lower() == "true",
            device=os.getenv("DEVICE", cls.device),
            pathway_data_dir=os.getenv("PATHWAY_DATA_DIR", cls.pathway_data_dir),
            top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", cls.top_k_retrieval)),
        )

