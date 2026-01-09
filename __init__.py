"""
Track A - End-to-End RAG + Reasoning Pipeline
Constraint-Verification RAG using Pathway and LLaMA-3.1-8B
"""

from .config import TrackAConfig
from .main_pipeline import TrackAPipeline
from .claim_decomposer import AtomicClaim, BackstoryClaimDecomposer
from .consistency_aggregator import FinalDecision, GlobalConsistencyAggregator
from .reasoning_layer import ClaimVerification, HybridReasoningLayer

__version__ = "1.0.0"
__all__ = [
    "TrackAConfig",
    "TrackAPipeline",
    "AtomicClaim",
    "BackstoryClaimDecomposer",
    "FinalDecision",
    "GlobalConsistencyAggregator",
    "ClaimVerification",
    "HybridReasoningLayer",
]

