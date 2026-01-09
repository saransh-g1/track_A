#!/usr/bin/env python3
"""
Standalone script to run Track A pipeline
This script can be run directly without package imports
"""
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import TrackAConfig
from pathway_ingestion import PathwayIngestionLayer, create_ingestion_pipeline
from document_store import PathwayDocumentStore
from claim_decomposer import BackstoryClaimDecomposer, ClaimPrioritizer
from reasoning_layer import HybridReasoningLayer, EvidenceChunk
from consistency_aggregator import GlobalConsistencyAggregator, FinalDecision


class TrackAPipeline:
    """
    Main pipeline for Track A: Constraint-Verification RAG + Reasoning
    """
    
    def __init__(self, config=None):
        self.config = config or TrackAConfig.from_env()
        
        # Initialize components
        print("Initializing Track A Pipeline...")
        
        # Ingestion
        self.ingestion = PathwayIngestionLayer(self.config.pathway_data_dir)
        
        # Document store
        print("Initializing document store...")
        self.document_store = PathwayDocumentStore(
            embedding_model_name=self.config.embedding_model_name,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            store_path=self.config.pathway_vector_store_path
        )
        
        # Claim decomposer
        print("Loading LLaMA-3.1-8B for claim decomposition...")
        self.claim_decomposer = BackstoryClaimDecomposer(
            model_name=self.config.llm_model_name,
            use_quantization=self.config.use_quantization,
            device=self.config.device,
            hf_token=self.config.hf_token
        )
        
        # Reasoning layer
        print("Loading LLaMA-3.1-8B for reasoning...")
        self.reasoning_layer = HybridReasoningLayer(
            llm_model_name=self.config.llm_model_name,
            use_quantization=self.config.use_quantization,
            device=self.config.device,
            hf_token=self.config.hf_token
        )
        
        # Consistency aggregator
        self.aggregator = GlobalConsistencyAggregator(
            consistency_threshold=self.config.consistency_threshold,
            contradiction_threshold=self.config.contradiction_threshold
        )
        
        # Claim prioritizer
        self.prioritizer = ClaimPrioritizer()
        
        print("Pipeline initialized successfully!")
    
    def process(
        self,
        novel_path: str,
        backstory_path: str,
        output_path=None
    ) -> FinalDecision:
        """
        Process a novel-backstory pair end-to-end.
        
        Args:
            novel_path: Path to full novel text file
            backstory_path: Path to backstory text file
            output_path: Optional path to save results JSON
            
        Returns:
            FinalDecision object with label (1/0) and rationale
        """
        print("\n" + "="*80)
        print("TRACK A PIPELINE: Constraint-Verification RAG + Reasoning")
        print("="*80)
        
        # Step 1: Ingestion
        print("\n[Step 1/6] Pathway Ingestion Layer")
        print("-" * 80)
        novel_table, backstory_table = create_ingestion_pipeline(
            novel_path,
            backstory_path
        )
        print(f"✓ Ingested novel: {novel_path}")
        print(f"✓ Ingested backstory: {backstory_path}")
        
        # Extract text from Pathway tables (simplified - in production use pw.run())
        # For now, read files directly
        with open(novel_path, 'r', encoding='utf-8') as f:
            novel_text = f.read()
        with open(backstory_path, 'r', encoding='utf-8') as f:
            backstory_text = f.read()
        
        print(f"  Novel length: {len(novel_text)} characters")
        print(f"  Backstory length: {len(backstory_text)} characters")
        
        # Step 2: Indexing
        print("\n[Step 2/6] Pathway Document Store + Vector Index")
        print("-" * 80)
        print("Chunking and indexing novel...")
        num_chunks = self.document_store.index_document(
            novel_text,
            document_id="novel",
            chapter_id=None
        )
        print(f"✓ Indexed {num_chunks} chunks with temporal metadata")
        
        # Step 3: Claim Decomposition
        print("\n[Step 3/6] Backstory Claim Decomposer (LLaMA-3.1-8B)")
        print("-" * 80)
        print("Decomposing backstory into atomic claims...")
        claims = self.claim_decomposer.decompose(
            backstory_text,
            max_claims=self.config.max_claims_per_backstory
        )
        print(f"✓ Extracted {len(claims)} atomic claims")
        
        # Refine and prioritize claims
        claims = self.claim_decomposer.refine_claims(claims)
        claims = self.prioritizer.prioritize(claims)
        
        print("\nSample claims:")
        for i, claim in enumerate(claims[:3], 1):
            print(f"  {i}. [{claim.claim_type}] {claim.claim_text[:80]}...")
        
        # Step 4: Evidence Retrieval (RAG)
        print("\n[Step 4/6] Claim-wise Evidence Retrieval (RAG)")
        print("-" * 80)
        print("Retrieving evidence for each claim...")
        
        claim_verifications = []
        from tqdm import tqdm
        import time
        
        # Process claims with progress bar
        print(f"\n  Processing {len(claims)} claims...")
        for idx, claim in enumerate(tqdm(claims, desc="  Claims", unit="claim"), 1):
            # Retrieve evidence
            evidence_results = self.document_store.retrieve_evidence(
                query=claim.claim_text,
                top_k=self.config.top_k_retrieval,
                multi_hop=True
            )
            
            # Convert to EvidenceChunk objects
            evidence_chunks = [
                EvidenceChunk(
                    chunk_text=chunk.text,
                    chunk_id=chunk.chunk_id,
                    position_ratio=chunk.position_ratio,
                    relevance_score=score,
                    document_id=chunk.document_id
                )
                for chunk, score in evidence_results
            ]
            
            # Step 5: Hybrid Reasoning
            verification = self.reasoning_layer.verify_claim(
                claim=claim,
                evidence_chunks=evidence_chunks,
                use_reranker=self.config.use_reranker
            )
            
            claim_verifications.append(verification)
            
            # Update progress bar description
            status = "✓" if verification.is_satisfied else "✗"
            tqdm.write(f"    [{idx}/{len(claims)}] {status} {claim.claim_text[:50]}... (conf: {verification.confidence:.2f})")
        
        # Step 6: Global Consistency Aggregation
        print("\n[Step 6/6] Global Consistency Aggregator")
        print("-" * 80)
        print("Aggregating results and making final decision...")
        
        final_decision = self.aggregator.aggregate(
            claim_verifications,
            generate_rationale=self.config.generate_rationale,
            max_rationale_length=self.config.max_rationale_length
        )
        
        # Print results
        print("\n" + "="*80)
        print("FINAL DECISION")
        print("="*80)
        print(f"Label: {final_decision.label} ({'COHERENT' if final_decision.label == 1 else 'NOT COHERENT'})")
        print(f"Confidence: {final_decision.confidence:.3f}")
        print(f"\nSatisfied Claims: {final_decision.satisfied_claims}/{final_decision.total_claims}")
        print(f"Violated Claims: {final_decision.violated_claims}/{final_decision.total_claims}")
        print(f"\nRationale:\n{final_decision.rationale}")
        
        if final_decision.conflict_summary.get("total_conflicts", 0) > 0:
            print(f"\nConflicts Detected: {final_decision.conflict_summary['total_conflicts']}")
            print(f"Contradiction Types: {final_decision.conflict_summary.get('contradiction_types', {})}")
        
        # Save results if requested
        if output_path:
            self._save_results(final_decision, output_path)
            print(f"\n✓ Results saved to: {output_path}")
        
        return final_decision
    
    def _save_results(self, decision: FinalDecision, output_path: str):
        """Save results to JSON file"""
        import json
        results = {
            "label": decision.label,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "satisfied_claims": decision.satisfied_claims,
            "violated_claims": decision.violated_claims,
            "total_claims": decision.total_claims,
            "claim_details": decision.claim_details,
            "conflict_summary": decision.conflict_summary
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Track A: End-to-End RAG + Reasoning Pipeline"
    )
    parser.add_argument(
        "novel_path",
        type=str,
        help="Path to full novel text file"
    )
    parser.add_argument(
        "backstory_path",
        type=str,
        help="Path to backstory text file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (optional)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = TrackAConfig.from_env() if not args.config else TrackAConfig()
    
    # Initialize pipeline
    pipeline = TrackAPipeline(config)
    
    # Process
    decision = pipeline.process(
        args.novel_path,
        args.backstory_path,
        args.output
    )
    
    return decision


if __name__ == "__main__":
    main()

