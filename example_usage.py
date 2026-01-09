"""
Example usage of Track A Pipeline
"""
from main_pipeline import TrackAPipeline
from config import TrackAConfig
import os


def example_basic():
    """Basic example with default config"""
    print("="*80)
    print("Example 1: Basic Usage")
    print("="*80)
    
    # Initialize pipeline with default config
    pipeline = TrackAPipeline()
    
    # Process example files (adjust paths as needed)
    novel_path = "../KDSH/examples/example_1_current.txt"
    backstory_path = "../KDSH/examples/example_1_backstory.txt"
    
    if not os.path.exists(novel_path) or not os.path.exists(backstory_path):
        print(f"Example files not found. Please provide valid paths.")
        print(f"Expected: {novel_path}, {backstory_path}")
        return
    
    # Process
    decision = pipeline.process(
        novel_path=novel_path,
        backstory_path=backstory_path,
        output_path="example_results.json"
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Decision: {'COHERENT' if decision.label == 1 else 'NOT COHERENT'}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"\nClaims: {decision.satisfied_claims} satisfied, {decision.violated_claims} violated out of {decision.total_claims}")
    print(f"\nRationale:\n{decision.rationale}")


def example_custom_config():
    """Example with custom configuration"""
    print("="*80)
    print("Example 2: Custom Configuration")
    print("="*80)
    
    # Create custom config
    config = TrackAConfig(
        llm_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_quantization=True,
        device="cuda",
        top_k_retrieval=10,  # Retrieve more evidence per claim
        max_claims_per_backstory=25,  # Extract more claims
        consistency_threshold=0.65,  # Stricter threshold
        contradiction_threshold=0.75,
        use_reranker=True,
        generate_rationale=True,
        max_rationale_length=800
    )
    
    # Initialize pipeline
    pipeline = TrackAPipeline(config)
    
    # Process
    novel_path = "../KDSH/examples/example_1_current.txt"
    backstory_path = "../KDSH/examples/example_1_backstory.txt"
    
    if not os.path.exists(novel_path) or not os.path.exists(backstory_path):
        print(f"Example files not found. Please provide valid paths.")
        return
    
    decision = pipeline.process(
        novel_path=novel_path,
        backstory_path=backstory_path,
        output_path="example_custom_results.json"
    )
    
    print(f"\nDecision: {decision.label} (confidence: {decision.confidence:.3f})")


def example_programmatic():
    """Example of programmatic usage for batch processing"""
    print("="*80)
    print("Example 3: Programmatic Batch Processing")
    print("="*80)
    
    pipeline = TrackAPipeline()
    
    # List of novel-backstory pairs
    pairs = [
        ("../KDSH/examples/example_1_current.txt", "../KDSH/examples/example_1_backstory.txt"),
        # Add more pairs here
    ]
    
    results = []
    for novel_path, backstory_path in pairs:
        if not os.path.exists(novel_path) or not os.path.exists(backstory_path):
            print(f"Skipping {novel_path} - file not found")
            continue
        
        print(f"\nProcessing: {novel_path}")
        decision = pipeline.process(
            novel_path=novel_path,
            backstory_path=backstory_path
        )
        
        results.append({
            "novel": novel_path,
            "backstory": backstory_path,
            "label": decision.label,
            "confidence": decision.confidence,
            "satisfied_claims": decision.satisfied_claims,
            "violated_claims": decision.violated_claims
        })
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH RESULTS")
    print("="*80)
    for result in results:
        print(f"{result['novel']}: Label={result['label']}, Confidence={result['confidence']:.3f}")
        print(f"  Claims: {result['satisfied_claims']} satisfied, {result['violated_claims']} violated")


def example_analyze_claims():
    """Example showing detailed claim analysis"""
    print("="*80)
    print("Example 4: Detailed Claim Analysis")
    print("="*80)
    
    pipeline = TrackAPipeline()
    
    novel_path = "../KDSH/examples/example_1_current.txt"
    backstory_path = "../KDSH/examples/example_1_backstory.txt"
    
    if not os.path.exists(novel_path) or not os.path.exists(backstory_path):
        print(f"Example files not found.")
        return
    
    decision = pipeline.process(
        novel_path=novel_path,
        backstory_path=backstory_path
    )
    
    # Analyze individual claims
    print("\n" + "="*80)
    print("DETAILED CLAIM ANALYSIS")
    print("="*80)
    
    for i, claim_detail in enumerate(decision.claim_details, 1):
        status = "✓" if claim_detail["satisfied"] else "✗"
        print(f"\n{i}. {status} [{claim_detail['type']}]")
        print(f"   Claim: {claim_detail['claim']}")
        print(f"   Confidence: {claim_detail['confidence']:.3f}")
        
        if claim_detail["violations"]:
            print(f"   Violations: {', '.join(claim_detail['violations'])}")
        
        if claim_detail["contradictions"]:
            print(f"   Contradictions: {len(claim_detail['contradictions'])} detected")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
    else:
        example_num = 1
    
    examples = {
        1: example_basic,
        2: example_custom_config,
        3: example_programmatic,
        4: example_analyze_claims
    }
    
    if example_num in examples:
        examples[example_num]()
    else:
        print(f"Invalid example number. Choose 1-4.")
        print("\nAvailable examples:")
        print("  1. Basic usage")
        print("  2. Custom configuration")
        print("  3. Programmatic batch processing")
        print("  4. Detailed claim analysis")
