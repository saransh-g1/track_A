#!/usr/bin/env python3
"""
Test script to run Track A pipeline with provided dataset files
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import TrackAConfig
from run_pipeline import TrackAPipeline


def test_small_files():
    """Test with small example files"""
    print("="*80)
    print("TEST 1: Small Example Files")
    print("="*80)
    
    novel_path = "files/novel1.txt"
    backstory_path = "files/backstory1.txt"
    
    if not os.path.exists(novel_path) or not os.path.exists(backstory_path):
        print(f"ERROR: Files not found!")
        print(f"  Looking for: {novel_path}, {backstory_path}")
        return False
    
    try:
        # Use smaller config for testing
        config = TrackAConfig(
            max_claims_per_backstory=10,  # Fewer claims for small backstory
            top_k_retrieval=3,
            use_quantization=True
        )
        
        pipeline = TrackAPipeline(config)
        decision = pipeline.process(
            novel_path=novel_path,
            backstory_path=backstory_path,
            output_path="results_small.json"
        )
        
        print(f"\n✓ Test completed successfully!")
        print(f"  Label: {decision.label} ({'COHERENT' if decision.label == 1 else 'NOT COHERENT'})")
        print(f"  Confidence: {decision.confidence:.3f}")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_large_files():
    """Test with large dataset files"""
    print("\n" + "="*80)
    print("TEST 2: Large Dataset Files")
    print("="*80)
    
    novel_path = "files/novel.txt"
    backstory_path = "files/backstory.txt"
    
    if not os.path.exists(novel_path) or not os.path.exists(backstory_path):
        print(f"ERROR: Files not found!")
        print(f"  Looking for: {novel_path}, {backstory_path}")
        return False
    
    # Check file sizes
    novel_size = os.path.getsize(novel_path) / (1024 * 1024)  # MB
    backstory_size = os.path.getsize(backstory_path) / (1024 * 1024)  # MB
    
    print(f"Novel size: {novel_size:.2f} MB")
    print(f"Backstory size: {backstory_size:.2f} MB")
    print(f"\nThis may take a while due to large file sizes...")
    
    try:
        # Use full config for large files
        config = TrackAConfig(
            max_claims_per_backstory=25,  # More claims for large backstory
            top_k_retrieval=5,
            chunk_size=512,
            use_quantization=True,
            consistency_threshold=0.6
        )
        
        pipeline = TrackAPipeline(config)
        decision = pipeline.process(
            novel_path=novel_path,
            backstory_path=backstory_path,
            output_path="results_large.json"
        )
        
        print(f"\n✓ Test completed successfully!")
        print(f"  Label: {decision.label} ({'COHERENT' if decision.label == 1 else 'NOT COHERENT'})")
        print(f"  Confidence: {decision.confidence:.3f}")
        print(f"  Satisfied: {decision.satisfied_claims}/{decision.total_claims} claims")
        print(f"  Violated: {decision.violated_claims}/{decision.total_claims} claims")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TRACK A PIPELINE TESTING")
    print("="*80)
    print("\nThis script will test the pipeline with your dataset files.")
    print("Note: First test uses small files, second test uses large files.")
    print("\nMake sure you have:")
    print("  1. Installed all dependencies (pip install -r requirements.txt)")
    print("  2. Set Hugging Face token: export HF_TOKEN=your_token")
    print("  3. Have sufficient GPU memory (or use CPU with use_quantization=True)")
    print("\n" + "="*80)
    
    # Ask user which test to run
    if len(sys.argv) > 1:
        test_choice = sys.argv[1]
    else:
        test_choice = input("\nWhich test to run? (1=small, 2=large, 3=both) [3]: ").strip() or "3"
    
    results = {}
    
    if test_choice in ["1", "3", "both"]:
        results["small"] = test_small_files()
    
    if test_choice in ["2", "3", "both"]:
        results["large"] = test_large_files()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed! Pipeline is ready to use.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

