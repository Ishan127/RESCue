"""
Test parallel verifier performance.

This script benchmarks the tournament verification with different numbers
of candidates to ensure parallelization is working.
"""
import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

os.environ["VERIFIER_API_BASE"] = os.environ.get("VERIFIER_API_BASE", "http://localhost:8000/v1")
os.environ["VERIFIER_VERBOSE"] = "1"

from src.verifier import Verifier

def create_dummy_masks(n, size=(512, 512)):
    """Create n different random masks for testing."""
    masks = []
    h, w = size
    for i in range(n):
        mask = np.zeros((h, w), dtype=np.uint8)
        # Create different sized squares at different positions
        x = (i * 50) % (w - 100)
        y = (i * 30) % (h - 100)
        side = 50 + (i * 10) % 100
        mask[y:y+side, x:x+side] = 255
        masks.append(Image.fromarray(mask))
    return masks

def main():
    print("Testing Parallel Tournament Verifier")
    print("=" * 60)
    
    verifier = Verifier(verbose=True)
    
    # Create a dummy image
    image = Image.new('RGB', (512, 512), color=(100, 150, 200))
    query = "the square shaped object on the left"
    
    for n in [4, 8, 16, 32, 64]:
        print(f"\n{'='*60}")
        print(f"Testing with N={n} candidates")
        print(f"{'='*60}")
        
        masks = create_dummy_masks(n)
        
        t0 = time.time()
        results = verifier.verify_batch_comparative(image, masks, query)
        elapsed = time.time() - t0
        
        print(f"\nResults:")
        print(f"  - Time: {elapsed:.2f}s")
        print(f"  - Time per candidate: {elapsed/n:.3f}s")
        
        # For tournament with N candidates:
        # - Rounds = ceil(log2(N))
        # - Comparisons = N - 1
        rounds = int(np.ceil(np.log2(n)))
        comparisons = n - 1
        print(f"  - Expected rounds: {rounds}")
        print(f"  - Total comparisons: {comparisons}")
        print(f"  - Time per comparison: {elapsed/comparisons:.3f}s")
        
        # Show top 5 ranked
        top5 = sorted(results, key=lambda r: r['rank'])[:5]
        print(f"  - Top 5: {[r['mask_idx'] for r in top5]}")
        
        # With max-num-seqs=64, we should be able to batch many parallel requests
        # Good throughput = time roughly scales with rounds, not comparisons

if __name__ == "__main__":
    main()
