import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rescue_pipeline import RESCuePipeline
from src.utils import plot_results

def main():
    parser = argparse.ArgumentParser(description="Run RESCue Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--query", required=True, help="Text query")
    parser.add_argument("--N", type=int, default=4, help="Number of reasoning paths")
    parser.add_argument("--output", default="output.jpg", help="Path to save result image")
    parser.add_argument("--dtype", default="auto", help="Model data type (auto, float16, etc.)")
    parser.add_argument("--quantization", default=None, help="Model quantization (awq, gptq, int8, etc.)")
    
    args = parser.parse_args()
    
    pipeline = RESCuePipeline(dtype=args.dtype, quantization=args.quantization)
    
    result = pipeline.run(args.image, args.query, N=args.N)
    
    if result:
        print(f"Best Score: {result['best_score']}")
        best_candidate = [c for c in result['all_candidates'] if c['score'] == result['best_score']][0]
        
        plot_results(
            result['image'], 
            [best_candidate['box']], 
            [best_candidate['mask']], 
            [best_candidate['score']], 
            output_path=args.output
        )
        print(f"Result saved to {args.output}")
    else:
        print("No candidates generated.")

if __name__ == "__main__":
    main()
