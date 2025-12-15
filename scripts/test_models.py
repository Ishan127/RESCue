import argparse
import sys
import os
import numpy as np
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.planner import Planner
from src.executor import Executor

def test_models(dtype="auto", quantization=None):
    print(f"Testing Model Loading... (dtype={dtype}, quantization={quantization})")
    
    try:
        print("Loading Qwen2.5-VL...")
        planner = Planner(dtype=dtype, quantization=quantization)
        img = Image.new('RGB', (100, 100), color='black')
        img_path = "temp_test_model.jpg"
        img.save(img_path)
        
        print("Running dummy generation...")
        hypotheses = planner.generate_hypotheses(img_path, "describe this black image", N=1)
        print("Planner Output:", hypotheses)
        
        os.remove(img_path)
        print("Planner Check Passed.")
    except Exception as e:
        print(f"Planner Failed: {e}")
        sys.exit(1)

    try:
        print("Loading SAM 3...")
        executor = Executor()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_box = [10, 10, 50, 50]
        masks = executor.execute(dummy_img, dummy_box, "black square")
        print(f"Executor produced {len(masks)} masks.")
        print("Executor Check Passed.")
    except Exception as e:
        print(f"Executor Failed: {e}")
        sys.exit(1)

    print("All Model Checks Passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="auto", help="Model data type")
    parser.add_argument("--quantization", default=None, help="Model quantization")
    args = parser.parse_args()
    
    test_models(dtype=args.dtype, quantization=args.quantization)
