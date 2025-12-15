import os
from datasets import load_dataset

def download_reasonseg():
    print("Downloading ReasonSeg Validation Set...")
    try:
        # Using the HF Hub dataset
        # Note: The dataset repo is named ReasonSeg_val but the internal split is named 'test'
        ds = load_dataset("Ricky06662/ReasonSeg_val", split="test")
        print(f"Dataset downloaded successfully. Size: {len(ds)} samples.")
        
        # Print sample structure
        print("Sample entry keys:", ds[0].keys())
        
        # Identify key columns
        # Usually: image, text/query, mask
        
        return ds
    except Exception as e:
        print(f"Error downloading ReasonSeg: {e}")
        return None

if __name__ == "__main__":
    download_reasonseg()
