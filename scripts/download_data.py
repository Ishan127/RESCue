import os
from datasets import load_dataset

def download_reasonseg():
    print("Downloading ReasonSeg Validation Set...")
    try:
        ds = load_dataset("Ricky06662/ReasonSeg_test", split="test")
        print(f"Dataset downloaded successfully. Size: {len(ds)} samples.")
        print("Sample entry keys:", ds[0].keys())
        return ds
    except Exception as e:
        print(f"Error downloading ReasonSeg: {e}")
        return None

if __name__ == "__main__":
    download_reasonseg()
