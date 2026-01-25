import json
import numpy as np

def analyze_output(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total Samples: {len(data['samples'])}")
    
    breakdowns = {
        "geo": [],
        "rating": [],
        "iou": [],
        "boundary": [],
        "semantic": []
    }
    
    raw_responses = []
    
    for s in data['samples']:
        for c in s['candidates']:
            bk = c.get('pointwise_breakdown')
            if bk:
                for k, v in bk.items():
                    breakdowns[k].append(v)
            
            # Check for empty/failed parsing
            raw = c.get('pointwise_raw', {})
            if not raw:
                raw_responses.append("EMPTY")
            else:
                raw_responses.append(raw)

    print("\n--- Metric Statistics ---")
    for k, v in breakdowns.items():
        if v:
            # Fix: Use sum with generator for proper float comparison instead of list.count(0)
            zeros_count = sum(1 for x in v if abs(x) < 0.001)
            print(f"{k.upper()}: Mean={np.mean(v):.2f}, Max={np.max(v)}, Min={np.min(v)}, Count={len(v)}")
            print(f"  Zeros: {zeros_count} ({zeros_count/len(v)*100:.1f}%)")
        else:
            print(f"{k.upper()}: NO DATA")

    print("\n--- Raw Response Analysis (First 5 Non-Empty) ---")
    non_empty = [r for r in raw_responses if r != "EMPTY"]
    for i, r in enumerate(non_empty[:5]):
        print(f"Response {i+1}: {r}")

    print(f"\nTotal Candidates Scored: {len(breakdowns['geo'])}")
    print(f"Total Raw Responses Captured: {len(non_empty)}")

if __name__ == "__main__":
    analyze_output("output.json")
