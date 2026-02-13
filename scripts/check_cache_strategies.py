import json
import os
from collections import Counter

def main():
    plans_path = "cache/plans/plans.json"
    if not os.path.exists(plans_path):
        print(f"File not found: {plans_path}")
        # Try D drive
        plans_path = "D:/BTP_Data/cache/cache/plans/plans.json"
        if not os.path.exists(plans_path):
             print(f"File not found: {plans_path}")
             return

    print(f"Loading {plans_path}...")
    try:
        with open(plans_path, 'r') as f:
            data = json.load(f)
            
        strategy_counts = Counter()
        total_hyps = 0
        
        # Check first 5 samples
        for i, (key, val) in enumerate(data.items()):
            hyps = val.get('hypotheses', [])
            for h in hyps:
                s = h.get('strategy', 'UNKNOWN')
                strategy_counts[s] += 1
                total_hyps += 1
                
        print(f"Total Hypotheses: {total_hyps}")
        print("Strategy Counts:")
        for s, count in strategy_counts.most_common():
            print(f"  '{s}': {count}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
