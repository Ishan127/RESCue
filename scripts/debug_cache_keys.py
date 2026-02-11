import os
import json
import glob

cache_dir = "cache"

# Try to find any score file
score_files = glob.glob(os.path.join(cache_dir, "scores", "*", "*.json"))
if not score_files:
    # Try D drive path directly?
    score_files = glob.glob("D:/BTP_Data/cache/cache/scores/*/*.json")

if not score_files:
    print("No score files found in 'cache/scores' or 'D:/BTP_Data/cache/cache/scores'")
    # List top level cache to see what's there
    if os.path.exists(cache_dir):
        print(f"Contents of {cache_dir}:")
        print(os.listdir(cache_dir))
    else:
        print(f"Directory {cache_dir} does not exist")
else:
    target_file = score_files[0]
    print(f"Inspecting: {target_file}")
    
    try:
        with open(target_file, 'r') as f:
            data = json.load(f)
            
        print("Keys found in JSON:")
        # Provide the top-level keys (usually indices like "0", "1")
        # And the nested keys (likely "v0", "v1")
        for k, v in list(data.items())[:3]:
            print(f"  Key '{k}': {list(v.keys())}")
            
    except Exception as e:
        print(f"Error reading file: {e}")
