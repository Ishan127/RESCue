import os
import time
import requests
import subprocess
import sys

def check_endpoint(url, name):
    print(f"Checking {name} at {url}...")
    try:
        if "v1" in url:
            # vLLM health check
            base = url.replace("/v1", "")
            resp = requests.get(f"{base}/health", timeout=2)
        else:
             # SAM server (FastAPI docs is a good check)
            resp = requests.get(f"{url}/docs", timeout=2)
        
        if resp.status_code == 200:
            print(f"{name} is UP.")
            return True
    except Exception as e:
        pass
    print(f"{name} is DOWN.")
    return False

def main():
    print("=== RESCue Benchmark Main Script ===")
    
    # 1. Download Data
    print("\n--- Step 1: Downloading Data ---")
    try:
        subprocess.run(["python", "scripts/download_data.py"], check=True)
    except subprocess.CalledProcessError:
        print("Data download failed or script not found.")
    
    # 2. Check Endpoints
    print("\n--- Step 2: Verifying Endpoints ---")
    planner_url = "http://localhost:8000/v1"
    executor_url = "http://localhost:8001"
    
    # We give the user a chance to start them if they haven't
    if not check_endpoint(planner_url, "Planner (vLLM)") or not check_endpoint(executor_url, "Executor (SAM)"):
        print("\nPlease ensure you are running the deployment scripts in separate shells:")
        print("  1. ./scripts/deploy_llm.sh")
        print("  2. ./scripts/deploy_sam.sh")
        print("Waiting 10 seconds before trying again...")
        time.sleep(10)
        
        if not check_endpoint(planner_url, "Planner (vLLM)") or not check_endpoint(executor_url, "Executor (SAM)"):
            print("Endpoints still not ready. Exiting.")
            sys.exit(1)

    # 3. Run Benchmark
    print("\n--- Step 3: Running Benchmark ---")
    subprocess.run([
        "python", "scripts/evaluate.py",
        "--fraction", "0.1",
        "--planner_url", planner_url,
        "--executor_url", executor_url
    ], check=True)

if __name__ == "__main__":
    main()
