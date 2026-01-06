import sys
import torch

def check_env():
    print(f"Checking PyTorch Environment...")
    print(f"Torch Version: {torch.__version__}")
    
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    print(f"ROCm (HIP) Support: {is_rocm}")
    
    if not is_rocm:
        print("ERROR: PyTorch is NOT installed with ROCm support. (Detected Generic/CUDA version)")
        return False
        
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False. GPU not accessible.")
        return False
        
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    return True

if __name__ == "__main__":
    if not check_env():
        sys.exit(1)
    sys.exit(0)
