import os
import sys

def find_amdsmi():
    search_roots = ['/opt/rocm', '/usr/lib', '/usr/local/lib']
    print(f"Searching for 'amdsmi' package in {search_roots}...")
    
    found_paths = []
    
    for root in search_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if 'amdsmi' in dirnames:
                full_path = os.path.join(dirpath, 'amdsmi')
                if os.path.exists(os.path.join(full_path, '__init__.py')):
                    print(f"FOUND: {full_path}")
                    found_paths.append(full_path)
                    
            # Don't go too deep to save time
            if dirpath.count(os.sep) - root.count(os.sep) > 4:
                del dirnames[:]
                
    if found_paths:
        print("\nPotential PYTHONPATH entries:")
        for p in found_paths:
            print(os.path.dirname(p))
    else:
        print("Could not find 'amdsmi' package.")

if __name__ == "__main__":
    find_amdsmi()
