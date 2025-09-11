#!/usr/bin/env python3
"""
Debug script to check button position and understand the next_to_act differences.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run the comparison script with 1 episode and capture early output
if __name__ == "__main__":
    import subprocess
    result = subprocess.run([
        "uv", "run", "tests/speed/compare_engines.py", 
        "--episodes", "1", 
        "--stop-on-first"
    ], capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
