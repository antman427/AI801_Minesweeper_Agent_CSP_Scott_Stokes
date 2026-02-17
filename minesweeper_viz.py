import os
import time
import numpy as np

def clear_screen():
    """Clears terminal: 'cls' for Windows, 'clear' for Mac/Linux"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(obs):
    """
    Prints the board state to the console with nice formatting.
    obs: The numpy array returned by env.step()
    """
    # Symbols mapping
    # -3: Flag, -2: Covered, -1: Exploded Mine, 0: Empty, 1-8: Numbers
    symbols = {
       -3: "F", # Flagged
        -2: " ", # Covered (Easier to see if blank)
        -1: "*", # Exploded Mine
        0: "."   # Empty Cell
    }
    
    # 1. Print Column Numbers (top header)
    h, w = obs.shape
    print("    " + " ".join([f"{i}" for i in range(w)])) 
    print("   " + "—" * (w * 2)) # Divider line

    # 2. Print Rows
    for r in range(h):
        row_str = f"{r} | " # Row Number + divider
        for val in obs[r]:
            if val > 0: 
                row_str += f"{val} " # Show number
            else:       
                row_str += f"{symbols.get(val, '?')} " # Show symbol
        print(row_str)
    print()

def print_header(title, step=None, info=None):
    """Optional helper to print a consistent header box."""
    print("=" * 30)
    print(f"   {title.upper()}")
    if step is not None:
        print(f"   Step: {step}")
    if info:
        print(f"   Status: {info}")
    print("=" * 30)
