import os

import numpy as np
import torch

def write_debug_log(input_data, append=True):
    """
        Writes the input_data to a debug.txt file if it exists. If not, the function will either 
        append to the existing file or create a new file named debug_{i}.txt based on the append flag.
    
        Args:
            input_data (Any): The data to be written to the debug file.
            append (bool): Flag to determine if the data should be appended to an existing debug.txt file.
    """
    file_exists = os.path.exists("debug.txt")
    
    if file_exists and append:
        with open("debug.txt", "a") as file:
            file.write(str(input_data) + "\n")
    elif file_exists and not append:
        i = 1
        while os.path.exists(f"debug_{i}.txt"):
            i += 1
        with open(f"debug_{i}.txt", "w") as file:
            file.write(str(input_data) + "\n")
    else:
        with open("debug.txt", "w") as file:
            file.write(str(input_data) + "\n")

def print_structure(data, indent=0):
    """
        Recursively prints the structure, dimensions, and min/max values of lists, tuples, numpy arrays, and torch tensors.
    """
    prefix = ' ' * indent

    if isinstance(data, np.ndarray):
        print(f"{prefix}NumPy Array: shape={data.shape}, dtype={data.dtype}")
        print(f"{prefix}  min={np.min(data)}, max={np.max(data)}")
    elif isinstance(data, torch.Tensor):
        print(f"{prefix}Torch Tensor: shape={tuple(data.shape)}, dtype={data.dtype}")
        print(f"{prefix}  min={torch.min(data)}, max={torch.max(data)}")
    elif isinstance(data, list):
        print(f"{prefix}List: length={len(data)}")
        if len(data) > 0:
            if isinstance(data[0], (list, tuple, np.ndarray, torch.Tensor)):
                for i, item in enumerate(data):
                    print(f"{prefix}  [{i}]:")
                    print_structure(item, indent + 4)
            else:
                print(f"{prefix}  min={min(data)}, max={max(data)}")
    elif isinstance(data, tuple):
        print(f"{prefix}Tuple: length={len(data)}")
        if len(data) > 0:
            for i, item in enumerate(data):
                print(f"{prefix}  [{i}]:")
                print_structure(item, indent + 4)
    else:
        print(f"{prefix}{type(data).__name__}: value={data}")

