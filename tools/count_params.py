# MatthewInkawhich

# This script counts the number of learnable parameters from a state_dict

import torch
import argparse
import os
import matplotlib.pyplot as plt



##################################
### MAIN
##################################
def main():
    # Get log path from arg
    parser = argparse.ArgumentParser(description="Training Loss Plotter")
    parser.add_argument(
        "path",
        metavar="path",
        type=str
    )
    args = parser.parse_args()

    # Read state_dict file into Python dict
    state_dict = torch.load(args.path, map_location=torch.device('cpu'))['model']

    total = 0
    for name, tensor in state_dict.items():
        c = tensor.numel()
        total += c
        print(name, c)

    print("\n\ntotal parameters:", total)


if __name__ == "__main__":
    main()
