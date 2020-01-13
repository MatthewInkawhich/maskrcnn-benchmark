# MatthewInkawhich

# This script plots a line graph that depicts the training loss.
# Designed to take a log.txt file (logger output) as input.

import argparse
import os
import matplotlib.pyplot as plt


##################################
### HELPERS
##################################




##################################
### MAIN
##################################
def main():
    # Get log path from arg
    parser = argparse.ArgumentParser(description="Training Loss Plotter")
    parser.add_argument(
        "logpath",
        metavar="logpath",
        type=str
    )
    parser.add_argument(
        "title",
        metavar="title",
        type=str
    )
    args = parser.parse_args()

    # Read file into list of lines
    lines = [line.rstrip('\n') for line in open(args.logpath)]
    
    # Filter everything but logger lines
    lines = [line for line in lines if "maskrcnn_benchmark.trainer INFO: eta:" in line]

    # Loop over lines, add to lists
    iters = []
    losses = []
    for line in lines:
        # Extract iter and loss
        it = int(line.split('iter: ')[1].split(' ')[0])
        loss = float(line.split('loss: ')[1].split('(')[0])

        # Append to lists
        if it % 100 == 0:
            iters.append(it)
            losses.append(loss)

    # Plot loss vs iter
    plt.plot(iters, losses)
    #plt.title("Training: {}".format(args.logpath))
    plt.title("Training Loss: {}".format(args.title))
    plt.xlabel("Iter")
    plt.ylabel("Loss")
    plt.show()







if __name__ == "__main__":
    main()
