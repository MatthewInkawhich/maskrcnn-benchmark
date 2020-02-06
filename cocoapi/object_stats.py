# MatthewInkawhich

"""
This script is a utility to iterate through COCO-style annotations and print statistics.
"""

import os
import json
import argparse
import math

### Print strings
#count = 0
#for i in range(10, 630, 10):
#    count += 1
#    print(i)
#print("count:", count)
#exit()

# Define Histogram class
class Histogram():
    def __init__(self, splits):
        self.splits = splits
        self.splits.append(10000)  # manually add upper bound split
        # Initialize bins dict to zeros
        self.bins = {}
        for s in self.splits:
            self.bins[s] = 0

    def insert(self, value):
        for k in self.bins:
            if value < k:
                self.bins[k] += 1
                break

    def get(self):
        return self.bins


# Process argument
parser = argparse.ArgumentParser(description="Print object stats")
parser.add_argument(
    "--annotation-file",
    default="annotations2017/annotations/instances_val2017.json",
    metavar="FILE",
    type=str,
    help="path to annotation file",
)
args = parser.parse_args()


# Initialize Histogram object
splits = list(range(5, 305, 5))
hist = Histogram(splits)
print(hist.get())


# Read json annotations file
data = json.load(open(args.annotation_file, 'r'))


# Iterate over object instances
total_count = 0
min_area = 100000000
max_area = -1
for obj in data['annotations']:
    total_count += 1
    if obj["area"] > max_area:
        max_area = obj["area"]
    if obj["area"] < min_area:
        min_area = obj["area"]
    hist.insert(math.sqrt(obj["area"]))



print("total_count:", total_count)
print("min_area:", min_area)
print("max_area:", max_area)
for k, v in hist.get().items():
    print(v)
