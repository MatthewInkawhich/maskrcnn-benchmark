# MatthewInkawhich

# Based on: https://gist.github.com/Nikasa1889/781a8eb20c5b32f8e378353cde4daa51#file-computereceptivefield-py

# [filter size, stride, padding]
#Assume the two dimensions are the same
#Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
# 
#Each layer i requires the following parameters to be fully represented: 
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

count = 1

#############################################################
### FUNCTIONS
#############################################################
def readModelFromFile(filepath):
    model = []
    with open(filepath) as f:
        for line in f:
            # Remove newline
            line = line.rstrip('\n')
            # filter out empty lines and comment lines
            if line and line[0] != '#':
                if '#' in line:
                    line = line.split('#')[0]
                layer = line.split(',')
                layer = [float(l) for l in layer]
                model.append(layer)
    return model


def outFromIn(conv, layerIn):
    n_in = layerIn[1]
    j_in = layerIn[2]
    r_in = layerIn[3]
    #start_in = layerIn[4]
    k = conv[0]
    s = conv[1]
    p = conv[2]
    d = conv[3]

    # adjust kernel according to dilation
    rf_k = d * (k - 1) + 1

    #n_out = math.floor((n_in - k + 2*p)/s) + 1
    n_out = math.floor((n_in + 2*p - (d * (k-1)) - 1) / s) + 1

    #actualP = (n_out-1)*s - n_in + k 
    #pR = math.ceil(actualP/2)
    #pL = math.floor(actualP/2)

    j_out = j_in * s
    r_out = r_in + (rf_k - 1)*j_in
    #start_out = start_in + ((k-1)/2 - pL)*j_in
    return n_in, n_out, j_out, r_out #, start_out
  

def printLayer(layer):
    global count
    print("\n\tcount:%s \n \t input_size: %s \n \t output_size %s \n \t rf: %s" % (count, layer[0], layer[1], layer[3]))
    count += 1
 

def plot_rf_growth(rfs, R, RFGC):
    layers = list(range(1, len(rfs)+1))
    plt.plot(layers, rfs)
    plt.fill_between(layers, rfs)
    plt.title(args.modelfile.split('/')[-1])
    plt.xlabel("Layer")
    plt.ylabel("Receptive Field")
    plt.xlim(1, len(rfs))
    plt.ylim(0, rfs[-1])
    plt.text(0.1*len(rfs), 0.85*rfs[-1], 'R={:.4f}\nRFGC={:.4f}'.format(R, RFGC), fontsize=12)
    plt.show()
    


#############################################################
### MAIN
#############################################################
imsize = 100
layerInfos = []


if __name__ == '__main__':
    # Read command line args
    parser = argparse.ArgumentParser(description="RF Calculation")
    parser.add_argument(
        "modelfile",
        metavar="modelfile",
        type=str
    )
    args = parser.parse_args()

    # Read model from file
    convnet = readModelFromFile(args.modelfile)

    # Collect Net Summary
    print ("-------Net summary------")
    # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    currentLayer = [0, imsize, 1, 1, 0.5]
    # Iterate over layers, append sizes and rf to layerInfos
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(list(currentLayer))
        printLayer(currentLayer)
    print ("------------------------")
    

    #print("layerInfos:", layerInfos)
    rfs = np.array(layerInfos)[:, 3]
    #print("\nrfs:", rfs, len(rfs))

    # Compute RFGC
    final_rf = rfs[-1]
    num_layers = len(rfs)
    auc = sum([rfs[l] + 0.5*(rfs[l+1]-rfs[l]) for l in range(num_layers-1)])
    R = auc / ((num_layers-1) * final_rf)
    RFGC = R * (final_rf / (num_layers-1))
    print("num layers:", num_layers)
    print("final RF:", final_rf)
    print("R:", R)
    print("RFGC:", RFGC)

    # Plot RF over layers
    plot_rf_growth(rfs, R, RFGC)

    
