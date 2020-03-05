# MatthewInkawhich

# Create "onion" plot comparing receptive field growth of different models.

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt


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
    k = conv[0]
    s = conv[1]
    p = conv[2]
    d = conv[3]
    # adjust kernel according to dilation
    rf_k = d * (k - 1) + 1
    #n_out = math.floor((n_in - k + 2*p)/s) + 1
    n_out = math.floor((n_in + 2*p - (d * (k-1)) - 1) / s) + 1
    j_out = j_in * s
    r_out = r_in + (rf_k - 1)*j_in
    return n_in, n_out, j_out, r_out
  


#############################################################
### MAIN
#############################################################
imsize = 100


if __name__ == '__main__':
    # Read command line args
    parser = argparse.ArgumentParser(description="RF Calculation")
    parser.add_argument(
        "modelfiles",
        metavar="modelfiles",
        type=str,
        nargs='*'
    )
    args = parser.parse_args()

    for modelfile in args.modelfiles:
        layerInfos = []
        # Read model from file
        convnet = readModelFromFile(modelfile)

        # Collect Net Summary
        # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
        currentLayer = [0, imsize, 1, 1, 0.5]
        # Iterate over layers, append sizes and rf to layerInfos
        for i in range(len(convnet)):
            currentLayer = outFromIn(convnet[i], currentLayer)
            layerInfos.append(list(currentLayer))
    
        
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


        # Plot RF growth for current model
        layers = list(range(1, len(rfs)+1))
        name = modelfile.split('/')[-1].split('_')[0]
        #label = "{}: R={:.4f}, RFGC={:.4f}".format(name, R, RFGC)
        label = "{}: R={:.4f}".format(name, R)
        plt.plot(layers, rfs, label=label)


    # Add axes labels, titles, etc and show    
    plt.title("Receptive Field Growth")
    plt.xlabel("Layer")
    plt.ylabel("Receptive Field")
    plt.xlim(1, len(rfs))
    plt.ylim(bottom=0)
    #plt.text(0.1*len(rfs), 0.85*rfs[-1], 'R={:.4f}\nRFGC={:.4f}'.format(R, RFGC), fontsize=12)
    plt.legend()
    plt.show()
        
    
