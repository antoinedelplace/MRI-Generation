# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 18/11/2019
"""
Post-processing program to visualize training outputs of ProGAN

Parameters
----------
directory_name : name of the folder containing all the outputs of the training 
                 (as regex generated_test_nblock_(\d+)_transition_(\d+\.\d+)_epoch_(\d+)_batch_(\d+)\.npy)
interval_img   : time in milliseconds between two images in the animation (default: 50)

Return
----------
Plot and save to directory_name.gif in the folder directory_name folder the animation of all the training outputs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import os
import sys
import re

if __name__ == "__main__":
    # Reading parameters
    if len(sys.argv) < 2:
        print("Need a directory_name")
        sys.exit(1)
    directory_name = sys.argv[1]
    print(directory_name)
    
    if len(sys.argv) > 2:
        interval_img = sys.argv[2]
    else:
        interval_img = 50
        print("Image interval not found: default ({}) is used.".format(interval_img))

    # Configuring plots
    fig = plt.figure(figsize=(7, 10))
    ax = []
    for i in range(0, 9):
        ax.append(fig.add_subplot('33{}'.format(str(i+1))))
    ims = []

    # Extracting output names
    data = []
    nblocks = []
    transitions = []
    epochs = []
    batchs = []
    for f in os.listdir(directory_name):
        r = re.search("generated_test_nblock_(\d+)_transition_(\d+\.\d+)_epoch_(\d+)_batch_(\d+)\.npy", f)
        if r:
            data.append(os.path.join(directory_name, f))
            nblocks.append(int(r.group(1)))
            transitions.append(float(r.group(2)))
            epochs.append(int(r.group(3)))
            batchs.append(int(r.group(4)))

    # Adding images to the animation in the chronological order
    for f, _, _, _, _ in sorted(zip(data, nblocks, transitions, epochs, batchs), key=lambda x: (x[1], x[2], x[3], x[4])):
        figures = np.load(f)
        ims.append([])
        
        for i in range(0, 9):
            im = ax[i].imshow(resize(figures[i, :, :], (256, 256), order=0), cmap='jet', vmin=-1.0, vmax=1.0, animated=True)
            ims[-1].append(im)

    # Plotting and saving
    ani = animation.ArtistAnimation(fig, ims, interval=int(interval_img), blit=True, repeat=False)
    plt.tight_layout()
    ani.save(os.path.join(directory_name, '{}.gif'.format(directory_name)))
    plt.show()
