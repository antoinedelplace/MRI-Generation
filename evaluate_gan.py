# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 18/11/2019
"""
Post-processing program to evaluate the performance of the GAN model

Parameters
----------
directory_name : name of the folder containing all the generated images
                 (outputs of *_generate_data.py or *_generate_interpolation.py for example)

Requirements
----------
mean_oasis_dataset.npy         : contains the mean of all images from the OASIS training set
eigenvectors_oasis_dataset.npy : contains all the normalized eigenvectors of the OASIS training manifold.

Return
----------
Print several evaluation measures : - number of generated images
                                    - mean realism (rho)
                                    - diversity extent (sigma)
                                    - number of relevant features (delta)
"""

import numpy as np
import os
import sys
import imageio

if __name__ == "__main__":
    # Reading parameters
    if len(sys.argv) < 2:
        print("Need a directory_name")
        sys.exit(1)
    dir_img = sys.argv[1]
    print(dir_img)

    ## Parameters
    input_size = 256

    ## Load dataset
    name_img = [f for f in os.listdir(dir_img) if os.path.isfile(os.path.join(dir_img, f))]
    print("Nb of images: ", len(name_img))

    tab_img = np.zeros((len(name_img), input_size*input_size))
    for i in range(0, len(name_img)):
        tab_img[i] = imageio.imread(os.path.join(dir_img, name_img[i])).flatten()

    print("Shape of tab_img: ", tab_img.shape)

    ## Normalize if not between -1 and 1
    tab_img -= 127.5
    tab_img *= 1./127.5

    ## Realism
    mean_real = np.load("mean_oasis_dataset.npy")
    eigenvectors_real = np.load("eigenvectors_oasis_dataset.npy")

    print("Shape of Mean real: ", np.shape(mean_real))
    print("Shape of Eigenvectors real: ", np.shape(eigenvectors_real))

    tab_img2 = (tab_img-mean_real).T
    tab_img2 /= np.linalg.norm(tab_img2, axis=0)

    proj = eigenvectors_real @ tab_img2
    print("Shape of proj: ", np.shape(proj))

    print("Mean realism: ", np.linalg.norm(proj, axis=0).mean())

    ## Diversity
    mean_img = tab_img.mean(axis=0)

    tab_img -= mean_img
    Mcov = np.cov(tab_img)
    w, v = np.linalg.eig(Mcov)
    w_perc = w/np.sum(w)

    print(w)
    print(w_perc)
    print("Diversity extent: ", np.sum(w))

    i = 0
    while w_perc[i] > 0.01:
        print(i, w_perc[i], w[i])
        i += 1

    print("Number of relevant features: ", i)
    print("Percentage all relevant eigens: ", w_perc[0:i].sum())
