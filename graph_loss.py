# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 18/11/2019
"""
Post-processing program to visualize loss functions and the processing time during training of DCGAN and SRResGAN

Parameters
----------
directory_name : name of the folder containing the log of the training program called "output.txt"
                 (contains regex like (\d+.\d+) - Epoch (\d+), Batch (\d+): d_loss=(-?\d+.\d+(e(\+|-)\d+)?|nan), g_loss=(-?\d+.\d+(e(\+|-)\d+)?|nan))

Return
----------
Plot and save the two graphs "learning_curve.pdf" and "processing_time.pdf" in the folder directory_name
"""

import numpy as np
import matplotlib.pyplot as plt
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

    # Reading output text file
    with open(os.path.join(directory_name, "output.txt")) as f:
        file_lines = f.readlines()
    file_lines = [x.strip() for x in file_lines]

    # Initialization
    i=0
    while " - Epoch 0, Batch 0: " not in file_lines[i]:
        i += 1

    tab_time_val = []
    tab_epoch_val = []
    tab_batch_val = []
    tab_d_loss_val = []
    tab_g_loss_val = []

    tab_time_test = []
    tab_epoch_test = []
    tab_batch_test = []
    tab_d_loss_test = []
    tab_g_loss_test = []

    # Extracting data
    while i < len(file_lines):
        if " - Testing: " in file_lines[i]:
            regex_group = re.search("(\d+.\d+) - Epoch (\d+), Batch (\d+) - Testing: d_loss=(-?\d+.\d+(e(\+|-)\d+)?|nan), g_loss=(-?\d+.\d+(e(\+|-)\d+)?|nan)", file_lines[i])
            #print(i, regex_group.group(1), regex_group.group(2), regex_group.group(3), regex_group.group(4), regex_group.group(7))
            
            tab_time_test.append(regex_group.group(1))
            tab_epoch_test.append(regex_group.group(2))
            tab_batch_test.append(regex_group.group(3))
            tab_d_loss_test.append(regex_group.group(4))
            tab_g_loss_test.append(regex_group.group(7))
        else:
            regex_group = re.search("(\d+.\d+) - Epoch (\d+), Batch (\d+): d_loss=(-?\d+.\d+(e(\+|-)\d+)?|nan), g_loss=(-?\d+.\d+(e(\+|-)\d+)?|nan)", file_lines[i])
            #print(i, regex_group.group(1), regex_group.group(2), regex_group.group(3), regex_group.group(4), regex_group.group(7))
            
            tab_time_val.append(regex_group.group(1))
            tab_epoch_val.append(regex_group.group(2))
            tab_batch_val.append(regex_group.group(3))
            tab_d_loss_val.append(regex_group.group(4))
            tab_g_loss_val.append(regex_group.group(7))
        i += 1

    tab_time_val = np.array(tab_time_val, dtype=float)
    tab_epoch_val = np.array(tab_epoch_val, dtype=int)
    tab_batch_val = np.array(tab_batch_val, dtype=int)
    tab_d_loss_val = np.array(tab_d_loss_val, dtype=float)
    tab_g_loss_val = np.array(tab_g_loss_val, dtype=float)

    tab_time_val_difference = np.diff(tab_time_val)
    tab_batch_glob_val = tab_batch_val+tab_epoch_val*(np.max(tab_batch_val)+1)
    tab_batch_glob_val_difference = np.diff(tab_batch_glob_val)

    tab_time_test = np.array(tab_time_test, dtype=float)
    tab_epoch_test = np.array(tab_epoch_test, dtype=int)
    tab_batch_test = np.array(tab_batch_test, dtype=int)
    tab_d_loss_test = np.array(tab_d_loss_test, dtype=float)
    tab_g_loss_test = np.array(tab_g_loss_test, dtype=float)

    tab_time_test_difference = np.diff(tab_time_test)
    tab_batch_glob_test = tab_batch_test+tab_epoch_test*(np.max(tab_batch_val)+1)
    tab_batch_glob_test_difference = np.diff(tab_batch_glob_test)

    # Plotting learning curve
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(tab_d_loss_val)
    plt.plot(tab_g_loss_val)
    plt.plot(tab_batch_glob_test, tab_d_loss_test, '.')
    plt.plot(tab_batch_glob_test, tab_g_loss_test, '.')
    for i in range(0, len(tab_batch_val)):
        if tab_batch_val[i] == 0:
            plt.axvline(x=tab_batch_glob_val[i], linestyle="--", color="gray")
    plt.xlabel("Batchs")
    plt.ylabel("Loss")
    if np.nanmin(tab_d_loss_test) < 0 or np.nanmin(tab_g_loss_test) < 0 or np.nanmin(tab_d_loss_val) < 0 or np.nanmin(tab_g_loss_val) < 0:
        plt.yscale('symlog')
    else:
        plt.yscale('log')
    plt.legend(["Discriminator validation loss", "Generator validation loss", "Discriminator test loss", "Generator test loss", "Epochs"])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(directory_name, "learning_curve.pdf"), format="pdf")
    plt.show()

    # Plotting processing time graph
    plt.figure(figsize=(8, 8))
    plt.title("Average processing time")
    plt.plot(np.cumsum(tab_batch_glob_val_difference), np.divide(tab_time_val_difference, tab_batch_glob_val_difference))
    plt.plot(np.cumsum(tab_batch_glob_test_difference), np.divide(tab_time_test_difference, tab_batch_glob_test_difference), '.')
    for i in range(0, len(tab_batch_val)):
        if tab_batch_val[i] == 0:
            plt.axvline(x=tab_batch_glob_val[i], linestyle="--", color="gray")
    plt.xlabel("Batchs")
    plt.ylabel("Time (sec/input)")
    plt.yscale('log')
    plt.legend(["Validation processing time", "Test processing time", "Epochs"])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(directory_name, "processing_time.pdf"), format="pdf")
    plt.show()
