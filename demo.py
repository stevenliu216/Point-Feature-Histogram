"""
Main file to demonstrate the algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

import pfh.utils as utils
from pfh.pfh import PFH, SPFH, FPFH

if __name__=='__main__':
    # Import the data
    source_pc = utils.load_pc_np('data/plant_source.npy')
    target_pc = utils.load_pc_np('data/plant_target3.npy')
    print("...Done loading point cloud. \n")

    # Display the starting data
    utils.view_pc([source_pc, target_pc], None, ['b', 'r'], ['o', '^'])
    plt.title('Starting Point Clouds')

    # Run ICP with some example parameters
    et = 0.1
    div = 2
    nneighbors = 8
    rad = 0.03
    #Icp = PFH(et, div, nneighbors, rad)   # Full PFH
    #Icp = SPFH(et, div, nneighbors, rad)  # Simplified PFH
    Icp = FPFH(et, div, nneighbors, rad)   # Fast PFH
    transformed_source = Icp.solve(source_pc, target_pc)
    R_list = Icp._Rlist
    t_list = Icp._tlist
    errorlist = Icp._error_list

    # Animation - can use R_list and t_list, and source_pc..

    # Calculate the final R and t which were applied to the source cloud
    R_final = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    for R in R_list:
        R_final = R.dot(R_final)
    i = 0
    t_final = t_list[0]
    for i in range(1, len(t_list)):
        t_final = R_list[i]*t_final + t_list[i]
    print('Final R: ', R_final)
    print('Final t: ', t_final)

    # Display the transformed starting cloud with the target cloud
    utils.view_pc([transformed_source, target_pc], None, ['b', 'r'], ['o', '^'])
    plt.title('Matched Point Clouds')
    plt.axis([-0.15, 0.15, -0.15, 0.15])

    # Plot error vs iterations
    plt.figure()
    plt.plot(range(0, len(errorlist)), errorlist, 'bo')
    plt.title('ICP Error per iteration')

    input('Hit any key to exit...')
