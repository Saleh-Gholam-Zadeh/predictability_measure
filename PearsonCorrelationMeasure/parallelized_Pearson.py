import os.path
import sys

import torch
import scipy.stats
import time

import multiprocessing
from multiprocessing import Pool

import numpy as np
import sys
print(sys.path)
sys.path.append(os.getcwd())
print(sys.path)
import random
from utils.synthetic_data_gen import sin_gen , white_noise
import math
import pandas as pd
from itertools import product
import concurrent.futures
import psutil



torch.manual_seed(2)
random.seed(2)
np.random.seed(2)



# Define a function to compute PearsonR correlation for a given pair of indices (i, j)
def compute_mapped_pearsonr(batch_pair):
    #print("inside the func ---> len(batch_pair):",len(batch_pair))
    pairs, X_mat, Y = batch_pair
    res = []
    for i,j in pairs:
        #print("i,j:",(i,j))
        r, pv = scipy.stats.pearsonr(X_mat[j, :], Y[i, :])
        res.append((i, j,pv,r))
    return res


def mapped_linear_correl(arr ,tresh = 0.01, number_output_functions=1,bonfer = True,batch_size = -1):

    '''
    implementation of PearsonR correlation
    m = data.shape[0]  # number of features (or more precisely features + outputs )
    n = data.shape[1]  # number of data points (windows)
    return p-value and r va
    '''
    arr=convert_to_numpy(arr)
    assert isinstance(tresh, float), "Variable is not of type float"
    assert len(arr.shape)==2
    m,n = arr.shape
    if bonfer :
        tresh = tresh/(number_output_functions*(m-number_output_functions))

    X_mat = arr[:-number_output_functions,:] # features
    Y = arr[-number_output_functions:,:] # labels
    res = []
    sum_r = 0
    print(tresh)


    pairs = [(i, j ) for i in range(number_output_functions) for j in range(m - number_output_functions)]
    # Divide pairs into batches
    if batch_size==-1:
        batch_size = len(pairs)//8


    all_batch_pairs = [pairs[i:i+batch_size] for i in range(0,len(pairs),batch_size)]


    input_to_parallel_pearson = [ (batch_pairs,X_mat,Y) for batch_pairs in all_batch_pairs]


    # Map the compute_pearsonr function to the pairs of indices
    # Use multiprocessing to parallelize computation
    with Pool() as pool:
        results = pool.map(compute_mapped_pearsonr, input_to_parallel_pearson)


    #results = map(compute_mapped_pearsonr, input_to_parallel_pearson)
    for partial_res in results:
        for  i, j , pv ,r in partial_res:
            res.append((i,j,pv,r))
            if pv<tresh:
                sum_r+=np.abs(r)

    return sum_r



