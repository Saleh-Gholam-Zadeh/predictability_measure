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


def convert_to_numpy(data):
    if isinstance(data, np.ndarray):
        # If it's already a NumPy array, no conversion needed
        return data
    elif isinstance(data, torch.Tensor):
        # If it's a PyTorch tensor, convert it to a NumPy array
        return data.numpy()
    else:
        return data
        #raise ValueError("Input data must be either a NumPy array or a PyTorch tensor")



# Define a function to compute PearsonR correlation for a given pair of indices (i, j)
def compute_batched_pearsonr(batch_pair):
    '''
    pairwise calcualtions for 1 batch of pairs
    '''
    #print("inside the func ---> len(batch_pair):",len(batch_pair))
    pairs, X_mat, Y = batch_pair
    res = []
    for i,j in pairs:
        #print("i,j:",(i,j))
        r, pv = scipy.stats.pearsonr(X_mat[j, :], Y[i, :])
        res.append((i, j,pv,r))
    return res


def parallel_perasonr(arr ,tresh = 0.01, number_output_functions=1,bonfer = True,batch_size = -1):

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
    print("treshold:",tresh)


    pairs = [(i, j ) for i in range(number_output_functions) for j in range(m - number_output_functions)]
    # Divide pairs into batches
    if batch_size==-1:
        batch_size = len(pairs)//8


    all_batch_pairs = [pairs[i:i+batch_size] for i in range(0,len(pairs),batch_size)]


    input_to_parallel_pearson = [ (batch_pairs,X_mat,Y) for batch_pairs in all_batch_pairs]


    # Map the compute_pearsonr function to the pairs of indices
    # Use multiprocessing to parallelize computation
    with Pool() as pool:
        results = pool.map(compute_batched_pearsonr, input_to_parallel_pearson)


    #results = map(compute_mapped_pearsonr, input_to_parallel_pearson)
    for partial_res in results:
        for  i, j , pv ,r in partial_res:
            res.append((i,j,pv,r))
            if pv<tresh:
                sum_r+=np.abs(r)

    return sum_r

if __name__ == "__main__":

    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)

    print("Testing Multi-Process")
    ctx_len = 100
    tar_len = 5
    n_features = 1
    B = 20000

    number_output_functions = tar_len * n_features

    # Number of jobs is set to the number of CPU cores
    #num_jobs = num_cpus


    noise    = white_noise(B,(ctx_len+tar_len)*n_features).reshape(B,(ctx_len+tar_len)*n_features) ## a timeseries of shape [B,70,1]
    clean_signal = sin_gen(B,(ctx_len+tar_len)*n_features).reshape(B,(ctx_len+tar_len)*n_features) # a timeseries of shape [B,70,1]
    operational_data = 0.2 * noise +  clean_signal # a timeseries of shape [B,70,1]
    #print(operational_data.shape)


    # Create a list of parameter tuples for each job
    print("starting parallel Pearsonr")
    t1 = time.time()
    sum_r = parallel_perasonr(operational_data,number_output_functions= number_output_functions)
    print("correl:",sum_r)

    t2 = time.time()
    print("Took:",t2-t1)





