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


def compute_pairwise_chisq(batch):

    '''

    (id_input,id_output),dependent or not, pv , n_bins_less_than5 ,n_bins_less_than1
    '''
    alpha_and_n_and_data,pairs = batch
    alpha , n , data = alpha_and_n_and_data


    results = []

    for args in pairs:
        j, id_output , freq_data,l  = args
        #print("inside_parallel: (j,id_output)",j,id_output)
        bins_less_than5 = 0
        bins_less_than1 = 0
        if len(freq_data[j]) > 1:
            dependent = 0
            counter_number_chi_square_tests_relevant_principal_features = 1

            #print("inside_parallel: (j,id_output)", (j,id_output))
            freq_data_product = np.histogram2d(data[id_output, :], data[j, :], bins=(l[id_output], l[j]))[0]

            #print("inside_parallel.. ", "data[id_output, :]")
            #print(data[id_output, :])

            #print("inside_parallel.. ", "data[j, :]")
            #print(data[j, :])


            expfreq = np.outer(freq_data[id_output], freq_data[j]) / n
            if sum(expfreq.flatten() < 5) > 0:
                bins_less_than5 = 1
            if sum(expfreq.flatten() < 1) > 0:
                bins_less_than1 = 1
            # print("debug_parallel .... freq_data_product.shape:",freq_data_product.shape)
            # print("debug_parallel .... freq_data_product:", freq_data_product)
            pv = scipy.stats.chisquare(freq_data_product.flatten(), expfreq.flatten(),ddof=(freq_data_product.shape[0] - 1) + (freq_data_product.shape[1] - 1))[1]
            #pval_list.append(pv)
            if pv <= alpha:
                dependent = 1
                #cnt_dep += 1
        else:
            pv = 1.1
            dependent = 0

        # if dependent == 1:
        #     intermediate_list_depending_on_system_state.append(j)
        # else:
        #     intermediate_list_not_depending_on_system_state.append(j)
        results.append(  ((j, id_output), dependent, pv , bins_less_than5 ,bins_less_than1)    )

    #return (j, id_output), dependent, pv , bins_less_than5 ,bins_less_than1
    return results

def run_parallel_chisquare_test(data ,min_n_datapoints_a_bin = None, number_output_functions=1 , alpha=0.01  ,bonfer= True):

    '''
        Implementation of MI code
        data: np.array m_row ---> ctx (features) + target : so each sample or datapoints is stored in a column
    /// n_col---> samples : each sample stored in one column
        i.e
        m = data.shape[0]  # number features (plus targets)
        n = data.shape[1]  # number of data points
    the whole length of target + context = m ---> target comes first
     each column of the data: [t1, t2, ..., t_num_output_function, ctx1, ctx2, ... m ]

     outputs:
    dep_list = calculate whci pairs are correalted (maximum number of elements = ctx*tar)
     pval_list= list of all pvalues: [ctx1-tar1 ctx1-tar2 ... ctx1-tarm ctx2-tar1 ctx2-tar2 ... ctxk-tarm]

     return
    '''

    data = convert_to_numpy(data)


    dep_list = [] # defined recently

    counter_bins_less_than5_relevant_principal_features=0 # number of chi-square tests with less than 5 datapoints a bin
    bins_less_than5_relevant_principal_features_ids =[]  # defined recently

    counter_bins_less_than1_relevant_principal_features=0 # number of chi-square tests with less than 1 datapoint a bin
    bins_less_than1_relevant_principal_features_ids =[]  # defined recently

    counter_number_chi_square_tests_relevant_principal_features=0 # nu

    #data = data.to_numpy()
    m = data.shape[0]  # number features
    n = data.shape[1]  # number of data points
    if (min_n_datapoints_a_bin is None):
        min_n_datapoints_a_bin = 0.05*n

    #alpha=0.01/m
    if bonfer:
        alpha = alpha/((m-number_output_functions)*number_output_functions)
        #print("bonfer_alpha after correction:",alpha)

    l = [0 for i in range(0, m)]  # list of lists with the points of support for the binning
    freq_data = [0 for i in range(0, m)]  # list of histograms
    left_features = [i for i in range(0, m)]  # list of features that is step by step reduced to the relevant ones
    constant_features = []

    # remove constant features and binning (discretizing the continuous values of our features)
    for i in range(0, m):
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print("Feature #f",i, "has only constant values")
            left_features.remove(i)
            constant_features.append(i)
            raise ValueError('WTF') #added by saleh
        else:
            # start the binning by sorting the data points
            list_points_of_support = []
            datapoints = data[i, :].copy()
            datapoints.sort()
            last_index = 0
            # go through the data points and bin them
            for point in range(0, datapoints.size):
                if point >= (datapoints.size - 1):  # if end of the data points leave the for-loop
                    break
                # close a bin if there are at least min_n_datapoints_a_bin and the next value is bigger
                if datapoints[last_index:point + 1].size >= min_n_datapoints_a_bin and datapoints[point] < datapoints[
                    point + 1]:
                    list_points_of_support.append(datapoints[point + 1])
                    last_index = point + 1
            if len(list_points_of_support) > 0:  # test that there is at least one point of support (it can be if there are only constant value up to the first ones which are less than min_n_datapoints_a_bin
                if list_points_of_support[0] > datapoints[
                    0]:  # add the first value as a point of support if it does not exist (less than min_n_datapoints_a_bin at the beginning)
                    list_points_of_support.insert(0, datapoints[0])
            else:
                list_points_of_support.append(datapoints[0])
            list_points_of_support.append(datapoints[-1] + 0.1)  # Add last point of support such that last data point is included (half open interals in Python!)
            if datapoints[datapoints >= list_points_of_support[-2]].size < min_n_datapoints_a_bin:  # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
                if len(list_points_of_support) > 2:  # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]


    #print("Binning done! (inside_parallel)")
    #print("l",l) # checked they were the same

    #print("freq_data (inside_parallel):", freq_data) # checked they were the same
    #print("List of features with constant values:")
    #print(constant_features)

    for id_output in range(0, number_output_functions):
        if id_output in constant_features or len(freq_data[id_output]) < 2:  # Warn if the output function is constant e.g. due to an unsuitable binning
            print("Warning: System state " + str(id_output) + " is constant!")
            raise ValueError('WTF') #added by saleh

    intermediate_list_depending_on_system_state=[]
    intermediate_list_not_depending_on_system_state=[]
    pval_list = []
    indices_principal_feature_values=np.zeros((1, 2))

    # Generate list of pairwise calculations
    batch_size = len(range(number_output_functions, m)) // multiprocessing.cpu_count()
    if batch_size==0:
        batch_size=1
    pairs = [(j, id_output, freq_data, l) for j in range(number_output_functions, m) for id_output in range(number_output_functions)]
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    alpha_n_data = (alpha, n, data) # we want to send them once


    #pairs = [(j, id_output, alpha,freq_data ,l,n ,data) for j in range(number_output_functions, m) for id_output in range(number_output_functions)]


    # Use multiprocessing to parallelize the calculations
    with Pool() as pool:
        #raw_results = pool.map(compute_pairwise_chisq, pairs)
        results = pool.map(compute_pairwise_chisq, [(alpha_n_data, batch) for batch in batches])

    #raw_results = list(map(compute_pairwise_chisq, pairs))

    for batch_results in results:
        for res in batch_results:
            (id_in,id_out), dep, pv, bin_less5, bin_less1  = res

            pval_list.append((pv, id_in, id_out))
            if dep:
                dep_list.append((id_in,id_out))

            if bin_less5:
                bins_less_than5_relevant_principal_features_ids.append((id_in,id_out))

            if bin_less1:
                bins_less_than1_relevant_principal_features_ids.append((id_in,id_out))

    return dep_list,pval_list,bins_less_than5_relevant_principal_features_ids ,bins_less_than1_relevant_principal_features_ids








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


    def circular_shift_features(data, t):
        # m = data.shape[0]  # number of features
        # n = data.shape[1]  # number of data points

        # Perform circular shift on the features
        shifted_data = np.roll(data, t, axis=0)

        return shifted_data

    operational_data = circular_shift_features( operational_data.swapaxes(0, 1), number_output_functions)
    #print("parallel binning...")

    # Create a list of parameter tuples for each job
    print("starting parallel chisquare")
    t1 = time.time()
    dep_list, pval, bin1,bin5 = run_parallel_chisquare_test(operational_data,number_output_functions= number_output_functions)
    print("dep_pairs (i,j):",dep_list)
    print("(pvl,i,j):",pval)
    print("bin_less_than_5:",bin5)
    print("bin_less_than_1:", bin1)
    t2 = time.time()
    print(t2-t1)


