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


def circular_shift_features(data, t):
    # m = data.shape[0]  # number of features
    # n = data.shape[1]  # number of data points

    # Perform circular shift on the features
    shifted_data = np.roll(data, t, axis=0)

    return shifted_data

def run_test(data, min_n_datapoints_a_bin = None, number_output_functions=1 , alpha=0.01 ,bonfer= True):

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
     cnt_dep = calculate how many of targets are correlated with how many of context (max = ctx*tar)
     pval_list= list of all pvalues: [ctx1-tar1 ctx1-tar2 ... ctx1-tarm ctx2-tar1 ctx2-tar2 ... ctxk-tarm]
    '''

    data = convert_to_numpy(data)
    data = circular_shift_features(data, number_output_functions)


    counter_bins_less_than5_relevant_principal_features=0 # number of chi-square tests with less than 5 datapoints a bin
    counter_bins_less_than1_relevant_principal_features=0 # number of chi-square tests with less than 1 datapoint a bin
    counter_number_chi_square_tests_relevant_principal_features=0 # nu



    #data = data.to_numpy()
    m = data.shape[0]  # number features
    n = data.shape[1]  # number of data points
    if (min_n_datapoints_a_bin is None):
        min_n_datapoints_a_bin = 0.05*n

    #alpha=0.01/m
    if bonfer:
        #print("old_alpha:",alpha)
        #((m-number_output_functions)*number_output_functions) --> number of experiments based on which we return a number or make a decision
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

    #print("Binning done! : inside serial")
    #print("l", l) # checked they were the same
    #print("freq_data (inside_serial):",freq_data) # checked they were the same
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
    cnt_dep = 0

    for j in range(number_output_functions,m):
        if len(freq_data[j]) > 1:
            dependent = 0 # Flag for the input feature j if there is a relation to one output-function
            for id_output in range(0,number_output_functions):
                counter_number_chi_square_tests_relevant_principal_features +=1
                #print("inside_serial: (j,id_output):", (j,id_output))
                freq_data_product = np.histogram2d(data[id_output, :], data[j, :],bins=(l[id_output], l[j]))[0]

                expfreq = np.outer(freq_data[id_output], freq_data[j]) / n
                if sum(expfreq.flatten() < 5) > 0:
                    counter_bins_less_than5_relevant_principal_features += 1
                if sum(expfreq.flatten() < 1) > 0:
                    counter_bins_less_than1_relevant_principal_features += 1

                # print("debug_serial .... freq_data_product.shape:", freq_data_product.shape)
                # print("debug_serial .... freq_data_product:", freq_data_product)
                pv = scipy.stats.chisquare(freq_data_product.flatten(), expfreq.flatten(),ddof=(freq_data_product.shape[0]-1)+(freq_data_product.shape[1]-1))[1]
                pval_list.append((pv,j,id_output))
                # According to the documentation of scipy.stats.chisquare, the degrees of freedom is k-1 - ddof where ddof=0 by default and k=freq_data_product.shape[0]*freq_data_product.shape[0].
                # According to literatur, the chi square test statistic for a test of independence (r x m contingency table) is approximately chi square distributed (under some assumptions) with degrees of freedom equal
                # freq_data_product.shape[0]-1)*(freq_data_product.shape[1]-1) = freq_data_product.shape[0]*freq_data_product.shape[1] - freq_data_product.shape[0] - freq_data_product.shape[1] + 1.
                # Consequently, ddof is set equal freq_data_product.shape[0]-1+freq_data_product.shape[1]-1 to adjust the degrees of freedom accordingly.

                # if p-value pv is less than alpha the hypothesis that j is independent of the output function is rejected
                if pv <= alpha:
                    dependent=1 # if the current feature is related to any of the outputs then it would become 1
                    cnt_dep += 1 # it counts the current feature is related to how many of the outputs. it is integer between 0 to num_output
                    #break
            if dependent==1:
                intermediate_list_depending_on_system_state.append(j)
            else:
                intermediate_list_not_depending_on_system_state.append(j)
        else:
            intermediate_list_not_depending_on_system_state.append(j)
            pv=1.1
        #indices_principal_feature_values= np.concatenate((indices_principal_feature_values, np.array([j, pv]).reshape((1, 2))), axis=0)
        indices_principal_feature_values = np.concatenate((indices_principal_feature_values, np.array([j, pv]).reshape((1, 2))), axis=0)


    return intermediate_list_depending_on_system_state,pval_list,cnt_dep



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

def run_parallel_test(data ,min_n_datapoints_a_bin = None, number_output_functions=1 , alpha=0.01  ,bonfer= True):

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
    data = circular_shift_features(data, number_output_functions)


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





##################################### MI STUFF #######################################

def permute_1d_array(arr,seed=None):
    if seed is not None:
        np.random.seed(seed)
    #print("inside permute_1d_array seed:",seed)
    assert len(arr.shape) < 2
    arr = np.array(arr).flatten()  # Ensure array is flattened to 1D
    permuted_arr = np.random.permutation(arr)
    return permuted_arr


def get_mutual_information(data, number_output_functions=1, min_n_datapoints_a_bin = None, perm_test_flag=True, N=10):
    # Calulate the Shannon mutual information
    def make_summand_from_frequencies(x, y):
        if x == 0:
            return 0
        else:
            return x * math.log2(x / y) / math.log2(basis_log_mutual_information)
    data = convert_to_numpy(data)
    data = circular_shift_features(data, number_output_functions)

    # Insert the the indices of the rows where the components of the output functions are stored
    # for i in range(0, number_output_functions):
    #     list_variables.insert(i, i)
    sum_list = []
    m = data.shape[0]
    n = data.shape[1]
    l = [0 for i in range(0, m)]
    freq_data = [0 for i in range(0, m)]
    left_features = [i for i in range(0, m)]
    list_variables = [i for i in range(0, m)]
    data = data[list_variables, :]

    if (min_n_datapoints_a_bin is None):
        min_n_datapoints_a_bin = 0.05*n


    constant_features = []

    for i in range(0, m):
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print("Feature #f",list_variables[i], " has only constant values")
            left_features.remove(i)
            constant_features.append(list_variables[i])
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
            list_points_of_support.append(datapoints[
                                              -1] + 0.1)  # Add last point of support such that last data point is included (half open interals in Python!)
            if datapoints[datapoints >= list_points_of_support[
                -2]].size < min_n_datapoints_a_bin:  # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
                if len(list_points_of_support) > 2:  # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]

    # Check for constant features
    if constant_features != []:
        print("List of features with constant values:")
        print(constant_features)
    for id_output in range(0, number_output_functions):
        if id_output in constant_features or len(
                freq_data[id_output]) < 2:  # Warn if the output function is constant e.g. due to an unsuitable binning
            print("Warning: Output function " + str(id_output) + " is constant!")


    # Calculate the mutual information for each feature with the corresponding component of the output function
    list_of_data_frames = []
    mutual_info = np.ones((1,len(left_features) - number_output_functions + 1))  # number of featuers plus one component of the output-function


# N times for loop here to shuffle N times and store the sum_MI
    if perm_test_flag:
        N=N+1

    total_MI_for_each_permutation = []
    for cnt in range(N):
        data_cp = data.copy()
        print("cnt: ",cnt,'/',N-1)
        sum_list =[]
        for i in range(0, number_output_functions):
            basis_log_mutual_information = len(freq_data[i])
            # shuffle (data [i,:] ) ---> data[i,:]

            if perm_test_flag and cnt>0:
                # print("before permutation....")
                # print("data[", i, ":]", data[i, :])
                perm_labels = permute_1d_array(data [i,:].copy(),seed=cnt)
                data_cp[i,:] = perm_labels
                # print("conter_serial= ",cnt)
                # print("after permutation....")
                # print("data[", i, ":]", data_cp[i, :])

            list_of_features = list(range(number_output_functions, len(left_features)))
            list_of_features.insert(0, i)
            id_features = np.array(list_variables)[list_of_features]

            for j in list_of_features:
                freq_data_product = ((
                np.histogram2d(data_cp[i, :], data_cp[left_features[j], :], bins=(l[i], l[left_features[j]]))[0])) / n
                expfreq = (np.outer(freq_data[i], freq_data[left_features[j]])) / (n * n)
                if j < number_output_functions:
                    mutual_info[0, 0] = np.sum(np.array(list(
                        map(make_summand_from_frequencies, freq_data_product.flatten().tolist(),
                            expfreq.flatten().tolist()))))
                else:
                    mutual_info[0, j - number_output_functions + 1] = np.sum(np.array(list(
                        map(make_summand_from_frequencies, freq_data_product.flatten().tolist(),
                            expfreq.flatten().tolist()))))

            sum_mi = np.sum(mutual_info[0,1:]) # the sum over all features for each output
            sum_list.append(sum_mi)
            #print("id_features:",id_features)
            #print("mutual information:",mutual_info.tolist()[0])
            pd_mutual_information = pd.DataFrame({"index feature": id_features, "mutual information": mutual_info.tolist()[0]})
            pd_mutual_information['index feature'] = pd_mutual_information['index feature'].astype(int)

            list_of_data_frames.append(pd_mutual_information)

        if cnt==0:
            actual_total_MI = sum(sum_list) # sum over all outputs (previously done on the features)
            actual_list_of_df = list_of_data_frames  # we can return this instead of None
        else: # permutation test ---> values come to the list to make a distribution:
            total_MI_for_each_permutation.append(sum(sum_list))


    if perm_test_flag==False:
        return list_of_data_frames, actual_total_MI,None   ,     None       ,         None
    else:
        avg_MI_permute = sum(total_MI_for_each_permutation) / len(total_MI_for_each_permutation)
        pvalue = np.sum(np.array(total_MI_for_each_permutation) > actual_total_MI)/len(total_MI_for_each_permutation)
        return None,                actual_total_MI, pvalue, avg_MI_permute  ,total_MI_for_each_permutation




# def compute_pairwise_MI(pair): # accept only 1 pair
#     #print("pair:",pair)
#     i, j, data_x, freq_data, l, number_output_functions, n, left_features , should_permute,counter = pair
#     permed_data_x = data_x.copy()
#
#     if should_permute:
#         perm_labels = permute_1d_array(data_x[i, :].copy(),seed=counter)
#         permed_data_x[i, :] = perm_labels
#
#
#
#     freq_data_product = np.histogram2d(permed_data_x[i, :], permed_data_x[left_features[j], :], bins=(l[i], l[left_features[j]]))[0] / n
#     expfreq = np.outer(freq_data[i], freq_data[left_features[j]]) / (n * n)
#     basis_log_mutual_information = len(freq_data[i])
#     if j < number_output_functions:
#         mutual_info = np.sum([mapped_make_summand_from_frequencies((x, y, basis_log_mutual_information)) for x, y in zip(freq_data_product.flatten(), expfreq.flatten())])
#     else:
#         mutual_info = np.sum([mapped_make_summand_from_frequencies((x, y, basis_log_mutual_information)) for x, y in zip(freq_data_product.flatten(), expfreq.flatten())])
#     return mutual_info, i, j

def mapped_make_summand_from_frequencies(params):
    x , y ,basis_log_mutual_information = params
    if x == 0:
        return 0
    else:
        return x * math.log2(x / y) / math.log2(basis_log_mutual_information)


def compute_pairwise_MI_batched(batch):
    data_x, freq_data, l, number_output_functions, n, left_features, should_permute, counter,pairs_batch  = batch
    permed_data_x = data_x.copy()



    results = []
    for pair in pairs_batch:
        i, j = pair
        if should_permute:
            perm_labels = permute_1d_array(data_x[i, :].copy(), seed=counter)
            permed_data_x[i, :] = perm_labels

        freq_data_product = np.histogram2d(permed_data_x[i, :], permed_data_x[left_features[j], :], bins=(l[i], l[left_features[j]]))[0] / n
        expfreq = np.outer(freq_data[i], freq_data[left_features[j]]) / (n * n)
        basis_log_mutual_information = len(freq_data[i])
        if j < number_output_functions:
            mutual_info = np.sum([mapped_make_summand_from_frequencies((x, y, basis_log_mutual_information)) for x, y in zip(freq_data_product.flatten(), expfreq.flatten())])
        else:
            mutual_info = np.sum([mapped_make_summand_from_frequencies((x, y, basis_log_mutual_information)) for x, y in zip(freq_data_product.flatten(), expfreq.flatten())])
        results.append((mutual_info, i, j))
    return results
#
# def get_parallel_mutual_information(data, number_output_functions=1, min_n_datapoints_a_bin = None, perm_test_flag=True, N=10):
#
#     # Calulate the Shannon mutual information
#
#     if perm_test_flag==False:
#         assert N==1 , "when you dont do permutation test N should be 1"
#     data = convert_to_numpy(data)
#     data = circular_shift_features(data, number_output_functions)
#
#     # Insert the the indices of the rows where the components of the output functions are stored
#     # for i in range(0, number_output_functions):
#     #     list_variables.insert(i, i)
#     sum_list = []
#     m = data.shape[0]
#     n = data.shape[1]
#     l = [0 for i in range(0, m)]
#     freq_data = [0 for i in range(0, m)]
#     left_features = [i for i in range(0, m)]
#     list_variables = [i for i in range(0, m)]
#     data = data[list_variables, :]
#
#     if (min_n_datapoints_a_bin is None):
#         min_n_datapoints_a_bin = 0.05*n
#
#
#     constant_features = []
#
#     for i in range(0, m):
#         mindata = min(data[i, :])
#         maxdata = max(data[i, :])
#         if maxdata <= mindata:
#             print("Feature #"f"{list_variables[i]}" " has only constant values")
#             left_features.remove(i)
#             constant_features.append(list_variables[i])
#         else:
#             # start the binning by sorting the data points
#             list_points_of_support = []
#             datapoints = data[i, :].copy()
#             datapoints.sort()
#             last_index = 0
#             # go through the data points and bin them
#             for point in range(0, datapoints.size):
#                 if point >= (datapoints.size - 1):  # if end of the data points leave the for-loop
#                     break
#                 # close a bin if there are at least min_n_datapoints_a_bin and the next value is bigger
#                 if datapoints[last_index:point + 1].size >= min_n_datapoints_a_bin and datapoints[point] < datapoints[
#                     point + 1]:
#                     list_points_of_support.append(datapoints[point + 1])
#                     last_index = point + 1
#             if len(list_points_of_support) > 0:  # test that there is at least one point of support (it can be if there are only constant value up to the first ones which are less than min_n_datapoints_a_bin
#                 if list_points_of_support[0] > datapoints[
#                     0]:  # add the first value as a point of support if it does not exist (less than min_n_datapoints_a_bin at the beginning)
#                     list_points_of_support.insert(0, datapoints[0])
#             else:
#                 list_points_of_support.append(datapoints[0])
#             list_points_of_support.append(datapoints[
#                                               -1] + 0.1)  # Add last point of support such that last data point is included (half open interals in Python!)
#             if datapoints[datapoints >= list_points_of_support[
#                 -2]].size < min_n_datapoints_a_bin:  # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
#                 if len(list_points_of_support) > 2:  # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
#                     list_points_of_support.pop(-2)
#             l[i] = list_points_of_support
#             freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]
#
#     # Check for constant features
#     if constant_features != []:
#         print("List of features with constant values:")
#         print(constant_features)
#     for id_output in range(0, number_output_functions):
#         if id_output in constant_features or len(
#                 freq_data[id_output]) < 2:  # Warn if the output function is constant e.g. due to an unsuitable binning
#             print("Warning: Output function " + str(id_output) + " is constant!")
#
#
#     # Calculate the mutual information for each feature with the corresponding component of the output function
#     list_of_data_frames = []
#     mutual_info = np.ones((1,len(left_features) - number_output_functions + 1))  # number of featuers plus one component of the output-function
#
#
# # N times for loop here to shuffle N times and store the sum_MI
#     if perm_test_flag:
#         N=N+1
#
#     total_MI_for_each_permutation = []
#     actual_total_MI = None
#     actual_list_of_df = None
#     list_of_data_frames = []
#
#     batch_size = len(list(product(range(number_output_functions), range(number_output_functions, len(left_features))))) // multiprocessing.cpu_count()
#     if (batch_size==0):
#         batch_size=1
#     print("batch_size:",batch_size)
#
#     with Pool() as pool:
#
#         for cnt in range(N):
#             pairs = list(product(range(number_output_functions), range(number_output_functions, len(left_features))))
#             #print("pairs:", pairs)
#             print("cnt: ",cnt,'/',N-1)
#             sum_list =[]
#             should_I_permute = perm_test_flag and (cnt!=0)
#
#
#             # Divide pairs into batches
#             pairs_batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
#             data_batch = [(data, freq_data, l, number_output_functions, n, left_features, should_I_permute, cnt,batch_pair) for batch_pair in pairs_batches]
#             results = pool.map(compute_pairwise_MI_batched, data_batch)
#
#             # results = []
#             #results = list(map(compute_pairwise_MI, [(i, j, data, freq_data, l, number_output_functions, n, left_features, should_I_permute ,cnt) for i, j in pairs]))
#             # for batch_pair in pairs_batches:
#             #     data_batch = (data, freq_data, l, number_output_functions, n, left_features, should_I_permute, cnt, batch_pair)
#             #     results.extend(map(compute_pairwise_MI_batched, [data_batch]))
#
#             for batch_result in results:
#                 for mutual_info, i, j in batch_result:
#                     sum_list.append(mutual_info)
#
#
#             #
#             # for mutual_info, i, j in results:
#             #     sum_list.append(mutual_info)
#
#                 # print("list_variables:", list_variables)
#                 # print("left_features:", left_features)
#                 #print("i,j:", (i,j))
#
#                 #print("list_variables[left_features[j]]:",list_variables[left_features[j]])
#                 #print("mutual_info",mutual_info)
#                 # pd_mutual_information = pd.DataFrame({"index feature": list_variables[left_features[j]], "mutual information": mutual_info})
#                 # pd_mutual_information['index feature'] = pd_mutual_information['index feature'].astype(int)
#                 # list_of_data_frames.append(pd_mutual_information)
#
#
#             if cnt == 0:
#                 actual_total_MI = sum(sum_list)
#                 actual_list_of_df = list_of_data_frames
#             elif cnt>0 and perm_test_flag : # permutation test: values comes to a list to make a distribution
#                 total_MI_for_each_permutation.append(sum(sum_list))
#                 #print(sum_list)
#             else:
#                 raise Exception("Sorry, a wrong combinations")
#
#         if perm_test_flag == False:
#             return list_of_data_frames, actual_total_MI, None, None, None
#         else:
#             avg_MI_permute = sum(total_MI_for_each_permutation) / len(total_MI_for_each_permutation)
#             pvalue = np.sum(np.array(total_MI_for_each_permutation) > actual_total_MI) / len(total_MI_for_each_permutation)
#         return  None, actual_total_MI, pvalue, avg_MI_permute, total_MI_for_each_permutation
#

def linear_correl(arr ,tresh = 0.01, number_output_functions=1,bonfer = True):

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
    for i in range(number_output_functions): # for loop on y
        for j in range(m-number_output_functions): # for loop on x --> all corelations between xi and yj will be considered
            r , pv = (scipy.stats.pearsonr(X_mat[j,:],Y[i,:]))
            res.append((pv,r))
            #print("(i,j,pv,r)",(i,j,pv,r))
            #print(pv,tresh)
            if pv <tresh:
                #print(i," pv:",pv,"  r:", r)
                sum_r = sum_r + np.abs(r)
    return sum_r

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




def get_parallel_ij_mutual_information(data, number_output_functions=1, min_n_datapoints_a_bin = None, perm_test_flag=True, N=10,num_cpu=None):

    # Calulate the Shannon mutual information
    if num_cpu==None or num_cpu > multiprocessing.cpu_count() -1:
        num_cpu = multiprocessing.cpu_count() -1
    print("available cpu in ij task:",num_cpu)
    if perm_test_flag==False:
        assert N==1 , "when you dont do permutation test N should be 1"
    data = convert_to_numpy(data)
    data = circular_shift_features(data, number_output_functions)

    # Insert the the indices of the rows where the components of the output functions are stored
    # for i in range(0, number_output_functions):
    #     list_variables.insert(i, i)
    sum_list = []
    m = data.shape[0]
    n = data.shape[1]
    l = [0 for i in range(0, m)]
    freq_data = [0 for i in range(0, m)]
    left_features = [i for i in range(0, m)]
    list_variables = [i for i in range(0, m)]
    data = data[list_variables, :]

    if (min_n_datapoints_a_bin is None):
        min_n_datapoints_a_bin = 0.05*n


    constant_features = []

    for i in range(0, m):
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print("Feature #f",list_variables[i]," has only constant values")
            left_features.remove(i)
            constant_features.append(list_variables[i])
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
            list_points_of_support.append(datapoints[
                                              -1] + 0.1)  # Add last point of support such that last data point is included (half open interals in Python!)
            if datapoints[datapoints >= list_points_of_support[
                -2]].size < min_n_datapoints_a_bin:  # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
                if len(list_points_of_support) > 2:  # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]

    # Check for constant features
    if constant_features != []:
        print("List of features with constant values:")
        print(constant_features)
    for id_output in range(0, number_output_functions):
        if id_output in constant_features or len(
                freq_data[id_output]) < 2:  # Warn if the output function is constant e.g. due to an unsuitable binning
            print("Warning: Output function " + str(id_output) + " is constant!")


    # Calculate the mutual information for each feature with the corresponding component of the output function
    list_of_data_frames = []
    mutual_info = np.ones((1,len(left_features) - number_output_functions + 1))  # number of featuers plus one component of the output-function


# N times for loop here to shuffle N times and store the sum_MI
    if perm_test_flag:
        N=N+1

    total_MI_for_each_permutation = []
    actual_total_MI = None
    actual_list_of_df = None
    list_of_data_frames = []

    #batch_size = len(list(product(range(number_output_functions), range(number_output_functions, len(left_features))))) // num_cpu
    batch_size = num_cpu*2048
    if (batch_size==0):
        batch_size=1
    print("batch_size:",batch_size)

    with Pool() as pool:

        for cnt in range(N):
            pairs = list(product(range(number_output_functions), range(number_output_functions, len(left_features))))
            #print("pairs:", pairs)
            print("cnt inside MI_ij : ",cnt,'/',N-1)
            sum_list =[]
            should_I_permute = perm_test_flag and (cnt!=0)


            # Divide pairs into batches
            pairs_batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
            data_batch = [(data, freq_data, l, number_output_functions, n, left_features, should_I_permute, cnt,batch_pair) for batch_pair in pairs_batches]
            results = pool.map(compute_pairwise_MI_batched, data_batch)

            # results = []
            #results = list(map(compute_pairwise_MI, [(i, j, data, freq_data, l, number_output_functions, n, left_features, should_I_permute ,cnt) for i, j in pairs]))
            # for batch_pair in pairs_batches:
            #     data_batch = (data, freq_data, l, number_output_functions, n, left_features, should_I_permute, cnt, batch_pair)
            #     results.extend(map(compute_pairwise_MI_batched, [data_batch]))

            for batch_result in results:
                for mutual_info, i, j in batch_result:
                    sum_list.append(mutual_info)


            #
            # for mutual_info, i, j in results:
            #     sum_list.append(mutual_info)

                # print("list_variables:", list_variables)
                # print("left_features:", left_features)
                #print("i,j:", (i,j))

                #print("list_variables[left_features[j]]:",list_variables[left_features[j]])
                #print("mutual_info",mutual_info)
                # pd_mutual_information = pd.DataFrame({"index feature": list_variables[left_features[j]], "mutual information": mutual_info})
                # pd_mutual_information['index feature'] = pd_mutual_information['index feature'].astype(int)
                # list_of_data_frames.append(pd_mutual_information)


            if cnt == 0:
                actual_total_MI = sum(sum_list)
                actual_list_of_df = list_of_data_frames
            elif cnt>0 and perm_test_flag : # permutation test: values comes to a list to make a distribution
                total_MI_for_each_permutation.append(sum(sum_list))
                #print(sum_list)
            else:
                raise Exception("Sorry, a wrong combinations")

        if perm_test_flag == False:
            return list_of_data_frames, actual_total_MI, None, None, None
        else:
            avg_MI_permute = sum(total_MI_for_each_permutation) / len(total_MI_for_each_permutation)
            pvalue = np.sum(np.array(total_MI_for_each_permutation) > actual_total_MI) / len(total_MI_for_each_permutation)
            return  None, actual_total_MI, pvalue, avg_MI_permute, total_MI_for_each_permutation





def get_parallel_ijn_mutual_information(data, number_output_functions=1, min_n_datapoints_a_bin = None, perm_test_flag=True,cnt=None , num_op_cpus=None):
    '''
    Important Notes:

    1)we dont Need N in this function. cnt does the job. we removed for loop over N. N is determined outside
    cnt is passed and fixed here
    2) we cant return p-values. since we have only 1 run here so we get one MI for the permuted array

    '''
    ts = time.time()
    if num_op_cpus ==None or num_op_cpus > multiprocessing.cpu_count() -1:
        num_op_cpus = int(multiprocessing.cpu_count()*0.95)

    #print("number of operational CPUS insdie get_parallel_ijn_mutual_information:",num_op_cpus)
    print("________________________________________________________________________________________________")
    print("started cnt:{} and num_op_cpus:{}".format(cnt,num_op_cpus))

    # Calulate the Shannon mutual information

    # if perm_test_flag==False:
    #     assert N==1 , "when you dont do permutation test N should be 1"
    if cnt==0:
        perm_test_flag=False


    data = convert_to_numpy(data)
    data = circular_shift_features(data, number_output_functions)

    # Insert the the indices of the rows where the components of the output functions are stored
    # for i in range(0, number_output_functions):
    #     list_variables.insert(i, i)
    sum_list = []
    m = data.shape[0]
    n = data.shape[1]
    l = [0 for i in range(0, m)]
    freq_data = [0 for i in range(0, m)]
    left_features = [i for i in range(0, m)]
    list_variables = [i for i in range(0, m)]
    data = data[list_variables, :]

    if (min_n_datapoints_a_bin is None):
        min_n_datapoints_a_bin = 0.05*n


    constant_features = []

    for i in range(0, m):
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print("Feature #f",list_variables[i]," has only constant values")
            left_features.remove(i)
            constant_features.append(list_variables[i])
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
            list_points_of_support.append(datapoints[
                                              -1] + 0.1)  # Add last point of support such that last data point is included (half open interals in Python!)
            if datapoints[datapoints >= list_points_of_support[
                -2]].size < min_n_datapoints_a_bin:  # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
                if len(list_points_of_support) > 2:  # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]

    # Check for constant features
    if constant_features != []:
        print("List of features with constant values:")
        print(constant_features)
    for id_output in range(0, number_output_functions):
        if id_output in constant_features or len(
                freq_data[id_output]) < 2:  # Warn if the output function is constant e.g. due to an unsuitable binning
            print("Warning: Output function " + str(id_output) + " is constant!")


    # Calculate the mutual information for each feature with the corresponding component of the output function
    list_of_data_frames = []
    mutual_info = np.ones((1,len(left_features) - number_output_functions + 1))  # number of featuers plus one component of the output-function


# N times for loop here to shuffle N times and store the sum_MI
#     if perm_test_flag:
#         N=N+1

    total_MI_for_this_permutation = []
    actual_total_MI = None
    actual_list_of_df = None
    list_of_data_frames = []


    # Determine the start and end CPUs for CPU affinity
    start_cpu = cnt  * num_op_cpus % multiprocessing.cpu_count() +1
    if start_cpu>=multiprocessing.cpu_count():
        start_cpu = 1

    end_cpu = min(start_cpu + num_op_cpus, multiprocessing.cpu_count())  # Ensure end_cpu doesn't exceed the maximum available CPU index

    # Calculate the average CPU usage for the subset of CPUs

    # Determine the threshold load for starting a new job
    threshold_load = 85  # 85% threshold for CPU usage

    linux_flag = 0 # later we determin it
    macos_flag = 0 # later we determin it

    batch_size = num_op_cpus* 2048 #  4*1024=4096
    if (batch_size==0):
        batch_size=1
    #print("batch_size:",batch_size)
    # while True:
    #     avg_cpu_usage, _ = get_avg_cpu_usage(start_cpu, end_cpu)
    #     if avg_cpu_usage < threshold_load:
    #         affinity_mask = list(range(start_cpu, end_cpu ))
    #
    #         try: # if os is linux:
    #             if not (macos_flag): # if it is macos still for the first time we dont know and we realize it here
    #                 avg_cpu_usage,usage_list = get_avg_cpu_usage(start_cpu, end_cpu)
    #                 if avg_cpu_usage < threshold_load :
    #                     affinity_mask = list(range(start_cpu, end_cpu))
    #
    #                     # when we have linux machine we can distribute the load over specified cpus
    #                     current_process = psutil.Process(os.getpid())
    #                     current_process.cpu_affinity(affinity_mask)
    #                     print("CPUs[{}:{}]".format(start_cpu, end_cpu))
    #                     print("inside MI_ijn:   os.getpid():",os.getpid() , ",cnt:{}  , batch_size:{}  ".format(cnt,batch_size) ," affinity mask_CPUS:{}".format(os.sched_getaffinity(os.getpid()) ) , "usage_list:",usage_list )
    #                     linux_flag=1
    #                 else: # if we are above treshold then we check the next few cpus
    #                     start_cpu = (end_cpu + 1) % multiprocessing.cpu_count()
    #                     end_cpu = min(start_cpu + num_op_cpus, multiprocessing.cpu_count())
    #                     print("assignee cpus were busier than 85% lets check the next subset ,cpus. batch_size: ",batch_size,list(range(start_cpu,end_cpu)) , "usage_list:",usage_list )
    #                     continue
    #         #print("affinity mask set CPUS:{}".format(os.sched_getaffinity(os.getpid()) ))
    #         except: # mac os:
    #             # the above feature doesnt work on macos --> we distribute load over all cpus
    #             start_cpu=1
    #             end_cpu = multiprocessing.cpu_count()
    #
    #             macos_flag = 1
    #             avg_cpu_usage,usage_list_mac = get_avg_cpu_usage(start_cpu, end_cpu)
    #             if avg_cpu_usage < threshold_load:
    #                 print("in macos we cant assign to a specific cpus so we distribute over all cpus. batch_size: ",batch_size ,list(range(start_cpu,end_cpu)) , "usage_list_mac:",usage_list_mac)
    #             else:
    #                 print("assignee cpus were busier than {} > 85%  , let's wait ,cpus:".format(avg_cpu_usage),"batch_size: ",batch_size ,list(range(start_cpu,end_cpu)), "usage_list_mac:",usage_list_mac )
    #                 continue
    #             #pass

    #with Pool(processes=num_op_cpus) as pool:
    with Pool() as pool:

        #for cnt in range(N): # cnt is fixed now and is passed as an argument
        pairs = list(product(range(number_output_functions), range(number_output_functions, len(left_features))))
        #print("pairs:", pairs)
        #print("cnt inside MI_ijn : ",cnt , " batch_size:",batch_size)
        sum_list =[]
        should_I_permute = perm_test_flag and (cnt!=0)



        # Divide pairs into batches
        pairs_batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
        data_batch = [(data, freq_data, l, number_output_functions, n, left_features, should_I_permute, cnt,batch_pair) for batch_pair in pairs_batches]
        results = pool.map(compute_pairwise_MI_batched, data_batch)

        # results = []
        #results = list(map(compute_pairwise_MI, [(i, j, data, freq_data, l, number_output_functions, n, left_features, should_I_permute ,cnt) for i, j in pairs]))
        # for batch_pair in pairs_batches:
        #     data_batch = (data, freq_data, l, number_output_functions, n, left_features, should_I_permute, cnt, batch_pair)
        #     results.extend(map(compute_pairwise_MI_batched, [data_batch]))

        for batch_result in results:
            for mutual_info, i, j in batch_result:
                sum_list.append(mutual_info)


        if cnt == 0:
            actual_total_MI = sum(sum_list)
            actual_list_of_df = list_of_data_frames
        elif cnt>0 and perm_test_flag : # permutation test: values comes to a list to make a distribution
            total_MI_for_this_permutation.append(sum(sum_list))
            #print(sum_list)
        else:
            raise Exception("Sorry, a wrong combinations")

        print("done with cnt:{} inside MI_ijn within {} seconds".format(cnt,time.time()-ts))
        print("________________________________________________________________________________________________")

        if perm_test_flag == False:
            return list_of_data_frames, actual_total_MI, None, None, None ,cnt
        else:
            #avg_MI_permute = sum(total_MI_for_this_permutation) / len(total_MI_for_this_permutation)


            return  None, None, None, None, total_MI_for_this_permutation ,cnt



def get_parallel_ijn_mutual_information_wrapper(params):
    data, number_output_functions, min_n_datapoints_a_bin, perm_test_flag, cnt, n_cpus = params
    #print("cnt_ijn_started:",cnt , " with {} CPUS".format(n_cpus))
    return get_parallel_ijn_mutual_information(data, number_output_functions, min_n_datapoints_a_bin,perm_test_flag, cnt, num_op_cpus=n_cpus)

    # Define a function to execute each job
def execute_job(params):
    return get_parallel_ijn_mutual_information_wrapper(params)

def submit_MI_ijn_jobs(operational_data,number_output_functions,perm_test_flag,N,num_cpus=None):

    if num_cpus ==None or num_cpus > multiprocessing.cpu_count() - 1:
        #num_cpus = int(multiprocessing.cpu_count() *0.9)
        #num_cpus = 4#int(multiprocessing.cpu_count() * 0.95)
        num_cpus = max(int(multiprocessing.cpu_count()//8) ,1)
    print("available cpus to submit_MI_ijn_jobs:",num_cpus)

    # Calculate the threshold load for each CPU
    threshold_load = calculate_threshold_load(multiprocessing.cpu_count() *0.95)

    job_parameters = [(operational_data, number_output_functions, None, perm_test_flag, cnt ,num_cpus) for cnt in range(N+1)]


    # batch_size = num_cpus
    # job_batches = [job_parameters[i:i + batch_size] for i in range(0, len(job_parameters), batch_size)]

    futures=[]
    # Create a ProcessPoolExecutor with a maximum number of worker processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit each batch of jobs to the executor
        #futures = []
        #for batch in job_batches:
        for params in job_parameters:
            # Check if it's safe to start a new job
            cpu_usage = get_cpu_usage(threshold_load=threshold_load)

            while not is_safe_to_start_job(cpu_usage, threshold_load):
                #time.sleep(0.01)
                print("we can NOT submit a new Job since cpu_usage{} is More than treshold{}".format(cpu_usage,threshold_load))
                cpu_usage = get_cpu_usage(threshold_load=threshold_load)
                if(is_safe_to_start_job(cpu_usage, threshold_load)):
                    print("we can submit a new Job since cpu_usage{} is less than treshold{}".format(cpu_usage,threshold_load))




            # Submit the job
            print("Job submitted .... with cnt:{}".format(params[-2]))
            future = executor.submit(execute_job, params)
            futures.append(future)


            # batch_futures = [executor.submit(execute_job, params) for params in batch]
            # futures.extend(batch_futures)
    # Wait for all jobs to complete
    concurrent.futures.wait(futures)




    # Extract results from completed jobs if needed
    perm_list_ijn = []
    for future in futures:
        result = future.result()
        #print("result:",result)
        _,act_MI,_,_,perm_MI,counter = result
        if counter==0:
            actual_ijn_MI = act_MI
        else:
            perm_list_ijn.append(perm_MI[0])

    avg_MI_permute = sum(perm_list_ijn) / len(perm_list_ijn)
    pvalue = np.sum(np.array(perm_list_ijn) > actual_ijn_MI) / len(perm_list_ijn)

    return  None, actual_ijn_MI, pvalue, avg_MI_permute, perm_list_ijn



# Define a function to get the CPU usage for each CPU
def get_cpu_usage(threshold_load=None):
    tt = psutil.cpu_percent(interval=0.1,percpu=True)
    ss=np.average(np.array(tt))

    if threshold_load is not None:
        if threshold_load<ss:
            print("cpu_usage:", tt, " ---> avg:", ss,">",threshold_load , "NOT possible to start a new Job")
            time.sleep(0.2)
        else:
            print("cpu_usage:", tt, " ---> avg:", ss, "<", threshold_load, " Let's Lunch a new job!")
    else:
        print("cpu_usage:", tt , " ---> avg:",ss)
    #print("cpu_usage_avg:", ss)
    return ss

# Function to calculate the average CPU usage for a subset of CPUs
def get_avg_cpu_usage(start_cpu, end_cpu):
    cpu_percentages = psutil.cpu_percent(interval=0.2, percpu=True)
    subset_percentages = cpu_percentages[start_cpu:end_cpu]
    #print("subset_percentages{}, for cpus:{} :".format(subset_percentages,list(range(start_cpu,end_cpu))))
    if len (subset_percentages) == 0:
        print("!!!!! WTF:  ,(start_cpu,end_cpu)=",(start_cpu,end_cpu))
    avg_usage = sum(subset_percentages) / len(subset_percentages)
    return avg_usage, cpu_percentages[start_cpu:end_cpu]

# Function to calculate the threshold load for each CPU
def calculate_threshold_load(num_cpus):
    return num_cpus * 90/multiprocessing.cpu_count()  # 90 means 90%.

# Function to check if it's safe to start a new job based on CPU load
def is_safe_to_start_job(cpu_usage, threshold_load):
    return cpu_usage < threshold_load



if __name__ == "__main__":

    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)

    print("Testing Multi-Process")
    ctx_len = 190
    tar_len = 720
    n_features = 1
    B = 20000
    N = 5
    num_ijn_cpus =None


    # ctx_len = 96
    # tar_len = 360
    # n_features = 4
    # B = 20000
    # N = 40
    # num_ijn_cpus =None

    # Parameters for get_parallel_mutual_information function
    number_output_functions = tar_len * n_features
    perm_test_flag = True

    # Number of jobs is set to the number of CPU cores
    #num_jobs = num_cpus


    noise    = white_noise(B,(ctx_len+tar_len)*n_features).reshape(B,(ctx_len+tar_len)*n_features) ## a timeseries of shape [B,70,1]
    clean_signal = sin_gen(B,(ctx_len+tar_len)*n_features).reshape(B,(ctx_len+tar_len)*n_features) # a timeseries of shape [B,70,1]
    operational_data = 0.2 * noise +  clean_signal # a timeseries of shape [B,70,1]
    #print(operational_data.shape)

    operational_data = operational_data.swapaxes(0, 1)
    #print("parallel binning...")

    # Create a list of parameter tuples for each job
    print("starting parallel IJN")
    t1 = time.time()
    _, actual_ijn_MI, pval_ijn, _, perm_list_ijn = submit_MI_ijn_jobs(operational_data, number_output_functions, perm_test_flag, N ,num_cpus=num_ijn_cpus)
    t2 = time.time()
    #


    # print("orig_total_MI",orig_total_MI)
    # print("orig_total_MI_for_each_permutation:",orig_total_MI_for_each_permutation)
    #
    print("Parallel_ijn func Took:", t2 - t1)
    # print("starting parallel IJ ...")

    # t5 = time.time()
    # _, actual_ij_MI, pval, _, perm_list_ij = get_parallel_ij_mutual_information(operational_data,number_output_functions=tar_len*n_features, perm_test_flag=True , N=N ,num_cpu=num_ijn_cpus)
    # t6 = time.time()

    # print("starting Serial")
    #
    t3 = time.time()
    _, orig_total_MI, orig_pval, _, orig_total_MI_for_each_permutation = get_mutual_information(operational_data, number_output_functions=tar_len * n_features, perm_test_flag=True, N=N )
    t4 = time.time()

    print("ijn_MI:",actual_ijn_MI)
    print("perm_list_ijn:",perm_list_ijn)

    # print("ij_MI:", actual_ij_MI)
    # print("perm_list_ij:", perm_list_ij)

    print("serial_MI:", orig_total_MI)
    print("perm_list_serial:", orig_total_MI_for_each_permutation)


    # t1 = time.time()
    # t2 = time.time()

    print("Parallel_ijn func Took:", t2 - t1)
    print("serial func Took:", t4 - t3)
    #print("Parallel_ij func Took:", t6 - t5)
    #print("speed gain:")
