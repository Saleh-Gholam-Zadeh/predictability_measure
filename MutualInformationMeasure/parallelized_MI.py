import os.path
import sys

import torch
import scipy.stats
import time

import multiprocessing
from multiprocessing import Pool

import numpy as np
import sys
#print(sys.path)
sys.path.append(os.getcwd())
#print(sys.path)
import random
from utils.synthetic_data_gen import sin_gen , white_noise
import math
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



##################################### MI STUFF #######################################

def permute_1d_array(arr,seed=None):
    if seed is not None:
        np.random.seed(seed)
    #print("inside permute_1d_array seed:",seed)
    assert len(arr.shape) < 2
    arr = np.array(arr).flatten()  # Ensure array is flattened to 1D
    permuted_arr = np.random.permutation(arr)
    return permuted_arr



def mapped_make_summand_from_frequencies(params):
    x , y ,basis_log_mutual_information = params
    if x == 0:
        return 0
    else:
        return x * math.log2(x / y) / math.log2(basis_log_mutual_information)


def compute_pairwise_MI_batched(batch):
    '''
    compute MI for a batch of i-j pairs
    '''
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
        if((len(freq_data[i]) > 1) and (len(freq_data[left_features[j]]) > 1)):
            mutual_info = np.sum([mapped_make_summand_from_frequencies((x, y, basis_log_mutual_information)) for x, y in zip(freq_data_product.flatten(), expfreq.flatten())])
        else:
            mutual_info = 0
        results.append((mutual_info, i, j))
    return results



def calculate_MI_job(data, number_output_functions=1, min_n_datapoints_a_bin = None, perm_test_flag=True,cnt=None , num_op_cpus=None):
    '''
    Recives one job (fixed cnt and all i-j pairs) and outsource it to compute_pairwise_MI_batched to run on many cpus
    Important Notes:

    1)we dont Need N in this function. cnt does the job. we removed for loop over N. N is determined outside.
    cnt is passed and fixed here
    2) we cant return p-values. since we have only 1 run here so we get one MI for the permuted array

    '''
    ts = time.time()
    if num_op_cpus ==None or num_op_cpus > multiprocessing.cpu_count() -1:
        num_op_cpus = int(multiprocessing.cpu_count()-1)

    # print("________________________________________________________________________________________________")
    # print("started cnt:{} and num_op_cpus:{}".format(cnt,num_op_cpus))

    # Calulate the Shannon mutual information

    # if perm_test_flag==False:
    #     assert N==1 , "when you dont do permutation test N should be 1"
    if cnt==0:
        perm_test_flag=False


    data = convert_to_numpy(data)
    # data = circular_shift_features(data, number_output_functions) ---> should be done before passing to this function

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


    total_MI_for_this_permutation = []
    actual_total_MI = None
    actual_list_of_df = None
    list_of_data_frames = []


    # Determine the start and end CPUs for CPU affinity
    # start_cpu = cnt  * num_op_cpus % multiprocessing.cpu_count() +1
    # if start_cpu>=multiprocessing.cpu_count():
    #     start_cpu = 1



    batch_size = num_op_cpus* 2048 #  4*1024=4096
    if (batch_size==0):
        batch_size=1

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

        # print("done with cnt:{} inside MI_ijn within {} seconds".format(cnt,time.time()-ts))
        # print("________________________________________________________________________________________________")

        if perm_test_flag == False:
            return list_of_data_frames, actual_total_MI, None, None, None ,cnt
        else:
            #avg_MI_permute = sum(total_MI_for_this_permutation) / len(total_MI_for_this_permutation)

            return  None, None, None, None, total_MI_for_this_permutation ,cnt


def calculate_MI_job_wrapper(params):
    '''
    extract params into several variables and pass it to the next function
    '''
    data, number_output_functions, min_n_datapoints_a_bin, perm_test_flag, cnt, n_cpus = params
    #print("cnt_ijn_started:",cnt , " with {} CPUS".format(n_cpus))
    #return get_parallel_ijn_mutual_information(data, number_output_functions, min_n_datapoints_a_bin,perm_test_flag, cnt, num_op_cpus=n_cpus)
    return calculate_MI_job(data, number_output_functions, min_n_datapoints_a_bin,perm_test_flag, cnt, num_op_cpus=n_cpus)

# Define a function to execute each job
def execute_job(params):
    return calculate_MI_job_wrapper(params)

def get_parallel_MI(operational_data,number_output_functions,perm_test_flag,N,num_cpus=None):

    if num_cpus ==None or num_cpus > multiprocessing.cpu_count() - 1:
        num_cpus = max(int(multiprocessing.cpu_count()//2) ,1)
    # print("available cpus to submit a job in get_parallel_MI():",num_cpus)

    # Calculate the threshold load for each CPU
    threshold_load = calculate_threshold_load(multiprocessing.cpu_count() *0.95)

    job_parameters = [(operational_data, number_output_functions, None, perm_test_flag, cnt ,num_cpus) for cnt in range(N+1)]


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
                # print("we can NOT submit a new Job since cpu_usage{} is More than treshold{}".format(cpu_usage,threshold_load))
                cpu_usage = get_cpu_usage(threshold_load=threshold_load)
                # if(is_safe_to_start_job(cpu_usage, threshold_load)):
                #     print("we can submit a new Job since cpu_usage{} is less than treshold{}".format(cpu_usage,threshold_load))

            # Submit the job
            # print("Job submitted .... with cnt:{}".format(params[-2]))
            future = executor.submit(execute_job, params)
            futures.append(future)


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
            #print("cpu_usage:", tt, " ---> avg:", ss,">",threshold_load , "NOT possible to start a new Job")
            #time.sleep(0.2)
            pass
        else:
            #print("cpu_usage:", tt, " ---> avg:", ss, "<", threshold_load, " Let's Lunch a new job!")
            pass
    else:
        #print("cpu_usage:", tt , " ---> avg:",ss)
        pass
    #print("cpu_usage_avg:", ss)
    return ss


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
    ctx_len = 50
    tar_len = 50
    n_features = 1
    B = 20000
    N = 5
    num_ijn_cpus =None


    # Parameters for get_parallel_mutual_information function
    number_output_functions = tar_len * n_features
    perm_test_flag = True


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
    print("starting parallel IJN")
    t1 = time.time()
    _, actual_ijn_MI, pval_ijn, _, perm_list_ijn = get_parallel_MI(operational_data, number_output_functions, perm_test_flag, N ,num_cpus=num_ijn_cpus)
    t2 = time.time()
    #


    # print("orig_total_MI",orig_total_MI)
    # print("orig_total_MI_for_each_permutation:",orig_total_MI_for_each_permutation)
    #
    print("Parallel_ijn func Took:", t2 - t1)

    print("ijn_MI:",actual_ijn_MI)
    print("perm_list_ijn:",perm_list_ijn)

