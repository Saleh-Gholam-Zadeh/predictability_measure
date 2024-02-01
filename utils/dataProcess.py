import torch
import numpy as np
import scipy.stats
import pandas as pd
import math

def convert_to_numpy(data):
    if isinstance(data, np.ndarray):
        # If it's already a NumPy array, no conversion needed
        return data
    elif isinstance(data, torch.Tensor):
        # If it's a PyTorch tensor, convert it to a NumPy array
        return data.numpy()
    else:
        raise ValueError("Input data must be either a NumPy array or a PyTorch tensor")

def get_statistics(data):


    re_shape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
    data = re_shape(data);
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # print("saleh_added: mean=",mean, "  std=",std)
    return mean, std





def ts2batch(data: np.ndarray , n_batch: int=10 ,len_hlaf_batch: int=16 , cent =None):

    if (len(np.squeeze(data).shape) >1):
        raise ValueError( 'Data should be 1D not' + str(len(np.squeeze(data).shape))  )



    data = np.squeeze(data)
    start = len_hlaf_batch  # 16
    end=  len(data) - len_hlaf_batch #17

    if cent is None:
        centers= np.random.randint(start,end,n_batch)  # (16,17)
    else:
        centers = cent

    trajs = np.zeros((n_batch,2*len_hlaf_batch))

    # print("centers:",centers)
    for i , center in enumerate(centers):
        trajs[i,:]=data[center-len_hlaf_batch:center+len_hlaf_batch]


    return trajs , centers


def ts2batch_ctx_tar(data: np.ndarray , n_batch: int=10 ,len_ctx: int=49,len_tar: int=1 , cent =None):
    '''
    Make batches from a flattend timesries
    '''
    data = convert_to_numpy(data)
    if len(data.shape) == 1: # if data doesnt have a feture dimension, we create a feature dimension of 1
        data = np.expand_dims(data,axis=-1)
    if (len(data.shape) > 2):
        print( 'Data should be at most 2D (time dimension and feature dimension) when calling this function but data is already have ' + str(len(data).shape) + ' dimensions: if the data is already batched you dont need to call this function' )
        print("we will return the same data")
        return data



    #data = np.squeeze(data)
    length , num_features = data.shape
    start = len_ctx
    end=  len(data) - len_tar

    if cent is None:
        centers= np.random.randint(start,end,n_batch)  # (16,17)
    else:
        centers = cent
    unique_centers = np.unique(centers)
    new_n_batch = len(unique_centers)
    trajs = np.zeros((new_n_batch,len_ctx+len_tar,num_features))

    # print("centers:",centers)
    for i , cent in enumerate(unique_centers):
        trajs[i,:,:]=data[cent-len_ctx:cent+len_tar,:]

    if cent is not None:
        return trajs , unique_centers
    else:
        return  trajs, None


def circular_shift_features(data, t):
    # m = data.shape[0]  # number of features
    # n = data.shape[1]  # number of data points

    # Perform circular shift on the features
    shifted_data = np.roll(data, t, axis=0)

    return shifted_data

def permute_1d_array(arr):
    assert len(arr.shape) < 3
    arr = np.array(arr).flatten()  # Ensure array is flattened to 1D
    permuted_arr = np.random.permutation(arr)
    return permuted_arr

def count_elements_greater_than_MI(MI , B):
    """
    MI = current MI
    B = list of permutations
    """
    return sum(1 for element in B if element > MI)


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
            print("Feature #"f"{list_variables[i]}" " has only constant values")
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
        print("cnt: ",cnt,'/',N-1)
        sum_list =[]
        for i in range(0, number_output_functions):
            basis_log_mutual_information = len(freq_data[i])
            # shuffle (data [i,:] ) ---> data[i,:]

            if perm_test_flag and cnt>0:
                perm_labels = permute_1d_array(data [i,:].copy())
                data[i,:] = perm_labels

            list_of_features = list(range(number_output_functions, len(left_features)))
            list_of_features.insert(0, i)
            id_features = np.array(list_variables)[list_of_features]

            for j in list_of_features:
                freq_data_product = ((
                np.histogram2d(data[i, :], data[left_features[j], :], bins=(l[i], l[left_features[j]]))[0])) / n
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


def run_test(data: np.ndarray, min_n_datapoints_a_bin = None, number_output_functions: int=1 , alpha=0.01 ,log: bool=True ,bonfer: bool= True):

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
        #((m-number_output_functions)*number_output_functions) --> number of example based on which we return a number or make a decision
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
            print("Feature #"f"{i}" " has only constant values")
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
            list_points_of_support.append(datapoints[
                                              -1] + 0.1)  # Add last point of support such that last data point is included (half open interals in Python!)
            if datapoints[datapoints >= list_points_of_support[
                -2]].size < min_n_datapoints_a_bin:  # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
                if len(list_points_of_support) > 2:  # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]
    #print("Binning done!")
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
                freq_data_product = np.histogram2d(data[id_output, :], data[j, :],
                                            bins=(l[id_output], l[j]))[0]
                expfreq = np.outer(freq_data[id_output], freq_data[j]) / n
                if sum(expfreq.flatten() < 5) > 0:
                    counter_bins_less_than5_relevant_principal_features += 1
                if sum(expfreq.flatten() < 1) > 0:
                    counter_bins_less_than1_relevant_principal_features += 1
                pv = scipy.stats.chisquare(freq_data_product.flatten(), expfreq.flatten(),ddof=(freq_data_product.shape[0]-1)+(freq_data_product.shape[1]-1))[1]
                pval_list.append(pv)
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


def linear_correl(arr ,tresh: float = 0.01, number_output_functions: int=1,bonfer = True):


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
            #print(pv,tresh)
            if pv <tresh:
                #print(i," pv:",pv,"  r:", r)
                sum_r = sum_r + np.abs(r)
    return sum_r



def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

