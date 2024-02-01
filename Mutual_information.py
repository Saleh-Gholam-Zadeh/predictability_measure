import numpy as np
import pandas as pd
import math



def permute_1d_array(arr):
    assert len(arr.shape) < 3
    arr = np.array(arr).flatten()  # Ensure array is flattened to 1D
    permuted_arr = np.random.permutation(arr)
    return permuted_arr

def get_mutual_information(data, number_output_functions=1, min_n_datapoints_a_bin = None, perm_test_flag=True, N=10):

    """
        data: a 2D numpy array in which the first "number_output_functions" rows are the values of the output function (or target values) and the other rows are the values of the features (input or context)
        Implementation of MI code
        samples : each sample stored in one column
        n_col---> number of samples
        n_rows ---> number of output + features (input)
        i.e
        m = data.shape[0]  # number features (plus targets)
        n = data.shape[1]  # number of data points
        the whole length of target + context = m ---> target comes first
        each column of the data: [t1, t2, ..., t_num_output_function, ctx1, ctx2, ... m ]
        number_output_functions: how many outputs we have
    """
    # Calulate the Shannon mutual information
    def make_summand_from_frequencies(x, y):
        if x == 0:
            return 0
        else:
            return x * math.log2(x / y) / math.log2(basis_log_mutual_information)


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

