import numpy as np
import scipy.stats


def chisquare_test(data: np.ndarray, min_n_datapoints_a_bin = None, number_output_functions: int=1 , alpha=0.01  ,bonfer: bool= True):

    '''
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

     outputs:
     cnt_dep = calculate how many of targets are correlated with how many of context (max = ctx*tar)
     pval_list= list of all pvalues: [ctx1-tar1 ctx1-tar2 ... ctx1-tarm ctx2-tar1 ctx2-tar2 ... ctxk-tarm]
    '''


    counter_bins_less_than5_relevant_principal_features=0 # number of chi-square tests with less than 5 datapoints a bin
    counter_bins_less_than1_relevant_principal_features=0 # number of chi-square tests with less than 1 datapoint a bin
    counter_number_chi_square_tests_relevant_principal_features=0 # nu

    m = data.shape[0]  # number features (inputs + outputs)
    n = data.shape[1]  # number of data points
    if (min_n_datapoints_a_bin is None):
        min_n_datapoints_a_bin = 0.05*n

    #alpha=0.01/m
    if bonfer:
        alpha = alpha/((m-number_output_functions)*number_output_functions)

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

