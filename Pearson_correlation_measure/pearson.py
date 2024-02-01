import torch
import numpy as np
import scipy.stats
import pandas as pd
import math


def pearson_test(arr ,tresh: float = 0.01, number_output_functions: int=1,bonfer = True):

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

    implementation of PearsonR correlation
    m = data.shape[0]  # number of features (or more precisely features + outputs )
    n = data.shape[1]  # number of data points (windows)

    return p-value and r va
    '''

    assert isinstance(tresh, float), "Variable is not of type float"
    assert len(arr.shape)==2
    m,n = arr.shape
    if bonfer :
        tresh = tresh/(number_output_functions*(m-number_output_functions))

    X_mat = arr[number_output_functions:,:] # features
    Y = arr[:number_output_functions,:] # labels
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
