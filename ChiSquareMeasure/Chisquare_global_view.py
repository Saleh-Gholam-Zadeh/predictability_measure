
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
from scipy.stats import chi2
import scipy
import warnings



def to_tensor_preserve_precision(x): # push it github
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, list):
        # Infer precision from the first element, default to float32 if unsure
        if len(x) == 0:
            return torch.tensor([], dtype=torch.float32)

        elif isinstance(x[0],float):
            print("we use float32 precision with torch, however float64 precision is passed to this function. we converted it to float32")
            return torch.tensor(x, dtype=torch.float32)

        elif isinstance(x[0], np.float64):
            return torch.tensor(x, dtype=torch.float64)
        elif isinstance(x[0], np.float32):
            return torch.tensor(x, dtype=torch.float32)
        elif isinstance(x[0], int):
            return torch.tensor(x, dtype=torch.int64)
        else:
            raise TypeError(f"Unsupported list element type: {type(x[0])}")
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")





def compute_frequencies_torch(data_x, data_y, boundaries_x, boundaries_y): #push to github
    """
    Compute `freq_data_product` and `expfreq` using PyTorch operations with histogramdd.

    Parameters:
        data_x (torch.Tensor): 1D Tensor of x-coordinates (data points).
        data_y (torch.Tensor): 1D Tensor of y-coordinates (data points).
        boundaries_x (torch.Tensor): 1D Tensor specifying bin boundaries for x.
        boundaries_y (torch.Tensor): 1D Tensor specifying bin boundaries for y.

    Returns:
        torch.Tensor, torch.Tensor: `freq_data_product` and `expfreq` as Tensors.
    """
    n = len(data_x)  # number of data points

    # Stack data_x and data_y to create 2D input for histogramdd
    data = torch.stack([data_y ,data_x], dim=1)

    # Compute 2D histogram using torch.histogramdd
    hist = torch.histogramdd(data, bins=[boundaries_y ,boundaries_x])
    # Extract the frequency counts from the histogram
    freq_data_product = hist.hist # P_joint[row,col]

    # Normalize by n to get probabilities
    freq_data_product = freq_data_product / n

    # Compute expected frequencies using outer product
    # expfreq_gt_left_out = torch.outer( freq_data_y,freq_data_x / (n * n))
    # print("later remove expfreq_gt_left_out ")

    return freq_data_product
    # return freq_data_product, expfreq_gt_left_out





def binning(data , number_output_functions=1,Rho=None): # push it to github

    '''
        data: np.array m_row ---> ctx (features) + target : so each sample or datapoints is stored in a column
    /// n_col---> samples : each sample stored in one column
        i.e
        :data: m*n
        m = data.shape[0]  # number features (plus targets)
        n = data.shape[1]  # number of data points (batches)
    the whole length of target + context = m ---> target comes first
     each column of the data: [t1, t2, ..., t_num_output_function, ctx1, ctx2, ... m ]

     outputs:
     cnt_dep = calculate how many of targets are correlated with how many of context (max = ctx*tar)
     pval_list= list of all pvalues: [ctx1-tar1 ctx1-tar2 ... ctx1-tarm ctx2-tar1 ctx2-tar2 ... ctx_k-tar_m]
    '''

    print("binning called!")

    if not type(data) == np.ndarray:
        data_cp = data.detach().cpu().numpy()
    else:
        data_cp = data

    # data_cp = circular_shift_features(data_cp,t=number_output_functions)


    counter_bins_less_than5_relevant_principal_features = 0  # number of chi-square tests with less than 5 datapoints a bin
    counter_bins_less_than1_relevant_principal_features = 0  # number of chi-square tests with less than 1 datapoint a bin
    counter_number_chi_square_tests_relevant_principal_features = 0  # nu

    # data = data.to_numpy()
    m = data_cp.shape[0]  # number features
    n = data_cp.shape[1]  # number of data points/windows
    if (Rho is None):
        min_n_datapoints_a_bin = int(0.05 * n)
    else:
        min_n_datapoints_a_bin = int(Rho * n)

    # alpha=0.01/m
    # if bonfer:
    #     # print("old_alpha:",alpha)
    #     # ((m-number_output_functions)*number_output_functions) --> number of experiments based on which we return a number or make a decision
    #     alpha = alpha / ((m - number_output_functions) * number_output_functions)
    #     # print("bonfer_alpha after correction:",alpha)

    l = [0 for i in range(0, m)]  # list of lists with the points of support for the binning
    freq_data = [0 for i in range(0, m)]  # list of histograms
    left_features = [i for i in range(0, m)]  # list of features that is step by step reduced to the relevant ones
    constant_features = []

    # remove constant features and binning (discretizing the continuous values of our features)
    for i in range(0, m):
        mindata = min(data_cp[i, :])
        maxdata = max(data_cp[i, :])
        # print("i,mindata , maxdata:",(i,mindata,maxdata))
        if maxdata <= mindata:
            print("Feature {} has only constant values".format(i))
            left_features.remove(i)
            constant_features.append(i)
            raise ValueError('WTF')  # added by saleh
        else:
            # start the binning by sorting the data points
            list_points_of_support = []
            datapoints = data_cp[i, :].copy()
            datapoints.sort()
            last_index = 0
            # go through the data points and bin them
            for point in range(0, datapoints.size):
                if point >= (datapoints.size - 1):  # if end of the data points leave the for-loop
                    break
                # close a bin if there are at least min_n_datapoints_a_bin and the next value is bigger
                if datapoints[last_index:point + 1].size >= min_n_datapoints_a_bin and datapoints[point] < datapoints[point + 1]:
                    list_points_of_support.append(datapoints[point + 1])
                    last_index = point + 1
            if len(list_points_of_support) > 0:  # test that there is at least one point of support (it can be if there are only constant value up to the first ones which are less than min_n_datapoints_a_bin
                if list_points_of_support[0] > datapoints[0]:  # add the first value as a point of support if it does not exist (less than min_n_datapoints_a_bin at the beginning)
                    # list_points_of_support.insert(0, datapoints[0])
                    list_points_of_support.insert(0, datapoints[0] - np.float32(0.2)) # subtracted by -0.2  [SALEH]
                    # print("warning: [SALEH] subtracted the first border by -0.2")

            else:
                list_points_of_support.append(datapoints[0])
            # list_points_of_support.append(datapoints[-1] + np.float32(0.1))  # Add last point of support such that last data point is included (half open interals in Python!)
            list_points_of_support.append(datapoints[-1] + np.float32(0.2))  # updated to 0.2 by [SALEH]
            if datapoints[datapoints >= list_points_of_support[-2]].size < min_n_datapoints_a_bin:  # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
                if len(list_points_of_support) > 2:  # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]
    # print("Binning done!")
    # print("List of features with constant values:")
    # print(constant_features)
    return l, freq_data, counter_bins_less_than1_relevant_principal_features, counter_bins_less_than5_relevant_principal_features


def compute_chisq_metric_hist_based_total(data_x, data_y, boundaries_x, boundaries_y): # push it to github
    '''
    Compute the chi-squared sum  for stochastic dependence between all input and output variables.
        Args:
        data_x (torch.Tensor): Input data for variable X (shape = [N]).
        data_y (torch.Tensor): Input data for variable Y (shape = [N]).
        boundaries_x (list): Discretization boundaries (bins) for X.
        boundaries_y (list): Discretization boundaries (bins) for Y.
        sigma (float): Gaussian kernel smoothing parameter.

    Returns:
        total_chisq_loss (torch.Tensor): The chi-squared-based loss for the pair (x, y).
        total_dof (int): (partial) Degrees of freedom for the computation.
        pvalue corresponding to the total_chisq_loss and total_dof.
    '''

    if isinstance(data_x,np.ndarray):
        data_x = torch.from_numpy(data_x)
    if isinstance(data_y,np.ndarray):
        data_y = torch.from_numpy(data_y)

    data_x = data_x.squeeze()
    data_y = data_y.squeeze()

    if len(data_x.shape)==2:
        N_x, T_x = data_x.shape
    elif len(data_x.shape)==1 :
        data_x = data_x.unsqueeze(-1)
        N_x, T_x = data_x.shape
    else:
        raise ValueError("Input data_x must be (effectively) 1D or 2D; if it is more please reshape it to (N,T) where N is the number of samples and T is the number of features")


    if len(data_y.shape)==2:
        N_y, T_y = data_y.shape
    elif len(data_y.shape)==1 :
        data_y = data_y.unsqueeze(-1)
        N_y, T_y = data_y.shape
    else:
        raise ValueError("Input data_y must be (effectively) 1D or 2D; if it is more please reshape it to (N,T) where N is the number of samples and T is the number of features")


    assert N_x==N_y, "the number of samples in x and y should be the same"

    assert len(boundaries_y) + len(boundaries_x) == T_x + T_y, "for every feature we should have corresponding (set of) boundaries"




    total_chisq_loss = 0
    total_dof = 0
    for i in range(T_x):
        for j in range(T_y):
            chisq_loss_ij, pxy_hist_ij, E_ij, dof_ij = compute_chisq_metric_hist_based_ij(data_x[:,i], data_y[:,j], boundaries_x[i], boundaries_y[j])
            total_chisq_loss += chisq_loss_ij.item()
            total_dof += dof_ij

    p_value_chisq_sum = 1 - chi2.cdf(total_chisq_loss, df=total_dof)


    return total_chisq_loss, total_dof , p_value_chisq_sum
def compute_chisq_metric_hist_based_ij(data_x, data_y, boundaries_x, boundaries_y): # push it to github

    """
    Compute the chi-squared loss for stochastic dependence between TWO RANDOM VARIABLES.


    Args:
        data_x (torch.Tensor): Input data for variable X (shape = [N]).
        data_y (torch.Tensor): Input data for variable Y (shape = [N]).
        boundaries_x (list): Discretization boundaries (bins) for X.
        boundaries_y (list): Discretization boundaries (bins) for Y.
        sigma (float): Gaussian kernel smoothing parameter.

    Returns:
        chisq_loss (torch.Tensor): The chi-squared-based loss for the pair (x, y).
        pxy_gaussian (torch.Tensor): Joint probability distribution matrix (observed distribution).
        E (torch.Tensor): Expected joint distribution matrix under independence.
        dof (int): (partial) Degrees of freedom for the computation.
    """
    # Joint PDF for (X, Y) using the histogram method

    # convert everything to torch tensr if they are already not
    data_x = to_tensor_preserve_precision(data_x)
    data_y = to_tensor_preserve_precision(data_y)
    boundaries_x = to_tensor_preserve_precision(boundaries_x)
    boundaries_y = to_tensor_preserve_precision(boundaries_y)

    pxy_hist = compute_frequencies_torch(data_x, data_y, boundaries_x, boundaries_y)

    # Marginal probabilities
    py = pxy_hist.sum(dim=1)  # Marginal P(Y) (row sum)
    px = pxy_hist.sum(dim=0)  # Marginal P(X) (col sum)

    # Total number of observations (for computing expected values)
    N = len(data_x)

    # Calculate the Expected probability matrix under independence
    E = torch.outer(py, px) * N

    # Observed frequencies (convert probabilities to frequencies by multiplying N)
    O = pxy_hist * N

    # Ensure no zero in E to avoid division issues
    if torch.any(E == 0):
        raise ValueError("Expected frequency matrix contains zeros. Adjust binning strategy or sigma.")

    # Check if any expected frequencies are less than 5 (common rule for chi-square test)
    if torch.any(E < 5):
        import warnings
        warnings.warn("Expected frequency matrix contains values less than 5. "
                      "Chi-square test may not be reliable. Consider adjusting binning or combining categories.")

    # Compute the chi-square statistic (χ²)
    chisq_stat_torch = ((O - E) ** 2 / E).sum()

    # Degrees of freedom: (l_x - 1) * (l_y - 1)
    l_x, l_y = len(boundaries_x) - 1, len(boundaries_y) - 1
    dof = (l_x - 1) * (l_y - 1)

    # Compute the p-value for this chi-square statistic

    # Define chisq_loss as the chi-square statistic (neglected p-value adjustment for simplicity)
    chisq_loss = chisq_stat_torch

    # p_value_torch = 1 - chi2.cdf(chisq_stat_torch.item(), df=dof)
    # chisq_scipy,pv_scipy = scipy.stats.chisquare(O.detach().cpu().flatten(), E.detach().cpu().flatten(), ddof=(O.shape[0] - 1) + (O.shape[1] - 1))
    #
    # print( torch.allclose(chisq_loss,torch.tensor(chisq_scipy ,dtype=torch.float32)) )
    # print(chisq_loss , chisq_scipy)
    # print(p_value_torch , pv_scipy)

    return chisq_loss, pxy_hist, E, dof


if __name__== '__main__':

    print("hi")
    my_data = np.linspace(0.99,500, 250000,dtype=np.float32)
    # print(my_data)
    print(len(my_data))
    my_data = my_data.reshape(-1,50)
    my_borders ,my_freq  , less5 ,less1 = binning(my_data.swapaxes(0,1))
    ctx_size = 49
    target_size = 1
    assert len(my_borders) == ctx_size + target_size
    total_chisq,total_dof,pv = compute_chisq_metric_hist_based_total(data_x=my_data[:,:ctx_size], data_y=my_data[:,ctx_size:] , boundaries_x=my_borders[:ctx_size], boundaries_y= my_borders[ctx_size:])
    print("---total_Chisquare:",total_chisq,"\t---total_dof:",total_dof,'\t---independence_pv:',pv)
