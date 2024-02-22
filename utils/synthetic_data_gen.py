import os

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
#from utils import generate_from_window
import sys
import random
import torch

torch.manual_seed(2)
random.seed(2)
np.random.seed(2)

def generate_gamma_samples(shape, scale, size=1):
    """
    Generate i.i.d. samples from the Gamma distribution.

    Parameters:
    shape (k) (float): Shape parameter (shape > 0).
    scale (theta) (float): Scale parameter (scale > 0).
    size (int, optional): Number of samples to generate (default is 1).

    Returns:
    numpy.ndarray: Array of Gamma distributed samples.
    """
    if shape <= 0 or scale <= 0:
        raise ValueError("Shape and scale parameters must be greater than 0.")

    return np.random.gamma(shape, scale, size)

def gamma_noise(m, n , shape=2, scale=2 ):

    noise = generate_gamma_samples(shape, scale, size=m*n)

    return noise


def generate_frechet_samples(alpha, s, m, size=1):
    """
    Generate i.i.d. samples from the Frechet distribution.

    Parameters:
    alpha (float): Shape parameter (alpha > 0).
    s (float): Scale parameter (s > 0). or called beta in literature
    m (float): Location parameter. (min or left side)
    size (int, optional): Number of samples to generate (default is 1).

    Returns:
    numpy.ndarray: Array of Frechet distributed samples.
    """
    if alpha <= 0 or s <= 0:
        raise ValueError("Alpha and scale parameters must be greater than 0.")

    return m + np.random.gamma(shape=alpha, scale=1/s, size=size)**(-1/alpha)


def white_noise(m,n):

    print("=========================================================== white noise Gen======================================================================")
    mean = 0
    std = 1
    num_samples = m * n
    white_noise = np.random.normal(mean, std, size=num_samples)#.reshape(m, n)
    return white_noise

def simulate_event_times(T, average_events):
    # Generate event times based on the Poisson process
    event_times = np.cumsum(np.random.exponential(1 / average_events, T))
    event_series = np.zeros(T, dtype=int)

    # Mark the time points where events occur
    event_indices = (event_times).astype(int)
    event_series[event_indices[event_indices < T]] = 1

    return event_series




def sin_gen(m,n):
    print("===================================================================  Sin Gen ==========================================================")


    dt=0.001
    t = np.arange(0, m*n*dt, dt)
    f=17.17
    #print('frequency of sin:',f)
    print("len(t):",len(t))
    data2 =  np.cos(2 * np.pi * f* t )
    #all_trajs = ts2batch(data2, n_batch=n, len_hlaf_batch=m//2)  # 5000* (len_half_batch+ 1 +len+half_batch)
    #all_trajs = np.swapaxes(all_trajs, 0, 1)
    #print("all_trajs.shape:", data2.shape)  ## 5000*33

    return data2

if __name__ == '__main__':
    import numpy as np

    # Generating zero-mean noise with a median around 0.3
    desired_median = 0.3
    size = 100000  # Adjust this as needed

    # Create noise centered around 0 with a normal distribution
    noise_centered = np.random.normal(loc=0, scale=0.1, size=size)
    print("mean_gauss:",np.mean(noise_centered))

    # Create additional positive noise biased towards 0.3
    additional_positive_noise = np.random.uniform(low=0.4, high=0.6, size=int(size ))
    print(additional_positive_noise.shape)
    print("mean_pos",np.mean(additional_positive_noise))

    # Create additional negative noise biased towards -0.6 (to counterbalance positive noise)
    additional_negative_noise = np.random.uniform(low=-1.2, high=-0.8, size=int(size * 0.5))
    print("mean_neg",np.mean(additional_negative_noise))

    # Combine noises
    combined_noise = np.concatenate((noise_centered, additional_positive_noise, additional_negative_noise))

    # Shuffle the combined noise
    np.random.shuffle(combined_noise)

    # Truncate to the original size
    combined_noise = combined_noise[:size]

    # Now the combined noise array might have a median around 0.3 and a mean close to zero
    actual_median = np.median(combined_noise)
    mean = np.mean(combined_noise)
    plt.plot(combined_noise[:100])
    plt.show()

    print(f"Desired Median: {desired_median}")
    print(f"Actual Median: {actual_median}")
    print(f"Mean: {mean}")









