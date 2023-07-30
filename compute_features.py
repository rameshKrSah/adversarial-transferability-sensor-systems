
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


import os
import sys
import getopt
import numpy as np
import pandas as pd
import pickle
import scipy.stats as stats
from entropy import spectral_entropy

feature_names = [
    "mean", 
    "median",
    "std",
    "min",
    "max",
    "IQR",
    "skew",
    "kurt",
    "var",
    "rms",
    "energy",
    "ptp",
    "crest-factor",
    "impulse-factor",
    "avg-derivatives",
    "ZCR",
    "MCR",
    "SPCE",
    "corr-x-y",
    "corr-x-z",
    "corr-y-z"
]

def get_zero_crossing_indices(data):
    """
    @brief:
        Return the indices where the signal value changed from positive to negative or vice versa. This returns the indices before the value change.
        For example if data = [1, -1, -2, 1], the return value will be [0, 2].

    @param: data, 1D array
    @return: indices of the zero crossing

    @Source: https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    """
    data = np.array(data)
    pos = data > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

def get_mean_crossing_indices(data):
    """
    @brief: Return the indices of elements in the array data(A), for which sign(a_i - U_A) and sign(a_i + U_A) are different.
    @param: data, 1D array
    @return: indices of the mean crossing
    @source: https://stackoverflow.com/questions/57501852/how-to-calculating-zero-crossing-rate-zcr-mean-crossing-rate-mcr-in-an-arr
    """
    return get_zero_crossing_indices(np.array(data) - np.mean(data))

"""
    Compute features from the 3 axial sensor window segment
"""
def compute_features(window_segments):
    n_channels = 3
    
    if window_segments.shape[1] != 3:
        raise ValueError("Window segments has 3 axes: X, Y, and Z")
    
    # container to store the features
    features = []
    
    for i in range(n_channels):
        seg = window_segments[:, i]

        # Mean: The DC component (average value) of the signal over the window
        features.append(np.mean(seg))

        # Median: The median signal value over the window
        features.append(np.median(seg))

        # Standard Deviation: Measure of the spreadness of the signal over the window
        features.append(np.std(seg))

        # Minimum: The minimum value of the signal over the window
        features.append(np.min(seg))

        # Maximum: The maximum value of the signal over the window
        features.append(np.max(seg))
        
        # Interquartile Range: Measure of the statistical dispersion, 
        # being equal to the difference between the 75th and the 25th percentiles of the signal over the window
        features.append(stats.iqr(seg))

        # Skewness: The degree of asymmetry of the sensor signal distribution
        features.append(stats.skew(seg))

        # Kurtosis: The degree of peakedness of the sensor signal distribution
        try:
            features.append(stats.kurtosis(seg))
        except:
            features.append(0)
        
        # Variance: The squared of standard deviation.
        features.append(np.var(seg))
        
        # Root Mean Square: THe quadratic mean value of the signal over the window.
        rms = np.sqrt(np.mean(seg**2))
        features.append(rms)
        
        # Energy: The energy of the signal over the window
        ene = np.sum(seg**2)
        features.append(ene)
        
        # Peak to Peak: the difference between the maximum and minimum over the window
        features.append(np.max(seg) - np.min(seg))
        
        # crest factor
        if rms != 0:
            features.append(np.max(np.abs(seg)) / rms)
        else:
            features.append(0)
        
        # impulse factor
        features.append(np.max(seg) / np.mean(seg))

        # averaged derivatives: The mean value of the first order derivatives of the signal over the window
        features.append(np.mean(np.diff(seg)))

        seg_len = len(seg)
        # Zero Crossing Rate (ZCR): The total number of times the signal changes from positive to negative or back or vice versa normalized by the window length
        zrc = len(get_zero_crossing_indices(seg)) / seg_len
        features.append(zrc)

        # Mean crossing Rate (MCR): The total number of times the signal changes from below average to above average or vice versa normalized by the window length
        mcr = len(get_mean_crossing_indices(seg)) / seg_len
        features.append(mcr)

        # Spectral Entropy (SPCE): Measure of the distribution frequency components.
        features.append(spectral_entropy(seg, sf=50.0, method='welch', normalize=False))
        
    
    # compute Pearson coefficients between the signal axes
    features.append(stats.pearsonr(window_segments[:, 0], window_segments[:, 1])[0])
    features.append(stats.pearsonr(window_segments[:, 0], window_segments[:, 2])[0])
    features.append(stats.pearsonr(window_segments[:, 1], window_segments[:, 2])[0])
    
    return features



"""
    Parse the arguments passed with the script
"""
def parse_arguments(argv):
    data_file = " "
    output_file = " "
    
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except:
        print("data_processing.py -i <sensor_window_segment_file> -o <output_file_as_pickle>")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == "-h":
            print("data_processing.py -i <sensor_window_segment_file> -o <output_file_as_pickle>")
            sys.exit()
            
        elif opt in ("-i", "--ifile"):
            data_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
            
    return data_file, output_file


# Perfrom the main action
if __name__ == "__main__":
    sensor_segment_path, save_path = parse_arguments(sys.argv[1:])
    
    if((len(sensor_segment_path) != 0) & (len(save_path) != 0)):
        # load the sensor window segments
        f = open(sensor_segment_path, "rb")
        sensor_segments, _ = pickle.load(f)
        f.close()
        
        total_segments_len = sensor_segments.shape[0]
        print("Shape of sensor segments {}".format(sensor_segments.shape))
        
        # compute features for each segment
        total_features = []
        i = 0
        for window_segment in sensor_segments:
            i += 1
            if (i % 1000 == 0):
                print("Processed {} sensor segments".format(i))
            total_features.append(compute_features(window_segment))
        
        # save the computed features
        total_features = np.array(total_features)
        print("Shape of feature data {}".format(total_features.shape))
        total_features_len = total_features.shape[0]
        
        nan_numbers = np.argwhere(np.isnan(total_features))
        print("Number of NAN values {}".format(len(nan_numbers)))

        if total_segments_len != total_features_len:
            raise ValueError("Error discrepancies in the features size and segments size")
            
        f = open(save_path, "wb")
        pickle.dump([total_features, _], f)
        f.close()
    else:
        print("data_processing.py -i <sensor_window_segment_file> -o <output_file_as_pickle>")
        sys.exit(2)




