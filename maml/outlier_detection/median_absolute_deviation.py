__author__ = 'epyzerknapp'
import numpy as np


def median_abs_deviation_oulier_detection(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        http://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

    Notes:
    ---------
    Unlike the references, especially the stack-overflow link, this implementation uses an L1 norm, not the L2
    norm implemented in the references.
    """
    points = np.atleast_2d(points)
    median = np.median(points, axis=0)
    diff = np.sum(np.abs(points - median), axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def double_median_abs_deviation_outlier_detection(points, thresh=3.5):
    """
    Returns a boolean array with True if the points are outliers and False
    otherwise.  This 'double MAD' should be used if you believe that the 
    distribution is skewed (i.e. one tail is longer than the other) in 
    which case the assumptions of a symmetrical cutoff does not hold.

    Parameters:
    ---------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ------------
    http://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
    http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers

    Notes:
    -------
    This method does not address problems if more than 50 % of the input data is identical. 
    If this occurs, every point in your dataset except those whose value is the same as the median 
    will be flagged as outliers, independent of the level of the outlier cutoff.

    """
    points = np.atleast_2d(points)
    median = np.median(points, axis=0)
    dev = np.abs(points - median)
    mad_left = np.median(np.abs(dev[points <= median]))
    mad_right = np.median(np.abs(dev[points >= median]))
    y_mad = np.zeros(len(points))
    y_mad[points < median] = mad_left
    y_mad[points > median] = mad_right
    modified_z_score = 0.6745 * dev / y_mad
    modified_z_score[y == m] = 0
    return mad_distance > thres
