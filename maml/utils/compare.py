# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:06:30 2014

@author: epyzerknapp
"""
import numpy as np

class Comparer(object):
    """
    Base class for comparison operators
    """
    def compare(self, hash1, hash2):
        pass


class ApproximateJaccard(Comparer):
    """
    Compares using the Approxinate Jaccard metric.

    Defined as the size of the union set normalized to the
    size of the set.

    """
    def compare(self, hash1, hash2):
        """
        Compares two sets using the Approximate Jaccard metric

        Arguments:
        hash1, hash2 : required, list, the hashes to be compared

        Example Usage:

        >>> h1 = [1,4,3,5,6]
        >>> h2 = [2,4,3,6,5]
        >>> comp = ApproximateJaccard()
        >>> c = comp.compare(h1, h2)
        >>> print c
        0.4

        See Also: maml.utils.lsh.py

        """

        assert len(hash1) == len(hash2)
        hash1 = np.array(hash1)
        hash2 = np.array(hash2)
        return  float(np.count_nonzero(hash1==hash2))/float(len(hash1))
