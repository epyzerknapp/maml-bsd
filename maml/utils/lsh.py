# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 12:50:03 2014

@author: epyzerknapp
"""
import minhash
import compare

class LshDistance(object):
    """
    A class for calculating LSH distances between two strings using
    an LSH based technique.
    """
    def __init__(self, comparison_type='ApproxJaccard',
                 hashing_type=None, **kwargs):
        """
        Initialization.

        Keyword Arguments:

        compaison_type: optional, str, default 'ApproxJaccard'
        the metric used for the hash comparison

        hashing_type: optional, str, default None, the hashing to
        implement on the strings before comparison.  None represents
        a null hasher.

        Non-specified Keyword Arguments: If hashing_type is not None,
        you must specify any keywords that are needed for the hashing
        algorithm specified

        See also: maml.utils.minhash

        """
        if comparison_type != 'ApproxJaccard':
            raise NotImplementedError
        self.comparison_type = comparison_type
        self.hashing_type = hashing_type
        if self.hashing_type:
            if self.hashing_type == 'MinHash':
                self.hasher = minhash.Hasher(**kwargs)

    def compare(self, string1, string2):
        """
        Compare two strings using LSH

        Arguments: string1, string2 , required, str, the strings
        that are to be compared

        Example Usage:

        >>> s1 = '1111111111'
        >>> s2 = '0000000000'
        >>> lsh = LhsDistance(hashing_type='MinHash', input_length=10, output_length=4)
        >>> compare = lsh.compare(s1,s2)
        >>> print compare
            0.0

        """
        if self.hashing_type:
            self.hasher.update(string1)
            hashed_s1 = self.hasher.digest()
            self.hasher.update(string2)
            hashed_s2 = self.hasher.digest()
        else:
            hashed_s1 = string1
            hashed_s2 = string2

        if self.comparison_type == 'ApproxJaccard':
            comp = compare.ApproximateJaccard()
            return comp.compare(hashed_s1, hashed_s2)


if __name__ == '__main__':
    lsh = LshDistance(hashing_type='MinHash', input_length=10, output_length=30)

    #s1 = '1010111110'
    #s2 = '0000011111'
    s1 = '1111111111'
    s2 = '0000000000'
    print lsh.compare(s1,s2)

    print lsh.compare(s2,s1)

    print lsh.compare(s1,s1)

    print lsh.compare(s2,s2)






