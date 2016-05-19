"""
Hasher.py

Created on Thu May 28 2015

@author: wendywoodin
"""

import numpy as np

class Hasher(object):
    '''
    This is a transcoder class that can be inherited to implement various hashes.
    '''
    def __init__(self, input_length=None, output_length=16, **kwargs):
        """
        Initialization.

        Keyword Arguments:

        input_length : required, int, the length of the string
        to be hashed.

        output_length : optional, int, default 16, the length of the
        output list

        """
        assert input_length is not None
        self.input_length = input_length
        self.output_length = output_length
        self.string = None

    def update(self, string):
        """
        Read in a string to digest.
        Nomeclature inspired by hashlib

        Arguments:
        string: required: str (or other type that can be converted
        to a numpy array), the string to be hashed, must only
        contain the characters 0,1

        Example Usage:

        >>> s1 = '1010001010'
        >>> hasher = Hasher(input_length=len(s1), output_length=5)
        >>> hasher.update(s1)
        >>> print hasher.string
        ['1' '0' '1' '0' '0' '0' '1' '0' '1' '0']

        """
        assert len(string) == self.input_length
        if type(string) is str:
            try:
                int(string,2)
            except ValueError:
                raise StandardError('Must be a binary string')
            string = np.array(list(string))
        else:
            try:
                string = np.array(string)
            except:
                raise StandardError("Must be convertible to a numpy array")
        self.string = string

    def digest(self):
        raise StandardError("Not implemented")
