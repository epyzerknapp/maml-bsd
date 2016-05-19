# -*- coding: utf-8 -*-
"""
Minhash.py

Created on Tue Nov 11 12:03:45 2014

@author: epyzerknapp
"""

import numpy as np
import numpy.random as npr
from hasher import Hasher

class MinHasher(Hasher):
    '''
    This is a transcoder class from string to minhash representation.
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
        npr.seed(42)
        self.indices = [npr.permutation(input_length) for _ in
        xrange(0,output_length)]
        self.input_length = input_length
        self.output_length = output_length
        self.string = None

    def digest(self):
        """
        Return the minhash of the input string as a list

        No Arguments

        Example Usage:
        >>> s1 = '1111111111'
        >>> hasher = MinHasher(input_length=len(s1), output_length=5)
        >>> hasher.update(s1)
        >>> minhash = hasher.digest()
        >>> print minhash
        [0, 0, 0, 0, 0]

        """
        output = []
        for index in self.indices:
            first_non_zero = np.argmax(self.string[index])
            if first_non_zero == 0 and (self.string[index][0] == '0' or self.string[index][0] == 0):
                first_non_zero = self.input_length
            output.append(first_non_zero)
        return output




if __name__ == '__main__':
    minhash = MinHasher(input_length=10,output_length=30)
    minhash.update('1001000000')
    print minhash.digest()
    minhash.update('0000000000')
    print minhash.digest()
    minhash.update('1111111111')
    print minhash.digest()
