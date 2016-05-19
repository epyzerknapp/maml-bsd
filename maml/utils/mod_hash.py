"""
mod_hash.py
Created on Thu May 21 2015
@author: wendywoodin
"""

import numpy as np
from hasher import Hasher


class ModHasher(Hasher):

    '''
    This is a transcoder class from string to a modular hash representation.
    Note: Most of the code is from minhash
    '''
    def __init__(self, input_length=None, modular=3, **kwargs):
        """
        Initialization.
        Keyword Arguments:
        input_length : required, int, the length of the string
        to be hashed.
        modular: optional, int, default 3, the number used for
        the modular arithmetic of the hash
        """
        assert input_length is not None
        assert modular > 0
        self.modular = modular
        self.input_length = input_length
        self.string = None

    def digest(self):
        """
        Return the modular hash of the input string as an np array
        No Arguments
        Example Usage:
        >>> s1 = '101010101010101'
        >>> hasher = ModHasher(input_length=len(s1), modular=2)
        >>> hasher.update(s1)
        >>> modhash = hasher.digest()
        >>> print modhash
        [ 8.  0.]
        """
        hashed = np.zeros(self.modular)
        for i in range(self.input_length):
            if self.string[i] == '1' or self.string[i] == 1:
                hashed[i % self.modular] += 1
        return hashed


if __name__ == '__main__':
    modhash = ModHasher(input_length=10, modular=3)
    modhash.update('1001000000')
    print modhash.digest()
    modhash.update('0000000000')
    print modhash.digest()
    modhash.update('1111111111')
    print modhash.digest()
