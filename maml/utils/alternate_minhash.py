"""
alternate_minhash.py
Created on Fri May 22 2015
@author: wendywoodin
"""

import numpy as np
import random
from hasher import Hasher


class AltMinHasher(Hasher):

    '''
    This is a transcoder class from string to an alternate minhash representation.
    '''
    def __init__(self, input_length=None, output_length=16, **kwargs):
        """
        Initialization.
        Keyword Arguments:
        input_length : required, int, the length of the string
        to be hashed, cannot be above 5000
        output_length : optional, int, default 16, the length of the
        output list

        Global Variables:
        self.prime: the prime used for the hash function. Must be greater
        than the input length. This restricts the input length to less
        than 5009
        """
        self.prime = 5009
        assert input_length is not None
        assert input_length < self.prime
        self.input_length = input_length
        self.output_length = output_length
        self.string = None
        hash_numbers = np.zeros((output_length, 1, 2))
        for i in range(output_length):
            random.seed(i)
            hash_numbers[i][0][0] = random.randint(1, self.prime-1)
            hash_numbers[i][0][1] = random.randint(0, self.prime-1)
        self.hash_numbers = hash_numbers

    def alt_hash(self, index, x):
        '''
        Function for hashing x, which is the position in the string,
        according to which part of the hash array we are in. This
        gives us a way to have as many hash functions as we want up
        to the prime we chose.

        We seed with our index in the hash array, which allows us
        to then get pseudo-random values for a and b dependent on
        which hash function we are in.

        Example Usage:
        >>> index = 4
        >>> x = 20
        >>> hasher = AltMinHasher(input_length = 30, output_length = 5)
        >>> newhash = hasher.alt_hash(index,x)
        >>> print(newhash)
        4140.0
        '''
        a = self.hash_numbers[index][0][0]
        b = self.hash_numbers[index][0][1]
        return (a * x+b) % self.prime

    def digest(self):
        """
        Return the minhash of the input string as a list
        No Arguments
        Example Usage:
        >>> s1 = '1111111111'
        >>> hasher = AltMinHasher(input_length=len(s1), output_length=5)
        >>> hasher.update(s1)
        >>> minhash = hasher.digest()
        >>> print minhash
        [  676.   283.  2758.   101.   239.]
        """
        hashes = self.prime * np.ones(self.output_length)
        for bit in range(self.input_length):
            if self.string[bit] == '1' or self.string[bit] == 1:
                for i in range(self.output_length):
                    newhash = self.alt_hash(i, bit)
                    if newhash < hashes[i]:
                        hashes[i] = newhash
        return hashes


if __name__ == '__main__':
    minhash = AltMinHasher(input_length=10, output_length=30)
    minhash.update('1001000000')
    print minhash.digest()
    minhash.update('0000000000')
    print minhash.digest()
    minhash.update('1111111111')
    print minhash.digest()
