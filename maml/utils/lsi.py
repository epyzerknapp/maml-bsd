__author__ = 'epyzerknapp', 'djasrasaria'

import numpy as np
import math
from scipy.sparse import lil_matrix, linalg

'''
Dimensionality reduction of vectors using Latent Semantic Indexing.
Can be used on sparse or dense vectors
'''

def build_counts(documents):
    '''
    Builds a counts matrix that shows how many times
    each word appears in each document.

    Parameters
    -----
    document: array of vectors
        each vector is given as the non-zero indices of
        the unreduced vector
        > ex: The vector [0 0 2 0 1] should be input
        >     as [2, 2, 4]

    Returns
    -----
    A: scipy lil_matrix
        counts matrix
    '''
    # dictionary of words
    word_dict = {}
    # the number of the document
    doc_no = 0
    # add document number to each word in dictionary
    for vector in documents:
        for elt in vector:
            if elt in word_dict:
                word_dict[elt].append(doc_no)
            else:
                word_dict[elt] = [doc_no]
        doc_no += 1

    # keep only words that appear in more than one document
    words = [w for w in word_dict.keys() if len(word_dict[w]) > 1]

    # initialize counts matrix
    A = lil_matrix((len(words), doc_no))
    # fill with word counts
    for i, word in enumerate(words):
        for d in word_dict[word]:
            A[i, d] += 1
    # return counts matrix
    return A

def tfidf(A):
    '''
    Modifies counts matrix using Term Frequency - Inverse Document
    Frequency. This gives a higher weight to less common words and
    a lower weight to more common words.

    Parameters
    -----
    A: scipy lil_matrix
        counts matrix

    Returns
    -----
    A: scipy lil_matrix
        modified counts matrix
    '''
    # total number of words in each documents
    tot_words = np.array(A.sum(axis=0))[0]
    # number of documents in which each word appears
    num_docs = A.getnnz(axis=1)

    # get matrix's nonzero coordinates
    coords = A.nonzero()
    # modify each nonzero coordinate using TF-IDF
    for i in range(0, len(coords[0])):
        x = coords[0][i]
        y = coords[1][i]
        A[x, y] = (A[x, y] / tot_words[y]) * math.log(float(A.shape[1]) / num_docs[x])
    # return modified matrix
    return A

def reduce_dims(A, k, return_svs=False):
    '''
    Reduces original vectors to 'k' dimensions by using
    Singular Value Decomposition to reduce the modified
    A matrix.

    Parameters
    -----
    A : scipy lil_matrix
        modified counts matrix
    k : int
        number of reduced dimensions
    return_svs : bool, default is False
        when True, returns the array of singular values
        for each reduced dimension

    Returns
    -----
    Aprime : matrix
        A matrix reduced to k dimensions
    s : array
        singular values for each reduced dimension
        only returned if return_svs=True
    '''
    u, s, vt = linalg.svds(A, k=k, return_singular_vectors='vh')
    Aprime = np.dot(np.diag(s), vt).T
    if return_svs:
        return Aprime, s
    else:
        return Aprime

def get_svs(documents, k=50):
    '''
    Returns the k singular values of the modified counts matrix.
    These values can be plotted to determine the optimal k value
    for LSI.

    Parameters
    -----
    documents : array of vectors
        each vector is given as the non-zero indices of
        the unreduced vector
        > ex: The vector [0 0 2 0 1] should ve input
        >     as [2, 2, 4]
    k : int, default is 50
        number of singular values

    Returns
    -----
    s : array
        array of k singular values
    '''
    # build counts matrix
    A = build_counts(documents)
    # modify using TF-IDF
    A2 = tfidf(A)
    # reduce using SVD
    s = linalg.svds(A2, k=k, return_singular_vectors=False)
    # return singular values
    return s[::-1]

def LSI(documents, k, return_svs=False):
    '''
    Dimensionality reduction of vectors using Latent
    Semantic Indexing.

    First, the function reads a document and builds a
    counts matrix. Then it modifies the counts using
    TF-IDF. Finally, it reduces the dimensionality of
    the original documents by performing Singular
    Value Decomposition and keeping only the k highest-
    contributing latent dimensions.

    Works for dense or sparse vectors.

    Parameters
    -----
    documents : array of vectors
        each vector is given as the non-zero indices of
        the unreduced vector
        > ex: The vector [0 0 2 0 1] should be input
        >     as [2, 2, 4]
    k : int
        number of reduced dimensions
    return_svs : bool, default is False
        when True, returns the array of singular values
        for each reduced dimension

    Returns
    -----
    reduced_doc: array of vectors
        each vector is reduced to length
    s : array
        singular values for each reduced dimension
        only returned if return_svs=True
    '''
    # build counts matrix
    A = build_counts(documents)
    # modify using TF-IDF
    A2 = tfidf(A)
    if return_svs:
        # reduce using SVD
        Aprime, s = reduce_dims(A2, k=k, return_svs=return_svs)
        # create array of reduced vectors
        reduced_doc = []
        for i in range(0, Aprime.shape[0]):
            reduced_doc.append(Aprime[i, :])
        return reduced_doc, s[::-1]
    else:
        # reduce using SVD
        Aprime = reduce_dims(A2, k=k, return_svs=return_svs)
        # create array of reduced vectors
        reduced_doc = []
        for i in range(0, Aprime.shape[0]):
            reduced_doc.append(Aprime[i, :])
        return reduced_doc