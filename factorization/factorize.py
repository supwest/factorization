import numpy as np
import scipy.sparse as sps
import theano.tensor a T


class Factorizer(object):
    '''
    Class for creating a factorization machine recommender

    '''

    def __init__(self, matrix, num_factors=5, loss='rss'):
        '''
        Input: A numpy sparse matrix
        '''
        self.matrix = matrix
        self.factors = num_factors
        self.loss = loss


    def factorize(self):
        pass




if __name__ == '__main__':
    mat = np.load('matrix.npy')

    users = mat[:,0]
    movies = mat[:,1]
    ratings = mat[:,2]

    sparse_matrix = sps.coo_matrix((ratings, (users, movies)), shape=(np.unique(users).shape[0]+1, movies.shape[0]+1))


    '''
    from theano import sparse
    x = sparse.csr_matrix(name='x', dtype='float32')
    z = sparse.csr_matrix(name='z', dtype='float32')
    y = sparse.structured_dot(x, z)
    f = theano.function([x,z], y)

    n = sps.csr_matrix(np.array([[1,0], [0,3]], dtype='float32'))
    a = f(n, n)
    n.toarray()
    a.toarray()


    cost = sparse.basic.sp_sum(x)
    T.grad(cost, x)
    '''
