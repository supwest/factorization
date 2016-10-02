import numpy as np
import scipy.sparse as sps
import theano.tensor as T
import theano
from theano import sparse
import pdb

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
        self.U = np.random.rand(self.matrix.shape[0], self.factors)
        self.V = np.random.rand(self.factors, self.matrix.shape[1])
        self.non_zeros = self.matrix.nonzero()
        self.mean_rating = self.matrix[self.non_zeros].mean()
        self.user_means = self._get_user_means()

    def _get_user_means(self):
        '''
        Find the mean of the user ratings
        Input: None
        Output: np.array of means
        '''
        m = np.zeros(self.matrix.shape[0])
        for i in range(self.matrix.shape[0]):
            m[i] = self.matrix[self.matrix.getrow(i).nonzero()].mean()
        return m

    def factorize(self, num_iter=5, verbose=True):
        '''
        Uses GD to find the UV decomposition of the matrix
        Input: num_iter (int) number of iterations to complete
               verbose (bool) whether to print iteration and error
        Output: none
        '''

        X = T.fmatrix('X')
        Uu = theano.shared(self.U)
        Vv = theano.shared(self.V)

        cost = T.sum(T.square(X - T.dot(Uu, Vv))[self.non_zeros])

        du, dv = theano.grad(cost, [Uu, Vv])
        train = theano.function(inputs = [X], outputs = cost, updates = ((Uu, Uu-0.0001*du), (Vv, Vv-0.0001*dv)))

        print "Training"
        for i in range(num_iter):
            train(self.matrix.toarray().astype('float32'))
            if verbose:
                print "Iteration {0} of {1}".format(i+1, num_iter)
                self._print_error(Uu, Vv) 
        self.U = Uu.get_value()
        self.V = Vv.get_value()
        

    def _print_error(self, Uu, Vv):
        '''
        Prints the error between the estimate and the real ratings
        Input: U (np.array), V (np.array)
        Output: None
        '''
        est = np.dot(Uu.get_value(), Vv.get_value())
        err = self.matrix - est
        err = err[self.non_zeros]
        print "Error: {}".format(np.mean(np.square(err)))


class FactorizationMachine(object):
    '''
    Class to create a Factorization Machine Recommender


    '''

    def __init__(self):
        pass



if __name__ == '__main__':
    mat = np.load('matrix.npy')

    users = mat[:,0]-1
    movies = mat[:,1]-1
    ratings = mat[:,2].astype(float)

    sparse_matrix = sps.coo_matrix((ratings, (users, movies)), shape=(np.unique(users).shape[0], movies.shape[0])).tocsr()

    fact = Factorizer(sparse_matrix)
    
    a = np.array([[1, 0, 0], [2, 0, 1], [0, 1, 0]])
    s = sps.csr_matrix(a)
    u2 = np.random.rand(3,3)
    d = (s-u2)
    d2 = np.array(d[s.nonzero()])[0]
    row = s.nonzero()[0]
    col = s.nonzero()[1]
    e = sps.csr_matrix((d2, (row, col)), shape=s.shape)


