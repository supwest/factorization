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
        self.bias = np.random.rand()
        self.fitted = False

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

    def fit(self, num_iter=5, user_learning_rate=.0001, item_learning_rate=.0001, verbose=True):
        '''
        Uses GD to find the UV decomposition of the matrix
        Input: num_iter (int) number of iterations to complete
               user_learning_rate (float) learning rate for U
               item_learning_rate (float) learning rate for V
               verbose (bool) whether to print iteration and error
        Output: none
        '''
        X = T.fmatrix('X')
        X_s = sparse.csr_matrix('Xs')
        Uu = theano.shared(self.U)
        Vv = theano.shared(self.V)
        #Bb = theano.shared(self.bias)
        cost =  T.sum(T.square((X - T.dot(Uu, Vv))[self.non_zeros]))
        #cost_s = Bb + sparse.sp_sum(sparse.sqr(sparse.csr_from_dense(X_s - T.dot(Uu, Vv))))
        du, dv = theano.grad(cost, [Uu, Vv])
        #du, dv, db = theano.grad(cost, [Uu, Vv, Bb])
        train = theano.function(inputs = [X], outputs = cost, updates = ((Uu, Uu-user_learning_rate*du), (Vv, Vv-item_learning_rate*dv)))
        #train = theano.function(inputs = [X], outputs = cost, updates = ((Uu, Uu-user_learning_rate*du), (Vv, Vv-item_learning_rate*dv), (Bb, Bb-0.0001*db)))

        print "Training"
        for i in range(num_iter):
            train(self.matrix.toarray().astype('float32'))
            #train(self.matrix)
            if verbose:
                print "Iteration {0} of {1}".format(i+1, num_iter)
                self._print_error(Uu, Vv) 
        self.U = Uu.get_value()
        self.V = Vv.get_value()
        self.fitted = True
        
    
    def fit_funk_svd(self, num_iter=1, fresh_start=False):
        '''
        use alternating least squares to find U and V
        Inputs: 
            num_iter (int) number of iterations
            fresh_start (bool) if True, initialize U and V as random matrices
        Output:
            None

        '''
        n_users = self.matrix.shape[0]
        n_items = self.matrix.shape[1]
        if fresh_start:
            self.U = np.random.rand(n_users, self.factors)
            self.V = np.random.rand(self.factors, n_items)
        #bias = np.random.rand()
        #w_i = np.random.rand(n_users)
        #w_j = np.random.rand(n_items)
        for iteration in xrange(num_iter):
            print "Iteration {}".format(iteration+1)

            #for u in xrange(n_users):
            #     self.U[u] = sps.linalg.lsqr(self.V.T, self.matrix[u].T)
            # for v in xrange(n_users):
            #     self.V[v] = sps.linalg.lsqr(self.U, self.matrix.T[v])
            for i in xrange(n_users):
                print "user {}".format(i+1)
                for j in xrange(n_items):
                    if self.matrix[i,j] > 0:
                        pred = self.bias + np.dot(self.U[i, :], self.V[:,j])
                        #print pred
                        err = self.matrix[i, j] - pred
                        #print err
                        for k in xrange(self.factors):
                            self.U[i,k] += 0.001*(2*err*self.V[k,j])
                            self.V[k,j] = self.V[k,j] + 0.001*(2*err*self.U[i,k])
                self.bias = self.bias + 0.1*err
                            
                            
    def predict_one(self, userid):
        '''
        returns predictions for one user
        
        Input: userid (int) id of user to provide predictions for
        Output: numpy array of predicted ratings
        '''
        if not self.fitted:
            print "Warning: fit has not been called. Predictions will be random"

        return np.dot(self.U[userid], self.V)

    def recommend_n(self, userid, n=10):
        '''
        recommend n items for one user

        Input: 
            userid (int) id of user to provide recommendations for
            n (int) number of items to recommend, default 10    
        Output:
            numpy array of indexes of top n items
        '''
        if not self.fitted:
            print "Warning: fit has not been called. Recommendations will be random"
        return np.argsort(self.predict_one(userid))[::-1][:n]


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

def make_sparse_matrix(mat_file):
    '''
    makes a sparse matrix from a ratings matrix
    Input: path to numpy array

    array should consist of:
    userids (ints) in first column
    item ids (ints) in second column
    ratings (float or int) in third column
    '''

    mat = np.load(mat_file)

    users = mat[:,0]
    items = mat[:,1]
    ratings = mat[:,2].astype(float)

    if users.min() != 0:
        users = users - users.min()
    if items.min() != 0:
        items = items - items.min()

    s = sps.coo_matrix((ratings, (users, items)), shape=(np.unique(users).shape[0], np.unique(items).shape[0])).tocsr()

    return s

if __name__ == '__main__':
    #sparse_matrix = make_sparse_matrix('matrix.npy')

    #fact = Factorizer(sparse_matrix)
    m = make_sparse_matrix('matrix.npy')
    #m = make_sparse_matrix('small.npy')
    m_fact = Factorizer(m, num_factors=10)
    #m_fact.fit()
    
