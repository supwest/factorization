import numpy as np
import scipy.sparse as sps
import theano.tensor as T
import theano
from theano import sparse

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


    
    from theano import sparse
    x = sparse.csr_matrix(name='x', dtype='float32')
    z = sparse.csr_matrix(name='z', dtype='float32')
    y = sparse.structured_dot(x, z)
    f = theano.function([x,z], y)
    n = sps.csr_matrix(np.array([[1,0], [0,3]], dtype='float32'))
    a = f(n, n)
    n.toarray()
    a.toarray()
    k = 4 #number of latent features
    #u = theano.shared(np.random.rand(n.shape[0], k))
    #v = theano.shared(np.random.rand(k, n.shape[1])

    Xv = np.array([[1,0,0], [1,0,0], [0,0,1], [0,0,0]], dtype="float32")
    Wv = np.random.rand(4,2).astype("float32")
    Vv = np.random.rand(2,4).astype("float32")
    X = sparse.csr_from_dense(Xv)
    #X = sps.coo_matrix(Xv)
    #X = X.tocsr()
    #X = sparse.csr_matrix(X)
    W = theano.shared(Wv)
    V = theano.shared(Vv)
    h = T.dot(W, V)
    cost = sparse.sub(X, h)

    dw, dv = theano.grad(cost.sum(), [W, V])
    print Xv
    print dw.eval()
    print dv.eval()
    train = theano.function(inputs = [X], outputs = cost, updates = ((W, W-dw), (V, V-dv)))

    for i in range(10):
        print i
        #X = X.toarray()
        train(X)

    print W
    print V

    #cost = sparse.basic.sp_sum(x)
    #T.grad(cost, x)

    a = np.array([[1, 0, 0], [2, 0, 1], [0, 1, 0]])
    s = sps.csr_matrix(a)
    u = np.random.rand((3,3))
    d = (s-u)[s.nonzero()]
    d = np.array(d).reshape(s.nonzero())
    e = sps.csr_matrix((d, s.nonzero(), shape=s.get_shape())
