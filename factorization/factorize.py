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


    def factorize(self):
        print "factorize"
        u = np.random.rand(self.matrix.shape[0], self.factors)
        v = np.random.rand(self.factors, self.matrix.shape[1])
        print u.shape
        for i in range(2):
            v, _, _, _ = np.linalg.lstsq(u, self.matrix.todense())
            u_temp, _, _, _ = np.linalg.lstsq(v.T, self.matrix.todense().T)
            u = u_temp.T
            print np.sum(np.square(np.dot(u, v)-self.matrix))
        print u
        print v
        print u.shape
        print v.shape
        return u, v
        pass




if __name__ == '__main__':
    mat = np.load('matrix.npy')

    users = mat[:,0]-1
    movies = mat[:,1]-1
    ratings = mat[:,2].astype(float)

    sparse_matrix = sps.coo_matrix((ratings, (users, movies)), shape=(np.unique(users).shape[0]+1, movies.shape[0]+1)).tocsr()

    fact = Factorizer(sparse_matrix)
    #u, v = fact.factorize()
    
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

    #Xv = np.array([[1,0,0], [1,0,0], [0,0,1], [0,0,0]], dtype="float32")
    Xv = sparse_matrix
    Xv_dense = sparse_matrix.toarray().astype('float32')
    Wv = np.random.rand(sparse_matrix.shape[0], 5).astype("float64")
    Vv = np.random.rand(5,sparse_matrix.shape[1]).astype("float64")
    X_dense = T.fmatrix('X')
    X = sparse.csr_matrix(name='X')
    W = theano.shared(Wv)
    V = theano.shared(Vv)
    h = T.dot(W, V)

    diff = sparse.sub(X,h)
    
    dense_cost = T.sum(T.square(T.sub(X_dense, T.dot(W, V)))[sparse_matrix.nonzero()])
    #cost = sparse.sp_sum(sparse.sqr(sparse.csr_from_dense(sparse.sub(X,T.dot(W,V)))),sparse_grad=True)
    #cost = sparse.sp_sum(sparse.sqr(X-T.dot(W,V)))

    #dw, dv = theano.grad(cost, [W, V])
    dw, dv = theano.grad(dense_cost, [W, V])
    #train = theano.function(inputs = [X], outputs = cost, updates = ((W, W-0.0001*dw), (V, V-0.0001*dv)))
    train = theano.function(inputs = [X_dense], outputs = dense_cost, updates = ((W, W-0.0001*dw), (V, V-0.0001*dv)))


    for i in range(20):
        print i
        train(Xv_dense)
        
        #train(Xv)
        ww = W.get_value()
        vv = V.get_value()

        print np.mean(np.square(np.dot(ww,vv) - sparse_matrix)[sparse_matrix.nonzero()])
    ww = W.get_value()
    vv = V.get_value()
    print np.dot(ww, vv)
        #print np.sum(np.square(np.dot(ww,vv)-sparse_matrix.toarray())[sparse_matrix.nonzero()])
    a = np.array([[1, 0, 0], [2, 0, 1], [0, 1, 0]])
    s = sps.csr_matrix(a)
    u2 = np.random.rand(3,3)
    d = (s-u2)
    d2 = np.array(d[s.nonzero()])[0]
    row = s.nonzero()[0]
    col = s.nonzero()[1]
    e = sps.csr_matrix((d2, (row, col)), shape=s.shape)


