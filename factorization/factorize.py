import numpy as np
import scipy.sparse as sps







if __name__ == '__main__':
    mat = np.load('matrix.npy')

    users = mat[:,0]
    movies = mat[:,1]
    ratings = mat[:,2]

    sparse_matrix = sps.coo_matrix((ratings, (users, movies)), shape=(np.unique(users).shape[0]+1, movies.shape[0]+1))

