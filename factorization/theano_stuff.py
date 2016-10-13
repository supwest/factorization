import numpy as np
import pandas as pd
import scipy.sparse as sps
import theano.tensor as T
import theano
from theano import sparse










if __name__ == '__main__':
    mat = np.load('small.npy')
    df = pd.DataFrame(mat).pivot(index=0, columns=1, values=2).fillna(0.0)

