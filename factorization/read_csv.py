import os, csv
import numpy as np


def read_big_csv(fileName,delimiter=",",asnumeric=[]):
    if not os.path.exists(fileName):
        print("... %s"%fileName)
        raise Exception("cannot find results file")

    ## get the number of lines                                                                                                                                                                      
    linecount = 0
    fid = open(fileName,"r")
    reader = csv.reader(fid,delimiter=",")
    header = reader.next()

    for linja in reader:
        linecount += 1
    fid.close()

    fid = open(fileName,"r")
    reader = csv.reader(fid,delimiter=",")
    header = reader.next()

    ## create an empty matrix                                                                                                                                                                       
    mat = np.zeros((linecount,len(header)),)

    ## read in results file                                                                                                                                                                         
    row = -1
    for linja in reader:
        row += 1
        for col,val in enumerate(linja):
            mat[row,col] = val

    fid.close()

    return mat


if __name__ == '__main__':
    mat = read_big_csv('../data/ratings.csv')
    np.save('matrix.npy', mat)
