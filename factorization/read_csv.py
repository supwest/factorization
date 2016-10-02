import os, csv
import numpy as np


def read_big_csv(fileName,delimiter=",",asnumeric=[]):
    if not os.path.exists(fileName):
        print("... %s"%fileName)
        raise Exception("cannot find results file")

    ## get the number of lines                                                                                                                                                                      
    linecount = 0
    fid = open(fileName,"r")
    reader = csv.reader(fid,delimiter=delimiter)
    header = reader.next()

    for linja in reader:
        linecount += 1
    fid.close()
    print "Linecount: {}".format(linecount)
    fid = open(fileName,"r")
    reader = csv.reader(fid,delimiter=delimiter)
    header = reader.next()

    ## create an empty matrix                                                                                                                                                                       
    #mat = np.zeros((linecount,len(header)),)
    
    ## linecount+1 because u.data has no header
    mat = np.zeros((linecount+1, len(header)),) 

    ## read in results file                                                                                                                                                                         
    row = -1
    for linja in reader:
        row += 1
        for col,val in enumerate(linja):
            mat[row,col] = val

    fid.close()

    return mat


if __name__ == '__main__':
    #mat = read_big_csv('../data/ratings.csv')
    mat = read_big_csv('../data/u.data', delimiter='\t')
    np.save('matrix.npy', mat)
