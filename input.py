import scipy.sparse as ss
import numpy as np

def input(file, row, col):
    matrix = ss.lil_matrix((row, col));
    input = open(file, "r")
    for row in input:
        doc_pos_fea = row.split()
        matrix[(int(doc_pos_fea[0]) -1, int(doc_pos_fea[1])-1)] = float(doc_pos_fea[2])
    return matrix

def label(file, row):
    label_array = np.zeros(row)
    input = open(file, "r")
    i=0
    for row in input:
        label_array[i] = int(row)
        i=i+1
    return label_array
