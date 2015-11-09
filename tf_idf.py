import scipy.sparse as ss
import numpy as np

def tf_idf(combine, tf):
    size = combine.shape[0]
    matrix = combine.tocsc()
    inverse_df = np.log(size / np.diff(matrix.indptr))
    idf_matrix = ss.lil_matrix((len(inverse_df),len(inverse_df)))
    idf_matrix.setdiag(inverse_df)
    tfidf = tf * idf_matrix
    return tfidf