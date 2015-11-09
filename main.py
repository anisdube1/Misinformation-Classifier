from input import *
from svm import *
from randomForest import *
from naiveBayes import *
from tf_idf import *
from baggingClassifier import *
from adaboost import *

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import scipy.sparse as ss
import numpy as np
import sys

def pca_preprocess(train, matrix):
    n_components = 100
    pca = PCA(n_components)
    pca.fit(train.todense())
    return pca.transform(matrix.todense())

def lca(train, matrix):
    lsa = TruncatedSVD(n_components=100)
    lsa.fit(train)
    return lsa.transform(matrix)

def cross_validation_print(model, train, label, val):
    scores = cross_validation.cross_val_score(model, train, label, cv=val)
    print("Scores", str(scores), "mean", str(np.mean(scores)))

def input_preprocess(training_matrix, testing_matrix):
    combine = ss.csc_matrix(np.vstack([training_matrix.todense(),testing_matrix.todense()]))
    zero_cols = np.where(np.array(combine.sum(0)).flatten())[0]
    training_matrix = training_matrix[:,zero_cols]
    testing_matrix = testing_matrix[:,zero_cols]
    combine = combine[:,zero_cols]
    return training_matrix, testing_matrix, combine

lca_flag =0 ; pca_flag = 0; svm_flag = 0; adaboo_flag = 0; rf_flag = 0; nb_flag =0; bagging_flag = 0;

if __name__ == "__main__":
    a = raw_input("Enter the arguments ")
    a = a.split()
    if "lca" in a[0]:
        lca_flag = 1
    elif "pca" in a[0]:
        pca_flag = 1
    if "svm" in a[1]:
        svm_flag = 1
    elif "adaboo" in a[1]:
        adaboo_flag = 1
    elif "rf" in a[1]:
        rf_flag = 1
    elif "nb" in a[1]:
        nb_flag = 1
    elif "bagging" in a[1]:
        bagging_flag = 1
    
    training_matrix = input("fall_2015_training.txt", 1842, 26364)
    testing_matrix = input("fall_2015_testing.txt", 952, 26364)
    training_label = label("fall_2015_label_training.txt", 1842)
    
    #Removing all columns which are zero
    #training_matrix = training_matrix[:,np.unique(training_matrix.nonzero()[1])]
    #testing_matrix = testing_matrix[:,np.unique(testing_matrix.nonzero()[1])]
    #X_test = X_test[:,np.unique(X_test.nonzero()[1])]
    
    #training_matrix, X_test, training_label, y_test = train_test_split(training_matrix, training_label, random_state=0)
    
    training_matrix, testing_matrix, combine = input_preprocess(training_matrix, testing_matrix)
    #Getting tf-idf of the matrix
    tf_idf_combine =  tf_idf(combine, combine)
    tf_idf_training_matrix = tf_idf(combine, training_matrix)
    tf_idf_testing_matrix = tf_idf(combine, testing_matrix)
    
    if (lca_flag):
        print("Doing LCA")
        train = lca(tf_idf_combine, tf_idf_training_matrix)
        test = lca(tf_idf_combine, tf_idf_testing_matrix)
    elif (pca_flag):
        print("Doing PCA")
        train = pca_preprocess(tf_idf_combine, tf_idf_training_matrix)
        test = pca_preprocess(tf_idf_combine, tf_idf_testing_matrix)
    else:
        print("Doing No Reduction")
        train = tf_idf_training_matrix
        test = tf_idf_testing_matrix
    
    if (svm_flag):
        print("Doing SVM")
        ans = support_vector_machine(train, training_label, test, 0.01)
    elif (adaboo_flag):
        print("Doing Adaboost")
        ans = adaboo(train, training_label, test)
    elif (rf_flag):
        print("Doing RF")
        ans = r_forest(train, training_label, test)
    elif (nb_flag):
        print("Doing Naive Bayes")
        ans = nb(train, training_label, test)
    elif (bagging_flag):
        print("Doing Bagging")
        ans =  bagging(train, training_label, test)
    else:
        print("Invalid argument specified")


    reverse_check = 0
    if (reverse_check):
        train = input("fall_2015_testing.txt", 952, 26364)
        test = input("fall_2015_training.txt", 1842, 26364)
        train_label = label("/home/anish/Desktop/Compare/output_svm_0.01_reduced_lca.txt", 952)
        test_label = label("fall_2015_label_training.txt", 1842)
        training_matrix, testing_matrix, combine = input_preprocess(train, test)
        tf_idf_combine =  tf_idf(combine, combine)
        tf_idf_training_matrix = tf_idf(combine, training_matrix)
        tf_idf_testing_matrix = tf_idf(combine, testing_matrix)
        train = lca(tf_idf_combine, tf_idf_training_matrix)
        test = lca(tf_idf_combine, tf_idf_testing_matrix)
        ans = support_vector_machine(train, train_label, test, 0.01)
        cm = confusion_matrix(test_label, ans)
        print("Confusion Matrix", cm)
