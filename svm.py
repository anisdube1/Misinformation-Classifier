from sklearn import *
import numpy as np
import os
from main import *

def support_vector_machine(train, label, test, c):
    clf = svm.SVC(kernel='linear', C=c)
    cross_validation_print(clf, train, label, 10)
    clf.fit(train,label)
    labels = clf.predict(test)
    f = open('svm_2.txt', 'w')
    for l in  labels:
        f.write(str(int(l))+"\n")
    return labels

def svm_linear(train, label, test, value):
    lin_clf = svm.SVC(kernel = value)
    lin_clf.fit(train, label)
    return lin_clf.predict(test)

def svm_tune():
    param_grid = [
         {'C': [0.01, 0.05, 0.1, 1], 'kernel': ['linear']},
         {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, param_grid)
    cross_validation_print(clf, train, label, 10)