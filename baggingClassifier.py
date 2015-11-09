from sklearn import *
import numpy as np
from main import *

def bagging(matrix, label, test):
    clf = ensemble.BaggingClassifier(bootstrap = False)
    cross_validation_print(clf, matrix, label, 10)
    clf = clf.fit(matrix, label)
    labels= clf.predict(test)
    f = open('bagging.txt', 'w')
    for l in  labels:
        f.write(str(int(l))+"\n")
    return labels