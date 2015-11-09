from sklearn.ensemble import *
from sklearn import tree
from sklearn import cross_validation
import numpy as np
from main import *

def r_forest(matrix, label, test):
    clf = RandomForestClassifier(n_estimators=100)
    # return rf.predict(test)
    #clf = tree.DecisionTreeClassifier()
    cross_validation_print(clf, matrix, label, 10)
    clf = clf.fit(matrix, label)
    labels= clf.predict(test)
    f = open('rf.txt', 'w')
    for l in  labels:
        f.write(str(int(l))+"\n")
    return labels