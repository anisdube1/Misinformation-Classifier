from sklearn.naive_bayes import *
from sklearn import cross_validation
import numpy as np
from main import *

def nb(train, label, test):
    clf = MultinomialNB().fit(train, label)
    #clf = GaussianNB()
    cross_validation_print(clf, train, label, 10)
    clf = clf.fit(train, label)
    labels = clf.predict(test)
    f = open('nb.txt', 'w')
    for l in  labels:
        f.write(str(int(l))+"\n")
    return labels