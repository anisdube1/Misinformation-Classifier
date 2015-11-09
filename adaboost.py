import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
from main import *

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


def adaboo(train, label, test):
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=1000)
    bdt.fit(train, label)
    labels =  bdt.predict(test)
    cross_validation_print(bdt, train, label, 10)
    f = open('adaboost.txt', 'w')
    for l in  labels:
        f.write(str(int(l))+"\n")
    return labels
