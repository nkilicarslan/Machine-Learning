import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC

dataset, labels = pickle.load(open("../data/part2_dataset1.data", "rb"))
# create 4 different configurations for the SVM classifier. The configurations are:
# 1. kernel = rbf, C = 1
# 2. kernel = rbf, C = 100
# 3. kernel = linear, C = 1
# 4. kernel = linear, C = 100
# create the all SVM classifier
svm_classifier = SVC(kernel="rbf", C=1)
svm_classifier.fit(dataset, labels)
svm_classifier2 = SVC(kernel="rbf", C=100)
svm_classifier2.fit(dataset, labels)
svm_classifier3 = SVC(kernel="linear", C=1)
svm_classifier3.fit(dataset, labels)
svm_classifier4 = SVC(kernel="linear", C=100)
svm_classifier4.fit(dataset, labels)
# add them to a list
svm_list = [svm_classifier, svm_classifier2, svm_classifier3, svm_classifier4]
# iterate over the list and plot the decision boundary for each classifier
# create the title list
title_list = ["kernel = rbf, C = 1", "kernel = rbf, C = 100", "kernel = linear, C = 1", "kernel = linear, C = 100"]
for svm in svm_list:
    # create a meshgrid
    x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    y_min, y_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # plot the decision boundary
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, s=20, edgecolor='k')
    plt.title(title_list[svm_list.index(svm)])
    plt.show()



