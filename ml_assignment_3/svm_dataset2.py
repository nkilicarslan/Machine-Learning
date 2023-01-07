import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC


dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

# create a stratified k-fold object with 5 splits
skf = StratifiedKFold(n_splits=10, shuffle=True)
# create a list of parameters to test
parameters = [{"kernel": ["linear", "sigmoid"], "C": [1, 100]}]
configurations = ["kernel: linear C: 1", "kernel: linear C: 100", "kernel: sigmoid C: 1", "kernel: sigmoid C: 100"]
# create a result list
result_list = []

# repeat the cross validation 5 times and store the mean results and then calculate the confidence interval
for i in range(5):
    # create a grid search object
    clf = GridSearchCV(SVC(), parameters, cv=skf, scoring="accuracy")
    # preprocess the data
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    # fit the data
    clf.fit(dataset, labels)
    # take the cv_results_ mean score and store it with parameters
    result_list.append([clf.cv_results_["mean_test_score"], clf.cv_results_["params"]])
result_for_the_std = []
result_with_mean = []
parameters = []
#
# iterater over the result list and calculate the mean and confidence interval
for result in range(4):
    result_with_mean.append((result_list[0][0][3-result] + result_list[1][0][3-result] + result_list[2][0][3-result] + result_list[3][0][3-result] + result_list[4][0][3-result]) / 5.0)
    parameters.append(result_list[0][1][3-result])
    result_for_the_std.append([result_list[0][0][3-result], result_list[1][0][3-result], result_list[2][0][3-result], result_list[3][0][3-result], result_list[4][0][3-result]])

# print the results with mean and std and confidence interval with using result_with_mean and parameters
for i in range(4):
    #calculate the confidence interval
    confidence_interval = 1.96 * np.std(result_for_the_std[i]) / np.sqrt(5)
    print("The mean accuracy for " + configurations[i] + " is " + str(result_with_mean[i]) + " with a confidence interval of " + str(result_with_mean[i]) +" +/- " + str(confidence_interval))







