import pickle

import numpy as np

from Distance import Distance
from Part1.Knn import KNN

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))


def create_example_dataset(dataset, labels):
    # lets create 9 different model for our project with different distance parameter and different K value
    exp_1 = KNN(dataset, labels, Distance.calculateMinkowskiDistance, 2, 5)
    exp_2 = KNN(dataset, labels, Distance.calculateMinkowskiDistance, 2, 10)
    exp_3 = KNN(dataset, labels, Distance.calculateMinkowskiDistance, 2, 30)
    exp_4 = KNN(dataset, labels, Distance.calculateMahalanobisDistance, np.linalg.inv(np.cov(dataset.T)), 5)
    exp_5 = KNN(dataset, labels, Distance.calculateMahalanobisDistance, np.linalg.inv(np.cov(dataset.T)), 10)
    exp_6 = KNN(dataset, labels, Distance.calculateMahalanobisDistance, np.linalg.inv(np.cov(dataset.T)), 30)
    exp_7 = KNN(dataset, labels, Distance.calculateCosineDistance, K=5)
    exp_8 = KNN(dataset, labels, Distance.calculateCosineDistance, K=10)
    exp_9 = KNN(dataset, labels, Distance.calculateCosineDistance, K=30)

    # Now it is time to add them examples to the list
    knn_list = [exp_1, exp_2, exp_3, exp_4, exp_5, exp_6, exp_7, exp_8, exp_9]
    return knn_list

experiment_names = ["MinkowskiDistance K=5", "MinkowskiDistance K=10", "MinkowskiDistance K=30",
                    "MahalanobisDistance K=5", "MahalanobisDistance K=10", "MahalanobisDistance K=30",
                    "CosineDistance K=5", "CosineDistance K=10", "CosineDistance K=30"]


# Shuffles the dataset at each split
skf = StratifiedKFold(n_splits=10, shuffle=True)

# keep all 45 accuracies in a list
keep_all_accuracies = []
# we have to calculate the mean for every 10 fold dataset and we have to do it 5 times for our each model
for j in range(9):
    for i in range(5):
        # for each fold we have to keep the value in a list and then calculate the mean and then add main list
        fold_accuracies = []
        # for each fold, calculate the accuracy
        for train_index, test_index in skf.split(dataset, labels):
            # lets get the label train and label test also dataset train and dataset test
            dataset_train = dataset[train_index]
            dataset_test = dataset[test_index]
            label_train = labels[train_index]
            label_test = labels[test_index]
            keep_all_knn = create_example_dataset(dataset_train, label_train)
            # lets create a model list for our predictions
            model_predict = []
            for instance in dataset_test:
                model_predict.append(keep_all_knn[j].predict(instance))
            # lets calculate the accuracy score for our model and add it to the list
            fold_accuracies.append(accuracy_score(label_test, model_predict))
        # find the mean of the accuracies and add it to the main list
        keep_all_accuracies.append(np.mean(fold_accuracies))

# calculate confidence interval for each classifier and then print it
for i in range(0, len(keep_all_accuracies), 5):
    confidence_interval_list = [keep_all_accuracies[i], keep_all_accuracies[i + 1], keep_all_accuracies[i + 2], keep_all_accuracies[i + 3], keep_all_accuracies[i + 4]]
    print("Confidence interval for " + str(experiment_names[i // 5])  + ": " + str(100 * np.mean(confidence_interval_list)) + " +/- " + str(100 * np.std(confidence_interval_list)))

# calculate the best configuration for our model
max_mean = 0
best_classifier = 0
for i in range(0, len(keep_all_accuracies), 5):
    confidence_interval_list = [keep_all_accuracies[i], keep_all_accuracies[i + 1], keep_all_accuracies[i + 2],
                                keep_all_accuracies[i + 3], keep_all_accuracies[i + 4]]
    # find best classifier
    if np.mean(confidence_interval_list) > max_mean:
        max_mean = np.mean(confidence_interval_list)
        best_classifier = i // 5 + 1

print("Best classifier is " + str(best_classifier))












