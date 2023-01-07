import numpy as np
from DataLoader import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import time

data_path = "../data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)

knn_config = {"n_neighbors": [1, 5, 10], "weights": ["uniform"]}
svm_config = {"kernel": ["rbf"], "C": [1, 10 ,100]}
dt_config = {"max_depth": [1, 5, 10], "criterion": ["gini"]}
rf_config = {"n_estimators": [10], "max_depth": [1, 5, 10]}
# add all the configurations to a list
config_list = [knn_config, svm_config, dt_config, rf_config]
# create a list of classifiers
classifier_list = [KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), RandomForestClassifier()]
knn_hyper_res = [[], [], []]
svm_hyper_res = [[], [], []]
dt_hyper_res = [[], [], []]
rf_hyper_res = [[], [], []]
hyper_res = [knn_hyper_res, svm_hyper_res, dt_hyper_res, rf_hyper_res]
knn_test_score = []
svm_test_score = []
dt_test_score = []
rf_test_score = []
test_score = [knn_test_score, svm_test_score, dt_test_score, rf_test_score]
knn_f1_score = []
svm_f1_score = []
dt_f1_score = []
rf_f1_score = []
f1_scores = [knn_f1_score, svm_f1_score, dt_f1_score, rf_f1_score]


# create a stratified k-fold for inner part 3 splits 5 repetitions
skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
for train_index, test_index in skf.split(dataset, labels):
    # split the dataset and labels into training and testing
    X_train, X_test = dataset[train_index], dataset[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # do the min-max normalization
    min = np.min(X_train)
    max = np.max(X_train)
    # normalize the training data
    X_train = (X_train - min) * 2 / (max - min) -1
    # normalize the testing data
    X_test = (X_test - min) * 2 / (max - min) -1

    # for outer part 5 splits 5 repetitions
    skf2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    for config in config_list:
        if config == rf_config:
            rff_tmp = []
            for i in range(5):
                # create a grid search for the classifier
                grid = GridSearchCV(classifier_list[config_list.index(config)], config, cv=skf2, scoring="accuracy")
                # fit the grid search to the training data
                grid.fit(X_train, y_train)
                rff_tmp.append(grid.cv_results_["mean_test_score"])
        else:
            # create a grid search for the classifier
            grid = GridSearchCV(classifier_list[config_list.index(config)], config, cv=skf2, scoring="accuracy")
            # fit the grid search to the training data
            grid.fit(X_train, y_train)
            # append the grid.cv_results_mean_test_score to the hyper_res list for each configuration
        for i in range(3):
            hyper_res[config_list.index(config)][i].append(grid.cv_results_["mean_test_score"][i])
        # evaluate the classifier for best parameters on the test data
        css = ""
        if config == rf_config:
            rff_tmp = np.array(rff_tmp)
            rff_tmp = np.mean(rff_tmp, axis=0)
            # take the best parameters from the grid search with maximum of the rff_tmp
            best_params = grid.cv_results_["params"][np.argmax(rff_tmp)]
            tmp = None
            if type(grid.best_estimator_) == RandomForestClassifier:
                tmp = RandomForestClassifier()
            css = tmp.set_params(**best_params)
        else:
            if type(grid.best_estimator_) == KNeighborsClassifier:
                tmp = KNeighborsClassifier()
            elif type(grid.best_estimator_) == SVC:
                tmp = SVC()
            elif type(grid.best_estimator_) == DecisionTreeClassifier:
                tmp = DecisionTreeClassifier()
            css = tmp.set_params(**grid.best_params_)
        css.fit(X_train, y_train)
        test_score[config_list.index(config)].append(css.score(X_test, y_test))
        f1_scores[config_list.index(config)].append(f1_score(y_test, css.predict(X_test)))

# iterate over the hyper_res list and calculate the mean and std for each configuration and print them
print("************************** **********************************")
print("Configuartion accuracy")
for i in range(len(hyper_res)):
    print("Classifier: ", classifier_list[i])
    for j in range(len(hyper_res[i])):
        # print the configuration in a nice way
        if i == 0:
            print("KNN: n_neighbors = ", knn_config["n_neighbors"][j], ", weights = ", knn_config["weights"][0])
        elif i == 1:
            print("SVM: kernel = ", svm_config["kernel"][0], ", C = ", svm_config["C"][j])
        elif i == 2:
            print("Decision Tree: max_depth = ", dt_config["max_depth"][j], ", criterion = ", dt_config["criterion"][0])
        elif i == 3:
            print("Random Forest: n_estimators = ", rf_config["n_estimators"][0], ", max_depth = ", rf_config["max_depth"][j])
        print("Mean: ", np.mean(hyper_res[i][j]))
        print("Std: ", np.std(hyper_res[i][j]))
        print("Confidence Interval: ", 1.96 * np.std(hyper_res[i][j]) / np.sqrt(len(hyper_res[i][j])))
        print("")
# iterate over the test_score list and calculate the mean and std for each configuration and print them
print("************************** **********************************")
print("Test accuracy")
for i in range(4):
    print("Classifier: ", classifier_list[i])
    print("Mean: ", np.mean(test_score[i]))
    print("Std: ", np.std(test_score[i]))
    print("Confidence Interval: ", 1.96 * np.std(test_score[i]) / np.sqrt(len(test_score[i])))
    print("")
# iterate over the f1_scores list and calculate the mean and std for each configuration and print them
print("************************** **********************************")
print("F1 score accuracy")
for i in range(4):
    print("Classifier: ", classifier_list[i])
    print("Mean: ", np.mean(f1_scores[i]))
    print("Std: ", np.std(f1_scores[i]))
    print("Confidence Interval: ", 1.96 * np.std(f1_scores[i]) / np.sqrt(len(f1_scores[i])))
    print("")


