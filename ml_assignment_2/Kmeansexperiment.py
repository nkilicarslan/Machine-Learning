import numpy as np

from Part2.KMeans import KMeans
import pickle



dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))

dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))

kmeans_list1 = []
kmeans_list2 = []
k_list = [2,3,4,5,6,7,8,9,10]

# for each k value run the kmeans algorithm 100 times
for k in k_list:
    # create a variable assign it to maximum integer value
    min_var1 = np.inf
    min_var2 = np.inf

    for i in range(100):
        # create a kmeans object
        kmeans1 = KMeans(dataset1, k)
        kmeans2 = KMeans(dataset2, k)
        # run the kmeans algorithm
        cluster_centers1, clusters1, calculateLoss1 = kmeans1.run()
        cluster_centers2, clusters2, calculateLoss2 = kmeans2.run()

        if calculateLoss1 < min_var1:
            min_var1 = calculateLoss1

        if calculateLoss2 < min_var2:
            min_var2 = calculateLoss2
        if (i + 1) % 10 == 0:
            kmeans_list1.append(min_var1)
            kmeans_list2.append(min_var2)

keep_mean1 = []
keep_mean2 = []
# Calculates the means and confidence intervals for each K value and prints them
for i in range(len(k_list)):
    # calculate the mean 10 by 10
    mean1 = np.mean(kmeans_list1[i * 10:(i + 1) * 10])
    mean2 = np.mean(kmeans_list2[i * 10:(i + 1) * 10])
    keep_mean1.append(mean1)
    keep_mean2.append(mean2)
    # calculate the confidence interval 10 by 10
    confidence_interval1 = 1.96 * np.std(kmeans_list1[i * 10:(i + 1) * 10]) / np.sqrt(10)
    confidence_interval2 = 1.96 * np.std(kmeans_list2[i * 10:(i + 1) * 10]) / np.sqrt(10)
    # print the results
    print("K = " + str(k_list[i]) + " dataset1: " + str(mean1) + " +- " + str(confidence_interval1))
    print("K = " + str(k_list[i]) + " dataset2: " + str(mean2) + " +- " + str(confidence_interval2))

# plot the results
import matplotlib.pyplot as plt

# plot the results
plt.plot(k_list, keep_mean1, label="dataset1")
plt.xlabel("K")
plt.ylabel("Loss")
plt.title("Loss vs K for dataset1")
plt.show()
plt.plot(k_list, keep_mean2, label="dataset2")
plt.xlabel("K")
plt.ylabel("Loss")
plt.title("Loss vs K for dataset2")
plt.show()
