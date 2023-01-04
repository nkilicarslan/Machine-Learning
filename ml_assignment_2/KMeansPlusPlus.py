import numpy as np
import math
from Distance import Distance
import copy
class KMeansPlusPlus:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class

    def findClosestCenterDistance(self, point):
        """Find the closest center to a point"""
        # calculate the distance nearest datapoint and return the distance

        # take the first center
        initial_center = self.cluster_centers[0]
        # calculate the distance between first center and point
        min_dist = self.calculateEuclidean(initial_center, point)
        # for each center
        for i in range(1, self.K):
            # control the center is not empty
            if self.cluster_centers[i] is not None and len(self.cluster_centers[i]) != 0:
                # calculate the distance between center and point
                new_dist = self.calculateEuclidean(self.cluster_centers[i], point)
                # if distance between is less than distance
                if new_dist < min_dist:
                    # change the distance
                    min_dist = new_dist
        return min_dist

    def calculateEuclidean(self,inst1,inst2):
        take_square = np.square(inst1-inst2)
        sum_square = np.sum(take_square)
        sqrt = np.sqrt(sum_square)
        return sqrt

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        # calculate the loss for each cluster
        # lets create a variable to keep the loss
        loss_value = 0
        # for each cluster do the following
        for i in range(self.K):
            if len(self.clusters[i]) != 0:
                # convert this array to the numpy array
                center_np = np.array(self.cluster_centers[i])
                cluster_np = np.array(self.clusters[i])
                # calculate the distance between cluster center and each point in the cluster
                distance = np.linalg.norm(cluster_np - center_np, axis=1)
                # take the square the distance
                distance_square = np.square(distance)
                # sum the distance square
                distance_square_sum = np.sum(distance_square)
                # add the loss value
                loss_value += distance_square_sum
        return loss_value

    def run(self):
        """Kmeans++ algorithm implementation"""
        # take the dataset count
        dataset_count = self.dataset.shape[0]
        initial_centers_index = np.random.choice(dataset_count, 1, replace=False)
        initial_center = self.dataset[initial_centers_index]
        self.cluster_centers[0] = initial_center[0].tolist()
        # for k times, calculate the distance between each point and nearest center with findClosestCenterDistance
        for i in range(1, self.K):
            # for each point
            keep_all_distnace = np.apply_along_axis(self.findClosestCenterDistance, 1, self.dataset)
            # take the square the distance
            distance_square = np.square(keep_all_distnace)
            # sum the distance square
            distance_square_sum = np.sum(distance_square)
            # take the probability
            probability = distance_square / distance_square_sum
            # chose a random number
            random_number = np.random.choice(dataset_count, 1, replace=False, p=probability)
            # take the data from random number
            random_data = self.dataset[random_number]
            # store the data in cluster centers
            self.cluster_centers[i] = random_data[0].tolist()
        while 1:
            instant_clusters = copy.deepcopy(self.cluster_centers)
            # for each data in dataset do the following
            for i in range(dataset_count):
                # create a var to max distance assign to max value
                min_distance = np.inf
                # lets create a variable in order to keep the min distance index
                min_index = -1
                for j in range(self.K):
                    # calculate distance between each point and each cluster center
                    distance = self.calculateEuclidean(self.dataset[i], self.cluster_centers[j])
                    # if distance is less than min distance
                    if distance < min_distance:
                        min_distance = distance
                        min_index = j
                # check the error is occured or not
                if min_index == -1:
                    print("Error")
                # if all is ok, append the data to the cluster
                else:
                    self.clusters[min_index].append(self.dataset[i])
                # update cluster centers by taking the mean of the points in each cluster
            for k in range(self.K):
                if len(self.clusters[k]) != 0:
                    self.cluster_centers[k] = list(np.sum(self.clusters[k], axis=0) / len(self.clusters[k]))
            # check the stopping criterion
            # if the cluster centers do not change, stop the algorithm
            # otherwise, repeat the above steps
            instant_clusters = np.array(list(instant_clusters.values()), dtype=object)
            new_cluster_centers = np.array(list(self.cluster_centers.values()), dtype=object)
            try:
                if (np.equal(instant_clusters,new_cluster_centers)).all():
                    return self.cluster_centers, self.clusters, self.calculateLoss()
                else:
                    # emtpy the clusters
                    for i in range(self.K):
                        self.clusters[i].clear()
            except:
                print("Error")
