import numpy as np


class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters



    def predict(self, instance):
        # calculate distances between instance and all samples in the dataset
        if self.similarity_function_parameters is not None:
            distances = [self.similarity_function(instance, sample, self.similarity_function_parameters) for sample in
                         self.dataset]

        elif self.similarity_function_parameters is None:
            distances = [self.similarity_function(instance, sample) for sample in self.dataset]
        # sort the distances and get the indices of the K nearest neighbors
        '''find the instance from list then remove and label it
        index = np.where(distances == 0)
        distances = np.delete(distances, index)
        self.dataset_label = np.delete(self.dataset_label, index)'''
        # sort distances and get indices of K nearest neighbors
        nearest_neighbors = np.argsort(distances)[:self.K]
        # remove first element from list, because it is the instance itself
        # get labels of K nearest neighbors
        nearest_neighbors_labels = self.dataset_label[nearest_neighbors]
        # get most frequent label
        # remove first element from list
        (values, counts) = np.unique(nearest_neighbors_labels, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]






