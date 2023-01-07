import numpy as np
# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label

class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...


    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        """
        Entropy calculations
        """
        # calculate the number of 0 and 1 in labels
        count = np.bincount(labels)
        # calculate the entropy of the labels
        entropy_value = -np.sum([p / len(labels) * np.log2(p / len(labels)) for p in count if p != 0])

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        """
            Average entropy calculations
        """
        #get the attribute index from the features
        attribute_index = self.features.index(attribute)
        #get the unique values of the attribute
        unique_values = np.unique([data[attribute_index] for data in dataset])
        #calculate the average entropy
        for value in unique_values:
            #get the indices of the instances with the value
            indices = np.where([data[attribute_index] == value for data in dataset])[0]
            #calculate the entropy of the instances with the value
            entropy = self.calculate_entropy__([dataset[i] for i in indices], [labels[i] for i in indices])
            #calculate the average entropy
            average_entropy += entropy * len(indices) / len(dataset)


        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain = 0.0
        """
            Information gain calculations
        """
        return self.calculate_entropy__(dataset, labels) - self.calculate_average_entropy__(dataset, labels, attribute)

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        """
            Intrinsic information calculations for a given attribute
        """
        # get the attribute index from the features
        attribute_index = self.features.index(attribute)
        # get the unique values of the attribute
        unique_values = np.unique([data[attribute_index] for data in dataset])
        # iterate over the unique values and calculate the intrinsic information
        intrinsic_information = 0.0
        for value in unique_values:
            # get the indices of the instances with the value
            indices = np.where([data[attribute_index] == value for data in dataset])[0]
            # calculate the intrinsic information
            intrinsic_information += -len(indices) / len(dataset) * np.log2(len(indices) / len(dataset))

        return intrinsic_information

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """
        gain_ratio = 0.0
        gain_ratio = self.calculate_information_gain__(dataset, labels, attribute) / self.calculate_intrinsic_information__(dataset, labels, attribute)

        return gain_ratio


    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
            Your implementation
        """
        # if there are no more instances, return a leaf node
        if len(dataset) == 0:
            return TreeLeafNode(dataset, labels[0])
        # if there are no more attributes to use, return a leaf node
        if len(used_attributes) == len(self.features):
            return TreeLeafNode(dataset, labels[0])
        #if all the labels are the same, return a leaf node
        if len(np.unique(labels)) == 1:
            return TreeLeafNode(dataset, labels[0])


        #calculate the information gain for each attribute
        information_gain = [self.calculate_information_gain__(dataset, labels, attribute) for attribute in self.features]
        #get the index of the attribute with the highest information gain
        attribute_index = np.argmax(information_gain)
        #get the attribute with the highest information gain
        attribute = self.features[attribute_index]
        #if the attribute is already used, return a leaf node
        if attribute in used_attributes:
            return TreeLeafNode(dataset, labels[0])
        #create a non-leaf node
        node = TreeNode(attribute)
        used_attributes.append(attribute)
        #get the unique values of the attribute
        unique_values = np.unique([data[attribute_index] for data in dataset])
        #iterate over the unique values
        for value in unique_values:
            #get the indices of the instances with the value
            indices = np.where([data[attribute_index] == value for data in dataset])[0]
            #create a subtree for the value
            subtree = self.ID3__([dataset[i] for i in indices], [labels[i] for i in indices], used_attributes)
            #add the subtree to the node
            node.subtrees[value] = subtree
        return node

    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        """
            Your implementation
        """
        #get the root node
        root_node = self.root
        #iterate over the tree
        while not isinstance(root_node, TreeLeafNode):
            #get the attribute index
            attribute_index = self.features.index(root_node.attribute)
            #get the value of the attribute
            value = x[attribute_index]
            #get the subtree
            root_node = root_node.subtrees[value]
        predicted_label = root_node.labels

        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")