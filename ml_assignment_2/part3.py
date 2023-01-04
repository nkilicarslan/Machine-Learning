import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

dataset = pickle.load(open("../data/part3_dataset.data", "rb"))



# create example dataset for our experiment. We will use 2 different distance functions and 2 different linkage methods for 4
# different cluster numbers
def create_example_dataset(dataset):
    # lets create 4 different model for our project with given spesiifc parameters
    exp_1 = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single', compute_distances=True)
    exp_2 = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete', compute_distances=True)
    exp_3 = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='single', compute_distances=True)
    exp_4 = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='complete', compute_distances=True)
    exp_5 = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single', compute_distances=True)
    exp_6 = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete', compute_distances=True)
    exp_7 = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='single', compute_distances=True)
    exp_8 = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='complete', compute_distances=True)
    exp_9 = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single', compute_distances=True)
    exp_10 = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete', compute_distances=True)
    exp_11 = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='single', compute_distances=True)
    exp_12 = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='complete', compute_distances=True)
    exp_13 = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single', compute_distances=True)
    exp_14 = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete', compute_distances=True)
    exp_15 = AgglomerativeClustering(n_clusters=5, affinity='cosine', linkage='single', compute_distances=True)
    exp_16 = AgglomerativeClustering(n_clusters=5, affinity='cosine', linkage='complete', compute_distances=True)
    # Fit the hierarchical clustering for each model
    exp_1.fit(dataset)
    exp_2.fit(dataset)
    exp_3.fit(dataset)
    exp_4.fit(dataset)
    exp_5.fit(dataset)
    exp_6.fit(dataset)
    exp_7.fit(dataset)
    exp_8.fit(dataset)
    exp_9.fit(dataset)
    exp_10.fit(dataset)
    exp_11.fit(dataset)
    exp_12.fit(dataset)
    exp_13.fit(dataset)
    exp_14.fit(dataset)
    exp_15.fit(dataset)
    exp_16.fit(dataset)
    # Now it is time to add them examples to the list
    agglomerative_list = [exp_1, exp_2, exp_3, exp_4, exp_5, exp_6, exp_7, exp_8, exp_9, exp_10, exp_11, exp_12, exp_13, exp_14, exp_15, exp_16]
    # return the result list
    return agglomerative_list
config_k = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
config_affinity = ['euclidean', 'euclidean', 'cosine', 'cosine', 'euclidean', 'euclidean', 'cosine', 'cosine', 'euclidean', 'euclidean', 'cosine', 'cosine', 'euclidean', 'euclidean', 'cosine', 'cosine']
config_linkage = ['single', 'complete', 'single', 'complete', 'single', 'complete', 'single', 'complete', 'single', 'complete', 'single', 'complete', 'single', 'complete', 'single', 'complete']
# iterate for every model in the list
model = create_example_dataset(dataset)


for i in range(len(model)):
    plt.title(
        "Dendrogram for k = " + str(config_k[i]) + " distance = " + config_affinity[i] + " linkage = " + config_linkage[
            i])
    # plot the top three levels of the dendrogram
    plot_dendrogram(model[i], truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

for i in range(len(model)):

    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(dataset) + (i//4 + 3) * 10])
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(dataset, model[i].labels_)




    print(
        "For " + str(config_k[i]) + " clusters, " + str(config_affinity[i]) + " distance and " + str(config_linkage[i]) + " linkage",
        "The average silhouette_score is :",
        silhouette_avg,
    )
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(dataset, model[i].labels_)
    y_lower = 10
    for j in range(i // 4 + 2):
        # Aggregate the silhouette scores for samples belonging to
        # cluster j, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[model[i].labels_ == j]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(j) / (i // 4 + 2))
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(
            -0.05,
            y_lower + 0.5 * size_cluster_i,
            str(j),
        )
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for " + str(config_k[i]) + " clusters, " + str(config_affinity[i]) + " distance and " + str(config_linkage[i]) + " linkage")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    fig.savefig('silhouette_' + str(config_k[i]) + '_' + str(config_affinity[i]) + '_' + str(config_linkage[i]) + '.png')
plt.show()
