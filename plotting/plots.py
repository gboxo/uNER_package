
#- Salida algoritmo (embeddings) para cada articulo i/o global
#- Distribucion de metrica de cada algoritmo y cada articulo
#- Bootstraping de la metrica media de cada algoritmo conforme añadimos articulos (bootstraping)
#- Clustering augmentation
#- Labeling
#- Wordmap
#- Wordmap mosaic



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch



def plot_pca_tensors(tensors, labels):
    """
    Plots a PCA of the given tensors annotated with labels.

    :param tensors: List of 768-dimensional tensors.
    :param labels: Corresponding list of labels for the tensors.
    """
    # Convert list of tensors to a numpy array
    tensors_array = np.array(tensors)

    # Perform PCA to reduce the tensors to 2 dimensions
    pca = PCA(n_components=2)
    reduced_tensors = pca.fit_transform(tensors_array)

    # Plotting the reduced tensors
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(reduced_tensors[i, 0], reduced_tensors[i, 1], label=label)

    # Annotating the points with labels
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_tensors[i, 0], reduced_tensors[i, 1]))

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Tensors')
    plt.legend()
    plt.show()



def plot_pca_tensors_with_clusters(tensors, labels, cluster_labels):
    """
    Plots a PCA of the given tensors, color-coding them based on cluster labels
    and annotating with provided labels.

    :param tensors: List of 768-dimensional tensors.
    :param labels: Corresponding list of labels for the tensors.
    :param cluster_labels: Labels used for color-coding the clusters.
    """
    # Convert list of tensors to a numpy array
    tensors_array = np.array(tensors)

    # Perform PCA to reduce the tensors to 2 dimensions
    pca = PCA(n_components=2)
    reduced_tensors = pca.fit_transform(tensors_array)

    # Create a scatter plot with colors based on cluster labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_tensors[:, 0], reduced_tensors[:, 1], c=cluster_labels, cmap='viridis', label=np.unique(cluster_labels))

    # Annotating the points with labels
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_tensors[i, 0], reduced_tensors[i, 1]))

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Tensors with Cluster Coloring')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()


def plot_histograms_metric(data):
    """
    Plots a three overlapping histograms for the metric provided by a 3xn dimensional array
    where n is the number of articles used in the experiment.

    :param data: Array of 3xn metrics
    """
    # Creating the plot with overlapped histograms
    plt.figure(figsize=(10, 6))
    plt.hist(data[0], bins=30, alpha=0.5, color='red', label='Trivial Algorithm')
    plt.hist(data[1], bins=30, alpha=0.5, color='blue', label='Fuzzy Algorithm')
    plt.hist(data[2], bins=30, alpha=0.5, color='green', label='uNER Algorithm')
    # Adding legend
    plt.legend()

    # Adding titles and labels
    plt.title('Overlapped Histograms with Transparency')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Displaying the plot
    plt.show()



def plot_piechart_labels(counter):
    """
    Plots a piechart diagram with the provided labels results of the clustering process for the downstream uNER tast.

    :param counter: A dictionar with the label and the number of occurances of each label
    """
    labels = list(counter.keys())
    sizes = list(counter.values())

    # Create a pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, labels=labels,)
    
    # Adding title
    plt.title('Pie Chart from Counter')
    plt.legend()

    # Display the plot
    plt.show()