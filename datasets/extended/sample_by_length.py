import json
import matplotlib.pyplot as plt
import random
random.seed(123)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import tqdm
import numpy as np

def plot_line_length_histogram(json_file_path):
    # Reading the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extracting the length of each line
    line_lengths = [len(line.split()) for line in data.values()]

    line_lengths = [l for l in line_lengths if l<2000]
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(line_lengths, bins=50, color='blue', edgecolor='black')
    plt.title('Histogram of Line Lengths in Text')
    plt.xlabel('Length of Lines')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

#plot_line_length_histogram('corpus_id.json') 



def sample_lines_and_save(json_file_path, sample_size=10000):
    # Reading the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Converting the dictionary to a list of (key, value) pairs
    items = list(data.items())

    # Sampling 10000 items while respecting the distribution of line lengths
    sampled_items = random.sample(items, sample_size)

    # Converting the sampled items back into a dictionary
    sampled_dict = {key: value for key, value in sampled_items}

    # Saving the sampled dictionary to a new JSON file
    with open('corpus_sample.json', 'w') as outfile:
        json.dump(sampled_dict, outfile, indent=4)

    return "Sampled JSON file created successfully."


sample_lines_and_save("corpus_id.json")


#plot_line_length_histogram('corpus_sample.json') 


def split_train_test(json_file_path, train_size=9000):
    # Reading the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Shuffling and splitting the data into train and test sets
    items = list(data.items())
    random.shuffle(items)
    train_items = items[:train_size]
    test_items = items[train_size:]

    # Converting the train and test sets back into dictionaries
    train_dict = dict(train_items)
    test_dict = dict(test_items)

    # Saving the train and test sets to separate JSON files
    with open('train_data.json', 'w') as train_file:
        json.dump(train_dict, train_file, indent=4)

    with open('test_data.json', 'w') as test_file:
        json.dump(test_dict, test_file, indent=4)

    return "Train and test JSON files created successfully."

# The function expects a JSON file path and an optional train size.
# Since I cannot access external files, this script can be executed in a local environment.
# Replace 'your_json_file.json' with the path to your JSON file to execute this.

split_train_test('corpus_sample.json')




def tfidf_clustering_and_elbow_method(json_file_path, max_k=200):
    # Loading the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extracting the text from the JSON file
    texts = list(data.values())

    # Applying TF-IDF to the text
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # Applying KMeans clustering with different values of k and calculating the inertia
    inertia = []
    for k in tqdm.tqdm(range(1, max_k + 1,10)):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(tfidf_matrix)
        inertia.append(kmeans.inertia_)

    # Plotting the elbow method graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1,10), inertia, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    #plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()

#tfidf_clustering_and_elbow_method('test_data.json')


def balanced_cluster_sampling_and_save(json_file_path, k_clusters, sample_size, output_file_path):
    # Loading the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extracting the text from the JSON file
    texts = list(data.values())

    # Applying TF-IDF to the text
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # Applying KMeans clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(tfidf_matrix)
    labels = kmeans.labels_

    # Calculating the number of samples per cluster
    samples_per_cluster = sample_size // k_clusters

    # Sampling balanced items from each cluster
    sampled_indices = []
    for cluster in range(k_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        sampled_indices.extend(np.random.choice(cluster_indices, samples_per_cluster, replace=False))

    # Extracting the sampled items
    sampled_items = {list(data.keys())[i]: texts[i] for i in sampled_indices}

    # Saving the sampled items to a JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(sampled_items, outfile, indent=4)

    return "Sampled items saved to JSON file."

balanced_cluster_sampling_and_save('train_data.json', 10, 2000, 'train_sampled_data.json')
