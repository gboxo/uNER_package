import numpy as np

import pandas as pd
import torch.nn.functional as F
from itertools import combinations

# The definition of an algorithm is an object that gets a string of text and returns a list of strings
# This substring output can be postprocessed into an embedding
# Cluster the substrings that are very  similar (eg. model and models)
# Asses the similarity of the extracted substrings (embedded and clustered if needed) and the ground truth (embedded and clustered if needed)
# Establish a dictionary of relationships between objects from the extracted and gt sets, a dicionary might be empty
# Compute the mean cosine similarity of the relation dictionaries. A mean cosine similarity close to 1 will indicate a good performing algorithm.
# The definition of a metric object is such that given a list of paired lists of strings returns a score

import torch
import torch.nn.functional as F

# Assuming `tensor1` and `tensor2` are the given tensors of shape [n, 768] and [m, 768]
# Placeholder tensors for the sake of demonstration
def metric(tensor1,tensor2):
    # Normalize the tensors along the last dimension
    tensor1_norm = F.normalize(tensor1, p=2, dim=1)
    tensor2_norm = F.normalize(tensor2, p=2, dim=1)

    # Compute pairwise cosine similarity
    similarity_matrix = torch.matmul(tensor1_norm, tensor2_norm.transpose(0, 1))

    # Select the highest similarity for each row
    highest_similarity, selected_indices = torch.max(similarity_matrix, dim=1)

    # Retrieve the selected tensors from tensor2
    selected_tensors = tensor2[selected_indices]
    return highest_similarity.mean()