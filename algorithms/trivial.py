

import pandas as pd
from collections import defaultdict
import os
import string

# Function to extract techniques from the corpus
def extract_techniques(corpus, techniques_df):
    # Convert the text to lowercase for case-insensitive matching
    corpus = corpus.lower()
    
    # Create a dictionary to store the results
    extracted_techniques = defaultdict(list)
    idx=0
    # Search for techniques in each article
    for _, row in techniques_df.iterrows():
        method_term = str(row['Specific Method/Term']).lower()
        synonym = str(row['Synonym for Terms to Group']).lower()
        group = row['Statistical Method Group']
        
        # If the method term or its synonym is found in the article, add it to the results
        if method_term in corpus or synonym in corpus:
            extracted_techniques[idx].append((method_term,synonym, group))
    
    return extracted_techniques

def remove_punctuation(text):
    # Define the set of punctuation characters
    punctuations = string.punctuation

    # Remove punctuation characters from the text
    cleaned_text = ''.join(char for char in text if char not in punctuations)

    return cleaned_text

def tokenize_string(string):
    return string.split()

def return_substrings(tokenized_string, tokenized_substrings):
    # Tokenize main string and substrings by whitespace
    #tokenized_string = tokenize_string(string.lower())
    #tokenized_substrings = [tokenize_string(sub) for sub in substrings]
    final_substring = list()
    for sub_tokens in tokenized_substrings:
        for index in range(len(tokenized_string) - len(sub_tokens) + 1):
            if tokenized_string[index:index + len(sub_tokens)] == sub_tokens:
                final_substring.append(" ".join(sub_tokens))
                break
    return final_substring