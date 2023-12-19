import pandas as pd
from collections import defaultdict
import os
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')



def load_data(file_path='/home/gerard/Desktop/Demo/STL_original_article.xls'):
    return pd.read_excel(file_path)


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



def get_substring_indices(string, substring):
    indices = []
    index = string.find(substring)
    
    while index != -1:
        indices.append(index)
        index = string.find(substring, index + 1)
    
    return indices


import string


def tokenize_string(string):
    return string.split()

def highlight_substrings(string, substrings):
    # Tokenize main string and substrings by whitespace
    tokenized_string = tokenize_string(string.lower())
    tokenized_substrings = [tokenize_string(sub) for sub in substrings]

    # Assigning unique colors to each substring
    colors = ['red' for _ in range(len(substrings))]  # Add more colors as needed
    color_mapping = {sub: colors[i % len(colors)] for i, sub in enumerate(substrings)}

    # Generate HTML with highlighted substrings
    highlighted_html = "<html><body><p>"
    
    index = 0
    while index < len(tokenized_string):
        highlighted = False
        for sub_tokens in tokenized_substrings:
            end_index = index + len(sub_tokens)
            if tokenized_string[index:end_index] == sub_tokens:
                color = color_mapping[' '.join(sub_tokens)]
                highlighted_html += f"<span style='background-color:{color};'>{' '.join(sub_tokens)+' '}</span>"
                index = end_index
                highlighted = True
                break
        
        if not highlighted:
            highlighted_html += tokenized_string[index] + " "
            index += 1
    
    highlighted_html += "</p></body></html>"
    return highlighted_html





def remove_punctuation(text):
    punctuations = string.punctuation
    cleaned_text = ''.join(char for char in text if char not in punctuations)
    return cleaned_text

def lemmatize_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text.lower())

    # Remove punctuation
    tokens = [remove_punctuation(token) for token in tokens]


    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(lemmatized_tokens)

