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

# Function to remove punctuation
def remove_punctuation(text):
    punctuations = string.punctuation
    cleaned_text = ''.join(char for char in text if char not in punctuations)
    return cleaned_text

# Function to lemmatize text
def lemmatize_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text.lower())

    # Remove punctuation
    tokens = [remove_punctuation(token) for token in tokens]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(lemmatized_tokens)

# Function to extract techniques from the corpus
def extract_techniques_fuzzy(corpus, techniques_df):
    # Lemmatize the corpus for searching
    lemmatized_corpus = lemmatize_text(corpus)

    # Convert both texts to lowercase for case-insensitive matching
    lemmatized_corpus = lemmatized_corpus.lower()
    corpus = corpus.lower()

    # Create a dictionary to store the results
    extracted_techniques = list()

    # Search for techniques in the corpus
    for _, row in techniques_df.iterrows():
        method_term = str(row['Specific Method/Term']).lower()
        synonym = str(row['Synonym for Terms to Group']).lower()
        group = row['Statistical Method Group']

        # Lemmatize the search terms
        lemmatized_method_term = lemmatize_text(method_term)
        lemmatized_synonym = lemmatize_text(synonym)

        # Search for lemmatized terms in the lemmatized corpus
        if lemmatized_method_term in lemmatized_corpus or lemmatized_synonym in lemmatized_corpus:
            # If found, add the original term from the non-lemmatized corpus
            if method_term in corpus:
                extracted_techniques.append(method_term)
            if synonym in corpus:
                extracted_techniques.append(synonym)

    return list(set(extracted_techniques))