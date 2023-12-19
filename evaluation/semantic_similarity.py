import numpy as np
import os
import glob
import importlib
import pandas as pd
from algorithms.trivial import extract_techniques

# The definition of an algorithm is an object that gets a string of text and returns a list of strings
# The definition of a metric object is such that given a list of paired lists of strings returns a score


class DataExtractor():
    def __init__(self, gt_folder, raw_corpus_folder):
        self.gt_folder = gt_folder
        self.raw_corpus_folder = raw_corpus_folder

        # List files in each set
        self.files_in_set1 = {folder: self.list_files_in_directory(os.path.join(self.gt_folder, folder)) 
                        for folder in os.listdir(self.gt_folder)}
        self.files_in_set2 = {folder: self.list_files_in_directory(os.path.join(self.raw_corpus_folder, folder)) 
                        for folder in os.listdir(self.raw_corpus_folder)}

        self.common_files = self.find_common_files()
        self.ground_truth = self.load_gt()
        self.corpus = self.load_corpus()

    def list_files_in_directory(self, directory):
        """List all text files in a given directory."""
        return {file for file in os.listdir(directory) if file.endswith('.txt')}

    def find_common_files(self):
        """Find common files in two sets."""
        common_files = {}
        for folder in self.files_in_set1:
            common_folder_files = self.files_in_set1[folder].intersection(self.files_in_set2.get(folder, set()))
            if common_folder_files:
                common_files[folder] = common_folder_files
        return common_files

    def load_gt(self):
        file_contents = {}
        for folder, files in self.common_files.items():
            for file in files:
                file_path = os.path.join(self.gt_folder, folder, file)
                with open(file_path, 'r') as f:
                    file_contents[file] = f.read().replace("- ", "").split("\n")[:-2]
        return file_contents
    def load_corpus(self):
        file_contents = {}
        for folder, files in self.common_files.items():
            for file in files:
                file_path = os.path.join(self.raw_corpus_folder, folder, file)
                with open(file_path, 'r') as f:
                    file_contents[file] = f.read().replace("\n", " ")
        return file_contents
    
    
class Algorithm:  
    def __init__(self,corpus,techniques_file):
        self.techniques_df = pd.read_excel(techniques_file)
        self.corpus = corpus
    def extract_techniques_from_corpus(self):
        """Apply the extract_techniques function to the entire corpus."""
        results = {}
        for file, text in self.corpus.items():
            results[file] = extract_techniques(text, self.techniques_df)
        return results
    

class MetricCalculator:
    def calculate_metric(self,ground_truth,algoithm_output):
        return






