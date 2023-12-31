
from abc import ABC, abstractmethod
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModel
import tqdm
# The structure of an algorithm in this application is the following
# The input is an string representing a scientific article
# The first step is to load the required data (usually in the form of an external file)
# The second step is to apply some kind of preprocessing to the text
# The third step is to apply the algorithm
# At the end the output will be a list of substrings from the original string representing the Statistical Terms





from utilities.utils import consolidate_labels, read_labels,load_original_data,read_all_text_files,mean_embed_sublist
from algorithms.trivial import extract_techniques,tokenize_string,return_substrings
from algorithms.fuzzy import remove_punctuation,lemmatize_text,extract_techniques_fuzzy
from algorithms.uNER_fast import get_predictions,get_predictions_mixed,generate_predictions,process_data

class Algorithm:
    def __init__(self, encode=False):
        self.encode = encode

    def load_data(self):
        # Generic data loading method, can be overridden in subclasses
        pass

    def run_algorithm(self):
        # Method to run the algorithm
        pass

    def encode_output(self, output):
        # Method for encoding, if required
        if self.encode:
            # Implement encoding logic here
            output = mean_embed_sublist(output)
        return output

class Trivial(Algorithm):
    def __init__(self, file_path, folder_path_corpus, encode=False):
        super().__init__(encode)
        self.file_path = file_path
        self.folder_path_corpus = folder_path_corpus

    def load_data(self):
        self.data = load_original_data(self.file_path)
        self.corpus = read_all_text_files(self.folder_path_corpus)

    def run_algorithm(self):
        if not hasattr(self, 'data') or not hasattr(self, 'corpus'):
            self.load_data()
        extracted_techniques = {key:extract_techniques(article, self.data) for key,article in self.corpus.items()}
        extracted_techniques = {key:[a[0] for a in value[0]]+[a[1] for a in value[0]] for key,value in extracted_techniques.items()}

        extracted_techniques = {key:return_substrings(tokenize_string(text),[tokenize_string(v) for v in value]) for (key,value),text in zip(extracted_techniques.items(),self.corpus.values())}
        self.output = {key:list(set([v for v in value if len(v)>0])) for key,value in extracted_techniques.items() }

        return self.output

class Fuzzy(Algorithm):
    def __init__(self, file_path, folder_path_corpus, encode=False):
        super().__init__(encode)
        self.file_path = file_path
        self.folder_path_corpus = folder_path_corpus

    def load_data(self):
        # Assuming load_original_data and read_all_text_files are defined elsewhere
        stl = load_original_data(self.file_path)
        
        # Lemmatizing the terms and synonyms in the data
        stl['Specific Method/Term'] = stl['Specific Method/Term'].apply(lemmatize_text)
        stl['Synonym for Terms to Group'] = stl['Synonym for Terms to Group'].apply(lemmatize_text)
        self.data = stl

        # Reading and processing corpus text
        texts = read_all_text_files(self.folder_path_corpus)
        texts_no_punc = {key: remove_punctuation(article) for key, article in texts.items()}
        self.corpus = {key: lemmatize_text(article) for key, article in texts_no_punc.items()}

    def run_algorithm(self):
        if not hasattr(self, 'data') or not hasattr(self, 'corpus'):
            self.load_data()

        # Extracting techniques from each article in the corpus
        extracted_techniques = {key: extract_techniques_fuzzy(article, self.data) for key, article in self.corpus.items()}
        
        extracted_techniques = {key:list(set([v for v in value if len(v)>0])) for key,value in extracted_techniques.items() }

        # Format the extracted techniques appropriately
        # Assuming a specific output format is needed
        return extracted_techniques

    def format_output(self, extracted_data):
        # Assuming a specific output format is needed
        # Implement the output formatting logic here
        pass



class uNER_fast(Algorithm):
    def __init__(self,bootstrap_file_path,folder_path_corpus,encode,model,tokenizer):
        super().__init__(encode)
        self.bootstrap_file_path = bootstrap_file_path
        self.folder_path_corpus = folder_path_corpus
        self.model = model
        self.tokenizer = tokenizer
        
    def load_data(self):
        self.terms_dict,self.lc_terms_dict=read_labels(self.bootstrap_file_path)
        corpus = read_all_text_files(self.folder_path_corpus)
        self.corpus = {key:value for key,value in corpus.items() if len(value)>0}


    def run_algorithm(self):
        if not hasattr(self, 'terms_dict') or not hasattr(self, 'corpus'):
            self.load_data()

        dictionary_predictions = {}
        for keys, text in tqdm.tqdm(self.corpus.items()):
            indices = self.tokenizer(text,return_tensors = "pt",add_special_tokens = False )['input_ids'][0]
            if (len(indices))<128:
                continue
            ind_list = np.array_split(indices, len(indices) // 128)
            stls = list()
            for ind in ind_list:
                topk_vals, topk_inds = generate_predictions(ind, self.tokenizer, self.model)
                max_entities = process_data(topk_inds, topk_vals, self.terms_dict, self.tokenizer)
                stl = ["STL", "MODEL", "DISTRB", "STUDY", "MEASURES", "SAMPLING", "D_TYPE", "STATISTICS", "TESTS", "SOFTWARE"]
                entity = [c for (a, b, c, d) in max_entities]
                entity = [ent if ent in stl else "UNTAGGED" for ent in entity]
                score = np.array([d for (a, b, c, d) in max_entities])
                score[score == -np.inf] = 0
                score = np.array([0 if ent == "UNTAGGED" else sc for sc, ent in zip(score, entity)])

                # Initialize list to store groups of joined words
                joined_words_list = []
                temp_words = []
                w = [self.tokenizer.convert_ids_to_tokens(i) for i in ind.numpy().tolist()]

                for word, sc in zip(w, score):
                    if sc != 0:
                        temp_words.append(word)
                    elif temp_words:
                        joined_words_list.append(' '.join(temp_words))
                        temp_words = []
                if temp_words:  # Check if any words are left in temp_words
                    joined_words_list.append(' '.join(temp_words))

                stls.append(joined_words_list)

            dictionary_predictions[keys] = stls

        return dictionary_predictions