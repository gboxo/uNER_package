from collections import OrderedDict
import numpy as  np
from transformers import BertTokenizer,AutoModel
import sys
import tqdm
import gc
import pickle as pkl
from utilities.utils import *
from sklearn.metrics import pairwise_distances
from collections import Counter


SINGLETONS_TAG  = "_singletons_ "
EMPTY_TAG = "_empty_ "
OTHER_TAG = "OTHER"
AMBIGUOUS = "AMB"
MAX_VAL = 20
TAIL_THRESH = 10
SUBWORD_COS_THRESHOLD = .2
MAX_SUBWORD_PICKS = 20

UNK_ID = 1
IGNORE_CONTINUATIONS=True
USE_PRESERVE=True


def is_subword(key):
        return True if str(key).startswith('#')  else False

def is_filtered_term(key): #Words selector. skiping all unused and special tokens
    if (IGNORE_CONTINUATIONS):
        return True if (is_subword(key) or str(key).startswith('[')) else False
    else:
        return True if (str(key).startswith('[')) else False

def filter_2g(term,preserve_dict):
    if (USE_PRESERVE):
        return True if  (len(term) <= 2 and term not in preserve_dict) else False
    else:
        return True if  (len(term) <= 2 ) else False

class BertEmbeds:
    def __init__(self, model_path,do_lower, terms_file,cache_embeds,normalize,labels_file,stats_file,preserve_2g_file,glue_words_file,bootstrap_entities_file,threshold_sim):
        do_lower = True if do_lower == 1 else False
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower)
        self.terms_dict = {self.tokenizer.convert_ids_to_tokens(i):i for i in range(len(self.tokenizer.get_vocab()))}
        self.labels_dict,self.lc_labels_dict = read_labels(labels_file)
        self.stats_dict = read_terms(stats_file) #Not used anymore
        self.preserve_dict = read_terms(preserve_2g_file)
        self.gw_dict = read_terms(glue_words_file)
        self.bootstrap_entities = read_entities(bootstrap_entities_file)
        self.model = AutoModel.from_pretrained(model_path)
        self.embeddings = self.model.state_dict()['embeddings.word_embeddings.weight']
        self.dist_threshold_cache = {}
        self.threshold_sim = threshold_sim
        self.n_elements = 20
        self.dist_zero_cache = {}
        self.top_n_indices = self.compute_top_indices()
        self.normalize = normalize

    def compute_top_indices(self):
        sim_matrix = pairwise_distances(self.embeddings, metric="cosine")
        below_threshold = sim_matrix <= self.threshold_sim
        sim_matrix[~below_threshold] = np.inf
        del below_threshold  # Free up memory from the mask
        gc.collect()
        sorted_indices = np.argsort(sim_matrix, axis=1)
        top_n_values = np.take_along_axis(sim_matrix, sorted_indices, axis=1)[:, :self.n_elements]
        del sim_matrix
        gc.collect()
        rows = np.arange(sorted_indices.shape[0])[:, None]
        top_n_indices = sorted_indices[rows, :self.n_elements]
        del sorted_indices
        gc.collect()
        return top_n_indices
    def create_entity_labels_file(self,full_entities_dict):
        with open("labels_final.txt","w") as fp:
            for term in self.terms_dict:
                if (term not in full_entities_dict and term.lower() not in self.bootstrap_entities):
                    fp.write("OTHER 0 " + term + "\n")
                    continue
                if (term not in full_entities_dict): #These are vocab terms that did not show up in a cluster but are present in bootstrap list
                    lc_term = term.lower()
                    counts_str = len(self.bootstrap_entities[lc_term])*"0/"
                    fp.write('/'.join(self.bootstrap_entities[lc_term]) + ' ' + counts_str.rstrip('/') + ' ' + term + '\n') #Note the term output is case sensitive. Just the indexed version is case insenstive
                    continue
                out_entity_dict = {}
                for entity in full_entities_dict[term]:
                    assert(entity not in out_entity_dict)
                    out_entity_dict[entity] = full_entities_dict[term][entity]
                sorted_d = OrderedDict(sorted(out_entity_dict.items(), key=lambda kv: kv[1], reverse=True))
                entity_str = ""
                count_str = ""
                for entity in sorted_d:
                    if (len(entity_str) == 0):
                        entity_str = entity
                        count_str =  str(sorted_d[entity])
                    else:
                        entity_str += '/' +  entity
                        count_str +=  '/' + str(sorted_d[entity])
                if (len(entity_str) > 0):
                    fp.write(entity_str + ' ' + count_str + ' ' + term + "\n")

    def subword_clustering(self):
        bootstrap_tokens=list(self.bootstrap_entities.keys())
        top_n_indices=self.top_n_indices.squeeze()
        top_n_tokens=[self.tokenizer.convert_ids_to_tokens(ind) for ind in top_n_indices]
        intersections = [list(set(tokens).intersection(bootstrap_tokens)) for tokens in top_n_tokens]
        label_groups = [[self.bootstrap_entities[token] for token in intersec] for intersec in intersections]
        flattened_label_groups = [ [label for sublist in group for label in sublist] for group in label_groups ]
        count_list=[Counter(l) for l in flattened_label_groups]
        full_entities_dict=OrderedDict()
        print("-----")
        print(len(top_n_tokens))
        print("------")
        for i in tqdm.tqdm(range(30522)):
            arr=top_n_tokens[i]
            curr_entities_dict=count_list[i]
            for term in arr:
                if term not in full_entities_dict: #This is case sensitive. We want vocab entries eGFR and EGFR to pick up separate weights for their entities
                    full_entities_dict[term] = OrderedDict()
                if len(curr_entities_dict.items())==0:
                    full_entities_dict[term]["OTHER"] = 0
                else:
                    for entity in curr_entities_dict:
                        #if  (entity not in term_entities): #aggregate counts only for entities present for this term in original manual harvesting list(bootstrap list)
                        #    continue
                        if (entity not  in full_entities_dict[term]):
                            full_entities_dict[term][entity] = curr_entities_dict[entity]
                        else:
                            full_entities_dict[term][entity] += curr_entities_dict[entity]


        return full_entities_dict



