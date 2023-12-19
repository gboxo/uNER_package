from scipy.special import softmax
import torch
import numpy as np
from transformers import BertTokenizer,BertForMaskedLM
from collections import OrderedDict
import pdb
import torch
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModel
import torch.nn.functional as F
import re
from collections import defaultdict, Counter
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib


model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
mskd_model=AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

def get_predictions(sentence,mask_position_list):
    tokens = tokenizer(sentence,return_tensors="pt",padding=True)
    with torch.no_grad():
        outputs = mskd_model(**tokens)
    predictions = torch.stack([outputs.logits[i,pos,:] for i,pos in  enumerate(mask_position_list) ])
    return predictions

def get_predictions_mixed(sentence):

    with torch.no_grad():
        tokens = tokenizer(sentence, return_tensors="pt", truncation=True)
        outputs = model(**tokens)    
        lhs=outputs.last_hidden_state   
        to_mskd=mskd_model.get_submodule("cls")(lhs)

    return to_mskd


def consolidate_labels(existing_node,new_labels,new_counts):
    """Consolidates all the labels and counts for terms ignoring casing

    For instance, egfr may not have an entity label associated with it
    but eGFR and EGFR may have. So if input is egfr, then this function ensures
    the combined entities set fo eGFR and EGFR is made so as to return that union
    for egfr
    """
    new_dict = {}
    existing_labels_arr = existing_node["label"].split('/')
    existing_counts_arr = existing_node["counts"].split('/')
    new_labels_arr = new_labels.split('/')
    new_counts_arr = new_counts.split('/')
    assert(len(existing_labels_arr) == len(existing_counts_arr))
    assert(len(new_labels_arr) == len(new_counts_arr))
    for i in range(len(existing_labels_arr)):
        new_dict[existing_labels_arr[i]] = int(existing_counts_arr[i])
    for i in range(len(new_labels_arr)):
        if (new_labels_arr[i] in new_dict):
            new_dict[new_labels_arr[i]] += int(new_counts_arr[i])
        else:
            new_dict[new_labels_arr[i]] = int(new_counts_arr[i])
    sorted_d = OrderedDict(sorted(new_dict.items(), key=lambda kv: kv[1], reverse=True))
    ret_labels_str = ""
    ret_counts_str = ""
    count = 0
    for key in sorted_d:
        if (count == 0):
            ret_labels_str = key
            ret_counts_str = str(sorted_d[key])
        else:
            ret_labels_str += '/' +  key
            ret_counts_str += '/' +  str(sorted_d[key])
        count += 1
    return {"label":ret_labels_str,"counts":ret_counts_str}

def read_labels(labels_file):
    terms_dict = OrderedDict()
    lc_terms_dict = OrderedDict()
    with open(labels_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            term = term.split()
            if (len(term) == 3):
                terms_dict[term[2]] = {"label":term[0],"counts":term[1]}
                lc_term = term[2]
                if (lc_term in lc_terms_dict):
                     lc_terms_dict[lc_term] = consolidate_labels(lc_terms_dict[lc_term],term[0],term[1])
                else:
                     lc_terms_dict[lc_term] = {"label":term[0],"counts":term[1]}
                count += 1
            else:
                print("Invalid line:",term)
                assert(0)
    print("count of labels in " + labels_file + ":", len(terms_dict))
    return terms_dict,lc_terms_dict

def generate_predictions(words, tokenizer, get_predictions):
    masked_sentences2 = []
    mask_position_list = []
    for i in range(len(words)):
        # Create a copy of the words list
        masked = words.copy()
        sent = 'A '+masked[i] + ' is a [MASK].'
        # Create the first type of masked sentence
        #masked_i = tokenizer.convert_ids_to_tokens((tokenizer(masked[i], add_special_tokens=False)['input_ids'][0]))
        inputs = tokenizer(sent, return_tensors="pt",padding=True)
        mask_token_id = tokenizer.mask_token_id
        mask_position = (inputs["input_ids"].squeeze() == mask_token_id).nonzero().item()
        mask_position_list.append(mask_position)
        masked_sentences2.append(sent)
    
    predictions1 = get_predictions(masked_sentences2,mask_position_list)
    # Convert lists of tensor predictions into tensors



    text2 = tokenizer.convert_tokens_to_string(words)
    predictions2 = get_predictions_mixed(text2)[0,1:-1,:]
    predictions = torch.stack((predictions1, predictions2))

    # Get top 10 predictions
    topk_vals, topk_inds = torch.topk(predictions, 5, dim=-1)

    return topk_vals, topk_inds

def process_data(topk_inds, topk_vals, terms_dict, tokenizer, F):
    max_entities = []
    
    for p in range(topk_inds.size(1)):
        # Extract tensors and entities for the current batch and prediction
        tensors_phr = [F.softmax(torch.tensor([int(i) for i in terms_dict[tokenizer.convert_ids_to_tokens(topk_inds[0,p,j].item())]['counts'].split("/")],dtype=torch.float),dim=-1) for j in range(topk_inds.size(2))]
        tensors_sent = [F.softmax(torch.tensor([int(i) for i in terms_dict[tokenizer.convert_ids_to_tokens(topk_inds[1,p,j].item())]['counts'].split("/")],dtype=torch.float),dim=-1) for j in range(topk_inds.size(2))]
        
        entities_lists_phr = [[terms_dict[tokenizer.convert_ids_to_tokens(topk_inds[0,p,i].item())]['label'].split("/")] for i in range(topk_inds.size(2))]
        entities_lists_sent = [[terms_dict[tokenizer.convert_ids_to_tokens(topk_inds[1,p,i].item())]['label'].split("/")] for i in range(topk_inds.size(2))]

        scalars_phr = topk_vals[0, p, :]
        scalars_sent = topk_vals[1, p, :]
        
        result_phr = [s * t for s, t in zip(scalars_phr, tensors_phr)]
        result_sent = [s * t for s, t in zip(scalars_sent, tensors_sent)]
        
        all_entities = set(entity for sublist in entities_lists_phr[0] for entity in sublist)
        aggregated_tensors_phr = defaultdict(lambda: torch.zeros(1))
        aggregated_tensors_sent = defaultdict(lambda: torch.zeros(1))
    
        for entities, tensor in zip(entities_lists_phr[0], result_phr):
            for entity, value in zip(entities, tensor):
                aggregated_tensors_phr[entity] += value
        for entities, tensor in zip(entities_lists_sent[0], result_sent):
            for entity, value in zip(entities, tensor):
                aggregated_tensors_sent[entity] += value
        
        aggregated_tensors = Counter(aggregated_tensors_phr) + Counter(aggregated_tensors_sent)
        aggregated_tensors = dict(aggregated_tensors)
        tensor = torch.tensor([t for t in aggregated_tensors.values()])
       
        mean_t = tensor.mean()
        std_t = tensor.std()
        normalized_tensor = (tensor - mean_t) / std_t
    
        max_value = float('-inf')
        max_entity = None
    
        for entity, tensor_value in zip(list(aggregated_tensors.keys()), tensor):
            if tensor_value.item() > max_value:
                max_value = tensor_value.item()
                max_entity = entity
    
        max_entities.append((0, p, max_entity, max_value))
    
    return max_entities



def create_html(strings, scores, strings2):
    # Create color map
    cmap = plt.get_cmap('coolwarm')

    # Normalize scores to range 0-1 for color map
    normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())

    html_str = '<p>'

    for string, score, string2 in zip(strings, normalized_scores, strings2):
        rgba = cmap(score)
        color = matplotlib.colors.rgb2hex(rgba)
        html_str += f'<span style="background-color: {color}; display: inline-block; margin-bottom: 10px;">{string}<br>{string2}</span> '

    html_str += '</p>'

    return html_str
