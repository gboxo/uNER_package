import pdb
from collections import OrderedDict
import numpy as  np
import json
import os
import glob
import torch
import pandas as pd


def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_list = json.loads(fp.read())
    arr = np.array(embeds_list)
    return arr


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
                lc_term = term[2].lower()
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


def read_entities(terms_file):
    ''' Read bootstrap entities file

    '''
    terms_dict = OrderedDict()
    with open(terms_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                nodes = term.split()
                assert(len(nodes) == 2)
                lc_node = nodes[1].lower()
                if (lc_node in terms_dict):
                    pdb.set_trace()
                    assert(0)
                    assert('/'.join(terms_dict[lc_node]) == nodes[0])
                terms_dict[lc_node] = nodes[0].split('/')
                count += 1
    print("count of entities in ",terms_file,":", len(terms_dict))
    return terms_dict



def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                terms_dict[term] = count
                count += 1
    print("count of tokens in ",terms_file,":", len(terms_dict))
    return terms_dict
    

def read_gt_folder(folder_path):
    # Construct the pattern to match all .txt files
    pattern = os.path.join(folder_path, '*.txt')

    # Find all files in the folder matching the pattern
    text_files = glob.glob(pattern)

    # Dictionary to hold file content with filenames as keys
    file_contents = {}

    # Loop through the files and read their contents
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_contents[os.path.basename(file_path)] = file.read().replace("- ","").split("\n")[:-2]
            content = file.read().replace("- ","").split("\n")[:-2]
            if len(content)>0:
                file_contents[os.path.basename(file_path)] = content
    file_contents = {key:value for key,value in file_contents.items() if len(value)>0}# The file cannot be empty
    file_contents = {key:[v for v in value if len(v)>0] for key,value in file_contents.items() }# the substring cannot be empty
    return file_contents



def load_original_data(file_path):
    return pd.read_excel(file_path)

def read_all_text_files(folder_path):
    # Construct the pattern to match all .txt files
    pattern = os.path.join(folder_path, '*.txt')

    # Find all files in the folder matching the pattern
    text_files = glob.glob(pattern)

    # Dictionary to hold file content with filenames as keys
    file_contents = {}

    # Loop through the files and read their contents
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_contents[os.path.basename(file_path)] = file.read().replace("\n"," ")[:-2]

    return file_contents


def mean_embed_sublist(model,tokenizer,sublist):
    tokens = tokenizer(sublist,padding=True,return_tensors = "pt")
    with torch.no_grad():
        output = model(**tokens).last_hidden_state
    mean_embeddings = torch.stack([output[i,torch.where(tokens['attention_mask'][i])[0],:][1:-1,:].mean(0) for i in range(output.size(0))])
    return mean_embeddings