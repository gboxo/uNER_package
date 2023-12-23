#from clustering.vocabulary_clustering import *
#b_embeds =BertEmbeds("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",0,"clustering/vocab.txt",True,True,"clustering/results/labels.txt","clustering/results/stats_dict.txt","clustering/preserve_1_2_grams.txt","clustering/glue_words.txt","clustering/bootstrap_entities.txt",0.8) #True - for cache embeds; normalize - True
#full_entities_dict = b_embeds.subword_clustering()
#b_embeds.create_entity_labels_file(full_entities_dict)
#----------------------------
#from algorithms.core_algorithm import *
#trivial = Trivial('datasets/STL_original_article.xls', 'datasets/raw articles/bmj/', encode=True)
#trivial.load_data()
#result = trivial.run_algorithm()
#encoded_result = trivial.encode_output(result)
#print(result)
#----------------------------
#from algorithms.core_algorithm import *
#fuzzy = Fuzzy('datasets/STL_original_article.xls', 'datasets/raw articles/bmj/', encode=True)
#fuzzy.load_data()
#result = fuzzy.run_algorithm()
#encoded_result = trivial.encode_output(result)
#print(result)
#-------------------------------
# fast uNER
#from algorithms.core_algorithm import uNER_fast

#uNER = uNER_fast('datasets/labels_final.txt','datasets/raw articles/all/',True,model,tokenizer)
#uNER.load_data()
#uNER_results = uNER.run_algorithm()
#from utilities.utils import read_gt_folder,mean_embed_sublist
#uNER_results = {key:[v for value in values for v in value] for key,values in uNER_results.items()}
#emb_dict_uNER = {key:mean_embed_sublist(model,tokenizer,value) for key,value in uNER_results.items() if len(value)>0}

#-------------
# random
#import random
#from algorithms.core_algorithm import *
#trivial = Trivial('datasets/STL_original_article.xls', 'datasets/raw articles/all/', encode=True)
#trivial.load_data()
#random_result = {key:random.sample(value.split(),5)for key,value in trivial.corpus.items() if len(value)>0}
#emb_dict_random = {key:mean_embed_sublist(model,tokenizer,value) for key,value in random_result.items() if len(value)>0}


#------------------------
#from plotting.plots import plot_pca_tensors
#import torch
#tensors = torch.rand((10,768))
#labels = ['model','binomial','distribution','PCA','residuals']
#plot_pca_tensors(tensors,labels)

#--------------------
#from plotting.plots import plot_pca_tensors_with_clusters
#import torch
#tensors = torch.rand((6,768))
#labels = ['model','binomial','distribution','PCA','residuals',"t test"]
#plot_pca_tensors_with_clusters(tensors,labels,[0,0,1,1,1,2])

#------------------
#import numpy as np
#from plotting.plots import plot_histograms_metric
# Sample data for the histograms
#data1 = np.random.normal(0, 1, 1000)
#data2 = np.random.normal(1, 1.5, 1000)
#data3 = np.random.normal(-1, 2, 1000)
#data = np.stack([data1, data2, data3])
#plot_histograms_metric(data)

#-----------
#from collections import Counter
#from utilities.utils import read_labels
#from plotting.plots import plot_piechart_labels
#a = read_labels("datasets/labels_final.txt")
#keys = a[0].keys()#['model']['label'].split("/") 
#entity_vec = [a[0][key]['label'].split("/") for key in keys]
#entity_vec_long = [label for sublist in entity_vec for label in sublist]
#plot_piechart_labels(Counter(entity_vec_long))


#-----------
#from plotting.plots import plot_word_cloud
#weights = {'Python': 100, 'Data': 80, 'Analysis': 60, 'Visualization': 50, 'Machine Learning': 40, 'AI': 30}
#plot_word_cloud(weights)

#-----------

#------------

from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")



from utilities.utils import read_gt_folder,mean_embed_sublist
ground_truth_gt = read_gt_folder("datasets/extracted section gpt/final/all/")
ground_truth_gt = {key:value for key,value in ground_truth_gt.items() if len(value)>0}
ground_truth_gt = {key:[v for v in value if len(v)>0] for key,value in ground_truth_gt.items() }
emb_dict_gt = {key:mean_embed_sublist(model,tokenizer,value) for key,value in ground_truth_gt.items()}
#----
from algorithms.core_algorithm import *
trivial = Trivial('datasets/STL_original_article.xls', 'datasets/raw articles/all/', encode=True)
trivial.load_data()
trivial_result = trivial.run_algorithm()
trivial_result = {key:list(set([v for v in value if len(v)>0])) for key,value in trivial_result.items() }
emb_dict_trivial = {key:mean_embed_sublist(model,tokenizer,value) for key,value in trivial_result.items() if len(value)>0}
#----
from algorithms.core_algorithm import *
fuzzy = Fuzzy('datasets/STL_original_article.xls', 'datasets/raw articles/all/', encode=True)
fuzzy.load_data()
fuzzy_result = fuzzy.run_algorithm()
fuzzy_result = {key:list(set([v for v in value if len(v)>0])) for key,value in fuzzy_result.items() }
emb_dict_fuzzy = {key:mean_embed_sublist(model,tokenizer,value) for key,value in fuzzy_result.items() if len(value)>0}

#----------
from algorithms.core_algorithm import uNER_fast

uNER = uNER_fast('datasets/labels_final.txt','datasets/raw articles/all/',True,model,tokenizer)
uNER.load_data()
uNER_results = uNER.run_algorithm()
from utilities.utils import read_gt_folder,mean_embed_sublist
uNER_results = {key:[v for value in values for v in value] for key,values in uNER_results.items()}
emb_dict_uNER = {key:mean_embed_sublist(model,tokenizer,value) for key,value in uNER_results.items() if len(value)>0}

#----------
import random
from algorithms.core_algorithm import *
trivial = Trivial('datasets/STL_original_article.xls', 'datasets/raw articles/all/', encode=True)
trivial.load_data()
random_result = {key:random.sample(value.split(),5)for key,value in trivial.corpus.items() if len(value)>0}
emb_dict_random = {key:mean_embed_sublist(model,tokenizer,value) for key,value in random_result.items() if len(value)>0}
#-----------

from evaluation.semantic_similarity import metric
from plotting.plots import plot_histograms_metric


result1 = np.array([metric(tensor1,tensor2) for tensor1,tensor2 in zip(emb_dict_gt.values(),emb_dict_trivial.values())])
result2 = np.array([metric(tensor1,tensor2) for tensor1,tensor2 in zip(emb_dict_gt.values(),emb_dict_fuzzy.values())])
result3 = np.array([metric(tensor1,tensor2) for tensor1,tensor2 in zip(emb_dict_gt.values(),emb_dict_uNER.values())])
result4 = np.array([metric(tensor1,tensor2) for tensor1,tensor2 in zip(emb_dict_gt.values(),emb_dict_random.values())])

plot_histograms_metric(result1,result2,result3,result4)
