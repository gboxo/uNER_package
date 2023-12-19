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
from collections import Counter
from utilities.utils import read_labels
from plotting.plots import plot_piechart_labels
a = read_labels("datasets/labels_final.txt")
keys = a[0].keys()#['model']['label'].split("/") 
entity_vec = [a[0][key]['label'].split("/") for key in keys]
entity_vec_long = [label for sublist in entity_vec for label in sublist]
plot_piechart_labels(Counter(entity_vec_long))