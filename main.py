import torch
import pandas as pd
from nltk.tokenize import word_tokenize
from sentence.model import HAN,train_batch, evaluate
from sentence.data_prepare import load_word_emb, snopes_pre_process, k_fold


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load word embedding
gloveFile = '/content/drive/My Drive/glove.6B.50d.txt'
glove, weights, word_index_dict, embedding_dim = load_word_emb(gloveFile)

# load dataset
r_snopes = '/content/drive/My Drive/snopes.tsv'
snopes = pd.read_csv(r_snopes, sep='\t')

# data pre-processing
tokenizer = word_tokenize
instances = snopes_pre_process(snopes,tokenizer, word_index_dict, weights,device)

# constructing train set and test set
instances_train, instances_test = k_fold(instances,10)

# training
hidden_size = 100
embedding_size = 50
han = HAN(embedding_size, hidden_size, 1).to(device)
train_batch(han, instances_train, n_epoches=100)

# test
p, r, f1 = evaluate(han, instances_test)
