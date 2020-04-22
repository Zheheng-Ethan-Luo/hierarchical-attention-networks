import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import nltk
import pandas as pd
from model import HAN
from train import train_batch


def load_wordem(wvFile):
  glove = {}
  word2index = {}
  weights = []
  f = open(wvFile,'r')
  for i,line in enumerate(f):
    splits = line.split()
    word = splits[0]
    word2index[word] = i
    embedding = np.array([float(val) for val in splits[1:]])
    glove[word] = embedding
    weights.append(embedding)
  embedding_dim = embedding.shape[0]
  weights.append(np.random.normal(scale=0.6,size=(embedding_dim,)))#for unknown token
  weights = np.asarray(weights)
  print("word embedding initialization finished")
  return glove,weights,word2index,embedding_dim

def wordtoindex(wl,word2index):
  wl_new = []
  for word in wl:
    if word in word2index:
      wl_new.append(word2index[word])
    else:
      wl_new.append(weights.shape[0]-1)
  return wl_new
  
def index2tensor(sent,weights):
  matrix = torch.zeros((len(sent),embedding_dim))
  for i in range(len(sent)):
    matrix[i] = torch.from_numpy(weights[sent[i]])
  return matrix
  
gloveFile = '/content/drive/My Drive/glove.6B.50d.txt'
glove,weights,word2index,embedding_dim = load_wordem(gloveFile)

r_snopes = '/content/drive/My Drive/snopes.tsv'
snopes = pd.read_csv(r_snopes,sep='\t')

claim_ids = snopes['<claim_id>']
claim_ids = claim_ids.drop_duplicates()
claim_ids = claim_ids.to_list()

claims = []
sentences = []
targets = []
for id in claim_ids:
  claim = snopes.loc[snopes['<claim_id>']==id]['<claim_text>'].drop_duplicates().to_list()[0]
  evidence = snopes.loc[snopes['<claim_id>']==id]['<evidence>'].to_list()
  label  = snopes.loc[snopes['<claim_id>']==id]['<cred_label>'].drop_duplicates().to_list()[0]
  claims.append(claim)
  sentences.append(evidence)
  targets.append(label)
 
def build_instance(claim,sentence,target):
  instance = []
  model_input = []
  claim_wl = tokenizer.tokenize(claim.lower())
  claim_indice = wordtoindex(claim_wl,word2index)
  sentence_wls = [tokenizer.tokenize(s.lower()) for s in sentence]
  sent_indice_l = [wordtoindex(s,word2index) for s in sentence_wls]
  claim_input = index2tensor(claim_indice,weights)
  sentence_input = [index2tensor(s,weights) for s in sent_indice_l]
  instance.append(claim_input)
  instance.append(sentence_input)
  if target in ['true', 'mostly true']:
    instance.append(torch.tensor([1.]).to(device))
  else:
    instance.append(torch.tensor([0.]).to(device))
  return instance

instances = []
instances = [build_instance(claims[i],sentences[i],targets[i]) for i in range(len(claims))]
#constructing train set and test set
instances_T = [instance for instance in instances if instance[2]==1]
instances_F = [instance for instance in instances if instance[2]==0]

divide_t = int(len(instances_T)/10)
divide_f = int(len(instances_F)/10)

instances_train = instances_T[divide_t:]+instances_F[divide_f:]
instances_test = instances_T[:divide_t]+instances_F[:divide_f]

hidden_size = 100
embedding_size = 50
han = HAN(embedding_size,hidden_size,1).to(device)
train_batch(han,instances,n_epoches=100)
