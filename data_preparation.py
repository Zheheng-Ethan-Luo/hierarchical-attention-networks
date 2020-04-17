import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import nltk

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
  weights.append(np.random.normal(scale=0.6,size=(embedding_dim,)))
  weights = np.asarray(weights)
  print("word embedding initialization finished")
  return glove,weights,word2index,embedding_dim

gloveFile = '/content/drive/My Drive/glove.6B.50d.txt'
glove,weights,word2index,embedding_dim = load_wordem(gloveFile)

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
    matrix[i] = torch.from_numpy(weight[sent[i]])
  return matrix
  
tokenizer = nltk.tokenize.WordPunctTokenizer()
claim_ori = "The test of a 5G cellular network is the cause of unexplained bird deaths occurring in a park in The Hague, Netherlands."
sentences = [
"Lots of tests going on with it in the Netherlands, but there haven’t been test done in The Haque during the time that the mysterious starling deaths occurred.", 
"One such test did occur in an area generally near Huijgenspark, but it took place on 28 June 2018.", 
"It’s not clear whether tests with 5G have been carried out again, but so far everything points in the direction of 5G as the most probable cause.", 
"Between Friday, 19 Oct and Saturday, 3 Nov 2018, 337 dead starlings and 2 dead common wood pigeons were found.", 
"The radiation created on the attempt of 5G cellular networks are not harmful only for birds but also for humans too.", 
"5G network developers promise faster data rates in addition to reduce energy and financial cost.", 
"Parts of the park are blocked and dogs are no longer allowed to be let out, the dead birds are always cleaned up as quickly as possible."]
claim_wl = tokenizer.tokenize(claim_ori.lower())
claim_indice = wordtoindex(claim_wl,word2index)
sentence_wls = [tokenizer.tokenize(s.lower()) for s in sentences]
sent_indice_l = [wordtoindex(s,word2index) for s in sentence_wls]
claim_input = index2tensor(claim_indice,weights)
sentence_input = [index2tensor(s,weights) for s in sent_indice_l]
