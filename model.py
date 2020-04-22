import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HAN(nn.Module):
  def __init__(self, embedding_size, hidden_size, output_size):
    super(HAN, self).__init__()
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.output_size = output_size
		
    self.claim_gru = nn.GRU(self.embedding_size, self.hidden_size)
    self.sent_grm = nn.GRU(self.embedding_size, self.hidden_size)
    self.gate_s = nn.Linear(self.hidden_size,1,bias=False)
    self.gate_c = nn.Linear(self.hidden_size,1,bias=False)
    self.atten_c = nn.Linear(self.hidden_size,self.hidden_size)
    self.atten_s = nn.Linear(self.hidden_size,1)
    self.coherence_atten = nn.Softmax(dim=1)
    self.extension = nn.Linear(self.hidden_size*2,self.hidden_size)
    self.joint = nn.Linear(self.hidden_size*4,self.hidden_size,bias=False)
    self.atten_entail = nn.Linear(self.hidden_size,1)
    self.entai_atten = nn.Softmax(dim=0)
    self.final = nn.Linear(self.hidden_size,self.output_size)
		
  def forward(self,claim,sentences):
    _, hc = self.claim_gru(torch.unsqueeze(claim,1))
    hc = torch.squeeze(hc,1)
    HS = torch.zeros(len(sentences),self.hidden_size).to(device)
    for i in range(len(sentences)):
      _,hs = self.sent_grm(torch.unsqueeze(sentences[i],1))
      hs = torch.squeeze(hs,1)
      HS[i] = hs
    gate_s = self.gate_s(HS)
    gate_c = self.gate_c(hc)
    g_c_S = torch.sigmoid(gate_s+gate_c)
    _HS = g_c_S*HS + (1-g_c_S)*hc
    coherence_atten = self.coherence_atten(torch.matmul(self.atten_c(_HS),_HS.t())+self.atten_s(_HS))
    H_apo_S = torch.matmul(coherence_atten,_HS)
    H_tilde_S = torch.tanh(self.extension(torch.cat((HS,H_apo_S),1)))
    H_c_s = torch.zeros_like(H_tilde_S).to(device)
    for i in range(H_tilde_S.size(0)):
      tmp = torch.cat((hc,torch.unsqueeze(H_tilde_S[i],0),hc*H_tilde_S[i],torch.abs(hc-H_tilde_S[i])),1)
      H_c_s[i] = torch.tanh(self.joint(tmp))
    atten_entail = torch.tanh(self.atten_entail(H_c_s))
    entai_atten = self.entai_atten(atten_entail)
    h_c_S = torch.matmul(entai_atten.t(),H_c_s)
    output = self.final(h_c_S)
    output = torch.sigmoid(output)
    return output
