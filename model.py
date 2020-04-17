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
		self.final = nn.Linear(hidden_size,self.output_size)
		
	def forward(self,claim,sentences):
		hc, _ = self.claim_gru(claim)
		HS, _ = self.sent_grm(sentences)
		g_c_S = F.sigmoid(self.gate_s(HS)+self.gate_c(hc))
		_HS = g_c_S*HS + (1-g_c_S)*hc
		coherence_atten = self.coherence_atten(self.atten_c(_HS)*_HS.t()+self.atten_s(_HS))
		H_apo_S = torch.matmul(coherence_atten,_HS)
		H_tilde_S = F.tanh(self.extension(torch.cat((HS,H_apo_S),1)))
		H_c_s = torch.zeros(H_tilde_S.size,dtype = torch.float)
		for i in range(H_tilde_S.size()[0]):
			H_c_s[i] = F.tanh(self.joint(torch.cat((hc,H_tilde_S[i],hc*H_tilde_S[i],torch.abs(hc-H_tilde_S[i])),1)))
		atten_entail = F.tanh(self.atten_entail(H_c_s))
		entai_atten = self.entai_atten(atten_entail)
		h_c_S = torch.matmul(entai_atten.t(),H_c_s)
		output = self.final(h_c_S)
		output = F.softmax(output,dim=1)
