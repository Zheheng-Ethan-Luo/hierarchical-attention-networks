import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import HAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(input,target,han,han_optmz,criterion):
	han_optmz.zero_grad()
	
	claim = input[0]
	sentences = input[1]
	
	loss = 0
	
	output = han(claim,sentences)
	loss += criterion(output,target)
	
	loss.backward()
	
	han_optmz.step()
	
	retun loss.item()
	
def trainIters(han, instances, n_epoches, print_every=100, learning_rate=0.005):
	start = time.time()
	print_loss_total = 0
	
	han_optmz = optim.Adagrad(han.parameters(), lr=learning_rate)
	criterion = nn.NLLLoss()
	for epoch in range(n_epoches):
		for i in range(len(instances)):
			input = instances[i][0]
			target = instances[i][1]
			
			loss = train(input,target,han,han_optmz,criterion)
			print_loss_total += loss
		
		if i % print_every ==0:
			print_loss_avg = print_loss_total/print_every
			print_loss_total = 0
			now = time.time()
			print('%s (%d %d%%) %.4f' % (now-start,i,i/len(instances)*n_epoches,print_loss_avg)

hidden_size = 100
han = HAN(300,100,1).to_device(device)
trainIters(han,instances,20)
