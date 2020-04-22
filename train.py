import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random

def train_batch(model,instances,n_epoches,batch_size=100,learning_rate=0.01):
	optmz = optim.Adagrad(model.parameters(), lr=learning_rate)
	criterion = nn.BCELoss()
	n_samples = len(instances)
	print_every = int(n_samples/20)
	print_average_loss = 0
	for epoch in range(n_epoches):
		print('epoch %s start' % epoch)
		optmz.zero_grad()
		random.shuffle(instances)
		for i in range(n_samples):
			claim = instances[i][0]
			sentences = instances[i][1]
			target = instances[i][2]
			output = model(claim,sentences)
			loss = criterion(output,target)
			print_average_loss += loss
			loss = loss/batch_size
			loss.backward()
			
			if (i+1)% batch_size == 0:
				optmz.step()
				optmz.zero_grad()
			if (i+1)% print_every == 0:
			  print_average_loss /= print_every
			  print('%s%% finished, loss: %.4f' % (int((i+1)/n_samples*100),print_average_loss))
			  print_average_loss = 0
			
def evaluate(model,instances):
	tp=fp=tn=fn = 0
	for instance in instances:
		claim = instance[0]
		sentences = instance[1]
		target = instance[2]
		output = model(claim,sentences)
		if output > 0.5:
			output = 1
		else:
			output = 0
		if target ==1:
			if output==1:
				tp+=1
			else:
				fn+=1
		else:
			if output==1:
				fp+=1
			else:
				tn+=1
	p = tp/(tp+fp)
	r = tp/(tp+fn)
	f1 = 2*p*r/(p+r)
	return p,r,f1
