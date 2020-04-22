import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import HAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_batch(model,instances,n_epoches,batch_size=100,learning_rate=0.01):
	optmz = optim.Adagrad(model.parameters(), lr=learning_rate)
	criterion = nn.BCELoss()
	optmz.zero_grad()
	n_samples = len(instances)
	print_every = int(n_samples/20)
	print_average_loss = 0
	for epoch in range(n_epoches):
		print('epoch %s start' % epoch)
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
			
hidden_size = 100
embedding_size = 50
han = HAN(embedding_size,hidden_size,1).to(device)
train_batch(han,instances,n_epoches=100)
