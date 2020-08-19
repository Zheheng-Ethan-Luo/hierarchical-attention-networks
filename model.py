import random
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import precision_score, recall_score, f1_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HAN(nn.Module):
    # This is Hierarchical Attention Networks model
    def __init__(self, embedding_size, hidden_size, output_size):
        super(HAN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.claim_gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.sent_gru = nn.GRU(self.embedding_size, self.hidden_size)

        self.gate_s = nn.Linear(self.hidden_size, 1, bias=False)
        self.gate_c = nn.Linear(self.hidden_size, 1, bias=False)

        self.att_c = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.att_s = nn.Linear(self.hidden_size, 1, bias=False)
        self.coherence_att = nn.Softmax(dim=1)
        self.extension = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.joint = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=False)

        self.att_entail = nn.Linear(self.hidden_size, 1)
        self.entai_atten = nn.Softmax(dim=0)
        self.final = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, claim, sentences, device=DEVICE):
        # gru
        _, claim_hidden = self.claim_gru(torch.unsqueeze(claim, 1))
        claim_hidden = torch.squeeze(claim_hidden, 1)
        sents_hidden = torch.zeros(len(sentences), self.hidden_size).to(device)
        for i in range(len(sentences)):
            _, sent_hidden = self.sent_gru(torch.unsqueeze(sentences[i], 1))
            sent_hidden = torch.squeeze(sent_hidden, 1)
            sents_hidden[i] = sent_hidden

        # conherence attention
        gate_s = self.gate_s(sents_hidden)
        gate_c = self.gate_c(claim_hidden)
        g_c_s = torch.sigmoid(gate_s + gate_c)
        gated_sents_hidden = g_c_s * sents_hidden + (1 - g_c_s) * claim_hidden

        coherence_att = self.coherence_att(
                                torch.matmul(self.atten_c(gated_sents_hidden), gated_sents_hidden.t())
                                + self.atten_s(gated_sents_hidden))
        coh_att_sents_hidden = torch.matmul(coherence_att, gated_sents_hidden)
        concat_sents_hidden = torch.tanh(self.extension(torch.cat((sents_hidden, coh_att_sents_hidden), 1)))

        joint_hidden = torch.zeros_like(concat_sents_hidden).to(device)
        for i in range(joint_hidden.size(0)):
            tmp = torch.cat((claim_hidden, torch.unsqueeze(joint_hidden[i], 0),
                             claim_hidden * joint_hidden[i],
                             torch.abs(claim_hidden - joint_hidden[i])), 1)
            joint_hidden[i] = torch.tanh(self.joint(tmp))

        entail_attention = self.entai_atten(torch.tanh(self.atten_entail(joint_hidden)))
        nli_hidden = torch.matmul(entail_attention.t(), sents_hidden)
        output = self.final(nli_hidden)
        output = torch.sigmoid(output)
        return output


def train_batch(model, instances, n_epochs, batch_size=128, learning_rate=0.01):
    optmz = optim.Adagrad(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    n_samples = len(instances)
    print_every = int(n_samples / 200)
    print_average_loss = 0
    for epoch in range(n_epochs):
        print('epoch %s start' % epoch)
        optmz.zero_grad()
        random.shuffle(instances)

        for i in range(n_samples):
            claim = instances[i][0]
            sentences = instances[i][1]
            target = instances[i][2]
            output = model(claim, sentences,device=DEVICE)
            loss = criterion(output, target)
            print_average_loss += loss
            loss /= batch_size
            loss.backward()

            if (i + 1) % batch_size == 0:
                optmz.step()
                optmz.zero_grad()
            if (i + 1) % print_every == 0:
                print_average_loss /= print_every
                print('%s%% finished, loss: %.4f' % (int((i + 1) / n_samples * 100), print_average_loss))
                print_average_loss = 0


def evaluate(model, instances):
    with torch.no_grad():
        y_list=  [instance[2] for instance in instances]
        y_pred_list = []
        for instance in instances:
            claim = instance[0]
            sentences = instance[1]
            target = instance[2]
            output = model(claim, sentences, device=DEVICE)
            if output > 0.5:
                y_pred_list.append(1)
            else:
                y_pred_list.append(0)
        precision = precision_score(y_list, y_pred_list)
        recall = recall_score(y_list, y_pred_list)
        f1score = f1_score(y_list, y_pred_list)
    return precision, recall, f1score
