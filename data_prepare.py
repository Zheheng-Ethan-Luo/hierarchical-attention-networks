import torch
import numpy as np


def load_word_emb(wvFile):
    glove = {}
    word_index_dict = {}
    weights = []
    f = open(wvFile, 'r')
    for i, line in enumerate(f):
        splits = line.split()
        word = splits[0]
        word_index_dict[word] = i
        embedding = np.array([float(val) for val in splits[1:]])
        glove[word] = embedding
        weights.append(embedding)

    emb_dim = embedding.shape[0]
    weights.append(np.random.normal(scale=0.6, size=(emb_dim,)))  # for unknown token
    weights = np.asarray(weights)
    print("word embedding initialization finished")
    return glove, weights, word_index_dict, emb_dim


def word2index(wl, word_index_dict, weights):
    wl_new = []
    for word in wl:
        if word in word2index:
            wl_new.append(word_index_dict[word])
        else:
            wl_new.append(weights.shape[0] - 1)
    return wl_new


def index2tensor(sent, weights, embedding_dim):
    matrix = torch.zeros((len(sent), embedding_dim))
    for i in range(len(sent)):
        matrix[i] = torch.from_numpy(weights[sent[i]])
    return matrix


def build_instance(claim, sentence, target, tokenizer, word_index_dict, emb_dim, weights,device):
    instance = []
    claim_wl = tokenizer(claim.lower())
    claim_indice = word2index(claim_wl, word_index_dict, weights)

    sentence_wls = [tokenizer(s.lower()) for s in sentence]
    sent_indice_l = [word2index(s, word_index_dict, weights) for s in sentence_wls]

    claim_input = index2tensor(claim_indice, weights, emb_dim)
    sentence_input = [index2tensor(s, weights, emb_dim) for s in sent_indice_l]

    instance.append(claim_input)
    instance.append(sentence_input)
    if target in ['true', 'mostly true']:
        instance.append(torch.tensor([1.]).to(device))
    else:
        instance.append(torch.tensor([0.]).to(device))
    return instance


def snopes_pre_process(snopes, tokenizer, word_index_dict, emb_dim, weights,device):
    claim_ids = snopes['<claim_id>']
    claim_ids = claim_ids.drop_duplicates()
    claim_ids = claim_ids.to_list()

    claims = []
    sentences = []
    targets = []
    for id in claim_ids:
        claim = snopes.loc[snopes['<claim_id>'] == id]['<claim_text>'].drop_duplicates().to_list()[0]
        evidence = snopes.loc[snopes['<claim_id>'] == id]['<evidence>'].to_list()
        label = snopes.loc[snopes['<claim_id>'] == id]['<cred_label>'].drop_duplicates().to_list()[0]
        claims.append(claim)
        sentences.append(evidence)
        targets.append(label)

    instances = [build_instance(claims[i],
                                sentences[i],
                                targets[i],
                                tokenizer,
                                word_index_dict,
                                emb_dim,
                                weights,
                                device
                                )
                 for i in range(len(claims))]

    return instances


def k_fold(instances, k):
    instances_T = [instance for instance in instances if instance[2] == 1]
    instances_F = [instance for instance in instances if instance[2] == 0]

    divide_t = int(len(instances_T) / 10)
    divide_f = int(len(instances_F) / 10)

    instances_train = instances_T[divide_t:] + instances_F[divide_f:]
    instances_test = instances_T[:divide_t] + instances_F[:divide_f]

    return instances_train, instances_test
