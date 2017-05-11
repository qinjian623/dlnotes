import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def window(self, C, offset):
        inputs = torch.LongTensor(C*2)
        for i in range(offset, offset+C):
            inputs[i-offset] = self.train[i]
        for i in range(offset+C+1, offset+2*C+1):
            inputs[i - (offset+C+1)] = self.train[i]
        return inputs, self.train[offset+C]
    
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class Word2Vec(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, d, ntoken, tie_weights=False):
        super(Word2Vec, self).__init__()
        self.encoder = nn.Embedding(ntoken, d)
        self.decoder = nn.Linear(d, ntoken)
        self.init_weights()
        self.d = d

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        sum_emb = torch.zeros(self.d)
        for input in inputs:
            emb = self.encoder(input)
            sum_emb.add_(emb)
        decoded = self.decoder(sum_emb)
        return F.log_softmax(decoded)


parser = argparse.ArgumentParser(description='PyTorch word2vec')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

model = Word2Vec()
model.cuda()
corpus = Corpus(args.data)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train():
    model.train()
    C = 2
    iters_of_epoch = corpus.train.size(0) - 2 * C
    total_loss = 0
    for i in range(iters_of_epoch):
        offset = C
        inputs, target = corpus.window(C, offset)
        offset += 1
        model.zero_grad()
        output = model(inputs)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        if i % 1000 == 0:
            print(total_loss/100)
            total_loss = 0


try:
    for epoch in range(1, 10):
        train()
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    with open(args.save, 'wb') as f:
        torch.save(model, f)
