import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


def weights_init(m):
    classname = m.__class__.__name__  # obtain the class name
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("inital  linear weight ")


class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(
            -1, 1, size=(vocab_length, embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(
            torch.from_numpy(w_embeding_random_intial))

    def forward(self, input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        #########################################
        # here you need to define the "self.rnn_lstm"  the input size is "embedding_dim" and the output size is "lstm_hidden_dim"
        # the lstm should have two layers, and the  input and output tensors are provided as (batch, seq, feature)
        # ???
        # self.w1 = nn.Parameter(torch.randn(embedding_dim, lstm_hidden_dim*4))
        # self.u1 = nn.Parameter(torch.randn(lstm_hidden_dim, lstm_hidden_dim*4))
        # self.b1 = nn.Parameter(torch.randn(lstm_hidden_dim*4))
        # self.w2 = nn.Parameter(torch.randn(lstm_hidden_dim, lstm_hidden_dim*4))
        # self.u2 = nn.Parameter(torch.randn(lstm_hidden_dim, lstm_hidden_dim*4))
        # self.b2 = nn.Parameter(torch.randn(lstm_hidden_dim*4))
        # 手写实现太慢了
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=2)

        ##########################################
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
        self.apply(weights_init)  # call the weights initial function.

        self.softmax = nn.LogSoftmax()  # the activation function.
        # self.tanh = nn.Tanh()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

    def _lstm_cell(self, x, h, c, w, u, b):
        gates = torch.mm(x, w) + torch.mm(h, u) + b
        i, f, o, g = torch.split(gates, gates.size(1) // 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def forward(self, sentence, is_test=False):
        batch_input = self.word_embedding_lookup(
            sentence).view(1, -1, self.word_embedding_dim)
        # print(batch_input.size())  # print the size of the input
        ################################################
        # here you need to put the "batch_input"  input the self.lstm which is defined before.
        # the hidden output should be named as output, the initial hidden state and cell state set to zero.
        # ???
        # batch_size, seq_len, _ = batch_input.size()
        # h1 = torch.zeros(batch_size, self.lstm_dim).to(self.device)
        # c1 = torch.zeros(batch_size, self.lstm_dim).to(self.device)
        # h2 = torch.zeros(batch_size, self.lstm_dim).to(self.device)
        # c2 = torch.zeros(batch_size, self.lstm_dim).to(self.device)

        # output = torch.zeros(batch_size, 0, self.lstm_dim).to(self.device)

        # for i in range(seq_len):
        #     x = batch_input[:, i, :]
        #     h1, c1 = self._lstm_cell(x, h1, c1, self.w1, self.u1, self.b1)
        #     h2, c2 = self._lstm_cell(h1, h2, c2, self.w2, self.u2, self.b2)

        #     output = torch.cat((output, h2.unsqueeze(1)), 1)
        output, _ = self.lstm(batch_input)

        ################################################
        out = output.contiguous().view(-1, self.lstm_dim)

        out = F.relu(self.fc(out))

        out = self.softmax(out)

        if is_test:
            prediction = out[-1, :].view(1, -1)
            output = prediction
        else:
            output = out
        # print(out)
        return output
