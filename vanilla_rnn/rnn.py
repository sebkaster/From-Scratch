import numpy as np


class RNN:
    def __init__(self, hidden_size: int, vocab_size: int, seq_length: int, learning_rate: float) -> None:
        # hyper parameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # model parameters
        self.U = np.random.uniform(-np.sqrt(1. / vocab_size), np.sqrt(1. / vocab_size), (hidden_size, vocab_size))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (vocab_size, hidden_size))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        self.b = np.zeros((hidden_size, 1))  # bias for hidden state
        self.c = np.zeros((vocab_size, 1))  # bias for output

    def forward(self, inputs, hprev):
        xs, hs, os, ycap = dict(), dict(), dict(), dict()
        hs[-1] = np.copy(hprev)

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1  # one hot encoding, 1 of k
            hs[t] = np.tanh(np.dot(self.V, hs[t]))  # hidden state
            os[t] = np.dot(self.V, hs[t])  # unnormalised log probs for next char
            ycap[t] = self.softmax(os[t])  # probs for next char

        return xs, hs, ycap

    @staticmethod
    def softmax(x):
        p = np.exp(x - np.max(x))
        return p / np.sum(p)

    def loss(self, ps, targets):
        """ cross-entropy loss for a sequence """
        return sum(-np.log(ps[t][targets[t], 0]) for t in range(self.seq_length))

    def backward(self, xs, hs, ycap, targets):
        """ backward pass: compute gradients going backward """

        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        db, dc = np.zeros_like(self.b), np.zeros_like(self.c)

        dh_next = np.zeros_like(hs[0])
        for t in reversed(range(self.seq_length)):
            # start with output
            dy = np.copy(ycap[t])

            # gradient through softmax
            dy[targets[t]] = - 1

            # dV and dc
            dV += np.dot(dy, hs[t].T)
            dc += dc

            # dh has two components, gradient flowing from output and next hidden state
            dh = np.dot(self.V.T, dy) + dh_next  # backprop into h

            dh_rec = (1 - hs[t] * hs[t]) * dh
            db += dh_rec

            dU += np.dot(dh_rec, xs[t].T)
            dW += np.dot(dh_rec, hs[t - 1].T)

            dh_next = np.dot(self.W.T, dh_rec)

        # to migitate gradient explosion, clip the gradients
        for dparam in [dU, dW, dV, db, dc]:
            np.clip(dparam, -5, -5, out=dparam)

        return dU, dW, dV, db, dc
