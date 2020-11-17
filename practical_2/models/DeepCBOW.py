import torch
from torch import nn
import torch.nn.functional as F
from practical_2.utils import t2i, Vocabulary, load_pretrained_weights
import numpy as np


def zero_init_function(n_vectors, vector_size):
    return [[0] * vector_size] * n_vectors


class DeepCBOW(nn.Module):
    """A simple bag-of-words model"""

    def __init__(self, vocab_size, embedding_dim, hidden_layer_dim, vocab, out_size=5):
        super(DeepCBOW, self).__init__()
        self.vocab = vocab

        # this is a trainable look-up table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.l1 = nn.Linear(embedding_dim, hidden_layer_dim)
        self.l2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.l3 = nn.Linear(hidden_layer_dim, 5)

        self.layers = [
            self.l1, self.l2, self.l3

        ]

        # this is a trainable bias term
        self.bias = nn.Parameter(torch.zeros(out_size), requires_grad=True)

    def forward(self, inputs):
        # this is the forward pass of the neural network
        # it applies a function to the input and returns the output

        # this looks up the embeddings for each word ID in inputs
        # the result is a sequence of word embeddings
        x = self.embed(inputs)

        for layer in self.layers:
            x = torch.tanh(layer(x))

        # the output is the sum across the time dimension (1)
        # with the bias term added
        logits = x.sum(1)

        return logits


class PTDeepCBOW(DeepCBOW):
    """A simple bag-of-words model"""

    def __init__(self, vocab_size, embedding_dim, hidden_layer_dim, vocab, out_size=5):
        super(PTDeepCBOW, self).__init__(vocab_size, embedding_dim, hidden_layer_dim, vocab, out_size=out_size)

    def load_pretrained_weights(self, vectors):
        '''
            Loads the pretrained weights from the given file.
            Returns the vocab that it created loading the weights
        :param ref:
        :return:
        '''
        self.embed.weight.data.copy_(torch.from_numpy(vectors))


def create_deep_cbow_model(v):
    vocab_size = len(v.w2i)
    n_classes = len(t2i)
    deep_cbow_model = DeepCBOW(vocab_size, 300, 100, v, n_classes)
    return deep_cbow_model


def create_glove_deep_cbow_model():
    ref = "embeddings/glove.840B.300d.sst.txt"
    v, vectors = load_pretrained_weights(ref, zero_init_function)
    vocab_size = len(v.w2i)
    n_classes = len(t2i)

    model = PTDeepCBOW(vocab_size, 300, 100, v, n_classes)
    model.load_pretrained_weights(vectors)
    return model



def create_w2v_deep_cbow_model():
    ref = "embeddings/googlenews.word2vec.300d.txt"
    v, vectors = load_pretrained_weights(ref, zero_init_function)
    vocab_size = len(v.w2i)
    n_classes = len(t2i)

    model = PTDeepCBOW(vocab_size, 300, 100, v, n_classes)
    model.load_pretrained_weights(vectors)
    return model