import torch
from torch import nn

from practical_2.utils import t2i


class CBOW(nn.Module):
    """A simple bag-of-words model"""

    def __init__(self, vocab_size, embedding_dim, vocab, out_size=5):
        super(CBOW, self).__init__()
        self.vocab = vocab

        # this is a trainable look-up table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.l1 = nn.Linear(embedding_dim, 5)

        # this is a trainable bias term
        self.bias = nn.Parameter(torch.zeros(out_size), requires_grad=True)

    def forward(self, inputs):
        # this is the forward pass of the neural network
        # it applies a function to the input and returns the output

        # this looks up the embeddings for each word ID in inputs
        # the result is a sequence of word embeddings
        embeds = self.embed(inputs)

        x = self.l1(embeds)

        # the output is the sum across the time dimension (1)
        # with the bias term added
        logits = x.sum(1) + self.bias

        return logits


def create_cbow_model(v):
    vocab_size = len(v.w2i)
    n_classes = len(t2i)
    bow_model = CBOW(vocab_size, 300, v, n_classes)
    return bow_model
