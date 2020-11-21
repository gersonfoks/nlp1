from torch import nn
import math
import torch

from practical_2.models.DeepCBOW import zero_init_function
from practical_2.utils import t2i, get_pretrained_weights


class LSTMClassifier(nn.Module):
    """Encodes sentence with an LSTM and projects final hidden state"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(LSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.rnn = MyLSTMCell(embedding_dim, hidden_dim)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),  # explained later
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        B = x.size(0)  # batch size (this is 1 for now, i.e. 1 single example)
        T = x.size(1)  # timesteps (the number of words in the sentence)

        input_ = self.embed(x)

        # here we create initial hidden states containing zeros
        # we use a trick here so that, if input is on the GPU, then so are hx and cx
        hx = input_.new_zeros(B, self.rnn.hidden_size)
        cx = input_.new_zeros(B, self.rnn.hidden_size)

        # process input sentences one word/timestep at a time
        # input is batch-major (i.e., batch size is the first dimension)
        # so the first word(s) is (are) input_[:, 0]
        outputs = []
        for i in range(T):
            hx, cx = self.rnn(input_[:, i], (hx, cx))
            outputs.append(hx)

        # if we have a single example, our final LSTM state is the last hx
        if B == 1:
            final = hx
        else:
            #
            # This part is explained in next section, ignore this else-block for now.
            #
            # We processed sentences with different lengths, so some of the sentences
            # had already finished and we have been adding padding inputs to hx.
            # We select the final state based on the length of each sentence.

            # two lines below not needed if using LSTM from pytorch
            outputs = torch.stack(outputs, dim=0)  # [T, B, D]
            outputs = outputs.transpose(0, 1).contiguous()  # [B, T, D]

            # to be super-sure we're not accidentally indexing the wrong state
            # we zero out positions that are invalid
            pad_positions = (x == 1).unsqueeze(-1)

            outputs = outputs.contiguous()
            outputs = outputs.masked_fill_(pad_positions, 0.)

            mask = (x != 1)  # true for valid positions [B, T]
            lengths = mask.sum(dim=1)  # [B, 1]

            indexes = (lengths - 1) + torch.arange(B, device=x.device, dtype=x.dtype) * T
            final = outputs.view(-1, self.hidden_dim)[indexes]  # [B, D]

        # we use the last hidden state to classify the sentence
        logits = self.output_layer(final)
        return logits

    def load_pretrained_weights(self, vectors):
        '''
            Loads the pretrained weights from the given file.
            Returns the vocab that it created loading the weights
        :param ref:
        :return:
        '''
        self.embed.weight.data.copy_(torch.from_numpy(vectors))
        self.embed.requires_grad = False


class MyLSTMCell(nn.Module):
    """Our own LSTM cell"""

    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(MyLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # YOUR CODE HERE
        self.g = InputHiddenLayer(input_size, hidden_size, torch.tanh)
        self.i = InputHiddenLayer(input_size, hidden_size, torch.sigmoid)
        self.f = InputHiddenLayer(input_size, hidden_size, torch.sigmoid)
        self.o = InputHiddenLayer(input_size, hidden_size, torch.sigmoid)

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, mask=None):
        """
        input is (batch, input_size)
        hx is ((batch, hidden_size), (batch, hidden_size))
        """
        prev_h, prev_c = hx

        # project input and prev state

        # main LSTM computation

        g = self.g(input_, prev_h)
        i = self.i(input_, prev_h)
        f = self.i(input_, prev_h)
        o = self.o(input_, prev_h)

        c = torch.add(torch.mul(g, i), torch.mul(prev_c, f))
        h = torch.mul(torch.tanh(c), o)

        return h, c

    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size)


class InputHiddenLayer(nn.Module):
    '''
    A module that combines the input and the hidden states.
    '''

    def __init__(self, input_dim, hidden_dim, activation_function):
        super(InputHiddenLayer, self).__init__()
        w_in = torch.empty(input_dim, hidden_dim)
        w_hidden = torch.empty(hidden_dim, hidden_dim)
        b = torch.zeros(hidden_dim)

        nn.init.kaiming_normal_(w_in, )
        nn.init.kaiming_normal_(w_hidden, )

        self.w_in = nn.Parameter(data=w_in, requires_grad=True)
        self.w_hidden = nn.Parameter(data=w_hidden, requires_grad=True)
        self.b = nn.Parameter(data=b, requires_grad=True)
        self.activation_function = activation_function

    def forward(self, x, h):
        return self.activation_function(torch.matmul(x, self.w_in) + torch.matmul(h, self.w_hidden) + self.b)


def create_lstm():
    ref = "embeddings/googlenews.word2vec.300d.txt"
    v, vectors = get_pretrained_weights(ref, zero_init_function)
    vocab_size = len(v.w2i)
    n_classes = len(t2i)
    lstm = LSTMClassifier(len(v.w2i), 300, 168, len(t2i), v)
    lstm.load_pretrained_weights(vectors)
    return lstm
