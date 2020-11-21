from torch import nn
from torch.nn.functional import softmax
import torch
import math

from practical_2.models.DeepCBOW import zero_init_function
from practical_2.utils import t2i, get_pretrained_weights


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]

        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class MultiheadAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_heads, vocab):
        super().__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)

        self.multihead_attention = MultiheadAttention(embedding_dim, hidden_dim, n_heads)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        embedding = self.embed(x)

        x = self.multihead_attention(embedding)
        x = x.sum(dim=1)

        out = self.output_layer(x)
        return out

    def load_pretrained_weights(self, vectors):
        '''
            Loads the pretrained weights from the given file.
            Returns the vocab that it created loading the weights
        :param ref:
        :return:
        '''
        self.embed.weight.data.copy_(torch.from_numpy(vectors))
        self.embed.requires_grad = False


def create_attention_classifier():
    ref = "embeddings/googlenews.word2vec.300d.txt"
    v, vectors = get_pretrained_weights(ref, zero_init_function)
    vocab_size = len(v.w2i)
    n_classes = len(t2i)
    model = MultiheadAttentionClassifier(len(v.w2i), 300, 200, len(t2i), 4, v)
    model.load_pretrained_weights(vectors)
    return model


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiheadAttentionClassifierWithPos(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_heads, vocab):
        super().__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        self.multihead_attention = MultiheadAttention(embedding_dim, hidden_dim, n_heads)

        self.output_layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        embedding = self.embed(x)
        pos_x = self.pos_encoding(embedding)
        x = self.multihead_attention(pos_x)
        x = x.sum(dim=1)

        out = self.output_layer(x)
        return out

    def load_pretrained_weights(self, vectors):
        '''
            Loads the pretrained weights from the given file.
            Returns the vocab that it created loading the weights
        :param ref:
        :return:
        '''
        self.embed.weight.data.copy_(torch.from_numpy(vectors))
        self.embed.requires_grad = False


def create_attention_classifier_with_pos():
    ref = "embeddings/googlenews.word2vec.300d.txt"
    v, vectors = get_pretrained_weights(ref, zero_init_function)
    vocab_size = len(v.w2i)
    n_classes = len(t2i)
    model = MultiheadAttentionClassifierWithPos(len(v.w2i), 300, 200, len(t2i), 4, v)
    model.load_pretrained_weights(vectors)
    return model
