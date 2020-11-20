import pickle
import re
import random
import time
import math
import numpy as np
import nltk
import matplotlib.pyplot as plt

plt.style.use('default')
from collections import namedtuple
from nltk import Tree
from collections import namedtuple
from nltk import Tree
from collections import Counter, OrderedDict, defaultdict

# A simple way to define a class is using namedtuple.
Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])


# this function reads in a textfile and fixes an issue with "\\"
def filereader(path):
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.sub(r"\([0-9] |\)", "", s).split()


SHIFT = 0
REDUCE = 1


def transitions_from_treestring(s):
    s = re.sub("\([0-5] ([^)]+)\)", "0", s)
    s = re.sub("\)", " )", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\)", "1", s)
    return list(map(int, s.split()))


def examplereader(path, lower=False):
    """Returns all examples in a file one by one."""
    for line in filereader(path):
        line = line.lower() if lower else line
        tokens = tokens_from_treestring(line)
        tree = Tree.fromstring(line)  # use NLTK's Tree
        label = int(line[1])
        trans = transitions_from_treestring(line)
        yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,
                           OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        '''
        min_freq: minimum number of occurrences for a word to be included
                  in the vocabulary
        '''
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)


# Now let's map the sentiment labels 0-4 to a more readable form
i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
# And let's also create the opposite mapping.
# We won't use a Vocabulary for this (although we could), since the labels
# are already numeric.
t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})


def save_history(history, file_ref):
    with open(file_ref, 'wb') as fp:
        pickle.dump(history, fp)


def load_history(file_ref):
    with open(file_ref, 'rb') as fp:
        history = pickle.load(fp)
    return history


### Utils for plotting history
def plot_history(history, title, flag=True):
    save_title = "".join(title.split())

    ## First plot the train and test loss
    x = np.arange(len(history['train_loss']))
    plt.plot(x, history['train_loss'], label='Train Loss')

    eval_freq = history['settings']['eval_every']
    x = np.arange(eval_freq * len(history['test_loss']), step=eval_freq)

    plt.plot(x, history['test_loss'], label='Test Loss')

    plt.legend()
    plt.title("Losses of {}".format(title))

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("{}_losses".format(save_title))
    plt.show()

    ## Secondly we plot all the other eval functions.
    ### First we get the names:
    names = list(history['train_eval'][0].keys())
    for name in names:
        x = np.arange(len(history['train_eval']))
        y = [history['train_eval'][i][name] for i in x]

        plt.plot(x, y, label='Train {}'.format(name))

        x = np.arange(eval_freq * len(history['test_eval']), step=eval_freq)
        y = [history['test_eval'][i][name] for i in range(len(history['test_eval']), )]
        plt.plot(x, y, label='Test {}'.format(name))
        plt.legend()
        plt.title("Accuracy of {}".format(title))
        plt.xlabel("Number of epochs")
        plt.ylabel(name)
        plt.savefig("{}_{}".format(save_title, name))
        plt.show()


def get_pretrained_weights(ref, init_func=np.zeros):
    v = Vocabulary()
    vectors = []
    v.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
    v.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

    with open(ref, mode='r', encoding="utf-8") as f:
        for line in f:
            line_list = line.split()
            v.add_token(line_list[0])
            vectors.append(line_list[1:])
    vector_len = len(vectors[0])

    vectors = np.array(init_func(2, vector_len) + vectors).astype(float)

    return v, vectors
