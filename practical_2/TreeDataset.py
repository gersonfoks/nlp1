import torch
from torch.utils.data import Dataset
from practical_2.utils import *


class TreeDataset(Dataset):

    def __init__(self, file_ref, transform=None, LOWER=False):
        """
        Args:
            file_ref (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = list(examplereader(file_ref, lower=LOWER))
        self.transform = transform

        ### this dataset automatically creates a vocab
        self.prepare_vocab()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = sample.tokens
        y = sample.label

        if self.transform:
            x, y = self.transform(sample)

        return x, y

    def prepare_vocab(self):
        self.v = Vocabulary()
        for data_set in (self.data,):
            for ex in data_set:
                for token in ex.tokens:
                    self.v.count_token(token)

        self.v.build()


def prepare_example(example, vocab):
    """
    Map tokens to their IDs for a single example
    """

    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    x = [vocab.w2i.get(t, 0) for t in example.tokens]

    x = torch.LongTensor(x)

    y = example.label

    return x, y


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def pad_batch(mb):
    """
    Minibatch is a list of examples.

    torch tensors to be used as input/targets.
    """
    x = [sample[0] for sample in mb]
    y = [sample[1] for sample in mb]

    batch_size = len(mb)
    maxlen = max([len(ex) for ex in x])

    # vocab returns 0 if the word is not there
    x = [pad(ex.tolist(), maxlen) for ex in x]

    x = torch.LongTensor(x)

    y = torch.LongTensor(y)



    return x, y
