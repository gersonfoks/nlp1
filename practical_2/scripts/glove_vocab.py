from practical_2.TreeDataset import TreeDataset
from practical_2.utils import Vocabulary

v = Vocabulary()
vectors = []
v.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
v.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

with open('embeddings/glove.840B.300d.sst.txt', mode='r', encoding="utf-8") as f:


    for line in f:
        line_list = line.split()
        v.add_token(line_list[0])
        vectors.append(line_list[1:])


print(v.i2w[:10])

train_v = TreeDataset("trees/train.txt").v
dev_v = TreeDataset("trees/dev.txt").v
test_v = TreeDataset("trees/test.txt").v

all_words = set(train_v.i2w + dev_v.i2w + test_v.i2w)


glove_words = set(v.i2w)

words_not_found = all_words - glove_words
print(len(all_words))
print(len(words_not_found))
print(len(words_not_found)/ len(all_words))

