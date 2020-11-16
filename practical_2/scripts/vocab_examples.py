from practical_2.utils import *

LOWER = False  # we will keep the original casing
train_data = list(examplereader("trees/train.txt", lower=LOWER))
dev_data = list(examplereader("trees/dev.txt", lower=LOWER))
test_data = list(examplereader("trees/test.txt", lower=LOWER))

v = Vocabulary()
for data_set in (train_data,):
  for ex in data_set:
    for token in ex.tokens:
      v.count_token(token)

v.build()
print("Vocabulary size:", len(v.w2i))

#First 10 words
print(v.i2w[:10])

# What are the 10 most common words?
print(v.i2w[2:12])

# And how many words are there with frequency 1?
# (A fancy name for these is hapax legomena.)
print(len([word for word in v.w2i.keys() if v.freqs[word] == 1]))



# Finally 20 random words from the vocabulary.
# This is a simple way to get a feeling for the data.
# You could use the `choice` function from the already imported `random` package
words = list(v.w2i.keys())
print([random.choice(words) for i in range(20)])