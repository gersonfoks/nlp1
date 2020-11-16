from torch.utils.data import DataLoader
from practical_2.TreeDataset import TreeDataset, prepare_example
from practical_2.models.BOW import *
from torch.optim import *
from practical_2.callbacks.callbacks import *
from practical_2.prepare import prepare

from practical_2.train import train_model

### For reproducibility.
prepare()

train_dataset = TreeDataset("trees/train.txt")
eval_testset = TreeDataset("trees/dev.txt")
### Now we need to set the tranformation function
transform = lambda example: prepare_example(example, train_dataset.v)

train_dataset.transform = transform
eval_testset.transform = transform

train_dataloader = DataLoader(train_dataset, batch_size=1)
eval_dataloader = DataLoader(eval_testset)
v = train_dataset.v

model = create_bow_model(v)

optimizer = Adam(model.parameters(), lr=0.0005)

eval_callback = ListCallback([
    AccuracyCallback()
])

history = train_model(model, optimizer, train_dataloader, eval_dataloader, eval_callback=eval_callback, n_epochs=10,
                      eval_every=1)

print(history)
