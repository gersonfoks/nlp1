from torch.utils.data import DataLoader
from practical_2.TreeDataset import TreeDataset, prepare_example, pad_batch
from practical_2.models.BOW import *
from torch.optim import *
from practical_2.callbacks.callbacks import *
from practical_2.models.CBOW import create_cbow_model
from practical_2.models.LSTM import create_lstm
from practical_2.models.MultiheadAttention import create_attention_classifier, create_attention_classifier_with_pos
from practical_2.prepare import prepare
from practical_2.utils import *
from practical_2.train import train_model

### For reproducibility.
prepare()

train_dataset = TreeDataset("trees/train.txt")
eval_testset = TreeDataset("trees/dev.txt")
### Now we need to set the tranformation function

model = create_attention_classifier()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
v = model.vocab

transform = lambda example: prepare_example(example, v)
train_dataset.transform = transform
eval_testset.transform = transform

train_dataloader = DataLoader(train_dataset, batch_size=512, collate_fn=pad_batch)
eval_dataloader = DataLoader(eval_testset, batch_size=512, collate_fn=pad_batch)


optimizer = Adam(model.parameters())

eval_callback = ListCallback([
    AccuracyCallback()
])

history = train_model(model, optimizer, train_dataloader, eval_dataloader, eval_callback=eval_callback, n_epochs=150,
                      eval_every=5)
print(history)
save_history(history, 'histories/attention_pos')
