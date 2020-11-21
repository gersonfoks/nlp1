from torch.utils.data import DataLoader
from practical_2.TreeDataset import TreeDataset, prepare_treelstm_minibatch
from practical_2.models.BOW import *
from torch.optim import *
from practical_2.callbacks.callbacks import *
from practical_2.models.TreeLSTM import create_tree_lstm
from practical_2.prepare import prepare
from practical_2.utils import *
from practical_2.train import train_model

### For reproducibility.
prepare()

train_dataset = TreeDataset("trees/train.txt")
eval_testset = TreeDataset("trees/dev.txt")
### Now we need to set the tranformation function

model = create_tree_lstm()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
v = model.vocab

collate_fn = lambda batch: prepare_treelstm_minibatch(batch, model.vocab)

train_dataloader = DataLoader(train_dataset, batch_size=512, collate_fn=collate_fn)
eval_dataloader = DataLoader(eval_testset, batch_size=512, collate_fn=collate_fn)

optimizer = Adam(model.parameters(), lr=2e-4)

eval_callback = ListCallback([
    AccuracyCallback()
])

history = train_model(model, optimizer, train_dataloader, eval_dataloader, eval_callback=eval_callback, n_epochs=10,
                      eval_every=5)
print(history)
save_history(history, 'histories/tree_lstm')
