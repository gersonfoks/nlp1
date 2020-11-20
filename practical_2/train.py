import torch
from torch import nn
from tqdm import tqdm


def train_model(model, optimizer, train_dataloader, eval_dataloader, n_epochs=20, print_every=5, eval_every=5,
                eval_callback=None):
    '''
    We make use of the wonderfull functions that pytorch has to offer to train our data.

    :param model:
    :param optimizer:
    :param train_dataloader:
    :param eval_dataloader:
    :param n_epochs:
    :param print_every:
    :param eval_every:
    :param eval_callbacks:
    :return:
    '''
    history = {"train_loss": [],
               "train_eval": [],
               "test_loss": [],
               "test_eval": [],
               "settings": {
                   "n_epochs": n_epochs,
                   'eval_every': eval_every
               }
               }  # Object that keeps track of the history of
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    criterion = nn.CrossEntropyLoss()  # loss function

    ### Training
    for epoch in tqdm(range(n_epochs)):

        ## The training loop
        loss_epoch = 0
        n_steps = 0
        callback_values = []
        for x, y in train_dataloader:
            optimizer.zero_grad()

            n_steps += 1

            # Put it on gpu
            x, y = x.to(device), y.to(device)

            # Forward pass
            out = model.forward(x)

            loss = criterion(out, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Keep track of statistics
            loss_epoch += loss.item()

            if eval_callback:
                eval_callback.forward(out, y)

        history["train_loss"].append(loss_epoch / n_steps)
        if eval_callback:
            history["train_eval"].append(eval_callback.accumulate())

        ### Evaluation
        if epoch % eval_every == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():

                loss_eval = 0
                n_steps = 0
                callback_values = []
                for x, y in eval_dataloader:

                    optimizer.zero_grad()

                    n_steps += 1

                    # Put it on gpu
                    x, y = x.to(device), y.to(device)

                    # Forward pass
                    out = model.forward(x)

                    loss = criterion(out, y)

                    # Keep track of statistics
                    loss_eval += loss.item()

                    if eval_callback:
                        eval_callback.forward(out, y)

                history["test_loss"].append(loss_eval / n_steps)
                if eval_callback:
                    history["test_eval"].append(eval_callback.accumulate())

                model.train()

    return history
