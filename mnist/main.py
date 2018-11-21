import configparser
import os

import click
import torch
import torch.nn as nn
import torch.optim as optim

import dataset.train1000
import model

_conf = None


def config(section, key, type=str):
    """Get a parameter from `./config`

    Parameters
    ----------
    section: str
    key: str
    """
    global _conf
    if _conf is None:
        _conf = configparser.ConfigParser(
            defaults=os.environ,
            interpolation=configparser.ExtendedInterpolation())
        _conf.read('config')
    return type(_conf.get(section, key))


def to_onehot_sparse(x):
    """make one-hot vector by argmax

    Parameters
    ----------
    Tensor(shape=(batches, n))

    Returns
    -------
    Tensor(shape=(batches,))
    """
    _, x = torch.max(x.data, 1)
    return x


@click.group()
def main():
    pass


@main.command()
@click.option('--epochs', default=200, type=int)
def train(epochs):

    # config
    name = config('global', 'name')
    model_name = config('global', 'model')
    if model_name == 'MixFeatConv':
        sigma = config('model', 'sigma', type=float)

    hyperparams = locals()
    name = config('global', 'name')
    click.secho(f"[{name}] {hyperparams}", fg='yellow')

    # gpu?
    device = None
    if torch.cuda.is_available():
        click.echo('running on GPU', err=True)
        device = 'cuda'
    else:
        click.echo('running on CPU', err=True)
        device = 'cpu'

    # dataset
    click.secho('Dataset loading...', fg='green', err=True)
    loader_train, loader_test = dataset.load()

    # model network
    click.secho('Network constructing...', fg='green', err=True)
    net = getattr(model, model_name)()
    if model_name == 'Conv':
        net = model.Conv()
    elif model_name == 'MixFeatConv':
        net = model.MixFeatConv(sigma=sigma)
    click.echo(str(net), err=True)
    net.to(device)

    criterion = nn.MultiLabelSoftMarginLoss(reduction='sum')
    criterion_eval = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # training
    click.secho('Training...', fg='green', err=True)
    for epoch in range(epochs):

        running_loss = 0.0
        running_entropy = 0.0
        running_count = 0
        running_acc = 0
        for i, (x, y) in enumerate(loader_train):

            x, y = x.to(device), y.to(device)

            # train
            net.train()
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_entropy += criterion_eval(y_pred, to_onehot_sparse(y)).item()
            running_count += len(x)

            _, y_pred = torch.max(y_pred.data, 1)
            _, y = torch.max(y.data, 1)
            c = (y_pred == y).squeeze()
            running_acc += c.sum().item()

        # reporting
        click.echo(f"Epoch: {epoch+1} "
                   f" Train-Loss: {running_loss / running_count :.5f}"
                   f" Train-Acc: {running_acc / running_count :.5f}"
                   f" Train-Entropy: {running_entropy / running_count :.5f}",
                   nl=False)

        # testing
        net.eval()
        testing_count = 0
        testing_entropy = 0.0
        testing_acc = 0
        for x, y in loader_test:
            x, y = x.to(device), y.to(device)
            y_pred = net(x)
            testing_entropy += criterion_eval(y_pred, to_onehot_sparse(y)).item()
            testing_count += len(x)

            _, y_pred = torch.max(y_pred.data, 1)
            _, y = torch.max(y.data, 1)
            c = (y_pred == y).squeeze()
            testing_acc += c.sum().item()

        click.echo(f" Test-Acc: {testing_acc / testing_count :.5f}"
                   f" Test-Entropy: {testing_entropy / testing_count :.5f}")


if __name__ == '__main__':
    main()
