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


@click.group()
def main():
    pass


@main.command()
@click.option('--epochs', default=100, type=int)
def train(epochs):

    name = config('global', 'name')
    click.secho(f"[{name}] {locals()}", fg='yellow')

    device = None
    if torch.cuda.is_available():
        click.echo('running on GPU', err=True)
        device = 'cuda'
    else:
        click.echo('running on CPU', err=True)
        device = 'cpu'

    click.secho('Dataset loading...', fg='green', err=True)
    loader_train, loader_test = dataset.load()

    click.secho('Network constructing...', fg='green', err=True)
    net = model.Conv()
    net.to(device)

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    click.secho('Training...', fg='green', err=True)
    for epoch in range(epochs):

        running_loss = 0.0
        running_count = 0
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
            running_count += 1

            # reporting
            click.echo(f"\rEpoch {epoch+1}, iteration {i+1};"
                       f" Train Loss: {running_loss / running_count :.5f};",
                       nl=False)

        # testing
        net.eval()
        testing_loss = 0.0
        testing_count = 0
        testing_acc = 0
        for x, y in loader_test:
            x, y = x.to(device), y.to(device)
            y_pred = net(x)
            loss = criterion(y_pred, y)

            testing_loss += loss.item()
            testing_count += 1

            _, y_pred = torch.max(y_pred.data, 1)
            _, y = torch.max(y.data, 1)
            c = (y_pred == y).squeeze()
            testing_acc += c.sum().item() / len(c)
        click.echo(f" Test Acc: {testing_acc / testing_count :.5f},"
                   f" Test Loss: {testing_loss / testing_count :.5f}")


if __name__ == '__main__':
    main()
