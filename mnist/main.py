import click
import torch
import torch.nn as nn
import torch.optim as optim

import dataset.train1000
import model


@click.group()
def main():
    pass


@main.command()
@click.option('--epochs', default=10, type=int)
def train(epochs):

    click.secho('Dataset loading...', fg='green', err=True)
    loader_train, loader_test = dataset.load()

    click.secho('Network constructing...', fg='green', err=True)
    net = model.Conv()
    if torch.cuda.is_available():
        click.echo('running on GPU', err=True)
        net.cuda()
    else:
        click.echo('running on CPU', err=True)

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    click.secho('Training...', fg='green', err=True)
    VIEW_INTERVAL = 1
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(loader_train):
            x, y = data
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # report loss
            running_loss += loss.item()
            if i % VIEW_INTERVAL == VIEW_INTERVAL - 1:

                running_loss /= VIEW_INTERVAL
                click.echo(f"Epoch {epoch+1}, iteration {i+1}; loss: {(running_loss):.3f}")
                running_loss = 0

                # testing
                # count_correct = 0
                # count_total = 0
                # for x, y in loader_test:
                #     if torch.cuda.is_available():
                #         x, y = x.cuda(), y.cuda()
                #     y_pred = net(x)
                #     _, y_pred = torch.max(y_pred.data, 1)
                #     c = (y_pred == y).squeeze()
                #     count_correct += c.sum().item()
                #     count_total += len(c)
                # click.echo(f"  Test Acc: {100.0 * count_correct / count_total :.3f}%")


if __name__ == '__main__':
    main()
