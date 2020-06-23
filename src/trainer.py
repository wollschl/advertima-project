from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import sys
from src.models.base_model import Model
from src.data.fashion_mnist import fashion_mnist
from torch.optim.adam import Adam
import numpy as np

class Trainer():
    def __init__(self, model, n_epochs=10, criterion=nn.CrossEntropyLoss):

        self.trainset, self.testset = fashion_mnist()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.criterion = criterion()
        self.optim = Adam(model.parameters(), lr=0.01)

    def train(self, n_epochs, batch_size, print_every=100):
        losses = []
        accuracies = []
        for e in range(n_epochs):
            iter = 0
            for x,y in tqdm(DataLoader(self.trainset, batch_size=batch_size)):
                x,y = x.to(self.device), y.to(self.device)

                def closure():
                    self.optim.zero_grad()
                    # returning output and feature map positive and negative
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    loss.backward()
                    losses.append(loss.item())
                    accuracies.append((pred.argmax(1) == y).float().mean())
                    if iter % print_every == 0:
                        print('\r' + ' epoch ' + str(e) +
                              ' |  loss : ' + str(loss.item()) +
                              ' | acc : ' + str(accuracies[-1]))
                    return loss

                self.optim.step(closure)
                iter += 1
    def test(self):
        pass


def main():
    model = Model()
    trainer = Trainer(model)
    trainer.train(n_epochs=10, batch_size=128)

if __name__ == "__main__":
    main()