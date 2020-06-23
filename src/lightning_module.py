import torch
from argparse import Namespace
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from src.models import *

def get_classifier(arch, pretrained, device='cpu'):
    if arch == 'base_model':
        model = get_base_model(pretrained, device=device)
    else:
        raise NotImplementedError
    return model

class LightModule(pl.LightningModule):
    def __init__(self, hparams, pretrained=False):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.hparams = hparams
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mean = [0.5]
        self.std = [0.2]
        self.model = get_classifier(hparams.classifier, pretrained)
        self.train_size = len(self.train_dataloader().dataset)
        self.val_size = len(self.val_dataloader().dataset)

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = torch.sum(torch.max(predictions, 1)[1] == labels.data).float() / batch[0].size(0)
        return loss, accuracy

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        logs = {'loss/train': loss, 'accuracy/train': accuracy}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_nb):
        avg_loss, accuracy = self.forward(batch)
        loss = avg_loss * batch[0].size(0)
        corrects = accuracy * batch[0].size(0)
        logs = {'loss/val': loss, 'corrects': corrects}
        return logs

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss/val'] for x in outputs]).sum() / self.val_size
        accuracy = torch.stack([x['corrects'] for x in outputs]).sum() / self.val_size
        logs = {'loss/val': loss, 'accuracy/val': accuracy}
        return {'val_loss': loss, 'log': logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        accuracy = self.validation_epoch_end(outputs)['log']['accuracy/val']
        accuracy = round((100 * accuracy).item(), 2)
        return {'progress_bar': {'Accuracy': accuracy}}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay)

        scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
                                                                      steps_per_epoch=self.train_size // self.hparams.batch_size,
                                                                      epochs=self.hparams.max_epochs),
                     'interval': 'step', 'name': 'learning_rate'}
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = FashionMNIST(root=self.hparams.data_dir, train=True, transform=transform_train)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True,
                                drop_last=True, pin_memory=True)
        return dataloader

    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        dataset = FashionMNIST(root=self.hparams.data_dir, train=False, transform=transform_val)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()