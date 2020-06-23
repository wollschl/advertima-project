import os
import torch
from pathlib import Path
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from src.lightning_module import LightModule


def main(hparams):
    seed_everything(0)

    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str:
        if len(hparams.gpus) == 2:  # GPU number and comma e.g. '0,' or '1,'
            torch.cuda.set_device(int(hparams.gpus[0]))

    # Model
    classifier = LightModule(hparams)

    # Trainer
    lr_logger = LearningRateLogger()
    logger = TensorBoardLogger("../logs", name=hparams.classifier)
    trainer = Trainer(callbacks=[lr_logger], gpus=hparams.gpus, max_epochs=hparams.max_epochs,
                      deterministic=True, early_stop_callback=False, logger=logger)
    trainer.fit(classifier)

    # Load best checkpoint
    checkpoint_path = os.path.join(Path(os.getcwd()).parent, 'logs', hparams.classifier, 'version_' + str(classifier.logger.version),
                                   'checkpoints')
    classifier = LightModule.load_from_checkpoint(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))

    # Save weights from checkpoint
    statedict_path = os.path.join(os.getcwd(), '..', 'models', hparams.classifier + '.pt')
    torch.save(classifier.model.state_dict(), statedict_path)

    # Test model
    trainer.test(classifier)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='base_model')
    parser.add_argument('--data_dir', type=str, default='../data/FashionMNIST')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--gpus', default='0,')  # use None to train on CPU
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    args = parser.parse_args()
    main(args)