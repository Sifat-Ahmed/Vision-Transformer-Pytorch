import numpy as np
import os
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.models as models
from Utils.helper import MetricMonitor, calculate_accuracy
from Dataset.utils import get_train_test
from Dataset.dataset import SmokeDataset
from Utils.visualize import visualize_augmentations, plot_curves
from config import Config
from collections import defaultdict

from Models.vit import ViT

import warnings
warnings.filterwarnings('ignore')

cudnn.benchmark = True


def train(train_loader, model, criterion, optimizer, epoch, cfg):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader, position=0, leave=True, colour='green')
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(cfg.device, non_blocking=True)
        target = target.to(
            cfg.device, non_blocking=True).float().view(-1, 1)
        output = model(images)
        loss = criterion(output, target)
        accuracy = calculate_accuracy(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train. {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor)
        )
    return metric_monitor


def validate(val_loader, model, criterion, epoch, cfg):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader, position=0, leave=True, colour='red')
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(cfg.device, non_blocking=True)
            target = target.to(cfg.device, non_blocking=True).float().view(-1, 1)
            output = model(images)
            loss = criterion(output, target)
            accuracy = calculate_accuracy(output, target)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
    return metric_monitor


def main():
    cfg = Config()

    history = defaultdict()
    history['val_loss'], history['train_loss'] = list(), list()
    history['val_acc'], history['train_acc'] = list(), list()

    print('Creating training, validation set')
    (train_images, train_labels), (val_images, val_labels), (test_images,
                                                             test_labels) = get_train_test(cfg.dataset_dir,
                                                                                           validation_size=0.01)
    # print(len(train_images), len(val_images), len(test_images))

    ## Visualize the training images
    ## some random samples are shown with the labels
    # display_image_grid(train_images, train_labels)

    train_dataset = SmokeDataset(
        train_images, train_labels, resize=cfg.resize, transform=cfg.train_transform)

    val_dataset = SmokeDataset(
        val_images, val_labels, resize=cfg.resize, transform=cfg.val_transform)

    ## Visualize the augmented images after augmentation
    visualize_augmentations(train_dataset, idx=233, name='augments.jpg')

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    model = ViT(in_channels=cfg.in_channels,
                image_size=cfg.image_size[0],
                embedding_size=cfg.embedding_size,
                patch_size=cfg.patch_size,
                depth=cfg.depth,
                n_classes=cfg.num_classes)
        #getattr(models, cfg.model_name)(
        #pretrained=False, num_classes=cfg.num_classes, )

    if os.path.isfile(cfg.model_path):
        model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
        print('Saved Model found at', cfg.model_path, ' and loaded')

    model = model.to(cfg.device)
    criterion = nn.BCEWithLogitsLoss().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    BEST_LOSS = np.inf

    for epoch in range(1, cfg.epochs + 1):
        hist = train(train_loader, model, criterion, optimizer, epoch, cfg)
        history['train_loss'].append(hist.get('Loss'))
        history['train_acc'].append(hist.get('Accuracy'))

        hist = validate(val_loader, model, criterion, epoch, cfg)
        if hist.get('Loss') < BEST_LOSS:
            print('Saving model...')
            torch.save(model.state_dict(), cfg.model_path)
            BEST_LOSS = hist.get('Loss')
        history['val_loss'].append(hist.get('Loss'))
        history['val_acc'].append(hist.get('Accuracy'))

    plot_curves(history['train_acc'], history['val_acc'], ylabel='Accuracy', name='accuracy.png')
    plot_curves(history['train_loss'], history['val_loss'], ylabel='Loss', name='loss.png')


if __name__ == '__main__':
    main()
