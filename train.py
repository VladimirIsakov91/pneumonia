import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy

from model import CNN
from dataset import ImageDataset


def update(engine, batch):

    sample, labels = batch
    prediction = net(sample.cuda())
    loss = loss_fn(prediction.cpu(), labels)
    loss.backward()
    optimizer.step()
    output = {'loss': loss.item(), 'labels': labels, 'prediction': prediction.cpu()}

    return output


def validate(engine, batch):

    sample, labels = batch
    prediction = net(sample.cuda())

    return prediction.cpu(), labels


def output_transform(output):

    y_pred = output['prediction']
    y = output['labels']

    return y_pred, y


def log_training(engine):

    loss = engine.state.output['loss']
    accuracy = engine.state.metrics['train_acc']
    lr = optimizer.param_groups[0]['lr']
    epoch = engine.state.epoch
    print('Epoch: {0}, Loss: {1}, Accuracy: {2}, Learning Rate: {3}'.format(epoch, loss, accuracy, lr))


def log_validation(engine):

    accuracy = engine.state.metrics['train_acc']
    print('Validation accuracy: {0}'.format(accuracy))


def scheduler_step():
    scheduler.step()


if __name__ == '__main__':

    net = CNN()
    data = ImageDataset(directory='/home/vladimir/MachineLearning/Datasets/chest_xray/train/NORMAL', labels=None)
    loader = DataLoader(dataset=data, batch_size=64, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(params=net.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

    trainer = Engine(update)

    Accuracy(output_transform=output_transform).attach(engine=trainer, name='train_acc')
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_training)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=scheduler_step)

    validator = Engine(validate)
    Accuracy().attach(engine=validator, name='val_acc')
    validator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_validation)



