import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy

from model import cnn
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
    tr_accuracy = engine.state.metrics['train_acc']
    lr = optimizer.param_groups[0]['lr']
    epoch = engine.state.epoch

    validator.run(loader, max_epochs=1)
    val_accuracy = validator.state.metrics['val_acc']

    print('Epoch: {0}, Loss: {1}, Training Accuracy: {2}, Validation Accuracy: {3}, Learning Rate: {4}'
          .format(epoch, loss, tr_accuracy, val_accuracy, lr))


def scheduler_step():
    scheduler.step()


if __name__ == '__main__':

    net = cnn()
    net.cuda()

    data = ImageDataset(directory='/home/vladimir/MachineLearning/Datasets/chest_xray/train/NORMAL', labels=None)
    loader = DataLoader(dataset=data, batch_size=64, shuffle=True, num_workers=6)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(params=net.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

    trainer = Engine(update)

    Accuracy(output_transform=output_transform).attach(engine=trainer, name='train_acc')
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_training)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=scheduler_step)

    validator = Engine(validate)
    Accuracy().attach(engine=validator, name='val_acc')

    trainer.run(loader, max_epochs=40)




