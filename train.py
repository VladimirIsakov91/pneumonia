import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, ConfusionMatrix
from ignite.handlers import EarlyStopping
import logging

from model import cnn
from dataset import ImageDataset


def update(engine, batch):

    sample, labels = batch

    net.train()
    optimizer.zero_grad()

    with autocast():
        prediction = net(sample.cuda())
        loss = loss_fn(prediction, labels.cuda())

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    output = {'loss': loss.item(), 'labels': labels, 'prediction': prediction.cpu()}

    return output


def validate(engine, batch):

    net.eval()
    sample, labels = batch
    with torch.no_grad():
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

    validator.run(val_loader, max_epochs=1)
    val_accuracy = validator.state.metrics['val_acc']

    logger.info('Epoch: {0}, Loss: {1}, Training Accuracy: {2}, Validation Accuracy: {3}, Learning Rate: {4}'
          .format(epoch, round(loss, 5), round(tr_accuracy, 3), round(val_accuracy, 3), lr))


def scheduler_step(): scheduler.step()


def confusion_matrix(engine):

    validator.run(val_loader, max_epochs=1)
    conf = validator.state.metrics['conf']
    logger.info(conf)


def score_function(engine):

    acc = engine.state.metrics['val_acc']
    return acc


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('ignite.engine.engine.Engine').propagate = False
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    net = cnn()
    net.cuda()

    data = ImageDataset('./output.zarr')
    val = ImageDataset('./val.zarr')

    loader = DataLoader(dataset=data, batch_size=64, shuffle=True, num_workers=6)
    val_loader = DataLoader(dataset=val, batch_size=64, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(params=net.parameters(), lr=0.001, weight_decay=0.0001)
    scaler = GradScaler()
    scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

    trainer = Engine(update)

    Accuracy(output_transform=output_transform).attach(engine=trainer, name='train_acc')

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_training)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=scheduler_step)
    trainer.add_event_handler(event_name=Events.COMPLETED, handler=confusion_matrix)

    validator = Engine(validate)

    Accuracy().attach(engine=validator, name='val_acc')
    ConfusionMatrix(num_classes=2).attach(engine=validator, name='conf')

    handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    validator.add_event_handler(event_name=Events.COMPLETED, handler=handler)

    trainer.run(loader, max_epochs=20)




