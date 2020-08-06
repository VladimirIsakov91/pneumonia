import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, ConfusionMatrix
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.neptune_logger import *
import logging
import os

from model import cnn
from dataset import ImageDataset
from transformations import train_tr, val_tr


def update(engine, batch):

    sample, labels = batch

    net.train()
    optimizer.zero_grad()

    #with autocast():
    prediction = net(sample.cuda())
    loss = loss_fn(prediction, labels.cuda())
    loss.backward()
    optimizer.step()
    #scaler.scale(loss).backward()
    #scaler.step(optimizer)
    #scaler.update()

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

    validator.run(val_loader)
    val_accuracy = validator.state.metrics['val_acc']

    logger.info('Epoch: {0}, Loss: {1}, Training Accuracy: {2}, Validation Accuracy: {3}, Learning Rate: {4}'
          .format(epoch, round(loss, 5), round(tr_accuracy, 3), round(val_accuracy, 3), lr))


def confusion_matrix(engine):

    conf = validator.state.metrics['conf']
    logger.info(conf)


def scheduler_step(): scheduler.step()
def score_function(engine): return engine.state.metrics['val_acc']
def end_logging(engine): nplogger.close()


if __name__ == '__main__':

    batch_size = 64
    lr = 0.001
    weight_decay = 0.0001
    step_size = 5
    gamma = 0.1
    epochs = 20
    patience = 3

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('ignite.engine.engine.Engine').propagate = False
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    nplogger = NeptuneLogger(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                           project_name="vladimir.isakov/sandbox",
                           experiment_name='pneumonia',
                           upload_source_files='./train.py',
                           params={'batch_size': batch_size,
                                   'epochs': epochs,
                                   'lr': lr,
                                   'step_size': step_size,
                                   'gamma': gamma,
                                   'weight_decay': weight_decay})

    net = cnn()
    net.cuda()

    data = ImageDataset('./train.zarr', transform=train_tr)
    val = ImageDataset('./val.zarr', transform=val_tr)

    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()
    scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    trainer = Engine(update)

    Accuracy(output_transform=output_transform).attach(engine=trainer, name='train_acc')
    Loss(loss_fn=loss_fn, output_transform=output_transform).attach(engine=trainer, name='loss')

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_training)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=scheduler_step)
    trainer.add_event_handler(event_name=Events.COMPLETED, handler=confusion_matrix)
    trainer.add_event_handler(event_name=Events.COMPLETED, handler=end_logging)

    validator = Engine(validate)

    Accuracy().attach(engine=validator, name='val_acc')
    ConfusionMatrix(num_classes=2).attach(engine=validator, name='conf')

    handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
    validator.add_event_handler(event_name=Events.COMPLETED, handler=handler)

    nplogger.attach(trainer,
                  log_handler=OutputHandler(tag='train',
                                            metric_names=['train_acc', 'loss']),
                  event_name=Events.EPOCH_COMPLETED)

    nplogger.attach(trainer,
                  log_handler=OptimizerParamsHandler(tag='optimizer',
                                                     optimizer=optimizer,
                                                     param_name='lr'),
                  event_name=Events.EPOCH_COMPLETED)

    nplogger.attach(validator,
                  log_handler=OutputHandler(tag='validate',
                                            metric_names=['val_acc'],
                                            global_step_transform=global_step_from_engine(trainer)),
                  event_name=Events.EPOCH_COMPLETED)

    trainer.run(loader, max_epochs=20)



