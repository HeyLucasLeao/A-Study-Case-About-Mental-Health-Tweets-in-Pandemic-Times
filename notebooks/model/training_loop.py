from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import yaml
import torch
from collections import defaultdict

from data import create_dataloader
from model import Classifier, criterion
from training_structure import train_epoch, eval_model, optimizer, scheduler

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

"Constantes determinadas pelo config.yml"

DATA_PATH = config['data']['path_to_data']
TRAIN = pd.read_csv(DATA_PATH + "\\" + config['data']['train_filename'])
TEST = pd.read_csv(DATA_PATH + "\\" + config['data']['test_filename'])
VALID = pd.read_csv(DATA_PATH + "\\" + config['data']['validation_filename'])


"Constantes para o modelo e para o treino"

MAX_LEN = config['model']['max_seq_length']
BS = config['training']['batch_size']

"""Normalização de datasets para leitura do modelo"""

TRAIN_DATA_LOADER = create_dataloader(df=TRAIN, max_len=MAX_LEN, bs=BS)
TEST_DATA_LOADER = create_dataloader(df=TEST, max_len=MAX_LEN, bs=BS)
VALID_DATA_LOADER = create_dataloader(df=VALID, max_len=MAX_LEN, bs=BS)


"""Calling Model and sending to CUDA"""

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = Classifier()
criterion = criterion
criterion.to(device)
model.to(device)

"""Training Loop"""

EPOCHS = config['training']['num_epochs']

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"-" * 10)
   
    train_acc, train_loss = train_epoch(
        model, 
        TRAIN_DATA_LOADER,
        criterion,
        optimizer,
        device,
        scheduler,
        len(TRAIN)
        )
    print(f"Train loss {train_loss} accuracy {train_acc}")

    val_acc, val_loss = eval_model(
        model, 
        VALID_DATA_LOADER,
        criterion,
        optimizer,
        device,
        scheduler,
        len(VALID)
        )
    print(f"Val loss {train_loss} accuracy {train_acc}")


history = defaultdict(list)
best_accuracy = 0

history['train_acc'].append(train_acc)
history['train_loss'].append(train_loss)

history['val_acc'].append(val_acc)
history['val_loss'].append(val_loss)

if val_acc > best_accuracy:
    torch.save(model, 'model.pth')
    best_accuracy = val_acc