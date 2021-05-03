from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import yaml
from transformers import AdamW, get_linear_schedule_with_warmup
from data import create_dataloader
import torch
from model import Classifier
from collections import defaultdict

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
model.to(device)

"""Training Loop"""

optimizer = AdamW(
    model.parameters(),
    lr=float(config['training']['learning_rate']),
    correct_bias=False
)

loss_fn = torch.nn.BCELoss()

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=float(config['training']['num_warmup_steps']),
    num_training_steps=int(config['training']['num_epochs'])
)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):

    model = model.train()
    losses = []
    correct_predictions = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['targets'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        _, pred = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):

    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in data_loader:

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )
            _, pred = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

history = defaultdict(list)
best_accuracy = 0

for epoch in range(int(config['training']['num_epochs'])):
    print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
    print(f"-" * 10)

    train_acc, train_loss = train_epoch(
        model, 
        TRAIN_DATA_LOADER,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(TRAIN)
        )
    print(f"Train loss {train_loss} accuracy {train_acc}")

    val_acc, val_loss = eval_model(
        model, 
        VALID_DATA_LOADER,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(VALID)
        )
    print(f"Val loss {train_loss} accuracy {train_acc}")

history['train_acc'].append(train_acc)
history['train_loss'].append(train_loss)

history['val_acc'].append(val_acc)
history['val_loss'].append(val_loss)

if val_acc > best_accuracy:
    torch.save(model, 'model.pth')
    best_accuracy = val_acc