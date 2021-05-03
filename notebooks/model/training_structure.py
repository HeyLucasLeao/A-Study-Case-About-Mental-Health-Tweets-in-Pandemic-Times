import pandas as pd
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from model import MODEL
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

optimizer = AdamW(
    MODEL.parameters(),
    lr=float(config['training']['learning_rate']),
    correct_bias=False
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config['training']['num_warmup_steps'],
    num_training_steps=config['training']['num_epochs']
)

def train_epoch(
                model, 
                data_loader, 
                criterion, 
                optimizer, 
                device, 
                scheduler, 
                n_examples):

    model.train()
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
        _, preds = torch.max(outputs, dim=1)
        
        
        loss = criterion(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    
    train_loss = (correct_predictions.double() / n_examples)
    train_acc = (np.mean(losses))

    return train_loss, train_acc

def eval_model(
            model, 
            data_loader, 
            criterion,
            device, 
            n_examples):

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
            _, preds = torch.max(outputs, dim=1)

            loss = criterion(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    valid_loss = (correct_predictions.double() / n_examples)
    valid_acc = (np.mean(losses))

    return valid_loss, valid_acc