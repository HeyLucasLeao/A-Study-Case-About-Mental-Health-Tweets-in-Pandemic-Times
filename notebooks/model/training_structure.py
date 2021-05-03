import pandas as pd
import numpy as np
import torch
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

def train_epoch(
                model, 
                data_loader, 
                criterion, 
                optimizer, 
                device, 
                scheduler):

    model.train()
    losses = np.array([])

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['targets'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )        
        
        #função de perda
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        #Back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #atualiza o learning rate
        scheduler.step()

        
    
    avg_loss = np.mean(losses)

    return avg_loss

def eval_model(
            model, 
            data_loader, 
            criterion,
            device):

    model.eval()
    losses = np.array([])

    with torch.no_grad():
        for data in data_loader:

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )

            #função de perda
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            #Back Propagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()

            #atualiza o learning rate
            scheduler.step()
 
    avg_loss = np.mean(losses)

    return avg_loss