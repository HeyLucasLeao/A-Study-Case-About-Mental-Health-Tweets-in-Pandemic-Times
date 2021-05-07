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
    losses = []

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = torch.flatten(data['targets'].to(device))

        outputs = model(
            input_ids=input_ids.squeeze(),
            attention_mask=attention_mask.squeeze()
            )

        #função de perda
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        #Train Accuracy
        
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
    losses = []
    with torch.no_grad():
        for data in data_loader:

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = torch.flatten(data['targets'].to(device))

            probability = model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )

            #função de perda
            loss = criterion(probability, targets)
            losses.append(loss.item())

            #Eval Accuracy


    avg_loss = np.mean(losses)
    return avg_loss