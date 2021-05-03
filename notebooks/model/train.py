from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import yaml
from transformers import AutoTokenizer, AutoModel
from data import create_dataloader
import torch
from model import Classifier


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
TOKENIZER = AutoTokenizer.from_pretrained(config['model']['model_name'], 
do_lower_case=config['model']['do_lower_case'])

"""Normalização de datasets para leitura do modelo"""

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
TRAIN_DATA_LOADER = create_dataloader(TRAIN, TOKENIZER, MAX_LEN, BS).to(device)
TEST_DATA_LOADER = create_dataloader(TEST, TOKENIZER, MAX_LEN, BS).to(device)
VALID_DATA_LOADER = create_dataloader(VALID, TOKENIZER, MAX_LEN, BS).to(device)

"""Calling Model and sending to CUDA"""

model = Classifier()
model.to(device)

"""Training Loop"""

