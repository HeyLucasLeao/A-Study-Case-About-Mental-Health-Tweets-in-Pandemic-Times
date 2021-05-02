from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import yaml
from transformers import AutoTokenizer, AutoModel
from data import create_dataloader

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

"Constantes determinadas pelo config.yml"

DATA_PATH = config['data']['path_to_data']
MAX_LEN = config['model']['max_seq_length']
BS = config['training']['batch_size']
TOKENIZER = AutoTokenizer.from_pretrained(config['model']['model_name'], do_lower_case=config['model']['do_lower_case'])
TRAIN = pd.read_csv(DATA_PATH + "\\" + config['data']['train_filename'])
TEST = pd.read_csv(DATA_PATH + "\\" + config['data']['test_filename'])
VALID = pd.read_csv(DATA_PATH + "\\" + config['data']['validation_filename'])

"""Normalização de datasets para leitura do modelo"""

train_data_loader = create_dataloader(TRAIN, TOKENIZER, MAX_LEN, BS)
test_data_loader = create_dataloader(TEST, TOKENIZER, MAX_LEN, BS)
valid_data_loader = create_dataloader(VALID, TOKENIZER, MAX_LEN, BS)