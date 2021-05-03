from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)


TOKENIZER = AutoTokenizer.from_pretrained(config['model']['model_name'], do_lower_case=True)

class ShapingDataset(Dataset):

    def __init__(self, text, target, max_len):
        super().__init__()
        self.text = text
        self.tokenizer = TOKENIZER
        self.target = target
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        encoding = self.tokenizer(
        text,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        max_length=int(config['model']['max_seq_length'])
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'targets': torch.tensor(self.target[item], dtype=torch.double) 
        }

def create_dataloader(df, max_len, bs, num_workers=4):
    dataset = ShapingDataset(
        text=df['text'].to_numpy(),
        target=df['target'].to_numpy(),
        max_len=max_len
    )
    data_loader = DataLoader(dataset, bs, num_workers)

    return data_loader