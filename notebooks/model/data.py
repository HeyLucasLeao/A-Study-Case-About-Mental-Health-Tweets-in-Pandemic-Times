from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):

    def __init__(self, text, target, tokenizer, max_len):
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'targets': torch.tensor(self.target[item],             dtype=torch.long) 
        }

def create_dataloader(df, tokenizer, max_len, bs, num_workers=4):
    dataset = Dataset(
        text=df['text'].to_numpy(),
        target=df['target'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    data_loader= DataLoader(dataset, bs, num_workers)

    return data_loader