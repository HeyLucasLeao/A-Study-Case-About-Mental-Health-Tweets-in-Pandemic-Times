import torch
from transformers import AutoModel
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

MODEL = AutoModel.from_pretrained(config['model']['model_name'])


for param in MODEL.parameters():
    MODEL.eval()
    param.requires_grad = False

class Classifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained_model = MODEL
        self.dropout = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(
            self.pretrained_model.config.hidden_size,
            out_features=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.pretrained_model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.sigmoid(output)

