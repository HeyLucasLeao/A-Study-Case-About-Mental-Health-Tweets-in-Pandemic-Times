import torch
from transformers import AutoModel
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

MODEL = AutoModel.from_pretrained(config['model']['model_name'])
criterion = torch.nn.BCELoss()


for param in MODEL.parameters():
    MODEL.eval()
    param.requires_grad = False

class Classifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained_model = MODEL
        self.linear1 = torch.nn.Linear(
            self.pretrained_model.config.hidden_size,
            out_features=1
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, output = self.pretrained_model.forward(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        output = self.dropout(output)
        output = self.linear1(output)
        output = self.sigmoid(output)
        return output

