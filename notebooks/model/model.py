import torch
from transformers import AutoModel, AutoTokenizer
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

MODEL = AutoModel.from_pretrained(config['model']['model_name'])


for param in MODEL.parameters():
    MODEL.eval()
    param.requires_grad = False

class NLP(torch.nn.Module):

    def __init__(self, pretrained_model):
        super(NLP).__init__()
        self.pretrained_model = MODEL
        self.dense1 = torch.nn.Linear(
            in_features=MODEL.config.hidden_size * 
            MODEL.config.max_position_embeddings, 
            out_features=1000
            )
        self.dense2 = torch.nn.Linear(
            in_features=1000,
            out_features=1
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        y_hat = self.pretrained_model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        y_hat = y_hat[0]
        y_hat = y_hat.flatten(start_dim=1)
        y_hat = y_hat = self.dense1(y_hat)
        y_hat = self.relu(y_hat)
        y_hat = self.dropout(y_hat)
        y_hat = self.dense2(y_hat)
        y_hat = self.sigmoid(y_hat)
        return y_hat
