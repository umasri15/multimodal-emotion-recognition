import torch
import torch.nn as nn
from transformers import BertModel

class TextEmotionModel(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = out.pooler_output
        return self.classifier(x)