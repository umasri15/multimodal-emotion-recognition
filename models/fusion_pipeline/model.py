import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # speech logits (7) + text logits (7) = 14
        self.fc1 = nn.Linear(14, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, speech_out, text_out):
        if speech_out.dim() == 1:
            speech_out = speech_out.unsqueeze(0)
        if text_out.dim() == 1:
            text_out = text_out.unsqueeze(0)

        x = torch.cat((speech_out, text_out), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x