import torch
import torch.nn as nn


class SpeechEmotionModel(nn.Module):

    def __init__(self, input_dim=168, num_classes=7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.float()

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if x.shape[1] != 168:
            x = x[:, :168]

        return self.net(x)