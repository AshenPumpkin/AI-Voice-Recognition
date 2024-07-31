import torch
import torch.nn as nn

class getModelVoice(nn.Module):
    def __init__(self, input_size=551053, hidden_size=128, num_layers=2, num_classes=2):
        super(getModelVoice, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

 
    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length dimension: (batch_size, 1, input_size)
        out, _ = self.lstm1(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        out = self.logsoftmax(out)  # Apply LogSoftmax here
        return out

    def initialize_weights(self):
        for name, param in self.lstm1.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)