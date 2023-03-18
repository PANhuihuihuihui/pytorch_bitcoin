import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()

        self.lstm = nn.GRU(input_size, hidden_dim, num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:,-1,:]
        output = self.linear(output)
        return output
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:,-1,:]
        output = self.linear(output)
        return output