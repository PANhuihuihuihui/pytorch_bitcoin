import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim,classification = False):
        super(GRU, self).__init__()

        self.classification = classification
        self.lstm = nn.GRU(input_size, hidden_dim, num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:,-1,:]
        output = self.linear(output)
        if self.classification:
            output =  self.sigmod(output)
        return output
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim,classification = False):
        super(LSTM, self).__init__()

        self.classification = classification
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:,-1,:]
        output = self.linear(output)
        if self.classification:
            output =  self.sigmod(output)
        return output
    
class Conv(nn.Module):
    def __init__(self,input_size, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size,out_channels=5,kernel_size=(input_size,input_size*2),padding=1)
        self.conv2 = nn.Conv2d(in_channels=5,out_channels=10,kernel_size=(input_size,input_size*2),padding=1)
        self.conv3 = nn.Conv2d(in_channels=10,out_channels=15,kernel_size=(input_size,input_size*2),padding=1)
        self.maxpool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(90,2)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()

    def forward(self,x):
        x = self.relu(self.conv1(x))
        print(x.shape)
        x = self.relu(self.conv2(x))
        print(x.shape)        
        x = self.relu(self.conv3(x))
        print(x.shape)
        x = self.maxpool(x)
        
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)

        return x
    
class MLP(nn.Module):
    def __init__(self,input_size, hidden_dim, num_layers, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_size,hidden_dim)
        self.fc1 = nn.Linear(hidden_dim,output_dim)
        self.lrelu = nn.LeakyReLU()

    def forward(self,x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x =  self.sigmod(x)
        return x