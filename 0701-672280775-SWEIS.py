import argparse
from torch.autograd import Variable 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import shutil
import numpy as np
import random

device = torch.device("cpu")
names = []
with open('names.txt', 'r') as f:
    names = f.readlines()
    # make lower case
    names = [x.lower()[:-1] for x in names]

# create character dictionary
characters = 'abcdefghijklmnopqrstuvwxyz'
char_to_index = {char: i+1 for i, char in enumerate(characters)}

# char_to_index = dict()
# EON = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def encoding(name):
    fin = []
    for i in range(11):
        encoded_list = np.zeros(27)
        if i < len(name):
            encode = char_to_index[name[i]]
        else:
            encode = 0
        encoded_list[encode] = 1
        fin.append(encoded_list)
    return np.array(fin)
    
# define LSTM RNN Here
class My_LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(My_LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
    
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 64)
        self.fc_2 = nn.Linear(64, num_classes)
    
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
    
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n = h_n.view(-1, self.hidden_size)
        out = self.relu(output)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        
        return out

# create sets X and Y
x = []
y = []
for name in names:
    x.append(encoding(name))
    y.append(encoding(name[1:]))

# vairable defs
num_epochs = 1000
learning_rate = 0.001
input_size = 27
hidden_size = 64
num_layers = 1
num_classes = 27

x_train = torch.tensor(np.array(x), dtype=torch.float32)
y_train = torch.tensor(np.array(y), dtype=torch.float32)

my_lstm = My_LSTM(num_classes, input_size, hidden_size, num_layers, x_train.shape[1])

print("Training Shape X and Y", x_train.shape, y_train.shape)

# Mean squares loss function chosen
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(my_lstm.parameters(), lr=learning_rate) # optimizer

loss_vs_epoch = np.zeros((2,num_epochs))
for epoch in range(num_epochs):
    outputs = my_lstm.forward(x_train)
    
    optimizer.zero_grad()
    
    loss = criterion(outputs, y_train)

    loss.backward() 
    optimizer.step() 
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
    loss_vs_epoch[0,epoch]=epoch
    loss_vs_epoch[1,epoch]=loss.item()
print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

#save the model
torch.save(lstm1.state_dict(), "0702-672280775-SWEIS.ZZZ")