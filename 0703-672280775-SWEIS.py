import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import StepLR

# LSTM Model
class My_LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(My_LSTM, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.seq_length = seq_length 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) 
        self.fc = nn.Linear(128, num_classes) 

        self.relu = nn.ReLU()
    
    def forward(self,x):
        # h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        output, (h_n, c_n) = self.lstm(x) 
        h_n = h_n.view(-1, self.hidden_size) 
        out = self.relu(output)
        out = self.fc_1(out) 
        out = self.relu(out) 
        out = self.fc(out) 
        
        return out

characters = ['EON']
characters.extend('abcdefghijklmnopqrstuvwxyz')
char_to_index = {char: i for i, char in enumerate(characters)}

def encode(name):
    fin = []
    for i in range(len(name)):
        encoded_list = np.zeros(27)
        if i < len(name):
            encode = char_to_index[name[i]]
        else:
            encode = 0
        encoded_list[encode] = 1
        fin.append(encoded_list)
    return np.array(fin)


def finding_char(x):
    TopK = 6
    output_tensor = x[-1]
    # print(output_tensor)
    _, ind = torch.topk(output_tensor, TopK)
    w = []
    for i in range(TopK):
        n = random.randint(1, 40)
        w.append(n)
    index = random.choices(ind.tolist(), weights = w)
    return characters[index[0]]

def generate_name(char, l, model):
    input = torch.tensor(np.array(encode(char)), dtype=torch.float32)
    output = model.forward(input)
    output_char = finding_char(output)
    
    if(output_char == 'EON' or len(char) == l):
        if(len(char) < 3):
            return generate_name(char, l, model)
        else:
            return char
    else:
        char = char + output_char
        return generate_name(char, l, model)
    
input_size = 27
hidden_size = 64
num_layers = 1
num_classes = 27
device = torch.device("cpu")

while True:
    letter = input('Input letter to start, enter done to quit')
    if letter == 'done':
        break
        
    model = My_LSTM(num_classes, input_size, hidden_size, num_layers, 11)
    model.eval()
    model_path = "0702-672280775-SWEIS.ZZZ"
    model.load_state_dict(torch.load(model_path,map_location=device))
    
    names_list = {letter:[]}
    while len(names_list[letter]) < 20:
        name = generate_name(letter, 11, model)
        if(name not in names_list[letter]):
            names_list[letter].append(name)
            
    print(f"20 names generated for {letter}: ")
    print(names_list[letter])
