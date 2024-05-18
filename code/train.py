import json
from utils import tokenize, stem, bagOfWords
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import NeuralNetwork

with open('intents.json','r') as f:
    intents = json.load(f)

words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        words.extend(w)
        xy.append((w, tag))

exclude_symbols = ['?','!','.',',',':',';','"',"'","-","/"]
words = [stem(i) for i in words if i not in exclude_symbols]
words = sorted(set(words))
tags = sorted(set(tags))

x_train = []
y_train = []
for (sentence, tg) in xy:
    bag = bagOfWords(sentence, words)
    x_train.append(bag)
    y_train.append(tags.index(tg))

x_train = np.array(x_train)
y_train = np.array(y_train)

class AthenaDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size=len(x_train[0])
learning_rate = 0.001
epochs = 1000

dataset = AthenaDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for (letters, labels) in  train_loader:
        letters = letters.to(device)
        labels = labels.to(device)

        output = model(letters)
        loss = criterion(output, labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": words,
    "tags": tags
}

file_name = "data.pth"
torch.save(data, file_name)
# verifies that model is saved to 'data.pt'
print(f"File saved to {file_name}")