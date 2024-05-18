import random
import json
import torch
from model import NeuralNetwork
from utils import bagOfWords, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as f:
    intents = json.load(f)

file = "data.pth"
data = torch.load(file)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Athena"
print(f"Hey, {bot_name} here. Let's chat :-)")
print("PS: type 'quit' to exit the chat")
while True:
    sentence = input("You: ")
    if sentence=="quit":
        print("Bye XD")
        break
    sentence = tokenize(sentence)
    X = bagOfWords(sentence, words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.9:
        for intent in intents["intents"]:
            if tag==intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Sorry, didn't really catch your meaning... Do you mind rephrasing your prompt?")