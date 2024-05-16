import json

with open('intents.json','r') as f:
    intents = json.load(f)

print(intents)