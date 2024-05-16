import json
from utils import tokenize, stem, bagOfWords

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
print(tags, words)