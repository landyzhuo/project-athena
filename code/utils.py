# imports NLTK library
import nltk
# imports the Porter Stemmer
from nltk.stem.porter import PorterStemmer
import numpy as np

# pre-trained AI tokenizer
# nltk.download("punkt")
# define your stemmer
stemmer = PorterStemmer()

# splits words and punctuations up and store into array
def tokenize(prompt):
    return nltk.word_tokenize(prompt)

# identify word roots and store into array
def stem(word):
    return stemmer.stem(word.lower())

# eliminate punctuations from array
def bagOfWords(tokenized_prompt, word_list):
    tokenized_prompt = [stem(w) for w in tokenized_prompt]
    bag = np.zeros(len(word_list), dtype=np.float32)
    for index, w in enumerate(word_list):
        if w in tokenized_prompt:
            bag[index] = 1.0
    return bag
