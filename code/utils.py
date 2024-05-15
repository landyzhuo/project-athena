# imports NLTK library
import nltk
# imports the Porter Stemmer
from nltk.stem.porter import PorterStemmer

# pre-trained AI tokenizer
nltk.download("punkt")
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
    pass
