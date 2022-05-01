import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import collections
import string

# Loading and merging the data
# Data preprocessing
def remove_punctuation(text):
    tokens = re.sub('[^a-zA-Z]', ' ', text).lower()
    return tokens
def remove_stop_words(text):
    stop_words = stopwords.words('english')
    word_list = [word for word in text.split() if word not in stop_words]
    return word_list