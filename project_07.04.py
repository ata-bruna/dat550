# from Canvas:
# Detect claims to fact check in political debates

# In this project you will implement various classifiers using both neural 
# and feature based technqiues to detect which sentences in political 
# debates should be fact checked.
# Dataset from ClaimBuster: https://zenodo.org/record/3609356 
# Evaluate your classifiers using the same metrics as 
# http://ranger.uta.edu/~cli/pubs/2017/claimbuster-kdd17-hassan.pdf 
# (Table 2)
# Classification report from sklearn provides everything
# Group crowdsourced.csv and groundtruth.csv into one dataset. Use debates 
# from 1960-2008 for training (27 first debates) and 2012-2016 for testing 
# (6 last debates)
# Create a baseline model: Should be fairly simple one, e.g. SVM, Random 
# Forest, Logistic regression using TF-IDF or other features of your choice. 
# Aim for 60% or more for f1 weighted average.
# Create advanced model(s) (suggestions are given below)
# Generate more features that a model can use. For example the context 
# around the sentence, sentiment, named entities etc.
# Rule based classifier. For example, if sentence contains certain words, 
# tags, statistics etc.
# Deep learning (word embeddings, transformer models etc.)
# Sub-sentence classifier. Long sentences may include several claims, so 
# the goal is to mark the span of claim(s) within a sentence


# ClaimBuster: A Benchmark Dataset of Check-worthy Factual Claims
# Fatma Arslan; Naeemul Hassan;  Chengkai Li; Mark Tremayne

# The ClaimBuster dataset consists of statements extracted from all
# U.S. general election presidential debates (1960-2016) along with 
# human-annotated check-worthiness labels. It contains 23,533 sentences 
# where each sentence is categorized into one of the three categories: 
# non-factual statement, unimportant factual statement, and check-worthy 
# factual statement. 

#%%
from tracemalloc import stop
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import collections
#%%

file1 = pd.read_csv("data/crowdsourced.csv", encoding='utf-8')
file2 = pd.read_csv("data/groundtruth.csv", encoding='utf-8')
df = pd.concat([file1, file2])

df["date"] = df["File_id"].str.strip(to_strip=".txt")

df["date"] = pd.to_datetime(df["date"])
df.sort_values("date", inplace= True)

#%%
df["mos_before_election"] = 11 - df["date"].dt.month

#%%
stops = set(stopwords.words('english'))
# vectorizer = TfidfVectorizer(stop_words=stops)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["Text"])

#%%

import string
numeric = '\n–0123456789‘’“”ƒš'
punct = string.punctuation
characters = numeric + punct
translator = str.maketrans('', '', characters)
a_list = []
for i in df['Text']:
    clean = i.translate(translator).replace('\n'," ").lower()
    u = [word for word in clean.split() if word not in (stops)]
    a_list.append(u)

flat_list = []
for sublist in a_list:
    for item in sublist:
        flat_list.append(item)

counter = collections.Counter(flat_list)
frequent_words = counter.most_common()


#%%
# splitting into training data:
mask = df["date"].dt.year < 2012

x_train = X[mask]
x_test = X[~mask]

y_train = df.loc[mask, "Verdict"].values
y_test = df.loc[~mask, "Verdict"].values

#%%  SVM
clf = svm.SVC(kernel='linear') 
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred, target_names= ["NFS", "UFS", "CFS"]))
comparison_svm = classification_report(y_test, y_pred, target_names= ["NFS", "UFS", "CFS"])

# labels verified according to QA section (3.2) by Bruna and Kurt
# ref. file2["Verdict"].value_counts()

#%%  Random Forest
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(x_train, y_train)
y_pred_rf = clf.predict(x_test)
# print(classification_report(y_test, y_pred_rf, target_names= ["NFS", "UFS", "CFS"]))
# %%

import random 
number_of_colors = 25

color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]



# %%
