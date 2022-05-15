# Final project DAT550 - Data mining and deep learning 

This repository contains the files for the project entitled "Detect claims to fact check in political debates"

## Table of contents


* [General Info](#general-information)
* [Additional Packages](#additional-packages)
* [Models Used](#models-used)
* [How to navigate the repository](#how-to-navigate-the-repository)
* [Room for Improvement](#room-for-improvement)



## General Information
- This project tackles the problem of text classification using multiple algorithms
- The aim is to identify claims that should be fact checked in political debates
- The data comprises claims extracted from all U.S. general election presidential debates (1960-2016)
- The original data is provided by ClaimBuster and can be found [here](https://zenodo.org/record/3609356)
- The project was created using Python 3.9


## Additional packages

### Importing the pretrained weights for the embedding matrix

To run the file [`2. word_embeddings_multi_class_pretrained.ipynb`](https://github.com/ata-bruna/dat550/blob/main/word%20embeddings/2.%20word_embeddings_multi_class_pretrained.ipynb) it is necessary to add the pretrained weights file to the data folder.

The pretrained weights used in this report are from the [GloVe](https://nlp.stanford.edu/projects/glove/) model Which can be downloaded [here ](https://nlp.stanford.edu/data/glove.6B.zip).

We chose the first option [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip), which contais Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download). This option contains 4 `.txt` files. We use the `glove.6B.100d.txt` as our sequences were padded to have `max_len =100`

Once the zip file is downloaded, place the zip into the [`data`](https://github.com/ata-bruna/dat550/tree/main/data) folder and run the following code:

```
import os
import zipfile
with zipfile.ZipFile('../data/glove.zip', 'r') as zip_ref:
    zip_ref.extractall('../data/glove')
```
The code is inidicated in the notebook, comment the block out and run it.


## Models Used

### Classification using sparse features - baseline model
Various classifiers were tested, data was preprocessed using TD-IDF.

1. SVM
2. KNN
3. Perceptron
4. Naive Bayes
5. Decision Tree 
5. Random Forest


### Deep learning techniques

#### Neural networks using embedding layer - word embedding
1. Bidirectional LSTM
2. Stacked Bi-LSTM
3. Convolutional Neural Network (CNN)
4. CNN + LSTM

#### Tranformers Model
1. BERT Model


## How to navigate the repository

- The raw data can be found in the [`data`](https://github.com/ata-bruna/dat550/tree/main/data) folder. 
- All data cleaning steps are presented in the notebook stored in the [`data_preprocessing`](https://github.com/ata-bruna/dat550/tree/main/data_preprocessing) folder
- The baseline model is stored in the [`baseline`](https://github.com/ata-bruna/dat550/tree/main/baseline) folder
- Word embedding in the [`word embeddings`](https://github.com/ata-bruna/dat550/tree/main/word%20embeddings) folder 
- Transformers are stored in the [`transformers`](https://github.com/ata-bruna/dat550/tree/main/transformers) folder

**NB!** The [GloVe zip file](https://nlp.stanford.edu/data/glove.6B.zip) must be stored in the is stored in [`data`](https://github.com/ata-bruna/dat550/tree/main/data) folder.

**NB! 2** Both word embeddings and transformers model are applied to two different datasets. The file numbered as "1." cover the case where stopwords were removed and "2." the case where these words were kept. For both models keeping stop words resulted in a better result.


## Room for Improvement

Room for improvement:
- Improvement to be done 1
- Improvement to be done 2




