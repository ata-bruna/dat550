# Final project DAT550 - Data mining and deep learning 

This repository contains the files for the project entitled "Detect claims to fact check in political debates"

## Table of contents


* [General Info](#general-information)
* [Models Used](#models-used)
* [How to navigate the repository](#how-to-navigate-the-repository)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)


## General Information
- This project tackles the problem of text classification using multiple algorithms
- The aim is to identify claims that should be fact checked in political debates
- The data comprises claims extracted from all U.S. general election presidential debates (1960-2016)
- The original data can be found at ClaimBuster's website [here](https://zenodo.org/record/3609356)
- The project was created using Python 3.9


## Models Used

### Feature based techniques - baseline model
Various classifiers were tested, data was preprocessed using TD-IDF.

1. SVM
2. KNN
3. Perceptron
4. Naive Bayes
5. Decision Tree 
5. Random Forest


### Deep learning techniques

#### Neural networks using embedding layer - word embedding
1. LSTM
2. Bidirectional LSTM
3. Stacked Bi-LSTM
4. Convolutional Neural Network (CNN)
5. CNN + LSTM

#### Tranformers Model


## How to navigate the repository
- The raw data can be found in the [`data`](https://github.com/ata-bruna/dat550/tree/main/data) folder. 
- All data cleaning steps are presented in the notebook stored in the [`data_preprocessing`](https://github.com/ata-bruna/dat550/tree/main/data_preprocessing) folder
- The baseline model is stored in the [`baseline`](https://github.com/ata-bruna/dat550/tree/main/baseline) folder
- Word embedding in the [`word embeddings`](https://github.com/ata-bruna/dat550/tree/main/word%20embeddings) folder 
- Transformers are stored in the `transformers` folder

The models import the data stored in data_preprocessing folder.


## Room for Improvement
Include areas you believe need improvement / could be improved. Also add TODOs for future development.

Room for improvement:
- Improvement to be done 1
- Improvement to be done 2

To do:
- Feature to be added 1
- Feature to be added 2


## Acknowledgements
Give credit here.
- This project was inspired by...
- This project was based on [this tutorial](https://www.example.com).
- Many thanks to...


