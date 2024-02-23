# Assignment 2: Argument Detection Classifier

This project is part of the Advanced NLP course and focuses on identifying and classifying arguments within sentences using logistic regression.

## Installation

The project used a virtual environment basing on python 3.7 or above. Please create and activate the virtual environment by:
``` bash
# Change directory to current folder
cd (current folder directory)

# Create a virtual environment named venv
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

## List of Content

1. Datasets

   - Training data: data/en_ewt-up-train.conllu

   - Development data: data/en_ewt-up-dev.conllu

   - Testing data: data/en_ewt-up-test.conllu

2. requirements.txt
   - Install the required packages by executing the following command in termial by:

``` bash
pip install -r requirements.txt
```

3. A2_main.ipynb
   - Jupyter Notebook format
   - Dataset description
   - Training and evaluation process of the argument detection classifier
4. main.py
   - Plain python script format
   - Training and evaluation process of the argument detection classifier