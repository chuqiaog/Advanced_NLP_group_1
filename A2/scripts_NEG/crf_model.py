import array
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from get_all_features import main as load_features
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
from itertools import chain
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract features for each token in a sentence
def extract_features(sentence):
    # Iterate through each token in the sentence and extract various features
    # Handles None values by replacing them with 0
    return [{
        'pos_tag_encoded': token['pos_tag_encoded'] if token['pos_tag_encoded'] is not None else 0,
        'is_capitalized': token['is_capitalized'] if token['is_capitalized'] is not None else 0,
        'distance_from_negation': token['distance_from_negation'] if token['distance_from_negation'] is not None else 0,
        'is_punctuation': token['is_punctuation'] if token['is_punctuation'] is not None else 0,
        'is_negation_cue': token['is_negation_cue'] if token['is_negation_cue'] is not None else 0,
        'token_embeddings_sum': sum(token['token_embeddings']) if token['token_embeddings'] is not None else 0,
        'lemma_embeddings_sum': sum(token['lemma_embeddings']) if token['lemma_embeddings'] is not None else 0,
        'Dependency Relation sum': sum(token['Dependency Relation']) if token['Dependency Relation'] is not None else 0,
        'Dependency Distance to Cue': token['Dependency Distance to Cue'] if token['Dependency Distance to Cue'] is not None else 0,
        'Dependency Path to Cue sum': sum(token['Dependency Path to Cue']) if token['Dependency Path to Cue'] is not None else 0,
    } for token in sentence]

# Function to extract labels for each token in a sentence
def extract_labels(sentence):
    # Returns a list of labels corresponding to each token in the sentence
    return [token['label'] for token in sentence]

# Load and process the dataset
filename = 'converted_dev.conll'
features = load_features(filename)

# Filter out empty sentences and extract features and labels
filtered_features = [sentence for sentence in features if len(sentence) > 0]
X = [extract_features(sentence) for sentence in filtered_features]
y = [extract_labels(sentence) for sentence in filtered_features]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',  # Using the LBFGS algorithm for optimization
    c1=0.1,  # Coefficient for L1 regularization
    c2=0.1,  # Coefficient for L2 regularization
    max_iterations=100,  # Maximum number of iterations for model training
    all_possible_transitions=True  # Consider all possible state transitions
)
crf.fit(X_train, y_train)  # Train the model on the training data

# Predict on the test data using the trained CRF model
y_pred = crf.predict(X_test)

# Verify that the number of predicted sequences matches the number of test sequences
assert len(y_pred) == len(y_test), "The number of predicted sequences does not match the number of test sequences."

# Flatten the lists of lists for evaluation
y_test_flat = list(chain.from_iterable(y_test))
y_pred_flat = list(chain.from_iterable(y_pred))

# Verify that the flattened true and predicted labels are of the same length
assert len(y_test_flat) == len(y_pred_flat), "The number of flattened true labels does not match the number of flattened predicted labels."

# Define 'labels' as the set of all unique labels in the test set
labels = sorted(set(y_test_flat))

# Calculate and print precision, recall, and F1 score using sklearn.metrics
precision = precision_score(y_test_flat, y_pred_flat, average='weighted', labels=labels, zero_division=1)
recall = recall_score(y_test_flat, y_pred_flat, average='weighted', labels=labels, zero_division=1)
f1 = f1_score(y_test_flat, y_pred_flat, average='weighted', labels=labels, zero_division=1)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Identify all non-'_' labels as 'positive'
positive_labels = set(labels) - {'_'}

# Function to calculate TP, FP, FN, TN for non-'_' labels
def calculate_confusion_matrix_elements(y_true, y_pred, positive_labels):
    # Calculate true positives, false positives, false negatives, and true negatives
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pred and true in positive_labels)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true != pred and pred in positive_labels)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true != pred and true in positive_labels)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == pred and true not in positive_labels)
    return tp, fp, fn, tn

# Calculate TP, FP, FN, TN for non-'_' labels
tp, fp, fn, tn = calculate_confusion_matrix_elements(y_test_flat, y_pred_flat, positive_labels)

print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives: {tn}")

# Calculate precision, recall, and F1 score for non-'_' labels
entity_precision = tp / (tp + fp) if tp + fp > 0 else 0
entity_recall = tp / (tp + fn) if tp + fn > 0 else 0
entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if entity_precision + entity_recall > 0 else 0

print("Entity Precision:", entity_precision)
print("Entity Recall:", entity_recall)
print("Entity F1 Score:", entity_f1)

# Calculate span-level metrics
span_precision, span_recall, span_f1, _ = precision_recall_fscore_support(
    y_test_flat, y_pred_flat, average='weighted', labels=labels, zero_division=1
)

print("Span Precision:", span_precision)
print("Span Recall:", span_recall)
print("Span F1 Score:", span_f1)

# Assume tp, fp, fn, tn are already calculated
conf_matrix = [[tp, fp],
               [fn, tn]]

