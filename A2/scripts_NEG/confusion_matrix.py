import pandas as pd
from convert_conll import *
from collections import defaultdict


def extract_spans(labels):
    spans = []
    current_span = []
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current_span:
                spans.append(current_span)
            current_span = [i]
        elif label.startswith("I-") and current_span:
            current_span.append(i)
        elif label == "O" and current_span:
            spans.append(current_span)
            current_span = []
    if current_span:
        spans.append(current_span)
    return spans

# def calculate_span_confusion_matrix(y_true, y_pred):
#     """
#     Calculate the 2x2 confusion matrix for span prediction.
#
#     Args:
#         y_true (dict): A dictionary containing true labels for each document and sentence.
#                        Keys are tuples (sent_id, doc_id), and values are lists of the true labels per sentence.
#         y_pred (dict): A dictionary containing predicted span labels for each document and sentence.
#                        Keys are tuples (sent_id, doc_id), and values are lists of the predicted labels per sentence.
#
#     Returns:
#         tuple: A tuple containing true positive (tp), false positive (fp),
#                true negative (tn), and false negative (fn) counts.
#
#     """
#     # Initialize counts for true positive, false positive, true negative, and false negative
#     tp, fp, tn, fn = 0, 0, 0, 0
#     for yt, yp in zip(y_true, y_pred):
#         true_spans = extract_spans(yt)
#         pred_spans = extract_spans(yp)
#
#     # Iterate over predicted spans for each document and sentence
#     for (sent_id, doc_id), yps in y_pred.items():
#         # Iterate over true and predicted sentences
#         for yt, yp in zip(y_true[(sent_id, doc_id)], yps):
#             # Check if true span has a negation (if not "***" is in the list)
#             if "***" not in list(yt):
#                 # Check if true and predicted spans match
#                 if list(yt) == list(yp):
#                     tp += 1  # Increment true positive count
#                 else:
#                     fn += 1  # Increment false negative count
#             else:
#                 # Check if true and predicted spans match
#                 if list(yt) == list(yp):
#                     tn += 1  # Increment true negative count
#                 else:
#                     fp += 1  # Increment false positive count
#
#     # Return the counts as a tuple
#     return tp, fp, tn, fn

def calculate_span_confusion_matrix(y_true, y_pred):
    tp, fp, tn, fn = 0, 0, 0, 0

    for yt, yp in zip(y_true, y_pred):
        true_spans = set(extract_spans(yt))
        pred_spans = set(extract_spans(yp))

        for span in pred_spans:
            if span in true_spans:
                tp += 1  # True positive: Predicted span is a true span
            else:
                fp += 1  # False positive: Predicted span is not a true span

        for span in true_spans:
            if span not in pred_spans:
                fn += 1  # False negative: True span not predicted

    # True negatives are typically not calculated for span-level metrics in NLP tasks,
    # but you can add logic here if applicable to your case.

    return tp, fp, tn, fn


def calculate_token_confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix for token-level prediction.

    Args:
        y_true (dict): A dictionary containing true tokens for each document and sentence.
                       Keys are tuples (sent_id, doc_id), and values are lists of true token labels.
        y_pred (dict): A dictionary containing predicted tokens for each document and sentence.
                       Keys are tuples (sent_id, doc_id), and values are lists of predicted token labels.

    Returns:
        tuple: A tuple containing true positive (tp), false positive (fp),
               true negative (tn), and false negative (fn) counts.

    """
    # Initialize counts for true positive, false positive, true negative, and false negative
    tp, fp, tn, fn = 0, 0, 0, 0

    # Iterate over predicted document and sentence
    for (sent_id, doc_id), yps in y_pred.items():
        # Iterate over true and predicted sentences
        for yt, yp in zip(y_true[(sent_id, doc_id)], yps):
            # Iterate over individual tokens in true and predicted sequences
            for yt_token, yp_token in zip(yt, yp):
                # Check if the true label is positive
                if yt_token != "***" and yt_token != "_":
                    # Check if true and predicted tokens match
                    if yt_token == yp_token:
                        tp += 1  # Increment true positive count
                    else:
                        fn += 1  # Increment false negative count
                else:
                    # Check if true and predicted tokens match
                    if yt_token == yp_token:
                        tn += 1  # Increment true negative count
                    else:
                        fp += 1  # Increment false positive count

    # Return the counts as a tuple
    return tp, fp, tn, fn


def calculate_performance(cm):
    """
    Calculate precision, recall, and F1 score based on the given 2x2 confusion matrix.

    Args:
        cm (tuple): A tuple containing true positive (tp), false positive (fp),
                    true negative (tn), and false negative (fn) counts.

    Returns:
        tuple: A tuple containing precision, recall, and F1 score.

    """
    # Unpack confusion matrix
    tp, fp, _, fn = cm

    # Calculate precision (P)
    precision = tp / (tp + fp) if tp + fp > 0 else 0

    # Calculate recall (R)
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Return precision, recall, and F1 score as a tuple
    return precision, recall, f1

def main(predictions_file, gold_file):
    """
    Compare predictions and gold annotations, calculate and write performance metrics to results.txt.

    Args:
        predictions_file (str): Path to the file containing predictions in CoNLL format.
        gold_file (str): Path to the file containing gold annotations in CoNLL format.

    Returns:
        None

    """
    # Create dataframes for file predictions and golden true
    df_P = read_conll(predictions_file)
    df_T = read_conll(gold_file)

    # Find the unique sentence_ids and document_ids of the prediction file
    ids_P = set(zip(df_P['document_id'], df_P['sentence_id']))

    # Define the columns to compare
    columns_to_compare = ['negation_word', 'negation_scope', 'negation_event']

    # Open a results file for writing
    with open("results/results.txt", 'w') as f:
        # Iterate over columns for comparison
        for column in columns_to_compare:
            f.write(f"Column: {column}\n")

            # Initialize defaultdicts to store true and predicted values for token and span comparisons
            y_true, y_pred = defaultdict(list), defaultdict(list)

            # Iterate through unique document and sentence IDs
            for doc_id, sent_id in ids_P:
                # Append true and predicted values for the specified column
                y_true[(doc_id, sent_id)].append(df_T[(df_T['sentence_id'] == sent_id) & (df_T['document_id'] == doc_id)][column])
                y_pred[(doc_id, sent_id)].append(df_P[(df_P['sentence_id'] == sent_id) & (df_P['document_id'] == doc_id)][column])

            # Calculate and write token-level performance metrics
            token_cm = calculate_token_confusion_matrix(y_true, y_pred)
            f.write("\t Confusion matrix tokens:\n")
            f.write("\t \t" + str(token_cm) + "\n \n")

            prec_token, recall_token, f1_token_score = calculate_performance(token_cm)
            f.write(f"\t Precision (Token Overlap): {prec_token}\n")
            f.write(f"\t Recall (Token Overlap): {recall_token}\n")
            f.write(f"\t F1-score (Token Overlap): {f1_token_score}\n \n ")

            # Calculate and write span-level performance metrics
            span_cm = calculate_span_confusion_matrix(y_true, y_pred)
            f.write("\t Confusion matrix spans:\n")
            f.write("\t \t" + str(span_cm) + "\n \n ")

            prec_span, recall_span, f1_span_score = calculate_performance(span_cm)
            f.write(f"Precision (Span Agreement): {prec_span}\n")
            f.write(f"Recall (Span Agreement): {recall_span}\n")
            f.write(f"F1-score (Span Agreement): {f1_span_score}\n")



