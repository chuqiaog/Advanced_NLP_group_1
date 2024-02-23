import os
import sys
import click
import argparse

import copy
from tqdm.auto import tqdm

import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction import DictVectorizer

import read_and_preprocess
import features_extraction
import data_preparation
import cpu_and_gpu_logreg


######################## ACCEPTING USER INPUT ########################
using_gpu = False
print("The default method uses sklearn's CPU implementation, which is slow to run and unlikely to converge.")
if click.confirm('Do you want to use GPU? (requires cuda device and cuml,cupy,cudf repo)', default=False):
    using_gpu = True
        
if using_gpu == True:
    import cudf
    import numpy as np
    import cupy as cp
    from cuml import LogisticRegression
if using_gpu == False:
    from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(
                    prog='A2_main',
                    description='This script trains logistic regression model for argument classification.',
                    epilog='Example usage: python3 A2_main.py /data/en_ewt-up-train.conllu data/en_ewt-up-dev.conllu 100')
parser.add_argument('train_data_path', help="Input path for the UP training dataset")  
parser.add_argument('test_data_path', help="Input path for the UP test dataset")  
parser.add_argument('output_path', help="Path where all the output are stored")
parser.add_argument('max_iter', help="Maximum iteration allowed for logistic regression")











def main(trainfile, testfile, output_path):
    using_gpu = gpu_usage()

    print('Reading and preprocessing data..')
    trainlist, testlist = read_conll(trainfile), read_conll(testfile)
    preprocessed_train, preprocessed_test = preprocess_list(trainlist), preprocess_list(testlist)
    print('Extracting training features')
    train_features = create_features(preprocessed_train)
    print('Extracting test features')
    test_features =  create_features(preprocessed_test)

    training_features, gold_labels = extract_feature_and_label(train_features) # single classifier

    if using_gpu == False:
        # Single logreg
        model_single, vec_single = create_log_classifier(training_features, gold_labels, max_iter)
        using_test_set, test_gold = extract_feature_and_label(test_features)
        single_predictions = classify_data(model_single, vec_single, using_test_set)
        
        outputpath = output_path + '/singlelogreg_cpu_' + str(max_iter) +'.csv'
        write_output_file(single_predictions, using_test_set, test_gold, outputpath)

        # Double logreg
        training_features_step1, gold_labels_step1 = extract_is_ARG_feature_and_label(train_features)
        reduced_training_features_step1 = reducing_features(training_features_step1)
        # first model
        model_double_1, vec_double_1 = create_log_classifier(reduced_training_features_step1, gold_labels_step1, max_iter)
        using_test_set_1, test_gold_1 = extract_is_ARG_feature_and_label(test_features)
        predictions_1 = classify_data(model_double_1, vec_double_1, using_test_set_1)
        # second model
        training_features_step2, gold_labels_step2 = extract_ARG_type_feature_and_label(train_features)
        model_double_2, vec_double_2 = create_log_classifier(training_features_step2, gold_labels_step2, max_iter)
        using_test_set_2, test_gold_2 = extract_ARG_type_feature_and_label_with_prediction(test_features, predictions_1)
        predictions_2 = classify_data(model_double_2, vec_double_2, using_test_set_2)
        
        outputpath = output_path + '/doublelogreg_cpu_' + str(max_iter) +'.csv'
        write_output_file(predictions_2, using_test_set_2, test_gold_2, outputpath)

    if using_gpu == True:
        mydict, mydict_rev = get_mappings_dict(gold_labels)
        # Single logreg
        vec_single, feat_vec_single = dict_vectorize(training_features)
        X_single, y_single = cp_feature_and_gold(feat_vec_single)
        reg_single = LogisticRegression(max_iter=max_iter,class_weight='balanced')
        reg_single.fit(X_single,y_single)
        using_test_set, test_gold = extract_feature_and_label(test_features)
        single_pred = classify_data_with_rewrite(using_test_set,vec_single,reg_single)

        outputpath = output_path + '/singlelogreg_gpu_' + str(max_iter) +'.csv'
        write_cp_output_file(single_pred, using_test_set, test_gold, outputpath)

        # Double logreg
        training_features_step1, gold_labels_step1 = extract_is_ARG_feature_and_label(train_features)
        reduced_training_features_step1 = reducing_features(training_features_step1)
        vec_double_1, feat_vec_double_1 = dict_vectorize(reduced_training_features_step1)
        X_double_1 = cp.sparse.csr_matrix(feat_vec_double_1)
        y_double_1 = cp.array(gold_labels_step1)
        reg_double_1 = LogisticRegression(max_iter=5000)
        reg_double_1.fit(X_double_1,y_double_1)
        using_test_set_1, test_gold_1 = extract_is_ARG_feature_and_label(test_features)
        using_test_set_1_vec = vec_double_1.transform(using_test_set_1)
        double_pred_1 = reg_double_1.predict(using_test_set_1_vec)


    
if __name__ == '__main__':
    args = parser.parse_args()
    trainfile = args.train_data_path
    testfile = args.test_data_path
    output_path = args.output_path
    max_iter = args.max_iter
    main(trainfile, testfile, output_path, max_iter)