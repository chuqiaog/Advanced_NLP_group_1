import cudf
import numpy as np
import cupy as cp
from cuml import LogisticRegression
import copy
from sklearn.feature_extraction import DictVectorizer

######################## GPU IMPLEMENTATION ########################
def dict_vectorize(train_features):
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    return vec, features_vectorized

def get_mappings_dict(inputlist):
    category_list = copy.deepcopy(inputlist)
    category_list.append(None)
    map_dict = dict(zip(set(category_list), range(len(set(category_list)))))
    map_dict_reverse = {v: k for k, v in map_dict.items()}
    return map_dict, map_dict_reverse

def numerical_mapping(category_list, list_dict):
    numerical_list = [list_dict[line] for line in category_list]
    return numerical_list

def cp_feature_and_gold(feat_vec, gold_labels, mydict):
    # This function convert array to cuml required cp_array type
    cpfeature = cp.sparse.csr_matrix(feat_vec)
    cpgold = [mydict[line] for line in gold_labels]
    cpgold = cp.array(cpgold, dtype=cp.int64)
    return cpfeature, cpgold

def classify_data_with_rewrite(using_test_set, vec, model, dict):  
    features = vec.transform(using_test_set)
    predictions = model.predict(features)
    rw_predictions = [dict[line] for line in predictions]
    return rw_predictions

def write_cp_output_file(predictions, training_features, gold_labels, outputfile):
    outfile = open(outputfile, 'w', encoding='utf8')
    # add headings
    outfile.write('word' + '\t' + 'gold' + '\t' + 'predict' + '\n')
    for i in range(len(predictions)):
        outfile.write(training_features[i]['form'] + '\t' + gold_labels[i] + '\t' + predictions[i] + '\n')
    outfile.close()

def write_cp_output_file_2(predictions, double_pred_1, training_features, gold_labels, outputfile):
    outfile = open(outputfile, 'w', encoding='utf8')
    # add headings
    outfile.write('word' + '\t' + 'gold' + '\t' + 'pred1' + '\t' + 'predict' + '\n')
    for i in range(len(predictions)):
        outfile.write(training_features[i]['form'] + '\t' + gold_labels[i] + '\t' + str(double_pred_1[i]) + '\t'+ predictions[i] + '\n')
    outfile.close()