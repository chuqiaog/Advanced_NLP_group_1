######################## CPU IMPLEMENTATION ########################
def create_log_classifier(train_features, train_targets, max_iter):
    logreg = LogisticRegression(max_iter=max_iter)
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    model = logreg.fit(features_vectorized, train_targets) 
    return model, vec

def classify_data(model, vec, features):  
    features = vec.transform(features)
    predictions = model.predict(features)
    return predictions

def write_output_file(predictions, training_features, gold_labels, outputfile):
    outfile = open(outputfile, 'w', encoding='utf8')
    # add headings
    outfile.write('word' + '\t' + 'gold' + '\t' + 'predict' + '\n')
    for i in range(len(predictions)):
        outfile.write(training_features[i]['form'] + '\t' + gold_labels[i] + '\t' + predictions[i] + '\n')
    outfile.close()


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

def write_cp_output_file(predictions, training_features, gold_labels, outputfile):
    outfile = open(outputfile, 'w', encoding='utf8')
    # add headings
    outfile.write('word' + '\t' + 'gold' + '\t' + 'predict' + '\n')
    for i in range(len(predictions)):
        outfile.write(training_features[i]['form'] + '\t' + gold_labels[i] + '\t' + predictions[i] + '\n')
    outfile.close()