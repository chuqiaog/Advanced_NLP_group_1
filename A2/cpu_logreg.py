from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

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