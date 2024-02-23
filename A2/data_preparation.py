import copy

######################## TRAINING PREPARATION ########################
def extract_feature_and_label(preplist):
    """
    This function extract features and label from extracted feature list of dicts.
    It will flattern list of sentences into list of tokens.
    Used for single classifier.
    """
    data = []
    targets = []
    flatlist = [x for xs in preplist for x in xs]
    for dict in flatlist:
        newdict = copy.deepcopy(dict)
        del newdict['ARG'] # Remove gold
        data.append(newdict)
        targets.append(dict['ARG'])

    return data, targets

def extract_is_ARG_feature_and_label(preplist):
    """
    This function extract features and label from preprocessed list
    """
    data = []
    targets = []
    flatlist = [x for xs in preplist for x in xs]
    for dict in flatlist:
        newdict = copy.deepcopy(dict)
        del newdict['ARG'] # Remove gold
        data.append(newdict)
        
        if dict['ARG'] != '_':
            targets.append(True)
        else:
            targets.append(False)

    return data, targets

def reducing_features(inputfeature):
    """
    This function reduce the amount of feature used. Input is the ready-to-use feature dict.
    """
    newfeature = copy.deepcopy(inputfeature)
    for newdicts in newfeature:
        del newdicts['ID'], newdicts['lemma'], newdicts['dup'], newdicts['trigram'], newdicts['is_named_entity']
    return newfeature

def extract_ARG_type_feature_and_label(preplist):
    """
    This function extract ARG_type feature from the training set.
    """

    data = []
    targets = []
    flatlist = [x for xs in preplist for x in xs]
    
    for dict in flatlist:
        newdict = copy.deepcopy(dict)
        del newdict['ARG'] # Remove gold
        if dict['ARG'] != '_':
            newdict['is_ARG'] = 'True'
        else:
            newdict['is_ARG'] = 'False'
        
        data.append(newdict)
        targets.append(dict['ARG'])

    return data, targets

def extract_ARG_type_feature_and_label_with_prediction(preplist, predictions_1):
    """
    This function add result from the first classifier to the feature list for the test sets.
    """

    data = []
    targets = []
    flatlist = [x for xs in preplist for x in xs]
    
    for dict, predictions in zip(flatlist, predictions_1):
        newdict = copy.deepcopy(dict)
        del newdict['ARG'] # Remove gold
        newdict['is_ARG'] = str(predictions)
        
        data.append(newdict)
        targets.append(dict['ARG'])

    return data, targets
