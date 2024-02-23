import copy
import spacy
from tqdm.auto import tqdm
nlp = spacy.load("en_core_web_sm")

######################## FEATURE EXTRACTION ########################
def dep_dist_to_head(input_text, conll_keys):
    '''
    This function calculates the dependency distance from token to head.
    Input: flat python string
    Return: A list of dict with {'token':token, 'dist_to_head':its distance to head}
            If negative, then the token is before the head.
    '''
    doc = nlp(input_text)
    token_full_info = doc.to_json()['tokens']
    dist_to_head = dict()
    token_combination = ''
    combining_tokens = False

    for i in range(len(doc)):
        
        if doc[i].text.strip() not in conll_keys or combining_tokens:
            combining_tokens = True
            token_combination += doc[i].text.strip()
            if token_combination in conll_keys:
                dist_to_head[token_combination] = token_full_info[i]['id']-token_full_info[i]['head']
                token_combination = ''
                combining_tokens = False
        elif doc[i].text.strip() in conll_keys and not combining_tokens:
            dist_to_head[doc[i].text] = token_full_info[i]['id']-token_full_info[i]['head']
            token_combination = ''

    return dist_to_head

def named_entity_recognition(input_text, conll_keys):
    """
    function to return all named entities from an input text.
        :input: flat python string
        :output: list of all named entities present in the input.
    """
    doc = nlp(input_text)
    ents = doc.ents
    NER_map = dict()
    token_combination = ''
    combining_tokens = False
    combined_is_NER = False

    for i in range(len(doc)):
        
        if doc[i].text.strip() not in conll_keys or combining_tokens:
            combining_tokens = True
            token_combination += doc[i].text.strip()
            if doc[i].text in ents:
                combined_is_NER = True
            if token_combination in conll_keys:
                NER_map[token_combination] = combined_is_NER
                token_combination = ''
                combining_tokens = False
                combined_is_NER = False

        elif doc[i].text.strip() in conll_keys and not combining_tokens:
            if doc[i] in ents:
                NER_map[doc[i].text] = True
            else:
                NER_map[doc[i].text] = False
            token_combination = ''

    return NER_map

def syntactic_head(input_text, conll_keys):
    '''
    This function finds the syntactic head for each word in the input text.
    Input: flat python string
    Return: heads (list): List of dictionaries with {word:head} relation
    '''
    doc = nlp(input_text)
    head_map = dict()
    token_combination = ''
    combining_tokens = False
    
    for i in range(len(doc)):
        if doc[i].text.strip() not in conll_keys or combining_tokens:
            combining_tokens = True
            token_combination += doc[i].text.strip()
            if token_combination in conll_keys:
                head_map[token_combination] = doc[i].head.text
                token_combination = ''
                combining_tokens = False
        elif doc[i].text.strip() in conll_keys and not combining_tokens:
            head_map[doc[i].text] = doc[i].head.text
            token_combination = ''

    return head_map

def extract_trigram(conll_keys):
    '''
    This function creates bigrams of the given text
    Input: flat python string
    Return: A list of dict with {'token':token, 'bigram': bigram}
    '''
    trigram_error = []
    try:
        trigram_dict = dict()
        trigram_dict[conll_keys[0]] = [f'SOS {conll_keys[0]} {conll_keys[1]}']
        for i in range(1,len(conll_keys)-1):
            trigram_dict[conll_keys[i]] = f'{conll_keys[i-1]} {conll_keys[i]} {conll_keys[i+1]}'
        trigram_dict[conll_keys[-1]] = f'{conll_keys[-2]} {conll_keys[-1]} EOS'

        return trigram_dict
    except:
        trigram_error.append(conll_keys)
        # return trigram_error
        
def follows_predicate(predicate, conll_keys):
    '''
    This function checks wether a word immediately follows (comes after) the predicate of the sentence
    '''
    bool_map = dict.fromkeys(conll_keys, False)

    #if there is no predicate in the sentence, return all values as False straight away
    if predicate is None:
        return bool_map
    
    for i, word in enumerate(conll_keys):

        if i == 0:
            continue
        elif conll_keys[i-1] == predicate:
            bool_map[word] = True

    return bool_map

def leads_predicate(predicate, conll_keys):
    '''
    This function checks wether a word immediately leads (comes before) the predicate of the sentence
    '''
    bool_map = dict.fromkeys(conll_keys, False)

    #if there is no predicate in the sentence, return all values as False straight away
    #otherwise if the predicate is at the start there cannot be a leading token, so return all as False
    if predicate is None or conll_keys[0] == predicate:
        return bool_map

    for i, word in enumerate(conll_keys):
        if conll_keys[i+1] == predicate:
            bool_map[word] = True
            break
    return bool_map

def create_features(preplist):
    """
    This function creates extra features by dependency parsing, using a preprocessed list.
    Also remove the features that are not needed from preprocessed dataset.
    """
    sent_with_feature = []
    
    for sentence in tqdm(preplist):
        # Extract sentence text
        sentence_text = ' '.join(word['form'] for word in sentence)
        conll_keys = [word['form'] for word in sentence]
        predicate = [word['form'] for word in sentence if word['pred'] != '_']
        if len(predicate) == 0:
            predicate = None
        else:
            predicate = predicate[0]

        try: # try processing each sentence and enriching, this only fails for broken data (URL instead of sentence, or single word conll entries)
            dist_to_head_map = dep_dist_to_head(sentence_text, conll_keys)
            NER_map = named_entity_recognition(sentence_text, conll_keys)
            syntactic_head_map = syntactic_head(sentence_text, conll_keys)
            trigram_map = extract_trigram(conll_keys)
            following_predicate_map = follows_predicate(predicate, conll_keys)
            leading_predicate_map = leads_predicate(predicate, conll_keys)
        except:
            continue
        # Add features back to dict
        news_sentence_dict = []
        for word_dict in sentence:
            new_word_dict = copy.deepcopy(word_dict) # Avoid changing the original file
            
            try:
                # add the distance for each word to head
                new_word_dict['dist_to_head'] = dist_to_head_map[word_dict['form']]                
                # add value for named entity of each word
                new_word_dict['is_named_entity'] = NER_map[word_dict['form']]
                # add syntactic head
                new_word_dict['syntactic_head'] = syntactic_head_map[word_dict['form']]
                # add word trigram
                new_word_dict['trigram'] = trigram_map[word_dict['form']]
                # following the predicate
                new_word_dict['follows_pred'] = following_predicate_map[word_dict['form']]
                # preceeding the predicate
                new_word_dict['preceeds_pred'] = leading_predicate_map[word_dict['form']]

                del new_word_dict['deprel'],new_word_dict['feats'],new_word_dict['misc'] # remove unused

                news_sentence_dict.append(new_word_dict)
                
            except Exception as e:
                # print(e)
                continue
                
        sent_with_feature.append(news_sentence_dict)

    return sent_with_feature
