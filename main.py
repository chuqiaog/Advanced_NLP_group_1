import spacy
import gensim
from nltk.corpus import brown

nlp = spacy.load("en_core_web_sm") 
try:
    word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('/home/ziggy/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
except FileNotFoundError:
    print("No local embedding found. Downloading instead...")
    import gensim.downloader as api
    word_embedding_model = api.load('word2vec-google-news-300')

./GoogleNews-vectors-negative300.bin.gz

def tokenize(input_text):
    """
    function to tokenize any input text.
        :input: flat python string
        :output: list of all tokens from the input string.
    """

    doc = nlp(input_text)
    return [token for token in doc]

def lemmatization(input_text):
    """
    function to return the lemmas of all words in the input text.
        :input: flat python string
        :output: list of all lemmas from the input string.
    """

    doc = nlp(input_text)
    return [token.lemma_ for token in doc]


def fetch_embedding(model, input_text):
    """
    function to return all embedding vectors for each word from an input text.
        :input: flat python string
        :output: list containing all embedded vectord for each word from the input string, if present in the model.
    """

    words = tokenize(input_text)
    vectors = dict()
    for word in words:
        try:
            vectors[word] = model.get_vector(word)
        except:
            vectors[word] = [0]*300
    return vectors
     


def named_entity_recognition(input_text):
    """
    function to return all named entities from an input text.
        :input: flat python string
        :output: list of all named entities present in the input.
    """

    doc = nlp(input_text)
    return [ent for ent in doc.ents]


# test_str = 'this input string is used as a test to see if these words can be embedded.'

# # print(fetch_embedding(model=word_embedding_model, input_text=test_str))


def sub_tree(input_text):
    '''
    This function generates the sub_tree of the input text.
    Input: flat python string
    Return: sub_tree_relation (dict): Dictionary of {word:sub_tree} relation
    '''
    doc = nlp(input_text)
    sub_tree_relation = {}  # Initiate a dictionary to store the sub_tree relation
    for token in doc:
        if list(token.children) == []:
            sub_tree = [token]
        else:
            sub_tree = list(token.children)
        sub_tree_relation[token.text] = [child.text for child in sub_tree]  # Store {token: sub_tree} relation
    return sub_tree_relation


# Extract Capitalization information, return a list of values, 0 for False, 1 for True

def capitalization(input_text):
    '''
    This function generates the capitalization of the input text.
    Input: flat python string
    Return: capitalization_list (list): Capitalization of the input text
    '''

    doc = nlp(input_text)
    capitalization_list = []  # Initialize a list to store capitalization information
    for token in doc:
        capitalization_list.append(1 if token.is_title else 0)  # Add capitalization information (0 for False, 1 for True) to the list
    return capitalization_list


def syntactic_head(input_text):
    '''
    This function finds the syntactic head for each word in the input text.
    Input: flat python string
    Return: heads (dict): Dictionary of {word:head} relation
    '''

    doc = nlp(input_text)
    heads = {}  # Initialize a dict to store {word:head} relation
    for token in doc:
        heads[token.text] = token.head.text  # Add {word:head} relation to the dict
    return heads


###
def PoS_tag(input_text):
    '''
    This function tags the Part-of-Speech tag of the words in the text
    Input: flat python string
    Return: A list of dict with {'token':token, 'pos':its pos tag}
    '''
    doc = nlp(input_text)
    pos = []
    for token in doc:
        dict={}
        dict['token'], dict['pos'] = token.text, token.pos_
        pos.append(dict)
    return pos

def dep_relations(input_text):
    '''
    This function labels the dependency for each word in the text
    Input: flat python string
    Return: A list of dict with {'token':token, 'dep': its dependency tag}
    '''
    doc = nlp(input_text)
    dep = []
    for token in doc:
        dict={}
        dict['token'], dict['pos'] = token.text, token.dep_
        dep.append(dict)
    return dep

def dep_path(input_text):
    '''
    This function shows the dependency path
    Input: flat python string
    Return: list of dicts. list: sentences in the text. 
                           dicts: token in the sentences as {'token':token, 
                                                             'head': the head of the token, 
                                                             'children': list of the childrens of the token}
    '''
    doc = nlp(input_text)
    deps = []
    for token in doc:
        dep={}
        dep['token'],dep['head'],dep['children'] = token.text, token.head.text, [child for child in token.children]
        deps.append(dep)
    return deps

def dep_dist_to_head(input_text):
    '''
    This function calculates the dependency distance from token to head.
    Input: flat python string
    Return: A list of dict with {'token':token, 'dist_to_head':its distance to head}
            If negative, then the token is before the head.
    '''
    doc = nlp(input_text)
    token_full_info, dist = doc.to_json()['tokens'], []
    for i in range(len(doc)):
        dict={}
        dict['token'],dict['dist_to_head'] = doc[i].text, token_full_info[i]['id']-token_full_info[i]['head']
        dist.append(dict)
    return dist



def Tag(input_text):
    '''
    This function provides more detailed part-of-speech tag.
    Input: flat python string
    Return: A list of dict with {'token':token, 'pos':its pos tag}
    '''
    doc = nlp(input_text)
    tag = []
    for token in doc:
        dict={}
        dict['token'], dict['tag'] = token.text, token.tag_
        tag.append(dict)
    return tag


def Tag(input_text):
    '''
    This function provides more detailed part-of-speech tag.
    Input: flat python string
    Return: A list of dict with {'token':token, 'pos':its pos tag}
    '''
    doc = nlp(input_text)
    tag = []
    for token in doc:
        dict={}
        dict['token'], dict['tag'] = token.text, token.tag_
        tag.append(dict)
    return tag


def extract_bigram(input_text):
    '''
    This function creates bigrams of the given text

    Input: flat python string
    Return: A list of dict with {'token':token, 'bigram': bigram}
    '''

    doc = nlp(input_text)
    bigram_list = []
        
    for i, token in enumerate(doc):
        dict={}
        if i < len(doc) - 1:
            ngram = str(token) + " " + str(doc[i+1])
            dict['token'], dict['ngram'] = token.text, ngram
            bigram_list.append(dict)
        else:
            dict['token'], dict['ngram'] = token.text, "EOS" #"end of sentence"
            bigram_list.append(dict)
    return bigram_list


def extract_morph_inform(input_text):
    '''
    This function extracts morphological information

    Input: flat python string
    Return: A list of dict with {'token':token, 'morph': morph}
    '''

    doc = nlp(input_text)
    morph = []

    for token in doc:
        dict={}
        dict['token'], dict['morph'] = token.text, token.morph
        morph.append(dict)
    return morph



def is_predicate (input_text):
    '''
    This function extracts the governing predicate, in this case the verb issued. 

    Input: Tflat python string
    Return: A list of dict with {'token':token, 'is_predicate': "VERB"/"_"}
    '''

    doc = nlp(input_text)
    predicate = []

    for token in doc:
        dict={}
        if token.pos_ == "VERB":
        
            dict['token'], dict['is_predicate'] = token.text, "VERB"
            predicate.append(dict)
        else:
            dict['token'], dict['is_predicate'] = token.text, "_"
            predicate.append(dict)
    return predicate

# "Noun phrases:" [chunk.text for chunk in doc.noun_chunks])
