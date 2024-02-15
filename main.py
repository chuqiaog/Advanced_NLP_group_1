import spacy
import gensim
from nltk.corpus import brown


nlp = spacy.load("en_core_web_sm") 
word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)



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

# print(fetch_embedding(model=word_embedding_model, input_text=test_str))


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
