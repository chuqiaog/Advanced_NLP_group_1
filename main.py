import spacy
import gensim
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import sys

nlp = spacy.load("en_core_web_sm") 


try:
    word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
except FileNotFoundError:
    print("No local embedding found. Downloading instead...")
    import gensim.downloader as api
    word_embedding_model = api.load('word2vec-google-news-300')



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


def extract_embedding(input_text):
    """
    function to return all embedding vectors for each word from an input text.
        :input: flat python string
        :output: list containing all embedded vectord for each word from the input string, if present in the model.
    """

    words = tokenize(input_text)
    vectors = dict()
    for word in words:
        try:
            vectors[word] = word_embedding_model.get_vector(word)
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


def sub_tree(input_text):
    '''
    This function generates the sub_tree of the input text.
    Input: flat python string
    Return: sub_tree_relations (list): List of dictionaries with {word:sub_tree} relation
    '''

    doc = nlp(input_text)
    sub_tree_relations = []  # Initialize a list to store {word:sub_tree} relations
    for token in doc:
        if list(token.children) == []:
            sub_tree = [token]
        else:
            sub_tree = list(token.children)
        sub_tree_relations.append({token.text: [child.text for child in sub_tree]})  # Store {word:sub_tree} relation
    return sub_tree_relations


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
    Return: heads (list): List of dictionaries with {word:head} relation
    '''
    doc = nlp(input_text)
    heads = []  # Initialize a list to store {word:head} relation
    for token in doc:
        heads.append({token.text: token.head.text})  # Add {word:head} relation to the list
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


def extract_wordnet_class (input_text):
    '''
    This function extracts the wordnet classes for all words in a sentence. 

    Input: flat python string
    Return: A list Synset objects containing all wordnet class matches for a given word.
    '''
    
    doc = nlp(input_text)
    return [wordnet.synsets(str(token)) for token in doc if token != '']


def main():
    # Check if user provided enough arguments
    if len(sys.argv) < 3:
        print("Usage: python main.py <function_name> <'input_string'>")
        sys.exit(1)

    # Get function number and input string from command line arguments
    function_name = sys.argv[1]
    input_string = sys.argv[2].replace("'",'')

    # Check if the function exists
    if function_name in globals() and callable(globals()[function_name]):
        # Call the function dynamically using exec
        exec(f"print({function_name}('{input_string}'))")
        
    else:
        print("Invalid function name, possible functions are:")
        print("""   tokenize
                    lemmatization
                    extract_embedding
                    named_entity_recognition
                    sub_tree
                    capitalization
                    syntactic_head
                    PoS_tag
                    Tag
                    dep_relations
                    dep_path
                    dep_dist_to_head
                    extract_bigram
                    extract_morph_inform
                    is_predicate
                    extract_wordnet_class
              """)

if __name__ == "__main__":
    main()

# "Noun phrases:" [chunk.text for chunk in doc.noun_chunks])
