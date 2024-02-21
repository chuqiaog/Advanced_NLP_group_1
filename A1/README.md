# Advanced_NLP_group_1

run the code by using the command: ' python main.py <function_name> <'input_string'> '
an example usage is: python main.py extract_embedding 'this is a test string'

the possible function names, along with a description are:

tokenize:
    function to tokenize any input text.
        :input: flat python string
        :output: list of all tokens from the input string.

lemmatization:
    function to return the lemmas of all words in the input text.
        :input: flat python string
        :output: list of all lemmas from the input string.

extract_embedding:
    function to return all embedding vectors for each word from an input text.
        :input: flat python string
        :output: list containing all embedded vectord for each word from the input string, if present in the model.
        
named_entity_recognition:
    function to return all named entities from an input text.
        :input: flat python string
        :output: list of all named entities present in the input.

sub_tree:
    This function generates the sub_tree of the input text.
    Input: flat python string
    Return: sub_tree_relations (list): List of dictionaries with {word:sub_tree} relation

capitalization:
    This function generates the capitalization of the input text.
    Input: flat python string
    Return: capitalization_list (list): Capitalization of the input text
  
syntactic_head:
    This function finds the syntactic head for each word in the input text.
    Input: flat python string
    Return: heads (list): List of dictionaries with {word:head} relation

PoS_tag:
    This function tags the Part-of-Speech tag of the words in the text
    Input: flat python string
    Return: A list of dict with {'token':token, 'pos':its pos tag}
    
Tag:
    This function provides more detailed part-of-speech tag.
    Input: flat python string
    Return: A list of dict with {'token':token, 'pos':its pos tag}

dep_relations:
    This function labels the dependency for each word in the text
    Input: flat python string
    Return: A list of dict with {'token':token, 'dep': its dependency tag}

def dep_path(input_text):
    This function shows the dependency path
    Input: flat python string
    Return: list of dicts. list: sentences in the text. 
                           dicts: token in the sentences as {'token':token, 
                                                             'head': the head of the token, 
                                                             'children': list of the childrens of the token}
dep_dist_to_head:
    This function calculates the dependency distance from token to head.
    Input: flat python string
    Return: A list of dict with {'token':token, 'dist_to_head':its distance to head}
            If negative, then the token is before the head.    

extract_bigram:
    This function creates bigrams of the given text
    Input: flat python string
    Return: A list of dict with {'token':token, 'bigram': bigram}

extract_morph_inform:
    This function extracts morphological information
    Input: flat python string
    Return: A list of dict with {'token':token, 'morph': morph}

is_predicate:
    This function extracts the governing predicate, in this case the verb issued. 
    Input: Tflat python string
    Return: A list of dict with {'token':token, 'is_predicate': "VERB"/"_"}

extract_wordnet_class:
    This function extracts the wordnet classes for all words in a sentence. 
    Input: flat python string
    Return: A list Synset objects containing all wordnet class matches for a given word.
