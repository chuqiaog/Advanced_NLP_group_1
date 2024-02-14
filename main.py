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
            continue
    return vectors
     


def named_entity_recognition(input_text):
    """
    function to return all named entities from an input text.
        :input: flat python string
        :output: list of all named entities present in the input.
    """

    doc = nlp(input_text)
    return [ent for ent in doc.ents]


test_str = 'this input string is used as a test to see if these words can be embedded.'

print(fetch_embedding(model=word_embedding_model, input_text=test_str))
