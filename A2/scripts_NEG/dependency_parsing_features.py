import spacy
import numpy as np
from spacy.tokens import Doc
from convert_conll import read_conll

def to_sentences(df):
    """
    Convert a DataFrame of tokenized text data to a list of sentences and corresponding negation cues. 
    If a sentence has more negation cues, it gets duplicate with a differen negation cue.
    The sentence is a list of tuples with the token and a boolean indicating if a space should be between the token and the next token.

    Args:
        df (pd.DataFrame): Input DataFrame with columns 'document_id', 'sentence_id', 'token_id', 'token', 'pos_tag',
                          'negation_word'.

    Returns:
        tuple: A tuple containing two lists:
               - List of sentences, where each sentence is represented as a list of tuples (token, needs_space).
               - List of negation cues, where each element is a list of negation words corresponding to a sentence.
    """
    ids = set(zip(df['document_id'], df['sentence_id']))
    sentences, neg_cues = [], []

    for doc_id, sent_id in ids:
        sentence = []
        neg_words = []
        for i, row in df[(df["document_id"] == doc_id) & (df["sentence_id"] == sent_id)].iterrows():
            # Check for negation words and add to the list
            if row['negation_word'] != '_' and row['negation_word'] != '***':
                neg_words.append(row['negation_word'])

            # Check for the start of a new sentence (token_id == '0')
            if row['token_id'] == '0':
                # Append the current sentence to the list of sentences
                sentences.append(sentence)
                # Start a new sentence with the current token
                sentence = [(row['token'], False if i == len(df)-1 else df['pos_tag'].iloc[i+1] == "PUNC")]

                # Append the list of negation words to the list of negation cues
                neg_cues.append(neg_words)
                # Clear the list of negation words
                neg_words = []
            else:
                # Add the current token to the current sentence
                sentence.append((row['token'], False if i == len(df)-1 else df['pos_tag'].iloc[i+1] == "PUNC"))

    return sentences, neg_cues


def spacy_preprocess(df):
    """
    Preprocess a DataFrame using SpaCy to extract features related to negation cues for each token in sentences.

    Args:
        df (pd.DataFrame): Input DataFrame with columns 'document_id', 'sentence_id', 'token_id', 'token', 'pos_tag',
                           'negation_word', and other relevant columns.

    Returns:
        list: List of lists of dictionaries. Each outer list corresponds to a sentence, and each inner list contains
              dictionaries representing features for each token in the sentence:
              - 'Dependency Relation': One-hot encoded vector representing the dependency relation of the token.
              - 'Dependency Distance to Cue': The distance to the closest cue token.
              - 'Dependency Path to Cue': Vector representing the dependency path to the closest cue token.
    """
    # Load the SpaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Access the DependencyParser component
    dependency_parser = nlp.get_pipe("parser")

    # Get all dependency labels in a dict for one hot encodings
    dependency_labels = ['acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'clf', 'complm', 'compound', 'conj', 'cop', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'discourse', 'dislocated', 'dobj', 'expl', 'fixed', 'flat', 'goeswith', 'hmod', 'hyph', 'infmod', 'intj', 'iobj', 'list', 'mark', 'meta', 'neg', 'nmod', 'nn', 'npadvmod', 'nsubj', 'nsubjpass', 'nounmod', 'npmod', 'num', 'number', 'nummod', 'oprd', 'obj', 'obl', 'orphan', 'parataxis', 'partmod', 'pcomp', 'pobj', 'poss', 'possessive', 'preconj', 'prep', 'prt', 'punct', 'quantmod', 'rcmod', 'relcl', 'reparandum', 'root', 'ROOT', 'vocative', 'xcomp']
    label2idx = {label: index for index, label in enumerate(dependency_labels)}

    sentences, neg_cues = to_sentences(df)
    feature_sentences = []
    for sentence, neg_cue in zip(sentences, neg_cues):
        feature_sentences.append(extract_cue_features(sentence, neg_cue, dependency_parser, label2idx))

    return feature_sentences

def extract_cue_features(sentence, cue_tokens, parser, label2idx):
    """
    Extract features related to negation cues for each token in a sentence.

    Args:
        sentence (list): List of tuples representing the tokenized sentence, where each tuple is (token, is_space).
        cue_tokens (list): List of cue tokens to consider for calculating features.
        parser (spacy.pipeline.dep_parser.DependencyParser): SpaCy dependency parser.
        label2idx (dict): Mapping of dependency relation labels to their corresponding indices.

    Returns:
        list: List of dictionaries, where each dictionary contains features for a token:
              - 'Dependency Relation': One-hot encoded vector representing the dependency relation of the token.
              - 'Dependency Distance to Cue': The distance to the closest cue token.
              - 'Dependency Path to Cue': Vector representing the dependency path to the closest cue token.

    """
    # Load the SpaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Process the input sentence using SpaCy
    doc = Doc(nlp.vocab, words=[t[0] for t in sentence], spaces=[t[1] for t in sentence])
    parser(doc)
    cue_tokens = [token for token in doc if token.text in cue_tokens]
    
    # Extract features per token
    features = []    
    
    
    for token in doc:
        # Extract the dependency relation for the cue token
        dependency_relation = np.zeros(len(label2idx))
        dependency_relation[label2idx[token.dep_]] = 1

        # Create the dependency distance and path vector
        dependency_distance = None
        dep_path = np.zeros(len(label2idx))

        if cue_tokens: 
            # Get the closest cue token
            cue_token, dependency_distance = min([(cue_token, abs(token.i - cue_token.i)) for cue_token in cue_tokens], key=lambda x: x[1])

            # Find the common ancestor of both tokens
            ancestors_token = [token]
            ancestors_idx_token = [token.i]
            ancestors_cue = [cue_token]

            # Traverse from the token to the root, storing ancestors
            current_token = token
            while current_token.head.i is not current_token.i:
                current_token = current_token.head
                ancestors_token.append(current_token)
                ancestors_idx_token.append(current_token.i)

            # Traverse from end_token to the first common ancester
            current_token = cue_tokens[0]
            while current_token.head.i not in ancestors_idx_token:
                current_token = current_token.head
                ancestors_cue.add(current_token)

            # Store the common ancestor
            common_ancestor = current_token

            # Traverse from the token to the common ancestor (going 'up')
            for current_token in ancestors_token:
                dep_path[label2idx[current_token.dep_]] = 1
                if current_token.i == common_ancestor.i:
                    break

            # Traverse from cue to the common ancestor (going 'down')
            for current_token in ancestors_cue:
                dep_path[label2idx[current_token.dep_]] = -1


        features.append({
            "Dependency Relation": dependency_relation,
            "Dependency Distance to Cue": dependency_distance,
            "Dependency Path to Cue": dep_path
        })
    
    return features

if __name__ == "__main__":
    df_dev = read_conll("converted_dev.conll")
    print(spacy_preprocess(df_dev))

