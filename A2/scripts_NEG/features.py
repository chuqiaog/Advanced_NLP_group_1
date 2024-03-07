import numpy as np
import string
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

from convert_conll import read_conll

import sys
import argparse

parser = argparse.ArgumentParser(
                    prog='convert_conll',
                    description='This script extract feature from the converted conll file.',
                    epilog='Example usage: python3 features.py converted_conll')
parser.add_argument('inputfile', help="Input path for the converted 10 column conll")



def check_punctuation(df, column='token'):
    '''
    For each token in the specified column, it checks if all characters in the token are punctuation. 
    If all characters are punctuation, it assigns 1 to the corresponding cell in the new column; otherwise, it assigns 0. 
    
    '''
    df['is_punctuation'] = df[column].apply(lambda x: int(all(char in string.punctuation for char in x)))
    
    return df



def check_negation_cue(df, column='token', negation_cues=['no', 'not', 'none']):
    '''
     For each token in the specified column, it checks if the token is in the list of negation cues. 
     If the token is a negation cue, it assigns 1 to the corresponding cell in the new column; otherwise, it assigns 0. 
    
    '''
    df['is_negation_cue'] = df[column].apply(lambda x: 1 if x in negation_cues else 0)
    
    return df


""" BELOW IS A PROBLEMATIC IMPLEMENTATION
def negation_word_distance(df):
    '''
    Creates a new column which contain the distance of each token from the negation word.
    By distance we mean the number of words between the token and the nearest negation word.
    
    '''
    
    df['distance_from_negation'] = len(df)

    # find the indices where 'negation_word' is not '_'
    negation_indices = df[df['negation_word'] != '_'].index

    for i in range(len(df)):
        # calculate the distance from the current token to each negation word
        distances = abs(negation_indices - i)
        
        # set 'distance_from_negation' to the minimum distance, if there are any negation words
        if len(distances) > 0:
            df.loc[i, 'distance_from_negation'] = min(distances)

    return df
"""

# This is the new implementation
def to_sentences_and_cue_position(df):
    """
    SUPPORT FUNCTION FOR THE FOLLOWING token_cue_distance(df) AND special_token_inbetween
    
    Creates two lists: a list of sentences; their corresponding negation cues.
    Both lists consits of the token and its position.
    If the sentence has no negation, cues will be ***
    """
    sent_list, sent, neg_list, neg = [],[],[],[]

    for x in range(len(df)): # x: sentences
        if df['token_id'][x] == '0': # if starting of sentence, clear containers
            sent_list.append(sent)
            sent = []

            cue='' # deal with multi word negation such as "by no means"
            for y in neg: # y: collected column 'negation_word' in sentence x
                if y[0] == '***':
                    neg_list.append(['***','***'])
                    break
                    
                if y[0] != '_':
                    cue = cue + y[0]
                    position = y[1] # define the last word as position (i.e. "means" in "by no means")
            if cue != '':
                neg_list.append([cue,position])
            neg = []
    
        sent.append([df['token'][x] , df['token_id'][x]])
        neg.append([df['negation_word'][x] , df['token_id'][x]]) # collect column 'negation_word' for one sentence

    sent_list.pop(0)
    sent_list.append(sent)
    cue=''
    for y in neg:
        if y[0] == '***':
            neg_list.append(['***','***'])
            break
                    
        if y[0] != '_':
            cue = cue + y[0]
            position = y[1] # define the last word as position (i.e. "means" in "by no means")
    if cue != '':
        neg_list.append([cue,position])

    return sent_list, neg_list

def token_cue_distance(df):
    """
    Calculates the distance from every token to the negation cue in this sentence.
    The cue to its own distance is 0.
    If the distance is positive, then the token is befor the cue and vise versa.
    If there's no negation in the sentence, the distance will be ***
    """
    sent_list, neg_list = to_sentences_and_cue_position(df)
    dist_list = []
    for x in range(len(neg_list)): # x: current sentence number
        for y in range(len(sent_list[x])): # y: current word in the sentence
            if neg_list[x][0] == '***':
                dist_list.append(None)
            else:
                dist = int(sent_list[x][y][1]) - int(neg_list[x][1])
                dist_list.append(dist)
                
    df['cue_distance'] = dist_list
    return df


def special_token_inbetween(df):
    '''
    Checks if there exist special token between the token and the negation cue.
    Special tokens are: punctuations and some of the conjunctions.
    If the token is the cue, the value is set as False.
    '''
    existence_list = []
    conj_punt_set = [',','.','?','"','\'','!','``',':',';','\'\'','-','--','`','(',')','[',']',
                    'for','and','nor','but','or','yet','while','when','whereas','whenever','wherever','whether','if','because','before',
                     'until','unless','since','so','although','after','as','','Accordingly','After','Also','Besides','Consequently',
                     'Conversely','Finally','Furthermore','Hence','However','Indeed','Instead','Likewise','Meanwhile','Moreover','Nevertheless',
                     'Nonetheless','Otherwise','Similarly','Still','Subsequently','Then','Therefore','Thus','except','rather']
    sent_list, neg_list = to_sentences_and_cue_position(df)
    
    for x in range(len(neg_list)): # x: the sentences                
        if neg_list[x][0] == '***':
            for y in range(len(sent_list[x])): # y: current word in the sentence
                existence_list.append(None)

        else:
            cue_position = int(neg_list[x][1])
            for y in range(0,cue_position+1): # y: tokens before cue and cue
                y_exist = False
                for z in range(y+1,cue_position+1): # z: all tokens between y and cue
                    if sent_list[x][y][0].casefold() in (token.casefold() for token in conj_punt_set):
                        y_exist = True
                if y == cue_position: # token is cue
                    y_exist = False # Set as False in this case
                existence_list.append(y_exist)

            for y in range(cue_position+1,len(sent_list[x])): # y: tokens after cue
                y_exist = False
                for z in range(cue_position+1,len(sent_list[x])): # z: all tokens between cue and y
                    if sent_list[x][y][0].casefold() in (token.casefold() for token in conj_punt_set):
                        y_exist = True
                existence_list.append(y_exist)

    df = df.copy()
    df.loc[:, "cp_existence"] = existence_list
    
    return df


def check_capitalized(df):
    '''
    Creates a new column which indicates whether each token is capitalized (1) or not (0).
    Useful as proper nouns and the beginning of sentences are often not within the scope of negation.
    
    '''
    
    df['is_capitalized'] = df['token'].apply(lambda x: 1 if isinstance(x, str) and x[0].isupper() else 0)

    return df



def word_embeddings(df, columns=['token', 'lemma'], size=100, window=5, min_count=1, workers=4):
    
    models = {}
    for column in columns:
        sentences = df[column].apply(str.split).tolist()

        # creates a Word2Vec model
        model = Word2Vec(sentences, vector_size=size, window=window, min_count=min_count, workers=workers)
        df[column + '_embeddings'] = df[column].apply(lambda x: model.wv[x] if x in model.wv else np.zeros(size))
        models[column] = model

    return df, models

def find_common_ancestor(tree1, tree2):
    """
    Find the common ancestor of two tokens based on parsing tree representations.

    Args:
        tree1 (str): Parsing tree representation for token1.
        tree2 (str): Parsing tree representation for token2.

    Returns:
        str or None: Common ancestor in the tree representation, or None if no common ancestor is found.
    """
    # Parse the tree representations into lists
    ancestors1 = parse_tree(tree1)
    ancestors2 = parse_tree(tree2)

    # Find the intersection of the lists (common ancestors)
    common_ancestors = set(ancestors1).intersection(ancestors2)

    # If there are common ancestors, return the one closest to the tokens
    if common_ancestors:
        return min(common_ancestors, key=lambda ancestor: ancestors1.index(ancestor))

    # No common ancestor found
    return None

def parse_tree(tree):
    """
    Parse a tree representation into a list of ancestors.

    Args:
        tree (str): Parsing tree representation.

    Returns:
        list: List of ancestors extracted from the tree representation.
    """
    ancestors = []
    current = ""

    for char in tree:
        if char == '(':
            ancestors.append(current.strip())
            current = ""
        elif char == ')':
            current = current.strip()
            if current:
                ancestors.append(current)
            current = ""
        else:
            current += char

    return ancestors



def common_ancestor(df):
    common_ancestors = []
    for i, row in df.iterrows():
        row['parsing_tree']


def label_encoding(df, columns=['pos_tag']):
    '''
    Each unique POS-tag value is assigned an integer value.
    
    '''
    for column in columns:
        le = LabelEncoder()
        df[column + '_encoded'] = le.fit_transform(df[column])
        
    return df

def main(df):
    (df, models) = word_embeddings(df, ['token', 'lemma'])
    features_df = label_encoding(df).pipe(check_capitalized).pipe(token_cue_distance).pipe(check_negation_cue).pipe(special_token_inbetween)
    
    return features_df

if __name__ == "__main__":
    args = parser.parse_args()
    input_file_path = args.inputfile
    
    df = read_conll(input_file_path)

    (df, models) = word_embeddings(df, ['token', 'lemma'])
    df = label_encoding(df).pipe(check_capitalized).pipe(token_cue_distance).pipe(special_token_inbetween).pipe(check_negation_cue)

    df = df.groupby('sentence_id').apply(lambda x: x.to_dict('records')).tolist()
    print(df[:5])  

