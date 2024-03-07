from features import main as fmain
from dependency_parsing_gpu import *
from tqdm import tqdm
import argparse
import pickle

parser = argparse.ArgumentParser(
                    prog='get_all_features',
                    description='This script get all features from conll or tsv.',
                    epilog='Example usage: python3 get_all_features.py converted_train.tsv')
parser.add_argument('inputfile', help="Input file path") 

def main(filename):
    
    raw_df = read_conll(filename)

    df = fmain(raw_df)

    ids = set(zip(df['document_id'], df['sentence_id']))
    sentences = []


    for doc_id, sent_id in ids:
        sentence = {}
        for _, row in tqdm(df[(df["document_id"] == doc_id) & (df["sentence_id"] == sent_id)].iterrows()):
            # Check for the start of a new sentence (token_id == '0')
            print("row", row['document_id'], row['sentence_id'])
            label = 1 if row['token'] == row['negation_scope'] else 0
            token_dict = {
                    "document_id": row["document_id"],
                    "sentence_id": row["document_id"],
                    "token_id": row["document_id"],
                    "pos_tag_encoded": row["pos_tag_encoded"],
                    "is_capitalized": row["is_capitalized"],
                    "distance_from_negation": row["cue_distance"],
                    "is_punctuation": row["cp_existence"],
                    "is_negation_cue": row["is_negation_cue"],
                    "token_embeddings": row["token_embeddings"],
                    "lemma_embeddings": row["lemma_embeddings"],
                    "label":label
                }
            
            if row['token_id'] == '0':
                # Append the current sentence to the list of sentences
                sentences.append(sentence)
                # Start a new sentence with the current token
                sentence = [token_dict]
            else:
                # Add the current feature of token to the current sentence
                sentence.append(token_dict)
    
    dep_features = spacy_preprocess(raw_df)

    features = []
    for fsen, depsen in tqdm(zip(sentences, dep_features)):
        feature_sentence = []
        for ftoken, deptoken in zip(fsen, depsen):
            #print(ftoken, deptoken)
            ftoken.update(deptoken)
            feature_sentence.append(ftoken)
        
        features.append(feature_sentence)
    
    return features
            
if __name__ == '__main__':
    args = parser.parse_args()
    input_file_path = args.inputfile
    features = main(input_file_path)
    with open('features_debug.pkl', 'wb') as f:
        pickle.dump(features, f)


