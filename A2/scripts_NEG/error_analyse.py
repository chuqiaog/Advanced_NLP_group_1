import pandas as pd

def read_conll(filename):
    column_names = ['document_id', 'sentence_id', 'token_id', 'token', 'lemma', 'pos_tag', 'parsing_tree', 'negation_word', 'negation_scope', 'negation_event']
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            columns = line.split('\t')
            data.append(columns)

    df = pd.DataFrame(data, columns=column_names)

    return df



# create dataframes for file A and file D
df_A = read_conll(r'converted_dev.conll')
df_D = read_conll(r'converted_D.conll')

print(df_A.head())
print(df_D.head())

# find the unique sentence_ids in both dataframes
ids_A = set(zip(df_A['document_id'], df_A['sentence_id']))

with open('anntotations_vs_gold.txt', 'w') as f:
    for doc_id, sent_id in ids_A:
        f.write("\n" + sent_id + ":")
        
        for idx, line in df_A[df_A['sentence_id'] == sent_id].iterrows():
            if line['token_id'] == '0':
                f.write('\n A:\t')
            
            token = line['token']
            if line['negation_word'] == token:
                f.write(f"NEG[{token}] ")
            elif line['negation_scope'] == token:
                f.write(f"SCO[{token}] ")
            else:
                f.write(token + " ")
        
        for idx, line in df_D[(df_D['sentence_id'] == sent_id) & (df_D['document_id'] == doc_id)].iterrows():
            if line['token_id'] == '0':
                f.write('\n D:\t')
            
            token = line['token']
            if line['negation_word'] == token:
                f.write(f"NEG[{token}] ")
            elif line['negation_scope'] == token:
                f.write(f"SCO[{token}] ")
            else:
                f.write(token + " ")