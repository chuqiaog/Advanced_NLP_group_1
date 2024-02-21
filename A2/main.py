from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import gensim
import numpy as np
import pandas as pd
import io

try:
    word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
except FileNotFoundError:
    print("No local embedding found. Downloading instead...")
    import gensim.downloader as api
    word_embedding_model = api.load('word2vec-google-news-300')

def embed_tokens(input_text):
    """
    function to return all embedding vectors for each word from an input text.
        :input: flat python string
        :output: list containing all embedded vectord for each word from the input string, if present in the model.
    """

    try:
        return word_embedding_model.get_vector(input_text)
    except:
        return np.zeros((300,))

def check_if_arg(arg):
    if arg in ['V','_']:
        return False
    else:
        return True

def read_conll(conllfile):
    """
    This function read and process the conllu file into list of sentences lists.
    """

    df_list = []

    output = io.StringIO()
    with open(conllfile, 'r', encoding='utf8') as infile:
        i = 0
        for line in infile:

            if line == '\n':
                print(i)
                output.seek(0)
                try:
                    df = pd.read_csv(output, sep='\t', header=None)
                except:
                    print('unable to read csv.')
                
                pred_col_template = ['_']*len(df)
                base_df = df.iloc[:,:10]
                
                for index, row in df.iterrows():

                    argument_list = list(row.values)

                    if row[10] != '_' and not pd.isna(row[10]):

                        pred_df = base_df.copy()
                        pred_col = pred_col_template.copy()
                        pred_col = [row[10]]*len(df)
                        #pred_col[index] = row[10]
                        pred_df['pred'] = pred_col
                        try:
                            pred_df['args'] = df.iloc[:,argument_list.index('V')].copy()
                        except:
                            print(df)
                        
                        df_list.append(pred_df)
                            
            elif line.startswith('#'):
                output = io.StringIO()
            else:
                output.write(line)

            i += 1

    return pd.concat(df_list, axis=0)

trainfile = 'data/en_ewt-up-train.conllu'
testfile = 'data/en_ewt-up-test.conllu'

traindf = read_conll(trainfile)
traindf.to_csv('data/train.csv')

#testdf = read_conll(testfile)
#testdf.to_csv('data/test.csv')

df = pd.read_csv('data/train.csv')

#add embedded vector for each word in the dataset
df['embed_vector'] = df['1'].apply(lambda x: embed_tokens(x))
new_columns = pd.DataFrame(df['embed_vector'].apply(pd.Series))
# Concatenate the new_columns DataFrame with the original DataFrame
df = pd.concat([df, new_columns], axis=1)
df.drop(['embed_vector'], axis=1, inplace=True)
#add column defining wether the word is an argument (gold data)
df['is_argument'] = df['args'].apply(lambda x: check_if_arg(x))

#one hot encode
one_hot = pd.get_dummies(df['7'])
df = df.drop('7',axis = 1)
df = df.join(one_hot)

#one hot encode
one_hot = pd.get_dummies(df['3'])
df = df.drop('3',axis = 1)
df = df.join(one_hot)

#drop irrelevant columns
df.drop(['Unnamed: 0','0','1','2','4','5','6','8','9','args','pred'],axis=1, inplace=True)

"""now that the dataset is prepared, 
we must split the gold target data 
from the input features and divide into
train and test sets"""

y = df['is_argument'].copy()
X = df.drop(['is_argument'], axis=1).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('fitting...')
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
print('fitted!')

predictions = logisticRegr.predict(X_test)

score = logisticRegr.score(X_test, y_test)
print(score)