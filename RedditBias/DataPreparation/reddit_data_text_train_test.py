"""
This script generates csv and text files of train and test split for biased reddit dataset
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def build_dataset_bos_eos(df, demo, dest_path):
    """
    Writes data from Dataframe to a text file, each dataframe row line by line in text file appending BOS and EOS token
    Parameters
    ----------
    df : pd.DataFrame
    Dataframe of biased reddit phrases
    demo : str
    Demographic name
    dest_path : str
    Path to store text file

    """
    f = open(dest_path, 'w')
    data = ''

    for idx, row in df.iterrows():
        bos_token = '<bos>'
        eos_token = '<eos>'
        comment = row['comments_2']
        data += bos_token + ' ' + comment + ' ' + eos_token + '\n'

    f.write(data)


def build_dataset_manual_annot(df, demo, dest_path):
    """
    Writes data from Dataframe to a text file, each dataframe row line by line in text file
    Parameters
    ----------
    df : pd.DataFrame
    Dataframe of biased reddit phrases
    demo : str
    Demographic name
    dest_path : str
    Path to store text file

    """
    f = open(dest_path, 'w')
    data = ''

    for idx, row in df.iterrows():
        comment = row['comments_processed']
        if demo == 'orientation':
            data += '<bos>' + ' ' + comment + '\n'
        else:
            data += comment + '\n'

    f.write(data)


def replace_with_caps(text, replacements):
    for i, j in replacements.items():
        text = text.replace(i, j)
    return text


pd.set_option('display.max_columns', 50)
data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'race' # 'gender' # 'orientation' # 'religion1' # 'religion2' #'gender' # 'religion'
demo_1 = 'black' # 'female' # 'lgbtq' # 'jews' # 'muslims'
demo_2 = 'white' # 'male' # 'straight' # 'christians'
input_file_suffix = '_processed_phrase_biased' # '_processed_phrase_biased_unbiased'
output_txt_train = '_bias_manual_train.txt' # '_bias_unbias_manual_train.txt' # '_bias_manual_lowercase_train.txt'
output_txt_test = '_bias_manual_valid.txt' # '_bias_unbias_manual_valid.txt' # '_bias_manual_lowercase_valid.txt'
output_csv_test = '_processed_phrase_biased_testset' # '_processed_phrase_biased_unbias_testset'
output_csv_train = '_processed_phrase_biased_trainset' # '_processed_phrase_biased_unbias_trainset'
type_data = 'bias' # 'bias_unbias'


df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_suffix + '.csv')
print('df shape {}'.format(df.shape))

if type_data == 'bias_unbias':
    df_bias_testset = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase_biased_testset_reduced' + '.csv')
    cond = df['comments_processed'].isin(df_bias_testset['comments_processed'])
    df = df.drop(df[cond].index)

print(df.shape)
if demo == 'gender':
    train_test_ratio = 0.75
else:
    train_test_ratio = 0.6

df_train, df_test = train_test_split(df, stratify=df['bias_phrase'], train_size=train_test_ratio, random_state=1)

print('Train {}'.format(df_train.shape))
print('Test {}'.format(df_test.shape))
print(df_train['bias_phrase'].value_counts())
print(df_test['bias_phrase'].value_counts())

'''
if demo == 'religion2':
    replace_dict = {'muslim': 'Muslim', 'islam': 'Islam', 'arab': 'Arab'}
    # df_train['comments_processed'] = df_train.apply(lambda x: replace_with_caps(df_train['comments_processed'], replace_dict), axis=1)
    # df_test['comments_processed'] = df_test.apply(lambda x: replace_with_caps(df_test['comments_processed'], replace_dict), axis=1)
    # df_train_new = df_train.apply(lambda x: replace_with_caps(x, replace_dict) if x.name == 'comments_processed' else x)
    # df_test_new = df_test.apply(lambda x: replace_with_caps(x, replace_dict) if x.name == 'comments_processed' else x)
    for idx, row in df_train.iterrows():
        df_train.loc[idx, 'comments_processed'] = replace_with_caps(row['comments_processed'], replace_dict)
    for idx, row in df_test.iterrows():
        df_test.loc[idx, 'comments_processed'] = replace_with_caps(row['comments_processed'], replace_dict)

print(df_train.head())
'''

desti_path = data_path + 'text_files/' + demo + '/'
build_dataset_manual_annot(df_train, demo, desti_path + demo + output_txt_train)
build_dataset_manual_annot(df_test, demo, desti_path + demo + output_txt_test)

df_test.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_csv_test + '.csv', index=False)
df_train.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_csv_train + '.csv', index=False)