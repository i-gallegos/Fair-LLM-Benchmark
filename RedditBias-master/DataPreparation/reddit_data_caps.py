"""
This script lowercased Muslim related target terms with uppercase in first character
"""
import pandas as pd


def replace_with_caps(text, replacements):
    for i, j in replacements.items():
        text = text.replace(i, j)
    return text


pd.set_option('display.max_columns', 50)
data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'religion2' # 'religion1' # 'race' # 'gender' #  # 'race'  # 'race' #'gender' # 'religion'
demo_1 = 'muslims' # 'jews'
demo_2 = 'christians'
input_file_suffix = '_processed_phrase_biased_testset'

df_demo_1 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_suffix + '.csv')
df_demo_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + input_file_suffix + '.csv')

print(df_demo_1.shape, df_demo_2.shape)
if demo == 'religion2':
    replace_dict_1 = {'muslim': 'Muslim', 'islam': 'Islam', 'arab': 'Arab'}
    replace_dict_2 = {'christian': 'Christian', 'america': 'America'}
    for idx, row in df_demo_1.iterrows():
        df_demo_1.loc[idx, 'comments_processed'] = replace_with_caps(row['comments_processed'], replace_dict_1)
    for idx, row in df_demo_2.iterrows():
        df_demo_2.loc[idx, 'comments_processed'] = replace_with_caps(row['comments_processed'], replace_dict_2)

print(df_demo_1.shape, df_demo_2.shape)

df_demo_1.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_suffix + '.csv', index=False)
df_demo_2.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + input_file_suffix + '.csv', index=False)