"""
This script generates text files of train and test datasets of Counter attribute data augmentation
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def build_dataset_manual_annot(df, dest_path):
    """
      Writes data from Dataframe to a text file, each dataframe row line by line in text file appending BOS and EOS token
      Parameters
      ----------
      df : pd.DataFrame
      Dataframe of biased reddit phrases
      dest_path : str
      Path to store text file
    """
    f = open(dest_path, 'w')
    data = ''

    for idx, row in df.iterrows():
        comment = row['comments_processed']
        data += comment + '\n'

    f.write(data)


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'race' # 'gender'  # 'orientation' # 'religion2' # 'religion1'
demo_1 = 'black' # 'female'  # 'lgbtq' # 'muslims' # 'jews'
demo_2 = 'white' # 'male'  # 'straight' # 'christians' # 'christians'
desti_path = data_path + 'text_files/' + demo + '/'


df_train_1 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase_biased_trainset' + '.csv')
df_train_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase_unbiased_trainset_pos_attr' + '.csv')

df_train_1 = df_train_1[['comments_processed']]
df_train_2 = df_train_2[['comments_processed']]

df_train = pd.concat([df_train_1, df_train_2])
build_dataset_manual_annot(df_train, desti_path + demo + '_bias_manual_swapped_attr_train.txt')

print(df_train.shape)

df_test_1 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase_biased_testset' + '.csv')
df_test_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase_unbiased_testset_pos_attr' + '.csv')

df_test_1 = df_test_1[['comments_processed']]
df_test_2 = df_test_2[['comments_processed']]

df_test = pd.concat([df_test_1, df_test_2])

print(df_test.shape)

build_dataset_manual_annot(df_test, desti_path + demo + '_bias_manual_swapped_attr_test.txt')
