"""
Create test set and validation set split on the test dataset with removed perplexity outliers
"""
import pandas as pd
from sklearn.model_selection import train_test_split


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
        # if demo == 'orientation':
        #     data += '<bos>' + ' ' + comment + '\n'
        # else:
        data += comment + '\n'

    f.write(data)


pd.set_option('display.max_columns', 50)
data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'orientation' # 'gender' # 'race' # 'religion2' # 'religion1' #'gender' # 'religion'
demo_1 = 'lgbtq' # 'female' # 'black' # 'jews' # 'muslims'
demo_2 = 'straight' # 'male' # 'white' # 'christians'
input_file_suffix = '_processed_phrase_biased_testset_reduced' # '_processed_phrase_biased_unbiased'

output_csv_valid = '_biased_valid_reduced' # '_processed_phrase_biased_unbias_testset'
output_csv_test = '_biased_test_reduced' # '_processed_phrase_biased_unbias_trainset'

df1 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_suffix + '.csv')
df2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + input_file_suffix + '.csv')

print('df1 shape {}'.format(df1.shape))
print('df2 shape {}'.format(df2.shape))

train_test_ratio = 0.5

df1_valid, df1_test, df2_valid, df2_test = train_test_split(df1, df2,
                                                            train_size=train_test_ratio, random_state=1)

print('Train {}'.format(df1_valid.shape))
print('Test {}'.format(df1_test.shape))
print(df1_valid['comments_processed'].head())
print(df1_test['comments_processed'].head())

print('Train {}'.format(df2_valid.shape))
print('Test {}'.format(df2_test.shape))
print(df2_valid['comments_processed'].head())
print(df2_test['comments_processed'].head())

desti_path = data_path + 'text_files/' + demo + '/'
build_dataset_manual_annot(df1_valid, demo, desti_path + demo + '_' + demo_1 + output_csv_valid + '.txt')
build_dataset_manual_annot(df2_valid, demo, desti_path + demo + '_' + demo_2 + output_csv_valid + '.txt')

build_dataset_manual_annot(df1_test, demo, desti_path + demo + '_' + demo_1 + output_csv_test + '.txt')
build_dataset_manual_annot(df2_test, demo, desti_path + demo + '_' + demo_2 + output_csv_test + '.txt')

df1_valid.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_csv_valid + '.csv', index=False)
df2_valid.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + output_csv_valid + '.csv', index=False)

df1_test.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_csv_test + '.csv', index=False)
df2_test.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + output_csv_test + '.csv', index=False)
