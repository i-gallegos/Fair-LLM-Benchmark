import pandas as pd
import re

demo = 'gender' # 'orientation' # 'religion1' # 'religion1' # '' # 'gender' # 'religion2' # '' # 'race'
demo_1 = 'female' # 'lgbtq' # 'jews' # 'muslims' # 'black' # 'black_pos' # '' #
train_file = '_processed_phrase_biased_trainset' # '_processed_phrase_biased_testset_reduced' # '_processed_phrase_biased' # '_processed_sent_biased' # '_processed'
test_file = '_processed_phrase_biased_testset'

data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'

demo_df_train = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + train_file + '.csv')
demo_df_test = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + test_file + '.csv')

with open(data_path + demo + '/' + demo + '_' + demo_1 + '.txt') as f:
    attribute_list = [re.sub('[*"]', '', line.split('\n')[0]) for line in f]

print(attribute_list)


def get_dist(attributes, df):
    attr_sent_count_dict = {}

    for attr in attributes:
        count = 0
        for idx, row in df.iterrows():
            if attr in row['comments_processed']:
                count += 1
        attr_sent_count_dict[attr] = count
    return attr_sent_count_dict


sent_dist_train = get_dist(attribute_list, demo_df_train)
sent_dist_test = get_dist(attribute_list, demo_df_test)

print('Trainset - Distribution of sentences over attribute words: {}'.format(sent_dist_train))
print('Testset - Distribution of sentences over attribute words: {}'.format(sent_dist_test))