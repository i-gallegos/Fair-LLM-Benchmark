"""
In this script, phrases with attributes related to career and interests are retained from the earlier extracted
Reddit phrases for demographic Gender
"""
import pandas as pd
import re
import numpy
import numpy as np

data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'gender'
demo_1 = 'female'

demo1_processed_phrase = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase' + '.csv')
print(demo1_processed_phrase.shape)


if demo == 'gender':
    data_career = []
    career_attributes = ['nurse', 'secretary', 'housekeep', 'cook', 'sewing', 'cleaning', 'waitress', 'maid', 'child-care',
                         'receptionist', 'clerk', 'home-care']
    for idx, row in demo1_processed_phrase.iterrows():
        # print(row.to_dict())
        if any(attr in row['phrase'] for attr in career_attributes):
            # print('yes')
            data_career.append(row.to_dict())

    print('Career dataset shape {}'.format(len(data_career)))

    demo1_processed_phrase_extra = pd.read_csv(
        data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase_extra' + '.csv')
    print('Art dataset shape {}'.format(demo1_processed_phrase_extra.shape))

    demo1_processed_phrase = demo1_processed_phrase_extra.append(data_career)
    print('Final career and art sentences for Females {}'.format(demo1_processed_phrase.shape))

    demo1_processed_phrase.reset_index(inplace=True)

drop_n = demo1_processed_phrase.shape[0] - 3000
drop_indices = np.random.choice(demo1_processed_phrase.index, drop_n, replace=False)
print(len(drop_indices))

demo1_reduced = demo1_processed_phrase.drop(drop_indices)

if demo == 'gender':
    demo1_reduced = demo1_reduced.drop(columns=['index', 'id'])
print(demo1_reduced.shape)
demo1_reduced.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase_for_annot' + '.csv', index=False)
