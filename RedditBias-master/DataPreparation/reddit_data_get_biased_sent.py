import pandas as pd
import numpy as np


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'religion1' # 'race' # 'religion2' # 'gender' #  # 'race'  # 'race' #'gender'
demo_1 = 'jews' # 'black_pos' # 'muslims' # 'female' # 'black'  # 'jews' # 'black' #'female' # 'jews'
demo_2 = 'christians' # 'white_pos' # 'male' #  # 'white'

demo1_df_processed = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase_annotated' + '.csv', encoding='Latin-1')

print(demo1_df_processed.head())
print(demo1_df_processed.shape)
print(demo1_df_processed.dtypes)

demo1_df_processed = demo1_df_processed.dropna(subset=['bias_sent'])
print(demo1_df_processed.head())
print(demo1_df_processed.shape)
print(demo1_df_processed.dtypes)

demo1_df_processed = demo1_df_processed[demo1_df_processed['bias_sent'].str.isnumeric()]

# demo1_df_processed = demo1_df_processed[demo1_df_processed['bias_sent'].applymap(np.isreal).all(1)]

# demo1_df_processed['bias_sent'] = demo1_df_processed.bias_sent.astype(float)
print(demo1_df_processed.head())
print(demo1_df_processed.shape)

demo1_df_processed = demo1_df_processed[demo1_df_processed['bias_sent'] == str(1)]
demo1_df_processed = demo1_df_processed.rename(columns={"comment": "comments_processed"})

print(demo1_df_processed.head())
print(demo1_df_processed.shape)

demo1_df_processed.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_sent_biased' + '.csv', index=False)

if demo == 'gender':
    colNames = ('id', 'comments_processed')
    demo2_df = pd.DataFrame(columns=colNames)

    gender_words = ['woman', 'women', 'girl', 'mother', 'daughter', 'wife', 'niece', 'mom', 'bride', 'lady', 'madam',
                    'hostess', 'female', 'wife', 'aunt', 'sister', 'man', 'men', 'boy', 'father', 'son', 'husband',
                    'nephew', 'dad', 'groom', 'gentleman', 'sir', 'host', 'male', 'husband', 'uncle', 'brother']
    comments_one_g = []
    for idx, row in demo1_df_processed.iterrows():
        s = row['comments_processed']
        match = {m for m in gender_words if m in s}
        print(match)
        if len(match) == 1:
            comments_one_g.append(s)
    demo2_df['comments_processed'] = comments_one_g
    print('gender one df {}'.format(demo2_df.shape))
    demo1_df_processed = demo2_df
    demo1_df_processed.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed' + '.csv', index=False)

demo2_df = pd.DataFrame(columns=['initial_demo', 'replaced_demo', 'comments', 'comments_processed'])

if demo == 'race':
    pairs = (('black', 'white'), ('african american', 'anglo american'), ('african-american', 'anglo-american'),
             ('afro-american', 'anglo-american'), ('african', 'american'), ('afroamericans', 'angloamericans'),
             ('negroes', 'caucasians'), ('dark-skin', 'light-skin'), ('dark skin', 'light skin'))
elif demo == 'religion1':
    pairs = (('jew ', 'christian '), ('jewish', 'christian'), ('jews ', 'christians '), ('judaism', 'christianity'))
elif demo == 'religion2':
    pairs = (('muslim', 'christian'), ('islamic', 'christian'), ('islam ', 'christianity '), ('arabs', 'americans'), ('islamism', 'christianity'))
elif demo == 'gender':
    pairs = \
    (('woman', 'man'), ('women', 'men'), ('girl', 'boy'), ('mother', 'father'), ('daughter', 'son'), ('wife', 'husband'),
    ('niece', 'nephew'), ('mom', 'dad'), ('bride', 'groom'), ('lady', 'gentleman'), ('madam', 'sir'),
    ('hostess', 'host'),
    ('female', 'male'), ('wife', 'husband'), ('aunt', 'uncle'), ('sister', 'brother'), (' she ', ' he '))
else:
    pairs = ()

for idx, row in demo1_df_processed.iterrows():
    initial_demo = []
    replaced_demo = []
    s = row['comments_processed']
    print(s)
    demo2_df.at[idx, 'comments'] = s

    for p in pairs:
        # s = s.replace(*p)
        if demo == 'race':
            if p[0] == 'african' and p[0] in s and ('anglo american' in s or 'anglo-american' in s):
                s = s.replace(*p)
            elif p[1] == 'american' and p[1] in s and ('anglo american' in s or 'anglo-american' in s):
                s = s.replace(*p)
            elif p[0] == 'afro-american' and p[0] in s:
                s = s.replace(*p)
            else:
                s = s.replace(p[0], '%temp%').replace(*reversed(p)).replace('%temp%', p[1])
        elif demo == 'religion1':
            if p[0] == 'jewish':
                if p[0] in s and ('christian' in s):
                    s = s.replace(*p)
                elif 'christian' in s:
                    s = s.replace(*p)
                else:
                    s = s.replace(p[0], '%temp%').replace(*reversed(p)).replace('%temp%', p[1])
            else:
                s = s.replace(p[0], '%temp%').replace(*reversed(p)).replace('%temp%', p[1])
        elif demo == 'religion2':
            if p[0] == 'islamic':
                if p[0] in s and ('christian' in s):
                    s = s.replace(*p)
                elif 'christian' in s:
                    s = s.replace(*p)
                else:
                    s = s.replace(p[0], '%temp%').replace(*reversed(p)).replace('%temp%', p[1])
            elif p[0] == 'islamism':
                if p[0] in s and ('christianity' in s):
                    s = s.replace(*p)
                elif 'christianity' in s:
                    s = s.replace(*p)
                else:
                    s = s.replace(p[0], '%temp%').replace(*reversed(p)).replace('%temp%', p[1])
            else:
                s = s.replace(p[0], '%temp%').replace(*reversed(p)).replace('%temp%', p[1])
        elif demo == 'gender':
            s = s.replace(*p)

        if p[1] in s:
            initial_demo.append(p[0])
            replaced_demo.append(p[1])
    demo2_df.at[idx, 'comments_processed'] = s
    demo2_df.at[idx, 'initial_demo'] = initial_demo
    demo2_df.at[idx, 'replaced_demo'] = replaced_demo

print('Shape of demo2 data {}'.format(demo2_df.shape))
demo2_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + '_processed_sent_biased' + '.csv', index=False)