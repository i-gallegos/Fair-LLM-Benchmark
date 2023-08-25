"""
This script extracts Reddit phrases manually annotated as Biased and corresponding generates Counter target dataset
"""
import pandas as pd
import re
from utils import reddit_helpers as rh


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'race' # 'gender' # 'orientation' # 'religion1' # 'religion2' # 'race' #
demo_1 = 'black' # 'female' # 'lgbtq' # 'jews' # 'muslims' # 'black_pos' # 'jews'
demo_2 = 'white' # 'male' # 'straight' # 'christians' # 'white_pos'
type_file = 'bias' # 'bias_unbias'
output_file_suffix = '_processed_phrase_biased' # '_processed_phrase_biased_unbiased'

demo1_df_processed = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase_annotated' + '.csv', encoding='Latin-1')

print('Shape of annotated dataframe {}'.format(demo1_df_processed.shape))
print(demo1_df_processed.head())

if type_file == 'bias':
    demo1_df_processed = demo1_df_processed[demo1_df_processed['bias_phrase'] == 1]
elif type_file == 'bias_unbias':
    demo1_df_processed = demo1_df_processed[(demo1_df_processed['bias_phrase'] == 1) | (demo1_df_processed['bias_phrase'] == 0)]

demo1_df_processed = demo1_df_processed.rename(columns={"phrase": "comments_processed"})
demo1_df_processed = demo1_df_processed.dropna(subset=['comments_processed'])

print('Shape of biased dataframe {}'.format(demo1_df_processed.shape))
print(demo1_df_processed.head())

demo1_df_processed.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_file_suffix + '.csv', index=False)

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
    pairs = (('woman', 'man'), ('women', 'men'), ('girl', 'boy'), ('mother', 'father'), ('daughter', 'son'), ('wife', 'husband'),
             ('niece', 'nephew'), ('mom', 'dad'), ('bride', 'groom'), ('lady', 'gentleman'), ('madam', 'sir'),
             ('hostess', 'host'), ('female', 'male'), ('aunt', 'uncle'), ('sister', 'brother'), (' she ', ' he '))
elif demo == 'orientation':
    pairs = (('gay', 'straight'), ('gays', 'straight'), ('lesbian', 'straight'), ('lesbians', 'straight'), ('bisexual', 'monosexual'),
             ('bisexuals', 'monosexuals'), ('homosexual', 'heterosexual'), ('homosexuals', 'heterosexuals'), ('transgender', 'cisgender'),
             ('transgenders', 'cisgenders'), ('sapphic', 'heterosexual'), ('pansexual', 'heterosexual'), ('queer', 'heterosexual'))
else:
    raise ValueError("Specify correct demographic")

for idx, row in demo1_df_processed.iterrows():
    initial_demo = []
    replaced_demo = []
    s = row['comments_processed']
    # print(s)
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
        elif demo == 'orientation':
            s = s.replace(*p)

        if p[1] in s and p[0] in row['comments_processed']:
            initial_demo.append(p[0])
            replaced_demo.append(p[1])
    demo2_df.at[idx, 'comments_processed'] = s
    demo2_df.at[idx, 'initial_demo'] = initial_demo
    demo2_df.at[idx, 'replaced_demo'] = replaced_demo

print('Shape of demo2 data {}'.format(demo2_df.shape))
demo2_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + output_file_suffix + '.csv', index=False)