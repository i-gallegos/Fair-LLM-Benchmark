"""
This script processes the raw Reddit comments for Target group 1(from reddit_data.py) and further creates Counter Target
dataset containing target group term replaced with Target group 2 terms. In case of demographic - Gender and
Sexual orientation, Reddit comments with only one Target group mention are retained.
"""
import pandas as pd
import re
from utils import reddit_helpers as rh


if __name__ == '__main__':

    data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
    demo = 'orientation' # 'gender' # 'race' # 'religion2' #  # 'race'  # 'race' #'gender' # 'religion'
    demo_1 = 'lgbtq' # 'female' # 'black_pos' # 'muslims' #  # 'jews' # 'black'  # 'jews' # 'black' #'female' # 'jews'
    demo_2 = 'straight' # 'male' # 'white_pos' # 'christians' #  # 'white'
    PROCESS_DEMO1 = True

    # Process Reddit comments in all raw files and store in processed file for Target group 1
    if PROCESS_DEMO1:
        print('Processing demo1 reddit files...')
        colNames = ('id', 'comments', 'comments_processed')

        demo1_df_processed = pd.DataFrame(columns=colNames)
        df_list = []
        if demo == 'gender' or demo == 'religion2':
            loops = 7
        elif demo == 'race' or demo == 'orientation':
            loops = 5
        elif demo == 'religion1':
            loops = 6
        else:
            loops = None
            print('Specify a correct demographic')

        for i in range(loops):
            demo1_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_raw_' + str(i)+'.csv')
            demo1_df = demo1_df.loc[:, ~demo1_df.columns.str.contains('^Unnamed')]

            demo1_df = demo1_df.dropna()

            demo1_df['comments_processed'] = demo1_df['comments'].apply(lambda x: rh.process_tweet(x))
            print('Before length filter {}'.format(demo1_df.shape))
            demo1_df = demo1_df[demo1_df['comments_processed'].str.len() < 150]
            # pd.concat([demo1_df_processed, demo1_df])
            print('After length filter {}'.format(demo1_df.shape))
            # demo1_df_processed.append(demo1_df, ignore_index=True)
            df_list.append(demo1_df)

        demo1_df_processed = pd.concat(df_list, ignore_index=True)
        print(demo1_df_processed.shape)
        demo1_df_processed = demo1_df_processed.dropna()
        demo1_df_processed = demo1_df_processed[demo1_df_processed['comments_processed'] != 'nan']
        print('After dropping nan {}'.format(demo1_df_processed.shape))

        demo1_df_processed.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed' + '.csv', index=False)

    # If demo is gender or orientation retain sentences with only one target group term
    if demo == 'gender':
        colNames = ('id', 'comments_processed')
        demo2_df = pd.DataFrame(columns=colNames)

        gender_words = ['woman', 'women', 'girl', 'mother', 'daughter', 'wife', 'niece', 'mom', 'bride', 'lady',
                        'madam',
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
        print('Shape of df with single target group comments {}'.format(demo2_df.shape))
        demo1_df_processed = demo2_df
        demo1_df_processed.to_csv(
            data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed' + '.csv', index=False)

    if demo == 'orientation':
        colNames = ('id', 'comments_processed')
        demo2_df = pd.DataFrame(columns=colNames)

        orientation_words = ['gay', 'lesbian', 'bisexual', 'homosexual', 'transgender', 'sapphic', 'pansexual', 'queer']
        comments_one_g = []
        for idx, row in demo1_df_processed.iterrows():
            s = row['comments_processed']
            match = {m for m in orientation_words if m in s}
            print(match)
            if len(match) == 1:
                comments_one_g.append(s)
        demo2_df['comments_processed'] = comments_one_g
        print('Shape of df with single target group comments {}'.format(demo2_df.shape))
        demo1_df_processed = demo2_df
        demo1_df_processed.to_csv(
            data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed' + '.csv', index=False)
    else:
        print('Reading processed demo1 reddit files...')
        demo1_df_processed = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed' + '.csv')
        print('Shape of demo1 data {}'.format(demo1_df_processed.shape))

    # Create Counter target data set
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
                 ('niece', 'nephew'), ('mom', 'dad'), ('bride', 'groom'), ('lady', 'gentleman'), ('madam', 'sir'), ('hostess', 'host'),
                 ('female', 'male'), ('wife', 'husband'), ('aunt', 'uncle'), ('sister', 'brother'), (' she ', ' he '))
    else:
        pairs = (('gay', 'straight'), ('gays', 'straights'), ('lesbian', 'straight'), ('lesbians', 'straights'), ('bisexual', 'monosexual'),
                 ('bisexuals', 'monosexuals'), ('homosexual', 'heterosexual'), ('homosexuals', 'heterosexuals'), ('transgender', 'cisgender'),
                 ('transgenders', 'cisgenders'), ('sapphic', 'heterosexual'), ('pansexual', 'heterosexual'), ('queer', 'heterosexual'))

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
    demo2_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + '_processed' + '.csv', index=False)
