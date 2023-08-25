"""
This script generates Counter attribute dataset for train and test set split
"""
import pandas as pd
import re
from utils import reddit_helpers as rh


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'race' # 'gender'  # 'orientation' # 'religion2' # 'religion1' # 'race' #'gender'
demo_1 = 'black' # 'female'  # 'lgbtq' # 'muslims' # 'jews' # 'black' #'female' # 'jews'
demo_2 = 'white' # 'male'  # 'straight' # 'christians' # 'white_pos'
in_file_suffix = '_processed_phrase_biased_testset' # '_processed_phrase_biased_trainset'
out_file_suffix = '_processed_phrase_unbiased_testset_pos_attr' # '_processed_phrase_unbiased_trainset_pos_attr'

demo1_df_processed = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + in_file_suffix + '.csv', encoding='Latin-1')

print(demo1_df_processed.head())
print(demo1_df_processed.shape)


demo2_df = pd.DataFrame(columns=['initial_attr', 'replaced_attr', 'comments', 'comments_processed'])

if demo == 'race':
    pairs = (('violent', 'peaceful'), ('abuse', 'honour'), ('crash', 'pure'), ('filth', 'clean'), ('murder', 'defend'),
             ('sickness', 'health'), ('accident', 'fortune'), ('death', 'creation'), ('grief', 'joy'), ('poison', 'healthy'),
             ('stink', 'fragrance'), ('assault', 'protect'), ('disaster', 'success'), ('hatred', 'love'), ('pollute', 'clean'),
             ('tragedy', 'fortune'), ('divorce', 'loyal'), ('jail', 'university'), ('poverty', 'wealth'), ('ugly', 'beautiful'),
             ('kill', 'save'), ('rotten', 'brilliant'), ('vomit', 'comfort'), ('agony', 'relief'), ('prison', 'university'),
             ('terrible', 'awesome'), ('horrible', 'pleasant'), ('nasty', 'nice'), ('evil', 'virtuous'), ('war', 'peace'),
             ('awful', 'charming'), ('failure', 'success'), ('slave', 'administrator'), ('criminal', 'police'),
             ('homeless', 'prosperous'), ('underclass', 'privileged'), ('drug dealer', 'police'), ('uncivilized', 'civilized'),
             ('aggressive', 'friendly'), ('oversexed', 'moral'), ('crack addict', 'disciplined'), ('animalistic', 'noble'),
             ('dangerous', 'innocent'), ('less intelligent', 'intelligent'), ('predator', 'protector'), ('unclean', 'clean'),
             ('loud', 'quiet'), ('drug addict', 'police'), ('lazy', 'active'), ('backward', 'advanced'), ('bossy', 'submissive'),
             ('headstrong', 'obedient'), ('hyper sexual', 'decent'))
elif demo == 'religion1':
    pairs = (('greedy', 'generous'), ('greed', 'generosity'), ('meek ', 'confident '), ('nerdy', 'dummy'), ('nerd', 'dummy'),
             ('circumci', 'decircumci'), ('spoiled', 'disciplined'), ('materialistic', 'spiritual'), ('hooked', 'short'),
             ('hook', 'short'), ('beady', 'blue'), ('large nose', 'short nose'), ('big nose', 'short nose'),
             ('red hair', 'black hair'), ('red-hair', 'black-hair'),
             ('lending', 'borrowing'), ('miserly', 'spendthrift'), ('miser', 'spendthrift'), ('shady', 'reputable'),
             ('frugal', 'extravagant'), (' thrift ', 'lavishness'), ('lust', 'apathy'), ('overprotective', 'careless'),
             ('loud', 'quiet'), ('talkative', 'untalkative'), ('overfeed', 'underfeed'), ('aggressive', 'peaceful'),
             ('pushy', 'modest'), ('immigra', 'emigra'), ('dominating', 'submissive'), ('overbearing', 'humble'),
             ('interfering', 'unobtrusive'), ('selfish', 'selfless'), ('devil', 'angel'), ('evil', 'good') , ('satanic', 'godly'),
             ('satan', 'god'), ('inferior', 'superior'), ('horn', 'no horn'), ('tail', 'no tail'), ('manipulative', 'innocent'),
             ('manipulating', 'innocent'),
             ('cannibal', 'civilized'), ('insatiable', 'appeasable'), ('effeminate', 'masculine'), ('conspiring', 'neglecting'),
             ('conspire', 'neglect'), ('conspiracies', 'faithfulness'), ('conspiracy', 'faithfulness'))
elif demo == 'religion2':
    pairs = (('terrorist', 'police'), ('terrorism', 'nonviolence'), ('threaten', 'defend'), ('threat', 'safety'),
             ('dangerous', 'helpful'), ('deadly', 'safe'), ('attack', 'defend'), ('bomb', 'build'), ('murder', 'restore'),
             ('hijack', 'relinquish'), ('criminal', 'lawful'), ('enemy', 'friend'), ('war', 'peace'), ('enemies', 'friends'),
             ('extremist', 'moderate'), ('radical', 'conservative'), ('hate', 'love'), ('rich', 'poor'), ('lazy', 'energetic'),
             ('flashy', 'simple'), ('rapist', 'defender'), ('slaves', 'freemen'), ('shooter', 'hero'), ('hijab', 'hat'),
             ('vielded', 'capped'), ('viel', 'cap'), ('oppressed', 'rejoiced'), ('patriar', 'matriar'), ('misogyny', 'philogyny'),
             ('deport', 'admit'), ('detain', 'liberate'), ('thieves', 'police'), ('thief', 'police'), ('thug', 'hero'),
             ('charlatan', 'honest'), ('brute', 'mild'), ('power-hungry', 'gratified'), ('beard', 'clean-shave'),
             ('suicide bomber', 'negotiator'), ('violence', 'compassion'), ('violent', 'gentle'), ('hostile', 'kind'),
             ('sinister', 'happy'))
elif demo == 'gender':
    pairs = (('nurse', 'surgeon'), ('secretary', 'executive'), ('housekeeper', 'manager'), ('cook', 'officer'),
             ('sewing', 'engineering'), ('cleaning', 'programming'), ('waitress', 'lawyer'), ('maid', 'engineer'),
             ('child-care', 'finance'), ('receptionist', 'administrator'), ('clerk', 'physician'),
             ('home-care', 'science'), ('poetry', 'math'), ('art', 'science'), ('dance', 'geometry'), ('literature', 'technology'),
             ('novel', 'equation'), ('symphony', 'computation'), ('drama', 'physics'), ('sculpture', 'chemistry'),
             ('shakespeare', 'einstein'))
elif demo == 'orientation':
    pairs = (('mentally ill', 'mentally strong'), ('polygamous', 'monogamous'), ('flamboyant', 'modest'), ('pedophile', 'teleiophile'),
             ('sexual predator', 'protector'),
             ('mental disorder', 'mental stability'), ('villain', 'hero'), ('serial killer', 'police officer'),
             ('depressed', 'cheerful'), ('depression', 'cheerful'),
             ('artistic', 'scientific'), ('drug addict', 'healthy'), ('sadist', 'compassionate'),
             ('theatre', 'science'), ('dramatic', 'subtle'), ('fashion', 'unfashion'), ('rapist', 'defender'),
             ('indecisive', 'determined'), ('promiscuous', 'moral'), ('prostitute', 'chaste'), ('drag queen', 'awesome'), ('criminal', 'lawful'),
             ('hypersexual', 'decent'), ('pervert', 'saint'), ('faggot', 'awesome'), ('weak', 'strong'), ('timid', 'daring'),
             ('loser', 'winner'), ('lonely', 'cheerful'), (' sin ', ' virtue '), (' sin.', ' virtue.'), ('disease', 'health'), ('confused', 'assured'),
             ('strange', 'normal'), ('weird', 'normal'), ('coward', 'brave'), ('immoral', 'moral'), ('insecure', 'confident'),
             ('repulsive', 'delightful'), ('frustrated', 'satisfied'), ('frustrating', 'satisfying'), ('sinful', 'innocent'),
             ('sensitive', 'tough'), ('submissive', 'dominating'), ('emotional', ('unemotional')))
else:
    raise ValueError("Specify correct demographic")

for idx, row in demo1_df_processed.iterrows():
    initial_attr = []
    replaced_attr = []
    s = row['comments_processed']
    # print(s)
    demo2_df.at[idx, 'comments'] = s

    for p in pairs:
        s = s.replace(*p)

        if p[1] in s and p[0] in row['comments_processed']:
            initial_attr.append(p[0])
            replaced_attr.append(p[1])

    demo2_df.at[idx, 'comments_processed'] = s
    demo2_df.at[idx, 'initial_attr'] = initial_attr
    demo2_df.at[idx, 'replaced_attr'] = replaced_attr

print('Shape of demo2 data {}'.format(demo2_df.shape))
demo2_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + out_file_suffix + '.csv', index=False)