"""
This script tokenizes the attribute words/phrases
"""
import re
from transformers import AutoModelWithLMHead, AutoTokenizer

# convert attribute words into tokens so that can be used for Cosine dist and Hard debiasing
data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'orientation' #'race' # 'gender' # 'religion2' # 'religion1'

pretrained_model = 'microsoft/DialoGPT-small' # 'gpt2' # 'roberta-base' # 'bert-base-uncased' # 'minimaxir/reddit' # 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelWithLMHead.from_pretrained(pretrained_model)

if demo == 'orientation':
    with open(data_path + demo + '/' + 'orientation_lgbtq.txt') as f:
        attributes = [re.sub('[*"]', '', line.split('\n')[0]) for line in f]

    with open(data_path + demo + '/' + 'orientation_lgbtq_not_used_annotation.txt') as f:
        ignore_attr = [re.sub('[*"]', '', line.split('\n')[0]) for line in f]

    extra_attr = ['depressed', 'depressing', 'indecisive', 'criminals', 'perverted', 'perverts', 'confused']
    extra_ignore_attr = ['depress', 'indecisi', 'criminal', 'pervert', 'confuse']
elif demo == 'religion1':
    with open(data_path + demo + '/' + 'religion1_jews.txt') as f:
        attributes = [re.sub('[*"]', '', line.split('\n')[0]) for line in f]
    ignore_attr = []
    extra_attr = ['nerdy', 'circumcise', 'hook', 'hooked', 'beady', 'lending', 'shady', 'immigrants']
    extra_ignore_attr = ['meek', 'violin', 'hook-nose', 'hooked-nose', 'beady eyes', 'red-hair', 'money-lending',
                         'shady practices', 'talkative', 'overfeed', 'caretak', 'guilt-inducement', 'spoiled brats',
                         'sexually repressive', 'long beard', 'philanthropist', 'jewface']
elif demo == 'religion2':
    with open(data_path + demo + '/' + 'religion2_muslims.txt') as f:
        attributes = [re.sub('[*"]', '', line.split('\n')[0]) for line in f]
    ignore_attr = []
    extra_attr = ['thugs', 'judaism', 'christianity', 'jewish']
    extra_ignore_attr = ['lazy sheik', 'oil sheik', 'belly dancers', 'vielded', 'viel', 'power-hungry',
                         'wealthy oilmen', 'harem maiden', 'headscarves', 'charlatan']
elif demo == 'gender':
    with open(data_path + demo + '/' + 'gender_female.txt') as f:
        attributes = [re.sub('[*"]', '', line.split('\n')[0]) for line in f]
    ignore_attr = ['maids']
    extra_attr = []
    extra_ignore_attr = []
elif demo == 'race':
    with open(data_path + demo + '/' + 'race_black.txt') as f:
        attributes = [re.sub('[*"]', '', line.split('\n')[0]) for line in f]
    ignore_attr = []
    extra_attr = []
    extra_ignore_attr = []
else:
    raise ValueError('Please specify valid demographic - religion1, religion2, orientation or gender')

ignore_attr.extend(extra_ignore_attr)
attributes.extend(extra_attr)

print(attributes)
tok_ids_list = []
for attr in attributes:
    print(attr)
    if attr not in ignore_attr:
        attr = ' ' + attr
        ids = tokenizer(attr, add_special_tokens=True, truncation=True, max_length=32)
        tokens = tokenizer.tokenize(attr)
        tok_ids_list.append(ids['input_ids'])
        print(ids['input_ids'])
        print(tokens)

print(len(tok_ids_list))
print(tok_ids_list)

# manual attr selection
# aggressive - Ġaggressive 8361, pushy 'Ġpush', 'y' 4574 88, immigrant Ġimmigrants Ġimmigrant Ġimmigration 7971 15773 6272,
# dominating Ġdominating 30651, overbearing 'Ġover', 'bearing' 625, 28655, selfish Ġselfish 20363, spoiled Ġspoiled 34781,
# insatiable 'Ġins', 'at', 'iable' 1035, 265, 3379, long noses Ġlong', 'Ġnoses' 890, 41760, overprotective 'Ġover', 'prot', 'ective',
# evil Ġevil 6181, devil Ġdevil 17118, satan 'Ġsat', 'an' 3332, 272, inferior Ġinferior 18536, manipulative Ġmanipulative 47859,
# manipulating Ġmanipulating 29349, cannibal Ġcannibal 39904, cannibals, effeminate 'Ġeff', 'em', 'inate' 914, 368, 4559,
# conspiring Ġconspiring 50186, greed Ġgreed 25474, greedy Ġgreedy 31828, nerd Ġnerd 34712, nerds Ġnerds 47088
# circumcision Ġcircumcision 25513, circumcise Ġcircumcised 45522, materialistic, hook Ġhook 8011, hooked Ġhooked 23373,
# beady eyes, large nose, red hair 'Ġred', 'Ġhair' 2266, 4190, frugal 'Ġf', 'rug', 'al', lust Ġlust 22279,

# left out attributes muslim - hijackers, lazy sheik, oil sheik, viel, vielded, deport, detain, thieves, charlatan,
# power-hungry, beard*, "wealthy oilmen", "harem maiden*", "suicide bomb*", headscarves