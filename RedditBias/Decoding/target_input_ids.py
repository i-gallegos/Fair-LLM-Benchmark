"""
This script tokenizes the target group terms/phrases
"""
import time
from transformers import AutoModelWithLMHead, AutoTokenizer


start = time.time()

data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'

pretrained_model = 'microsoft/DialoGPT-small' # 'gpt2' # 'roberta-base' # 'bert-base-uncased' # 'minimaxir/reddit' # 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelWithLMHead.from_pretrained(pretrained_model)

with open(data_path + 'text_files/race/race_bias_manual_train.txt') as f:
    lines = [line.rstrip() for line in f]

for sent in lines:
    print(sent)
    input_ids = tokenizer(sent, add_special_tokens=True, truncation=True, max_length=32)
    tokens = tokenizer.tokenize(sent)
    print(input_ids)
    print(tokens)

sent = [' blacks whites', ' africans americans', ' african american', ' black white', ' negro caucasian',
        ' negroes caucasians', 'black is', 'white is', 'blacks are', 'whites are', 'african is', 'american is',
        'africans are', 'americans are', 'negro is', 'caucasian is', 'negroes are', 'caucasians are', ' islam is',
        ' christianity is']

for s in sent:
    print(s)
    input_ids = tokenizer(s, add_special_tokens=True, truncation=True, max_length=32)
    tokens = tokenizer.tokenize(s)
    print(input_ids)
    print(tokens)