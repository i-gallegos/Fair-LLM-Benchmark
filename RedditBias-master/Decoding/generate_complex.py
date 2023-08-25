"""
This script generates model responses based on given context using Sampling or beam search
"""
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import pandas as pd
import math
import time
import torch

torch.manual_seed(0)
start = time.time()

data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'

pretrained_model = 'microsoft/DialoGPT-small' # 'minimaxir/reddit' # 'gpt2' # 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelWithLMHead.from_pretrained(pretrained_model)

input_sentence = pd.read_csv(data_path + 'complex_conv.csv')
print('Before dropping NaN {}'.format(input_sentence.shape))

input_sentence = input_sentence.dropna(axis=0, how='all')

print('After dropping NaN {}'.format(input_sentence.shape))

output_sentence = pd.DataFrame()

demography = ['man', 'woman', 'black person', 'white person', 'gay person', 'straight person']

for i, sent in input_sentence.iterrows():
    # input_context = 'Women are supposed to be '
    # print(sent)
    if not pd.isnull(sent['Initial sentence']):
        for idx, d in enumerate(demography):
            input_context = sent['Initial sentence']
            input_context = input_context.replace('XYZ', d)
            # print('Replaced sentence')
            # print(input_context)
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context

            # outputs = model.generate(input_ids=input_ids, num_beams=8, num_return_sequences=6, temperature=1.2)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            outputs = model.generate(input_ids=input_ids, do_sample=True, top_k=50, temperature=0.7, top_p=0.95,
                                     max_length=50, num_return_sequences=6, early_stopping=True,
                                     pad_token_id=tokenizer.eos_token_id)

            for j in range(6): # 5 output sequences were generated
                # print('Generated {}: {}'.format(j, tokenizer.decode(outputs[j], skip_special_tokens=True)))
                label = 'Response_' + str(j)
                output_sentence.at[i*len(demography)+idx, label] = tokenizer.decode(outputs[j], skip_special_tokens=True)
            # print(input_sentence.head())
            output_sentence.at[i*len(demography)+idx, 'Initial_sentence'] = input_context


file = pretrained_model.split('/')[1] if '/' in pretrained_model else pretrained_model
output_sentence.to_csv(data_path + 'dialog_response_' + file + '_complex.csv')
print('Time taken to run code %s min', (time.time() - start)/60)
