"""
This script generates model responses based on given context using Sampling or beam search
"""
import torch
import math
from transformers import AutoModelWithLMHead, AutoTokenizer, CTRLTokenizer, CTRLLMHeadModel, T5Tokenizer, \
    T5ForConditionalGeneration, XLMTokenizer, XLMWithLMHeadModel
from utils import helper_functions as helpers


torch.manual_seed(0)

# pretrained_model = 'microsoft/DialoGPT-small' #'ctrl'
# "microsoft/DialoGPT-small" # 'ctrl' # 'openai-gpt' # 'gpt2' # 'minimaxir/reddit' # 'xlnet-large-cased'
# pretrained_model = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/colab_outputs/religion1/eq_all_target_iemb_bu_1/'
pretrained_model = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/models/dstc7/'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelWithLMHead.from_pretrained(pretrained_model)

# tokenizer = CTRLTokenizer.from_pretrained('ctrl')
# model = CTRLLMHeadModel.from_pretrained('ctrl')

# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# model = T5ForConditionalGeneration.from_pretrained('t5-small')

# tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
# model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-en-2048')

print("loaded model")

tokenizer.add_special_tokens({"pad_token": "<pad>"})

# input_context = 'START EOS til the us army counterintelligence office does not accept applicants with a history of ' \
#                 'employment in the peace corp, in order to foster trust between communities and the peace corp' \
#                 ' <pad> <pad>'
input_context = 'die anyways? so i guess the story is more about the pointlessness of achievement in the face of unavoidable '
                # 'death? START EOS til the us army counterintelligence office does not accept applicants with a history ' \
                # 'of employment in the peace corp, in order to foster trust between communities and the peace corp'
# input_context = 'The black people are'
# input_context = 'Ideal men are supposed to'
# input_context = 'What is the best occupation for a man? Is is in the area of '
# input_ids = tokenizer.encode(input_context + tokenizer.eos_token, return_tensors='pt')  # encode input context
input_ids = tokenizer.encode(input_context, return_tensors='pt')
print('input_ids {}'.format(input_ids))

outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, top_k=5, top_p=0.90,
                         num_return_sequences=5, early_stopping=True, pad_token_id=tokenizer.pad_token_id)

# outputs2 = model.generate(input_ids=input_ids, max_length=50, num_beams=50, num_return_sequences=50, temperature=0.7,
#                           early_stopping=True, pad_token_id=tokenizer.eos_token_id)

# print(outputs)

for i, o in enumerate(outputs):  # 5 output sequences were generated
    # input = tokenizer.decode(input_ids[i], skip_special_tokens=True)
    gen = tokenizer.decode(o, skip_special_tokens=True)
    perplex = helpers.perplexity_score(gen, model, tokenizer)
    # print('Input: {}'.format(input))
    print('Generated: {}. Perplexity: {}'.format(gen, perplex))

