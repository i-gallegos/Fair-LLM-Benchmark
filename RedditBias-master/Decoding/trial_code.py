import torch
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
'''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2', return_dict=True)

# Add a [CLS] to the vocabulary (we should train it also!)
num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
encoded_choices = [tokenizer.encode(s) for s in choices]
cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
print(cls_token_location)
print(encoded_choices)

input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
print(input_ids)
mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

# print(mc_token_ids)
print(input_ids)
outputs = model(input_ids, labels=input_ids, mc_token_ids=mc_token_ids, mc_labels=torch.tensor(1).unsqueeze(0))

lm_logits = outputs.logits
mc_logits = outputs.mc_logits

# print(lm_logits)
print(mc_logits)
print(outputs.loss)
print(outputs.mc_loss)

from transformers import GPT2Model, GPT2Config

# Initializing a GPT2 configuration
configuration = GPT2Config(summary_activation='tanh')
# print(configuration)
# Initializing a model from the configuration
model = GPT2Model(configuration)

# Accessing the model configuration
configuration = model.config
print(configuration)

from transformers import GPT2DoubleHeadsModelCustomLoss
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small', return_dict=True)

# special_tokens_dict = {'cls_token': '[CLS]'}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

model.resize_token_embeddings(len(tokenizer))

with torch.no_grad():
    model.eval()
    inputs = tokenizer("the jew is so greedy the christian is greedy", return_tensors="pt")
    print(inputs)
    vocab = tokenizer.get_vocab()
    # print(vocab)

    tokens = tokenizer.tokenize("the jew is so greedy the christian is greedy")
    print(tokens)

    j_id = tokenizer.convert_tokens_to_ids('jew')
    j_token = tokenizer.convert_ids_to_tokens(j_id)
    j_string = tokenizer.convert_tokens_to_string(j_token)
    print(j_id, j_token, j_string)

    c_id = tokenizer.convert_tokens_to_ids('Christian')
    c_token = tokenizer.convert_ids_to_tokens(c_id)
    c_string = tokenizer.convert_tokens_to_string(c_token)
    print(c_id, c_token, c_string)

    # inputs_2 = tokenizer.encode("[CLS] Hello, my dog is cute hello, my cat is cute")
    #
    # print(inputs_2)
    # print(tokenizer.cls_token_id)
    # print(inputs_2.index(tokenizer.cls_token_id))
    # cls_location = inputs_2.index(tokenizer.cls_token_id)


    # inputs_uns = torch.tensor(inputs['input_ids']).unsqueeze(0)
    # outputs = model(inputs_uns, labels=inputs_uns, output_hidden_states=True)
    input_embed = model.get_input_embeddings()
    # print(input_embed(torch.LongTensor([13624, 20298, 4302, 9316, 13624])))
    input_embed = input_embed(torch.LongTensor([13624, 20298, 4302, 9316, 16659]))

    outputs = model(**inputs, labels=inputs['input_ids'], output_hidden_states=True)

    last_hidden_states = outputs.hidden_states
    # print(np.shape(last_hidden_states))

    # print(last_hidden_states[0])
    print((last_hidden_states[0]).shape)

    # print(last_hidden_states[-1][0])
    print((last_hidden_states[-1][0]).shape)
    # print(last_hidden_states[-1][0][0])
    print(len(last_hidden_states[-1][0][0]))

    m_h = torch.mean(last_hidden_states[-1][0], 0)
    # print(m_h)
    print(m_h.shape)

    loss = outputs.loss
    print(loss)

    l = (last_hidden_states[-1][0][0]).tolist()
    print(l)


standard = StandardScaler()
x = input_embed.detach().cpu().numpy()
x_std = standard.fit_transform(x)
plt.figure()


tsne = TSNE(n_components=2, random_state=0)  # n_components means you mean to plot your dimensional data to 2D
x_test_2d = tsne.fit_transform(x_std)

print()

markers = ('s', 'd', 'o', '^', 'v', '8', 's', 'p', "_", '2')
color_map = {0: 'red', 1: 'blue', 2: 'lightgreen', 3: 'purple', 4: 'cyan', 5: 'black', 6: 'yellow', 7: 'magenta',
         8: 'plum', 9: 'yellowgreen'}

plt.scatter(x=x_test_2d[:, 0], y=x_test_2d[:, 1])
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()