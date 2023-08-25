# File name: main.py
# Description:
# Author: Marion Bartl
# Date: 18-5-20


import argparse
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from nltk import sent_tokenize
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup

from bias_utils.utils import model_evaluation, mask_tokens, input_pipeline, format_time, statistics


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', help='provide language, either EN or DE', required=True)
    parser.add_argument('--eval', help='.tsv file with sentences for bias evaluation (BEC-Pro or transformed EEC)', required=True)
    parser.add_argument('--tune', help='.tsv file with sentences for fine-tuning (GAP flipped)', required=False)
    parser.add_argument('--out', help='output directory + filename', required=True)
    parser.add_argument('--model', help='which BERT model to use', required=False)
    parser.add_argument('--batch', help='fix batch-size for fine-tuning', required=False, default=1)
    parser.add_argument('--seed', required=False, default=42)
    args = parser.parse_args()
    return args


def fine_tune(model, dataloader, epochs, tokenizer, device):
    model.to(device)
    model.train()

    # ##### NEXT part + comments from tutorial:
    # https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP#scrollTo=oCYZa1lQ8Jn8&forceEdit=true
    # &sandboxMode=true
    # Note: AdamW is a class from the huggingface transformers library (as opposed to pytorch) I
    # believe the 'W' stands for 'Weight Decay fix'
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8)  # args.adam_epsilon  - default is 1e-8.

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    for epoch_i in range(0, epochs):
        print('')
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        for step, batch in enumerate(dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

            # mask inputs so the model can actually learn something
            b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            b_input_mask = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            masked_lm_labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the 'exploding gradients' problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # torch.exp: Returns a new tensor with the exponential of the elements of the input tensor.
        # perplexity = torch.exp(torch.tensor(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print('')
        print('  Average training loss: {0:.2f}'.format(avg_train_loss))
        print('  Training epoch took: {:}'.format(format_time(time.time() - t0)))

    print('Fine-tuning complete!')

    return model


if __name__ == '__main__':

    args = parse_arguments()

    # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # determine which pre-trained BERT model to use:
    # if none is provided, use the provided default case for the respective language
    if args.model is None:
        if args.lang == 'EN':
            pretrained_model = 'bert-base-uncased'
        elif args.lang == 'DE':
            pretrained_model = 'bert-base-german-dbmdz-cased'
        else:
            raise ValueError('language could not be understood. Use EN or DE.')
    else:
        pretrained_model = args.model

    print('-- Prepare evaluation data --')
    # import the evaluation data; data should be a tab-separated dataframe
    eval_data = pd.read_csv(args.eval, sep='\t')

    # eval_data = eval_data[:100]

    print('-- Import BERT model --')
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    # set up the model
    model = BertForMaskedLM.from_pretrained(pretrained_model,
                                            output_attentions=False,
                                            output_hidden_states=False)

    print('-- Calculate associations before fine-tuning --')
    st = time.time()
    # calculate associations before fine-tuning
    pre_associations = model_evaluation(eval_data, tokenizer, model, device)

    et = time.time()
    print('Calculation took {0:.2f} minutes'.format((et - st) / 60))
    # Add the associations to dataframe
    eval_data = eval_data.assign(Pre_Assoc=pre_associations)

    print('-- Import fine-tuning data --')
    if args.tune:
        if 'gap' in args.tune:
            tune_corpus = pd.read_csv(args.tune, sep='\t')
            tune_data = []
            for text in tune_corpus.Text:
                tune_data += sent_tokenize(text)
        else:
            raise ValueError('Can\'t deal with other corpora besides GAP yet.')

        # make able to handle
        # tune_data = tune_data[:5]

        # as max_len get the smallest power of 2 greater or equal to the max sentence length
        max_len_tune = max([len(sent.split()) for sent in tune_data])
        pos = math.ceil(math.log2(max_len_tune))
        max_len_tune = int(math.pow(2, pos))
        print('Max len tuning: {}'.format(max_len_tune))

        # get tokens and attentions tensor for fine-tuning data
        tune_tokens, tune_attentions = input_pipeline(tune_data, tokenizer, max_len_tune)
        assert tune_tokens.shape == tune_attentions.shape

        # set up Dataloader
        batch_size = int(args.batch)
        train_data = TensorDataset(tune_tokens, tune_attentions)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        print('-- Set up model fine-tuning --')
        epochs = 3
        model = fine_tune(model, train_dataloader, epochs, tokenizer, device)

        print('-- Calculate associations after fine-tuning --')
        # calculate associations after fine-tuning
        post_associations = model_evaluation(eval_data, tokenizer, model, device)

        # add associations to dataframe
        eval_data = eval_data.assign(Post_Assoc=post_associations)

    else:
        print('No Fine-tuning today.')

    # save df+associations in out-file (to be processed in R)
    eval_data.to_csv(args.out + '_' + args.lang + '.csv', sep='\t', encoding='utf-8', index=False)

    if 'Prof_Gender' in eval_data.columns:
        # divide by gender of person term
        eval_m = eval_data.loc[eval_data.Prof_Gender == 'male']
        eval_f = eval_data.loc[eval_data.Prof_Gender == 'female']

        print('-- Statistics Before --')
        statistics(eval_f.Pre_Assoc, eval_m.Pre_Assoc)
        if args.tune:
            print('-- Statistics After --')
            statistics(eval_f.Post_Assoc, eval_m.Post_Assoc)
        print('End code.')
    else:
        print('Statistics cannot be printed, code ends here.')
