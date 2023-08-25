# File name: BERT_utils.py
# Description: additional functionality for my BERT scripts to keep them (relatively) small
# Author: Marion Bartl
# Date: 03/03/2020
import datetime
import math
from typing import Tuple

import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from scipy import stats
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import PreTrainedTokenizer


# taken from https://github.com/allenai/dont-stop-pretraining/blob/master/scripts/mlm_study.py
def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability=0.15) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the "
            "--mlm flag if you want to use this tokenizer. "
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults
    # to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def attention_mask_creator(input_ids):
    """Provide the attention mask list of lists: 0 only for [PAD] tokens (index 0)
    Returns torch tensor"""
    attention_masks = []
    for sent in input_ids:
        segments_ids = [int(t > 0) for t in sent]
        attention_masks.append(segments_ids)
    return torch.tensor(attention_masks)


def statistics(group1, group2):
    """take 2 groups of paired samples and compute either a paired samples t-test or
    a Wilcoxon signed rank test
    prints out a description of the two groups as well as the statistic and p value of the test"""
    assert len(group1) == len(group2), "The two groups do not have the same length"

    print('Group 1:')
    print(group1.describe())
    print('Group 2:')
    print(group2.describe())

    dif = group1.sub(group2, fill_value=0)

    SW_stat, SW_p = stats.shapiro(dif)
    print(SW_stat, SW_p)

    if SW_p >= 0.05:
        print('T-Test:')
        statistic, p = stats.ttest_rel(group1, group2)
    else:
        print('Wilcoxon Test:')
        statistic, p = stats.wilcoxon(group1, group2)

    print('Statistic: {}, p: {}'.format(statistic, p))

    effect_size = statistic / np.sqrt(len(group1))
    print('effect size r: {}'.format(effect_size))

    return


def tokenize_to_id(sentences, tokenizer):
    """Tokenize all of the sentences and map the tokens to their word IDs."""
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'

            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
    return input_ids


def input_pipeline(sequence, tokenizer, MAX_LEN):
    """function to tokenize, pad and create attention masks"""
    input_ids = tokenize_to_id(sequence, tokenizer)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long",
                              value=tokenizer.mask_token_id,
                              truncating="post", padding="post")
    input_ids = torch.tensor(input_ids)

    attention_masks = attention_mask_creator(input_ids)

    return input_ids, attention_masks


def prob_with_prior(pred_TM, pred_TAM, input_ids_TAM, original_ids, tokenizer):
    pred_TM = pred_TM.cpu()
    pred_TAM = pred_TAM.cpu()
    input_ids_TAM = input_ids_TAM.cpu()

    probs = []
    for doc_idx, id_list in enumerate(input_ids_TAM):
        # see where the masks were placed in this sentence
        mask_indices = np.where(id_list == tokenizer.mask_token_id)[0]
        # now get the probability of the target word:
        # first get id of target word
        target_id = original_ids[doc_idx][mask_indices[0]]
        # get its probability with unmasked profession
        target_prob = pred_TM[doc_idx][mask_indices[0]][target_id].item()
        # get its prior probability (masked profession)
        prior = pred_TAM[doc_idx][mask_indices[0]][target_id].item()

        probs.append(np.log(target_prob / prior))

    return probs


def model_evaluation(eval_df, tokenizer, model, device):
    """takes professional sentences as DF, a tokenizer & a BERTformaskedLM model
    and predicts the associations"""

    # as max_len get the smallest power of 2 greater or equal to the max sentence lenght
    max_len = max([len(sent.split()) for sent in eval_df.Sent_TM])
    pos = math.ceil(math.log2(max_len))
    max_len_eval = int(math.pow(2, pos))

    print('max_len evaluation: {}'.format(max_len_eval))

    # create BERT-ready inputs: target masked, target and attribute masked,
    # and the tokenized original inputs to recover the original target word
    eval_tokens_TM, eval_attentions_TM = input_pipeline(eval_df.Sent_TM,
                                                        tokenizer,
                                                        max_len_eval)
    eval_tokens_TAM, eval_attentions_TAM = input_pipeline(eval_df.Sent_TAM,
                                                          tokenizer,
                                                          max_len_eval)
    eval_tokens, _ = input_pipeline(eval_df.Sentence, tokenizer, max_len_eval)

    # check that lengths match before going further
    assert eval_tokens_TM.shape == eval_attentions_TM.shape
    assert eval_tokens_TAM.shape == eval_attentions_TAM.shape

    # make a Evaluation Dataloader
    eval_batch = 20
    eval_data = TensorDataset(eval_tokens_TM, eval_attentions_TM,
                              eval_tokens_TAM, eval_attentions_TAM,
                              eval_tokens)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, batch_size=eval_batch,
                                 sampler=eval_sampler)

    # put everything to GPU (if it is available)
    # eval_tokens_TM = eval_tokens_TM.to(device)
    # eval_attentions_TM = eval_attentions_TM.to(device)
    # eval_tokens_TAM = eval_tokens_TAM.to(device)
    # eval_attentions_TAM = eval_attentions_TAM.to(device)
    model.to(device)

    # put model in evaluation mode & start predicting
    model.eval()
    associations_all = []
    for step, batch in enumerate(eval_dataloader):
        b_input_TM = batch[0].to(device)
        b_att_TM = batch[1].to(device)
        b_input_TAM = batch[2].to(device)
        b_att_TAM = batch[3].to(device)

        with torch.no_grad():
            outputs_TM = model(b_input_TM,
                               attention_mask=b_att_TM)
            outputs_TAM = model(b_input_TAM,
                                attention_mask=b_att_TAM)
            predictions_TM = softmax(outputs_TM[0], dim=2)
            predictions_TAM = softmax(outputs_TAM[0], dim=2)

        assert predictions_TM.shape == predictions_TAM.shape

        # calculate associations
        associations = prob_with_prior(predictions_TM,
                                       predictions_TAM,
                                       b_input_TAM,
                                       batch[4],  # normal inputs
                                       tokenizer)

        associations_all += associations

    return associations_all


#### TAKEN FROM TUTORIAL

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Helper function for formatting elapsed times.
def format_time(elapsed):
    """ Takes a time in seconds and returns a string hh:mm:ss"""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

#### COPY END
