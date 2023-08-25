import torch
import math
import re


def perplexity_score(sentence, model, tokenizer):
    """
    Finds perplexity score of a sentence based on model
    Parameters
    ----------
    sentence : str
    Given sentence
    model :
    Pre-trained language model
    tokenizer :
    Pre-trained tokenizer

    Returns
    -------
    Perplexity score
    """
    with torch.no_grad():
        model.eval()
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = model(tensor_input, labels=tensor_input)
        return math.exp(loss[0])


def model_perplexity(sentences, model, tokenizer):
    """
     Finds model perplexity based on average model loss over all sentences
     Parameters
     ----------
     sentences : list
     sentence set
     model :
     Pre-trained language model
     tokenizer :
     Pre-trained tokenizer

     Returns
     -------
     Model perplexity score
     """
    total_loss = 0
    for sent in sentences:
        with torch.no_grad():
            model.eval()
            tokenize_input = tokenizer.tokenize(sent)
            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
            loss = model(tensor_input, labels=tensor_input)
            total_loss += loss[0]
    return math.exp(total_loss/len(sentences))


def process_tweet(sent):
    """
    Pre-processes a given sentence
    Parameters
    ----------
    sent : str
    Given sentence

    Returns
    -------
    Processed sentence
    """
    sent = sent.encode("ascii", errors="ignore").decode() # check this output
    # print(sent)
    sent = re.sub('@[^\s]+', '', sent)
    sent = re.sub('https: / /t.co /[^\s]+', '', sent)
    sent = re.sub('http: / /t.co /[^\s]+', '', sent)
    sent = re.sub('http[^\s]+', '', sent)

    # split camel case combined words
    sent = re.sub('([A-Z][a-z]+)', r'\1', re.sub('([A-Z]+)', r' \1', sent))

    sent = sent.lower()

    # remove numbers
    sent = re.sub(' \d+', '', sent)
    # remove words with letter+number
    sent = re.sub('\w+\d+|\d+\w+', '', sent)

    # remove spaces
    sent = re.sub('[\s]+', ' ', sent)
    sent = re.sub(r'[^\w\s,.!?]', '', sent)

    # remove 2 or more repeated char
    sent = re.sub(r"(.)\1{2,}", r"\1", sent)
    sent = re.sub(" rt ", "", sent)

    sent = re.sub('- ', '', sent)
    sent = sent.strip()

    # print(sent)
    return sent
