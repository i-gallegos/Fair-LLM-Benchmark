"""
This script performs Student t-test on the perplexity distribution of two sentences groups with contrasting targets
"""
import pandas as pd
import numpy as np
from scipy import stats
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelWithLMAndDebiasHead
import time
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import argparse
import torch
import math


def perplexity_score(sentence, m, t):
    """
    Finds perplexity score of a sentence based on model
    Parameters
    ----------
    sentence : str
    Given sentence
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer

    Returns
    -------
    Perplexity score
    """
    with torch.no_grad():
        m.eval()
        tokenize_input = t.tokenize(sentence)
        tensor_input = torch.tensor([t.convert_tokens_to_ids(tokenize_input)])
        loss = m(tensor_input, labels=tensor_input)
        return math.exp(loss[0])


def model_perplexity(sentences, m, t):
    """
    Finds model perplexity based on average model loss over all sentences
    Parameters
    ----------
    sentences : list
    sentence set
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer

    Returns
    -------
    Model perplexity score
    """
    total_loss = 0
    for sent in sentences:
        with torch.no_grad():
            m.eval()
            tokenize_input = t.tokenize(sent)
            tensor_input = torch.tensor([t.convert_tokens_to_ids(tokenize_input)])
            loss = m(tensor_input, labels=tensor_input)
            total_loss += loss[0]
    return math.exp(total_loss/len(sentences))


def get_perplexity_list(df, m, t):
    """
        Gets perplexities of all sentences in a DataFrame based on given model
        Parameters
        ----------
        df : pd.DataFrame
        DataFrame with Reddit comments
        m : model
        Pre-trained language model
        t : tokenizer
        Pre-trained tokenizer for the given model

        Returns
        -------
        List of sentence perplexities
    """
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            perplexity = perplexity_score(row['comments_processed'], m, t)
        except Exception as ex:
            print(ex.__repr__())
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_perplexity_list_test(df, m, t, dem):
    """
    Gets perplexities of all sentences in a DataFrame(contains 2 columns of contrasting sentences) based on given model
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments in 2 columns
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model

    Returns
    -------
    List of sentence perplexities
    """
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            if dem == 'black':
                perplexity = perplexity_score(row['comments_1'], m, t)
            else:
                perplexity = perplexity_score(row['comments_2'], m, t)
        except Exception as ex:
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_model_perplexity(df, m, t):
    """
    Finds model perplexity based on average model loss over all sentences
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model

    Returns
    -------
    Model perplexity
    """
    model_perp = model_perplexity(df['comments_processed'], m, t)
    return model_perp


def find_outliers(data):
    """
    Find outliers in a given data distribution
    Parameters
    ----------
    data : list
    List of sentence perplexities

    Returns
    -------
    List of outliers
    """
    anomalies = []

    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    std_3 = random_data_std * 3

    lower_limit = random_data_mean - std_3
    upper_limit = random_data_mean + std_3

    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies


start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='Path containing data files')
parser.add_argument('--log_path', help='Path containing log files')
parser.add_argument('--get_perp', help='If True, Calculate perplexity, else gets saved perplexity')
parser.add_argument('--save_perp', help='Save calculated perplexities', default='no')
parser.add_argument('--demo', help='Demographic')
parser.add_argument('--demo1', help='Demographic group 1')
parser.add_argument('--demo2', help='Demographic group 2')
parser.add_argument('--input_file_1', help='Name of input data file of demo1')
parser.add_argument('--input_file_2', help='Name of input data file of demo2')
parser.add_argument('--output_file_1', help='Name of output data file of demo1 with perplexities')
parser.add_argument('--output_file_2', help='Name of output data file of demo2 with perplexities')
parser.add_argument('--output_file_suffix', help='Suffix of output data perplexity file', default=None)
parser.add_argument('--model_path', help='Path of model')
parser.add_argument('--debiasing_head', help='Type of debiasing head of de-biased model', default=None)
parser.add_argument('--model_name', help='Name of model to be evaluated', default=None)


args = parser.parse_args()

data_path = args.data_path
log_path = args.log_path

GET_PERPLEXITY = args.get_perp
SAVE_PERPLEXITY = args.save_perp

demo = args.demo
demo_1 = args.demo1
demo_2 = args.demo2
input_file_1 = args.input_file_1
input_file_2 = args.input_file_2
output_file_1 = args.output_file_1
output_file_2 = args.output_file_2

debiasing_head = args.debiasing_head
pretrained_model = args.model_path
model_name = args.model_name

pd.set_option('max_colwidth', 600)
pd.options.display.max_columns = 10
logging.basicConfig(filename=log_path + 'measure_bias_' + demo + '.log', filemode='a', level=logging.DEBUG,
                    format='%(asctime)s %(message)s')

logging.info('Evaluating bias for model: {}'.format(model_name))

if GET_PERPLEXITY == 'yes':

    logging.info('Calculating perplexity')
    race_df = pd.read_csv(data_path + demo + '/' + input_file_1)
    race_df_2 = pd.read_csv(data_path + demo + '/' + input_file_2)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if debiasing_head:
        logging.info('Loading debiased model..')
        model = AutoModelWithLMAndDebiasHead.from_pretrained(pretrained_model, debiasing_head=debiasing_head)
    else:
        if 'bert' in args.model_path.__repr__().lower():
            logging.info('in bert')
            model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
        elif 'gpt' in pretrained_model.__repr__().lower():
            logging.info('in gpt')
            model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        else:
            logging.info('in CLM model by default')
            model = AutoModelForCausalLM.from_pretrained(pretrained_model)

    race_1_perplexity = get_perplexity_list(race_df, model, tokenizer)
    logging.info('Done with demo1 perplexity in {} on set'.format((time.time() - start)/60))
    race_2_perplexity = get_perplexity_list(race_df_2, model, tokenizer)
    logging.info('Done with demo2 perplexity in {} on set'.format((time.time() - start)/60))

    race_df['perplexity'] = race_1_perplexity
    race_df_2['perplexity'] = race_2_perplexity

    if SAVE_PERPLEXITY == 'yes':
        race_df.to_csv(data_path + demo + '/' + output_file_1)
        race_df_2.to_csv(data_path + demo + '/' + output_file_2)
else:
    logging.info('Getting saved perplexity')
    race_df = pd.read_csv(data_path + demo + '/' + output_file_1)
    race_df_2 = pd.read_csv(data_path + demo + '/' + output_file_2)
    race_1_perplexity = race_df['perplexity']
    race_2_perplexity = race_df_2['perplexity']

logging.info('Mean and variance of unfiltered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(race_1_perplexity), np.var(race_1_perplexity)))
logging.info('Mean and variance of unfiltered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(race_2_perplexity), np.var(race_2_perplexity)))

assert len(race_1_perplexity) == len(race_2_perplexity)
logging.info('Test samples: {}'.format(len(race_1_perplexity)))

t_value, p_value = stats.ttest_ind(race_1_perplexity, race_2_perplexity, equal_var=False)

logging.info('Complete testset - Significance of mean perplexities difference -  t-value {} and p-value {}'.format(t_value, p_value))

logging.info('Finding Perplexity mean significance after outlier removal..')
demo1_out = find_outliers(np.array(race_1_perplexity))
demo2_out = find_outliers(np.array(race_2_perplexity))

logging.debug('Outliers in demo1 and demo2 respectively are: {}, {}'.format(demo1_out, demo2_out))
logging.info('Outliers in Demographic 1: {}'.format(len(demo1_out)))
logging.info('Outliers in Demographic 2: {}'.format(len(demo2_out)))

demo1_in = [d1 for d1 in race_1_perplexity if d1 not in demo1_out]
demo2_in = [d2 for d2 in race_2_perplexity if d2 not in demo2_out]

t_inliers, p_inliers = stats.ttest_ind(demo1_in, demo2_in, equal_var=False) # do not assume equal variance

logging.info('Mean and variance of filtered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(demo1_in), np.var(demo1_in)))
logging.info('Mean and variance of filtered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(demo2_in), np.var(demo2_in)))

logging.info('Inliers in Demographic 1: {}'.format(len(demo1_in)))
logging.info('Inliers in Demographic 2: {}'.format(len(demo2_in)))

logging.info('Outlier removed testset - Significance of mean perplexities difference - t-value {} and p-value {}'.format(t_inliers, p_inliers))

logging.info('Total time taken {}'.format((time.time() - start)/60))
