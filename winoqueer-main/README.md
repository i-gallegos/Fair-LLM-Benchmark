# WinoQueer

## Paper
Our paper, *WinoQueer: A Community-in-the-Loop Benchmark for Anti-LGBTQ+ Bias in Large Language Models* can be found on [arXiV](TODO add link)

## Benchmark
The winoqueer benchmark dataset is at: `data/winoqueer_final.csv.` 

Columns are: 
| Col. Name | Meaning |
|---|---|
| Gender\_ID\_X | bias target category |
| Gender\_ID\_Y | counterfactual target category |
| sent\_x | biased/offensive sentence |
| sent\_y | counterfactual sentence |

To run WinoQueer evaluation on your own language model, use `code/metric.py` for masked LMs and `code/metric\_autoregressive.py` for autoregressive LMs. 

Usage: `python metric(\_autoregressive).py --input\_file <path to winoqueer\_final.csv> --lm\_model\_path <path to model directory> --output\_file <path to CSV for detailed output> --summary\_file <path to file for summary output (optional)>`

Evaluation scripts are forked from CrowS-Pairs evaluation script: https://github.com/nyu-mll/crows-pairs and `metric_autoregressive.py` was modified for autoregressive models.

## Finetuning Data

`data/article_urls.csv`: metadata and URLs of news articles used to finetune models. Due to licensing requirments, we are not allowed to share the full text of the articles, but you can "rehydrate" the URLs using tools from Media Cloud: https://www.mediacloud.org/search-tool-guide.

Once rehydrated, segment sentences with `code/preproc/segment_articles.py`, then train a model with one of `code/finetune_*.py`.

`data/tweetIDs.csv.zip`: TweetIDs for Twitter data used to finetune models. TweetIDs must be "rehydrated" using the Twitter API before they can be used. TweetIDs are provided for non-commericial research purposes only. Provided as ZIP due to large file size.

## Finetuning Scripts

### Preprocessing

`code/preproc/segment_articles.py` : script for sentence segmenting news articles.

`code/TweetNormalizer.py`: Script for normalizing tweets. Called from finetune\_*.py, there is no need to call this as a separate preprocessing step. Fork of BERTweet tweet normalizer: https://github.com/VinAIResearch/BERTweet

### Training

`code/ds_config_general.json`: DeepSpeed configuration file.

`code/finetune_model.py`: Used to finetune all versions of BERT, RoBERTa, and ALBERT
Usage: `python finetune_model.py <path to model dir>  <path to training data> {'n' for news, 't' for twitter}`

`code/finetune_autoregressive.py`: Used to finetune all versions of GPT2 and BLOOM (requires DeepSpeed).

Usage: `deepspeed  --num_gpus=<desired number of GPUS> finetune_autoregressive.py <path to model dir>  <path to training data> {'n' for news, 't' for twitter} --deepspeed ds_config_general.json`
DeepSpeed defaults to port 29500. If you want to launch two deepspeed training runs on the same machine, use `--master_port=<something other than 29500>` to avoid a port conflict.

`code/finetune_opt_350m.py`: Used to finetune opt-350m.

Usage: `python finetune_opt_350m.py <path to model dir>  <path to training data> {'n' for news, 't' for twitter}`

## General

`code/requirements.txt`: versioning information for python packages. We used Python 3.9.12 with pip 22.0.4.
