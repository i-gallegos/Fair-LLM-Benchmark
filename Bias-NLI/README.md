# On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings (AAAI 2020)

This branch contains code for ELMo-based debiasing. For BERT-based debiasing, checkout the [bert_debias](https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings/tree/bert_debias) branch.


For citing our work:

@misc{dev2019measuring,
    title={On Measuring and Mitigating Biased Inferences of Word Embeddings},
    author={Sunipa Dev and Tao Li and Jeff Phillips and Vivek Srikumar},
    year={2019},
    eprint={1908.09369},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}


# Preprocessing

First have glove.840B.300d.txt located at ``./data/glove.840B.300d.txt``, and SNLI data txt files located at ``./data/nli_bias/``.
Then do preprocessing:
```
python3 preprocess.py --glove ./data/glove.840B.300d.txt --batch_size 48 --dir ./data/nli_bias/ --output snli
python3 get_pretrain_vecs.py --glove ./data/glove.840B.300d.txt --dict ./data/nli_bias/snli.word.dict \
	--output ./data/snli.glove
```

For unlabeled data (i.e. bias probing data), preprocess like this:
```
DATA_NAME=occupation_gender_templates.stitch
python3 preprocess_unlabeled.py --glove ./data/glove.840B.300d.txt --batch_size 48 --dir ./data/nli_bias/ \
	--sent1 ${DATA_NAME}.sent1.txt --sent2 ${DATA_NAME}.sent2.txt \
	--vocab ./data/nli_bias/snli.word.dict --vocab_all ./data/nli_bias/snli.allword.dict \
	--output ${DATA_NAME}
python3 get_pretrain_vecs.py --glove ./data/glove.840B.300d.txt --dict ./data/nli_bias/${DATA_NAME}.word.dict \
	--output ./data/${DATA_NAME}.glove

```

# Training

To train a baseline RNN model on SNLI, use the following:
```
MODEL=./models/baseline
python3 -u train.py --gpuid 0 --dir data/nli_bias/ \
	--train_data snli-train.hdf5 --val_data snli-val.hdf5 \
	--word_vecs snli.glove.hdf5 --dict snli.word.dict \
	--encoder rnn --use_elmo_post 0 \
	--save_file $MODEL | tee $MODEL.log.txt
```

# Evaluation

To evaluate a trained model on SNLI test set, use:
```
MODEL=./models/baseline
python3 -u eval.py --gpuid 0 --dir data/nli_bias/ \
	--data snli-test.hdf5 \
	--word_vecs snli.glove.hdf5 --dict snli.word.dict \
	--encoder rnn --use_elmo_post 0 \
	--load_file $MODEL
```


To evlauate on unlabeled data, use:
```
DATA_NAME=occupation_gender_templates.stitch
MODEL=./models/baseline
python3 -u predict_unlabeled.py --gpuid 0 --dir data/nli_bias/ \
	--data ${DATA_NAME}.hdf5 --res ${DATA_NAME}.sent1.txt,${DATA_NAME}.sent2.txt,${DATA_NAME}.x_pair.txt \
	--word_vecs ${DATA_NAME}.glove.hdf5 --dict ${DATA_NAME}.word.dict \
	--encoder rnn --use_elmo_post 0 \
	--pred_output models/${DATA_NAME}.pred.txt \
	--load_file $MODEL
```