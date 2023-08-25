#/bin/bash

export ENGINEDIR=$1 # this is the directory where your data and model are
./NMTScripts/train_fairseq_trans_bpe.sh

