#/bin/bash

export SRCLANG=$src;
export TRGLANG=$trg;

export ENGINEDIR=$1 # the directory where the model and the data is stored

./NMTScripts/dictionary_bpe_joint.sh

echo "Done"
