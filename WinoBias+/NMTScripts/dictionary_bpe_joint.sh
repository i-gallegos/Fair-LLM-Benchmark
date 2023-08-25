#!/bin/sh
if [ -z $ENGINEDIR ]
then
    if [ ! -z "$1" ]
    then
        ENGINEDIR=$1
    else
        echo 'Specify or export ENGINEDIR'
        exit 1
    fi
fi

DATADIR=${ENGINEDIR}/data

NUMSYM=32000

cat $DATADIR/train.tok.src $DATADIR/train.tok.trg > $DATADIR/train.tok.src-trg
# train BPE
subword-nmt learn-bpe -s $NUMSYM < $DATADIR/train.tok.src-trg > $DATADIR/bpe.src-trg

# apply BPE
for FILE in 'train' 'test' 'dev'
do
    subword-nmt apply-bpe -c $DATADIR/bpe.src-trg < $DATADIR/$FILE.tok.src > $DATADIR/${FILE}.tc.bpe.src
    subword-nmt apply-bpe -c $DATADIR/bpe.src-trg < $DATADIR/$FILE.tok.trg > $DATADIR/${FILE}.tc.bpe.trg
done
