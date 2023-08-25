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

if [ ! -z $SKIPPREPROCESS ]
then
    SKIPPREPROCESS=1
else
    SKIPPREPROCESS=0
fi

MODELDIR=$ENGINEDIR/model
mkdir -p $MODELDIR

export CUDA_VISIBLE_DEVICES=0,1,2

echo "Prepare the data..."

SRC=src
TRG=trg

if [ "$SKIPPREPROCESS" -eq "0" ] || [ ! -d $ENGINEDIR/data/ready_to_train ];
then
    echo $SKIPPREPROCESS
    rm -r $ENGINEDIR/data/ready_to_train

    fairseq-preprocess --source-lang $SRC --target-lang $TRG \
	--trainpref $ENGINEDIR/data/train.tc.bpe --validpref $ENGINEDIR/data/dev.tc.bpe --testpref $ENGINEDIR/data/test.tc.bpe \
	--destdir $ENGINEDIR/data/ready_to_train
else
    echo "Skipping preprocessing"
fi

echo "Launching GPU monitoring"
GPUMONPID=$( nvidia-smi dmon -i 0,1,2 -s mpucv -d 1 -o TD > $MODELDIR/gpu.log & echo $! )

echo "Train..."
echo "Options derived from: http://opennmt.net/OpenNMT-py/FAQ.html "
fairseq-train $ENGINEDIR/data/ready_to_train \
    --arch transformer_iwslt_de_en \
    --lr 0.0005 --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --max-tokens 4096 \
    --dropout 0.3 \
    --update-freq=1 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --min-lr 1e-09 --warmup-updates 4000 \
    --save-dir $MODELDIR \
    --skip-invalid-size-inputs-valid-test \
    --patience 5

kill -s 9 $GPUMONPID
echo "Done."
