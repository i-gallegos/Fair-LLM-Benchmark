#!/bin/bash

set -e

gpuid=${gpuid:-0}
d=${d}

if [ "$1" == "-h" ]; then
  echo "Generate predictions using all models and all datasets we have."
  echo "   --gpuid       The GPU device index to use, default to 0"
  echo "   --d           A list of dataset types, separated by comma, must be in {gender, country, religion, ethnicity}"
  echo "   -h            Print the help message and exit"
  exit 0
fi


while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi

  shift
done

## mased LMs
./scripts/generate_lm_predictions.sh --m roberta-base --m_name robertabase_lm --d $d --gpuid $gpuid
./scripts/generate_lm_predictions.sh --m roberta-large --m_name robertalarge_lm --d $d --gpuid $gpuid
./scripts/generate_lm_predictions.sh --m distilbert-base-uncased --m_name distilbert_lm --d $d --gpuid $gpuid
./scripts/generate_lm_predictions.sh --m bert-base-uncased --m_name bertbase_lm --d $d --gpuid $gpuid
./scripts/generate_lm_predictions.sh --m bert-large-uncased-whole-word-masking --m_name bertlarge_lm --d $d --gpuid $gpuid

## SQuAD models
./scripts/generate_qa_predictions_hf.sh --m tli8hf/unqover-roberta-base-squad --m_name robertabase --d $d --gpuid $gpuid
./scripts/generate_qa_predictions_hf.sh --m tli8hf/unqover-roberta-large-squad --m_name robertalarge --d $d --gpuid $gpuid
./scripts/generate_qa_predictions_hf.sh --m tli8hf/unqover-distilbert-base-uncased-squad --m_name distilbert --d $d --gpuid $gpuid
./scripts/generate_qa_predictions_hf.sh --m tli8hf/unqover-bert-base-uncased-squad --m_name bertbase --d $d --gpuid $gpuid
./scripts/generate_qa_predictions_hf.sh --m tli8hf/unqover-bert-large-uncased-squad --m_name bertlarge --d $d --gpuid $gpuid

# NewsQA models
#   here we use models trained on our own
./scripts/generate_qa_predictions_hf.sh --m tli8hf/unqover-roberta-base-newsqa --m_name robertabase --extra newsqa --d $d --gpuid $gpuid
./scripts/generate_qa_predictions_hf.sh --m tli8hf/unqover-roberta-large-newsqa --m_name robertalarge --extra newsqa --d $d --gpuid $gpuid
./scripts/generate_qa_predictions_hf.sh --m tli8hf/unqover-distilbert-base-uncased-newsqa --m_name distilbert --extra newsqa --d $d --gpuid $gpuid
./scripts/generate_qa_predictions_hf.sh --m tli8hf/unqover-bert-base-uncased-newsqa --m_name bertbase --extra newsqa --d $d --gpuid $gpuid
./scripts/generate_qa_predictions_hf.sh --m tli8hf/unqover-bert-large-uncased-newsqa --m_name bertlarge --extra newsqa --d $d --gpuid $gpuid
