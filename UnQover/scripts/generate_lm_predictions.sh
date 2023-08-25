#!/bin/bash

set -e

m=${m}
m_name=${m_name}
d=${d}
gpuid=${gpuid:-0}

if [ "$1" == "-h" ]; then
  echo "Generate predictions from masked LMs from HuggingFace's transformers."
  echo "   --m           Name of the pre-trained transformer model"
  echo "   --m_name      A brief name of the masked LM, used to compose output path, must be in {robertabase, robertalarge, bertbase, bertlarge, distilbert}"
  echo "   --d           A list of dataset types, separated by comma, must be in {gender, country, religion, ethnicity}"
  echo "   --gpuid       The GPU device index to use, default to 0"
  echo "   -h           Print the help message and exit"
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

echo "======================================="
echo "        PREDICTING QUESTIONS"
echo "======================================="


d=$(echo $d | tr "," "\n")
echo ">> Datasets to process: "$d

# e.g. for robertabase, the family is roberta
M_FAMILY=$(echo ${m_name//base})
case "$m_name" in
    "robertabase_lm" | "robertalarge_lm")
        M_FAMILY="roberta"
        ;;
    "bertbase_lm" | "bertlarge_lm" | "distilbert_lm")
        M_FAMILY="bert"
        ;;
    *)
        echo "Can not handle model name: ${m_name}"
        exit 1
        ;;
esac

for di in $d
do
    echo "======================================="
    echo ">> Running masked LM (HuggingFace) "${m}" on "${di}" data"
    echo ">> Will dump predictions to ./data/"${m_name}_${di}.output.json

    TYPE=slot_act_map
    case "${di}" in
        "gender")
            SUBJ=mixed_gender_${M_FAMILY}
            SLOT=gender_noact_lm
            ACT=occupation_rev1
            USE_HESHE=1
           ;;
        "country" | "religion" | "ethnicity")
            SUBJ=${di}_${M_FAMILY}
            SLOT=${di}_noact_lm
            ACT=biased_${di}
            USE_HESHE=0
            ;;
        *)
            echo "Can not handle subject class: ${di}"
            exit 1
            ;;
    esac

    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    echo ">> Input file "${FILE}.source.json

    python3 -u -m lm.predict --gpuid $gpuid --transformer_type $m --use_he_she $USE_HESHE \
      --input ${FILE}.source.json --output ./data/${m_name}_${di}.output.json

done


exit 0