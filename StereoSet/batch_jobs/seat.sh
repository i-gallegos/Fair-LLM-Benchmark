#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:15:00" ["albert-base-v2"]="00:10:00" ["roberta-base"]="00:10:00" ["gpt2"]="00:15:00"
declare -A time=(["bert-base-uncased"]="00:30:00" ["albert-base-v2"]="00:30:00" ["roberta-base"]="00:30:00" ["gpt2"]="00:30:00")


for model in ${models[@]}; do
    experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
        echo ${experiment_id}
        sbatch \
            --time ${time[${model_to_model_name_or_path[${model}]}]} \
            --partition=long \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/seat.py \
                --tests ${seat_tests} \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]}
    fi
done
