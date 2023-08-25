#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:10:00" ["albert-base-v2"]="00:10:00" ["roberta-base"]="00:10:00" ["gpt2"]="00:10:00"
declare -A time=(["bert-base-uncased"]="00:20:00" ["albert-base-v2"]="00:20:00" ["roberta-base"]="00:20:00" ["gpt2"]="00:20:00")


for model in ${masked_lm_models[@]}; do
    experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        sbatch \
            --time ${time[${model_to_model_name_or_path[${model}]}]} \
            --partition=long \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/stereoset.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]}
    fi
done


for model in ${causal_lm_models[@]}; do
    experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        sbatch \
            --time ${time[${model_to_model_name_or_path[${model}]}]} \
            --partition=long \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/stereoset.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]}
    fi
done
