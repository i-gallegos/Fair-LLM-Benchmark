#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:05:00" ["albert-base-v2"]="00:05:00" ["roberta-base"]="00:05:00" ["gpt2"]="00:05:00"
declare -A time=(["bert-base-uncased"]="00:10:00" ["albert-base-v2"]="00:10:00" ["roberta-base"]="00:10:00" ["gpt2"]="00:10:00")


for model in ${masked_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --gres=gpu:32gb:1 \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/crows.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_type ${bias_type}
        fi
    done
done


for model in ${causal_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --gres=gpu:32gb:1 \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/crows.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_type ${bias_type}
        fi
    done
done
