#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:20:00" ["albert-base-v2"]="00:20:00" ["roberta-base"]="00:20:00" ["gpt2"]="00:20:00"
declare -A time=(["bert-base-uncased"]="01:00:00" ["albert-base-v2"]="01:00:00" ["roberta-base"]="01:00:00" ["gpt2"]="01:00:00")

for model in ${models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="projection_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0"
        if [ ! -f "${persistent_dir}/results/projection_matrix/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            sbatch \
                --gres=gpu:32gb:1 \
                --mem 24G \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/inlp_projection_matrix.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_type ${bias_type} \
                    --n_classifiers ${model_to_n_classifiers[${model}]} \
                    --seed 0
        fi
    done
done
