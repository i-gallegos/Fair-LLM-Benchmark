#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["gender"]="01:00:00" ["race"]="00:10:00" ["religion"]="00:10:00"
declare -A time=(["gender"]="03:00:00" ["race"]="02:00:00" ["religion"]="02:00:00")


for model in ${models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="subspace_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/subspace/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${bias_type}]} \
                --mem 48G \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/sentence_debias_subspace.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_type ${bias_type}
        fi
    done
done
