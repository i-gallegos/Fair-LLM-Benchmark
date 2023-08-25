#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:15:00" ["albert-base-v2"]="00:15:00" ["roberta-base"]="00:10:00" ["gpt2"]="00:15:00"
declare -A time=(["bert-base-uncased"]="01:00:00" ["albert-base-v2"]="01:00:00" ["roberta-base"]="01:00:00" ["gpt2"]="01:00:00")


for model in ${sentence_debias_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/seat_debias.py \
                    --tests ${seat_tests} \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}.pt" \
                    --bias_type ${bias_type}
        fi
    done
done


for model in ${inlp_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/seat_debias.py \
                    --tests ${seat_tests} \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0.pt" \
                    --bias_type ${bias_type}
        fi
    done
done


for model in ${cda_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/seat_debias.py \
                    --tests ${seat_tests} \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --load_path "${persistent_dir}/checkpoints/cda_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0" \
                    --bias_type ${bias_type}
        fi
    done
done


for model in ${dropout_models[@]}; do
    experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
        echo ${experiment_id}
        sbatch \
            --time ${time[${model_to_model_name_or_path[${model}]}]} \
            --partition=long \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/seat_debias.py \
                --tests ${seat_tests} \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --load_path "${persistent_dir}/checkpoints/dropout_c-${model_to_model_name_or_path[${model}]}_s-0"
        fi
done
