#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:15:00" ["albert-base-v2"]="00:15:00" ["roberta-base"]="00:15:00" ["gpt2"]="00:15:00"
declare -A time=(["bert-base-uncased"]="00:30:00" ["albert-base-v2"]="00:30:00" ["roberta-base"]="00:30:00" ["gpt2"]="00:30:00")


for model in ${inlp_masked_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
             echo ${experiment_id}
             sbatch \
                 --time ${time[${model_to_model_name_or_path[${model}]}]} \
                 --partition=long \
                 -J ${experiment_id} \
                 -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                 -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                 python_job.sh experiments/stereoset_debias.py \
                     --model ${model} \
                     --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                     --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0.pt" \
                     --bias_type ${bias_type}
        fi
    done
done


for model in ${cda_masked_lm_models[@]}; do
    for seed in 0 1 2; do
        for bias_type in ${bias_types[@]}; do
            experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-${seed}"
            if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                echo ${experiment_id}
                sbatch \
                    --time ${time[${model_to_model_name_or_path[${model}]}]} \
                     --partition=long \
                    -J ${experiment_id} \
                    -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                    -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                    python_job.sh experiments/stereoset_debias.py \
                        --model ${model} \
                        --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                        --load_path "${persistent_dir}/checkpoints/cda_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-${seed}" \
                        --bias_type ${bias_type} \
                        --seed ${seed}
            fi
        done
    done
done


for model in ${dropout_masked_lm_models[@]}; do
    for seed in 0 1 2; do
        experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_s-${seed}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --load_path "${persistent_dir}/checkpoints/dropout_c-${model_to_model_name_or_path[${model}]}_s-${seed}" \
                    --seed ${seed}
        fi
    done
done


for model in ${sentence_debias_masked_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                 --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}.pt" \
                    --bias_type ${bias_type}
        fi
    done
done


for model in ${self_debias_masked_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                 --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_type ${bias_type}
        fi
    done
done


for model in ${inlp_causal_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                 --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0.pt" \
                    --bias_type ${bias_type}
        fi
    done
done


for model in ${sentence_debias_causal_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}.pt" \
                    --bias_type ${bias_type}
        fi
    done
done


for model in ${cda_causal_lm_models[@]}; do
    for seed in 0 1 2; do
        for bias_type in ${bias_types[@]}; do
            experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-${seed}"
            if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                echo ${experiment_id}
                sbatch \
                    --time ${time[${model_to_model_name_or_path[${model}]}]} \
                    --partition=long \
                    -J ${experiment_id} \
                    -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                    -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                    python_job.sh experiments/stereoset_debias.py \
                        --model ${model} \
                        --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                        --load_path "${persistent_dir}/checkpoints/cda_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-${seed}" \
                        --bias_type ${bias_type} \
                        --seed ${seed}
            fi
        done
    done
done


for model in ${dropout_causal_lm_models[@]}; do
    for seed in 0 1 2; do
        experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_s-${seed}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --load_path "${persistent_dir}/checkpoints/dropout_c-${model_to_model_name_or_path[${model}]}_s-${seed}" \
                    --seed ${seed}
        fi
    done
done


for model in ${self_debias_causal_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --time ${time[${model_to_model_name_or_path[${model}]}]} \
                --partition=long \
                -J ${experiment_id} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_type ${bias_type}
        fi
    done
done
