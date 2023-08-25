#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"


experiment_id="glue_m-BertForSequenceClassification_c-bert-base-uncased"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "BertForSequenceClassification" \
                    --model_name_or_path "bert-base-uncased" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-AlbertForSequenceClassification_c-albert-base-v2"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "AlbertForSequenceClassification" \
                    --model_name_or_path "albert-base-v2" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-RobertaForSequenceClassification_c-bert-base-uncased"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "RobertaForSequenceClassification" \
                    --model_name_or_path "roberta-base" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-CDABertForSequenceClassification_c-bert-base-uncased_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "CDABertForSequenceClassification" \
                    --model_name_or_path "${persistent_dir}/checkpoints/cda_c-bert-base-uncased_t-gender_s-0" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-CDAAlbertForSequenceClassification_c-albert-base-v2_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "CDAAlbertForSequenceClassification" \
                    --model_name_or_path "${persistent_dir}/checkpoints/cda_c-albert-base-v2_t-gender_s-0" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-CDARobertaForSequenceClassification_c-roberta-base_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "CDARobertaForSequenceClassification" \
                    --model_name_or_path "${persistent_dir}/checkpoints/cda_c-roberta-base_t-gender_s-0" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-DropoutBertForSequenceClassification_c-bert-base-uncased"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "DropoutBertForSequenceClassification" \
                    --model_name_or_path "${persistent_dir}/checkpoints/dropout_c-bert-base-uncased_s-0" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-DropoutAlbertForSequenceClassification_c-albert-base-v2"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "DropoutAlbertForSequenceClassification" \
                    --model_name_or_path "${persistent_dir}/checkpoints/dropout_c-albert-base-v2_s-0" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-DropoutRobertaForSequenceClassification_c-albert-base-v2"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "DropoutRobertaForSequenceClassification" \
                    --model_name_or_path "${persistent_dir}/checkpoints/dropout_c-roberta-base_s-0" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-SentenceDebiasBertForSequenceClassification_c-bert-base-uncased_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                --partition=long \
                -c 4 \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "SentenceDebiasBertForSequenceClassification" \
                    --model_name_or_path "bert-base-uncased" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model["SentenceDebiasBertForSequenceClassification"]}_c-${model_to_model_name_or_path["SentenceDebiasBertForSequenceClassification"]}_t-gender.pt" \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-SentenceDebiasAlbertForSequenceClassification_c-albert-base-v2_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                --partition=long \
                -c 4 \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "SentenceDebiasAlbertForSequenceClassification" \
                    --model_name_or_path "albert-base-v2" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model["SentenceDebiasAlbertForSequenceClassification"]}_c-${model_to_model_name_or_path["SentenceDebiasAlbertForSequenceClassification"]}_t-gender.pt" \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-SentenceDebiasRobertaForSequenceClassification_c-roberta-base_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                --partition=long \
                -c 4 \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "SentenceDebiasRobertaForSequenceClassification" \
                    --model_name_or_path "roberta-base" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model["SentenceDebiasRobertaForSequenceClassification"]}_c-${model_to_model_name_or_path["SentenceDebiasRobertaForSequenceClassification"]}_t-gender.pt" \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-INLPBertForSequenceClassification_c-bert-base-uncased_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                --partition=long \
                -c 4 \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "INLPBertForSequenceClassification" \
                    --model_name_or_path "bert-base-uncased" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model["INLPBertForSequenceClassification"]}_c-${model_to_model_name_or_path["INLPBertForSequenceClassification"]}_t-gender_s-0.pt" \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-INLPAlbertForSequenceClassification_c-albert-base-v2_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                --partition=long \
                -c 4 \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "INLPAlbertForSequenceClassification" \
                    --model_name_or_path "albert-base-v2" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model["INLPAlbertForSequenceClassification"]}_c-${model_to_model_name_or_path["INLPAlbertForSequenceClassification"]}_t-gender_s-0.pt" \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-INLPRobertaForSequenceClassification_c-roberta-base_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "INLPRobertaForSequenceClassification" \
                    --model_name_or_path "roberta-base" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model["INLPRobertaForSequenceClassification"]}_c-${model_to_model_name_or_path["INLPRobertaForSequenceClassification"]}_t-gender_s-0.pt" \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-GPT2ForSequenceClassification_c-gpt2"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "GPT2ForSequenceClassification" \
                    --model_name_or_path "gpt2" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-CDAGPT2ForSequenceClassification_c-gpt2"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "CDAGPT2ForSequenceClassification" \
                    --model_name_or_path "${persistent_dir}/checkpoints/cda_c-gpt2_t-gender_s-0" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-DropoutGPT2ForSequenceClassification_c-gpt2"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "DropoutGPT2ForSequenceClassification" \
                    --model_name_or_path "${persistent_dir}/checkpoints/dropout_c-gpt2_s-0" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-INLPGPT2ForSequenceClassification_c-gpt2_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "INLPGPT2ForSequenceClassification" \
                    --model_name_or_path "gpt2" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model["INLPGPT2ForSequenceClassification"]}_c-${model_to_model_name_or_path["INLPGPT2ForSequenceClassification"]}_t-gender_s-0.pt" \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done


experiment_id="glue_m-SentenceDebiasGPT2ForSequenceClassification_c-gpt2_t-gender"
for seed in 0 1 2; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            sbatch \
                --time "24:00:00" \
                --gres=gpu:1 \
                -c 4 \
                --partition=long \
                -J ${experiment_id}_g-${task}_s-${seed} \
                -o $HOME/scratch/debias-eval/logs/%x.%j.out \
                -e $HOME/scratch/debias-eval/logs/%x.%j.err \
                python_job.sh experiments/run_glue.py \
                    --model "SentenceDebiasGPT2ForSequenceClassification" \
                    --model_name_or_path "gpt2" \
                    --task_name ${task} \
                    --do_train \
                    --do_eval \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --seed ${seed} \
                    --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model["SentenceDebiasGPT2ForSequenceClassification"]}_c-${model_to_model_name_or_path["SentenceDebiasGPT2ForSequenceClassification"]}_t-gender.pt" \
                    --output_dir "${persistent_dir}/checkpoints/${experiment_id}/${seed}/${task}"
        fi
    done
done
