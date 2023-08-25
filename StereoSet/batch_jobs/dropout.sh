#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"


for seed in 0 1 2; do
    experiment_id="dropout_c-bert-base-uncased_s-${seed}"
    if [ ! -d "${persistent_dir}/checkpoints/${experiment_id}" ]; then
        echo ${experiment_id}
        sbatch \
            --time "48:00:00" \
            --gres=gpu:48gb:1 \
            -c 4 \
            --partition=long \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/run_mlm.py \
                --model_name_or_path "bert-base-uncased" \
                --do_train \
                --train_file "${persistent_dir}/data/text/wikipedia-10.txt" \
                --max_steps 2000 \
                --per_device_train_batch_size 32 \
                --gradient_accumulation_steps 16 \
                --max_seq_length 512 \
                --save_steps 500 \
                --preprocessing_num_workers 4 \
                --dropout_debias \
                --seed ${seed} \
                --output_dir "${persistent_dir}/checkpoints/${experiment_id}"
    fi
done


for seed in 0 1 2; do
    experiment_id="dropout_c-albert-base-v2_s-${seed}"
    if [ ! -d "${persistent_dir}/checkpoints/${experiment_id}" ]; then
        echo ${experiment_id}
        sbatch \
            --time "48:00:00" \
            --gres=gpu:48gb:1 \
            -c 4 \
            --partition=long \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/run_mlm.py \
                --model_name_or_path "albert-base-v2" \
                --do_train \
                --train_file "${persistent_dir}/data/text/wikipedia-10.txt" \
                --max_steps 2000 \
                --per_device_train_batch_size 32 \
                --gradient_accumulation_steps 16 \
                --max_seq_length 512 \
                --save_steps 500 \
                --preprocessing_num_workers 4 \
                --dropout_debias \
                --seed ${seed} \
                --output_dir "${persistent_dir}/checkpoints/${experiment_id}"
    fi
done


for seed in 0 1 2; do
    experiment_id="dropout_c-roberta-base_s-${seed}"
    if [ ! -d "${persistent_dir}/checkpoints/${experiment_id}" ]; then
        echo ${experiment_id}
        sbatch \
            --time "48:00:00" \
            --gres=gpu:48gb:1 \
            -c 4 \
            --partition=long \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/run_mlm.py \
                --model_name_or_path "roberta-base" \
                --do_train \
                --train_file "${persistent_dir}/data/text/wikipedia-10.txt" \
                --max_steps 2000 \
                --per_device_train_batch_size 32 \
                --gradient_accumulation_steps 16 \
                --max_seq_length 512 \
                --save_steps 500 \
                --preprocessing_num_workers 4 \
                --dropout_debias \
                --seed ${seed} \
                --output_dir "${persistent_dir}/checkpoints/${experiment_id}"
    fi
done


for seed in 0 1 2; do
    experiment_id="dropout_c-gpt2_s-${seed}"
    if [ ! -d "${persistent_dir}/checkpoints/${experiment_id}" ]; then
        echo ${experiment_id}
        sbatch \
            --time "48:00:00" \
            --gres=gpu:48gb:1 \
            -c 4 \
            --partition=long \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/run_clm.py \
                --model_name_or_path "gpt2" \
                --do_train \
                --train_file "${persistent_dir}/data/text/wikipedia-10.txt" \
                --max_steps 2000 \
                --per_device_train_batch_size 8 \
                --gradient_accumulation_steps 32 \
                --save_steps 500 \
                --preprocessing_num_workers 4 \
                --dropout_debias \
                --seed ${seed} \
                --output_dir "${persistent_dir}/checkpoints/${experiment_id}"
    fi
done
