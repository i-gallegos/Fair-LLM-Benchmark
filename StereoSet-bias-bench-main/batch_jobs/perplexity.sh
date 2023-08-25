#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"


experiment_id="perplexity_m-BertForMaskedLM_c-bert-base-uncased"
if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}.json" ]; then
    echo ${experiment_id}
    sbatch \
        --time "24:00:00" \
        --gres=gpu:32gb:1 \
        --mem=16GB \
        --partition=long \
        -J ${experiment_id} \
        -o $HOME/scratch/debias-eval/logs/%x.%j.out \
        -e $HOME/scratch/debias-eval/logs/%x.%j.err \
        python_job.sh experiments/perplexity_mlm.py \
            --model "BertForMaskedLM" \
            --model_name_or_path "bert-base-uncased"
fi


experiment_id="perplexity_m-CDABertForMaskedLM_c-bert-base-uncased"
if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}.json" ]; then
    echo ${experiment_id}
    sbatch \
        --time "24:00:00" \
        --gres=gpu:32gb:1 \
        --mem=16GB \
        --partition=long \
        -J ${experiment_id} \
        -o $HOME/scratch/debias-eval/logs/%x.%j.out \
        -e $HOME/scratch/debias-eval/logs/%x.%j.err \
        python_job.sh experiments/perplexity_mlm.py \
            --model "CDABertForMaskedLM" \
            --model_name_or_path "bert-base-uncased" \
            --load_path "${persistent_dir}/checkpoints/cda_c-bert-base-uncased_t-gender_s-0"
fi


experiment_id="perplexity_m-DropoutBertForMaskedLM_c-bert-base-uncased"
if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}.json" ]; then
    echo ${experiment_id}
    sbatch \
        --time "24:00:00" \
        --gres=gpu:32gb:1 \
        --mem=16GB \
        --partition=long \
        -J ${experiment_id} \
        -o $HOME/scratch/debias-eval/logs/%x.%j.out \
        -e $HOME/scratch/debias-eval/logs/%x.%j.err \
        python_job.sh experiments/perplexity_mlm.py \
            --model "DropoutBertForMaskedLM" \
            --model_name_or_path "bert-base-uncased" \
            --load_path "${persistent_dir}/checkpoints/dropout_c-bert-base-uncased_s-0"
fi


experiment_id="perplexity_m-INLPBertForMaskedLM_c-bert-base-uncased"
if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}.json" ]; then
    echo ${experiment_id}
    sbatch \
        --time "24:00:00" \
        --gres=gpu:32gb:1 \
        --mem=16GB \
        --partition=long \
        -J ${experiment_id} \
        -o $HOME/scratch/debias-eval/logs/%x.%j.out \
        -e $HOME/scratch/debias-eval/logs/%x.%j.err \
        python_job.sh experiments/perplexity_mlm.py \
            --model "INLPBertForMaskedLM" \
            --model_name_or_path "bert-base-uncased" \
            --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-BertModel_c-bert-base-uncased_t-gender_s-0.pt"
fi


experiment_id="perplexity_m-SelfDebiasBertForMaskedLM_c-bert-base-uncased"
if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}.json" ]; then
    echo ${experiment_id}
    sbatch \
        --time "24:00:00" \
        --gres=gpu:32gb:1 \
        --mem=16GB \
        --partition=long \
        -J ${experiment_id} \
        -o $HOME/scratch/debias-eval/logs/%x.%j.out \
        -e $HOME/scratch/debias-eval/logs/%x.%j.err \
        python_job.sh experiments/perplexity_mlm.py \
            --model "SelfDebiasBertForMaskedLM" \
            --model_name_or_path "bert-base-uncased" \
            --self_debias

fi


experiment_id="perplexity_m-SentenceDebiasBertForMaskedLM_c-bert-base-uncased"
if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}.json" ]; then
    echo ${experiment_id}
    sbatch \
        --time "24:00:00" \
        --gres=gpu:32gb:1 \
        --mem=16GB \
        --partition=long \
        -J ${experiment_id} \
        -o $HOME/scratch/debias-eval/logs/%x.%j.out \
        -e $HOME/scratch/debias-eval/logs/%x.%j.err \
        python_job.sh experiments/perplexity_mlm.py \
            --model "SentenceDebiasBertForMaskedLM" \
            --model_name_or_path "bert-base-uncased" \
            --bias_direction "${persistent_dir}/results/subspace/subspace_m-BertModel_c-bert-base-uncased_t-gender.pt"
fi


for model in ${causal_lm_models[@]}; do
    experiment_id="perplexity_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}/results.txt" ]; then
        echo ${experiment_id}
        sbatch \
            --time "01:00:00" \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/perplexity.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --output_dir "${persistent_dir}/results/perplexity/${experiment_id}"
    fi
done


for model in ${cda_causal_lm_models[@]}; do
    experiment_id="perplexity_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-gender"
    if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}/results.txt" ]; then
        echo ${experiment_id}
        sbatch \
            --time "01:00:00" \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/perplexity.py \
                --model ${model} \
                --model_name_or_path "${persistent_dir}/checkpoints/cda_c-gpt2_t-gender_s-0" \
                --output_dir "${persistent_dir}/results/perplexity/${experiment_id}"
    fi
done


for model in ${dropout_causal_lm_models[@]}; do
    experiment_id="perplexity_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}/results.txt" ]; then
        echo ${experiment_id}
        sbatch \
            --time "01:00:00" \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/perplexity.py \
                --model ${model} \
                --model_name_or_path "${persistent_dir}/checkpoints/dropout_c-gpt2_s-0" \
                --output_dir "${persistent_dir}/results/perplexity/${experiment_id}"
    fi
done


for model in ${inlp_causal_lm_models[@]}; do
    experiment_id="perplexity_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-gender"
    if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}/results.txt" ]; then
        echo ${experiment_id}
        sbatch \
            --time "01:00:00" \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/perplexity.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-gender_s-0.pt" \
                --output_dir "${persistent_dir}/results/perplexity/${experiment_id}"
    fi
done


for model in ${sentence_debias_causal_lm_models[@]}; do
    experiment_id="perplexity_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-gender"
    if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}/results.txt" ]; then
        echo ${experiment_id}
        sbatch \
            --time "01:00:00" \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/perplexity.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-gender.pt" \
                --output_dir "${persistent_dir}/results/perplexity/${experiment_id}"
    fi
done


for model in ${self_debias_causal_lm_models[@]}; do
    experiment_id="perplexity_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-gender"
    if [ ! -f "${persistent_dir}/results/perplexity/${experiment_id}/results.txt" ]; then
        echo ${experiment_id}
        sbatch \
            --time "01:00:00" \
            -J ${experiment_id} \
            -o $HOME/scratch/debias-eval/logs/%x.%j.out \
            -e $HOME/scratch/debias-eval/logs/%x.%j.err \
            python_job.sh experiments/perplexity.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --is_self_debias \
                --output_dir "${persistent_dir}/results/perplexity/${experiment_id}"
    fi
done
