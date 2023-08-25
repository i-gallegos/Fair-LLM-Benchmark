import argparse
import os

import torch
import transformers

from bias_bench.dataset import load_sentence_debias_data
from bias_bench.debias import (
    compute_gender_subspace,
    compute_race_subspace,
    compute_religion_subspace,
)
from bias_bench.model import models
from bias_bench.util import generate_experiment_id

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(
    description="Computes the bias subspace for SentenceDebias."
)
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertModel",
    choices=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"],
    help="Model (e.g., BertModel) to compute the SentenceDebias subspace for. "
    "Typically, these correspond to a HuggingFace class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    choices=["gender", "religion", "race"],
    required=True,
    help="The type of bias to compute the bias subspace for.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=32,
    help="Batch size to use while encoding.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="subspace",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
    )

    print("Computing bias subspace:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - model: {args.model}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - batch_size: {args.batch_size}")

    # Get the data to compute the SentenceDebias bias subspace.
    data = load_sentence_debias_data(
        persistent_dir=args.persistent_dir, bias_type=args.bias_type
    )

    # Load model and tokenizer.
    model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Specify a padding token for batched SentenceDebias subspace computation for
    # GPT2.
    if args.model == "GPT2Model":
        tokenizer.pad_token = tokenizer.eos_token

    if args.bias_type == "gender":
        bias_direction = compute_gender_subspace(
            data, model, tokenizer, batch_size=args.batch_size
        )
    elif args.bias_type == "race":
        bias_direction = compute_race_subspace(
            data, model, tokenizer, batch_size=args.batch_size
        )
    else:
        bias_direction = compute_religion_subspace(
            data, model, tokenizer, batch_size=args.batch_size
        )

    print(
        f"Saving computed PCA components to: {args.persistent_dir}/results/subspace/{experiment_id}.pt."
    )
    os.makedirs(f"{args.persistent_dir}/results/subspace", exist_ok=True)
    torch.save(
        bias_direction, f"{args.persistent_dir}/results/subspace/{experiment_id}.pt"
    )
