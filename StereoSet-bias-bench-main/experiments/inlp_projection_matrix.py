import argparse
import os

import torch
import transformers

from bias_bench.dataset import load_inlp_data
from bias_bench.debias.inlp import compute_projection_matrix
from bias_bench.model import models
from bias_bench.util import generate_experiment_id

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Computes the projection matrix for INLP.")
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
    help="Model (e.g., BertModel) to compute the INLP projection matrix for. "
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
    default="gender",
    choices=["gender", "race", "religion"],
    help="What type of bias to compute the INLP projection matrix for.",
)
parser.add_argument(
    "--n_classifiers",
    action="store",
    type=int,
    default=80,
    help="Number of classifiers to train when computing projection matrix.",
)
parser.add_argument("--seed", action="store", type=int, default=0, help="Seed for RNG.")


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="projection",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        seed=args.seed,
    )

    print("Computing projection matrix:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - n_classifiers: {args.n_classifiers}")
    print(f" - seed: {args.seed}")

    # Load data for INLP classifiers.
    data = load_inlp_data(args.persistent_dir, args.bias_type, seed=args.seed)

    # Load model and tokenizer.
    model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    projection_matrix = compute_projection_matrix(
        model,
        tokenizer,
        data,
        bias_type=args.bias_type,
        n_classifiers=args.n_classifiers,
    )

    print(
        f"Saving computed projection matrix to: {args.persistent_dir}/results/projection_matrix/{experiment_id}.pt"
    )
    os.makedirs(f"{args.persistent_dir}/results/projection_matrix", exist_ok=True)
    torch.save(
        projection_matrix,
        f"{args.persistent_dir}/results/projection_matrix/{experiment_id}.pt",
    )
