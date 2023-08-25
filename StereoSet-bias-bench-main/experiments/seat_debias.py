import argparse
import json
import os

import torch
import transformers

from bias_bench.benchmark.seat import SEATRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id


thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs SEAT benchmark for debiased models.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--tests",
    action="store",
    nargs="*",
    help="List of SEAT tests to run. Test files should be in `data_dir` and have "
    "corresponding names with extension .jsonl.",
)
parser.add_argument(
    "--n_samples",
    action="store",
    type=int,
    default=100000,
    help="Number of permutation test samples used when estimating p-values "
    "(exact test is used if there are fewer than this many permutations).",
)
parser.add_argument(
    "--parametric",
    action="store_true",
    help="Use parametric test (normal assumption) to compute p-values.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="SentenceDebiasBertModel",
    choices=[
        "SentenceDebiasBertModel",
        "SentenceDebiasAlbertModel",
        "SentenceDebiasRobertaModel",
        "SentenceDebiasGPT2Model",
        "INLPBertModel",
        "INLPAlbertModel",
        "INLPRobertaModel",
        "INLPGPT2Model",
        "CDABertModel",
        "CDAAlbertModel",
        "CDARobertaModel",
        "CDAGPT2Model",
        "DropoutBertModel",
        "DropoutAlbertModel",
        "DropoutRobertaModel",
        "DropoutGPT2Model",
    ],
    help="Debiased model (e.g., SentenceDebiasModel) to evaluate.",
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
    "--bias_direction",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed bias direction for SentenceDebias.",
)
parser.add_argument(
    "--projection_matrix",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed projection matrix for INLP.",
)
parser.add_argument(
    "--load_path",
    action="store",
    type=str,
    help="Path to saved CDA or Dropout model checkpoint.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    choices=["gender", "religion", "race"],
    help="The type of bias to mitigate.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="seat",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
    )

    print("Running SEAT benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - tests: {args.tests}")
    print(f" - n_samples: {args.n_samples}")
    print(f" - parametric: {args.parametric}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_direction: {args.bias_direction}")
    print(f" - projection_matrix: {args.projection_matrix}")
    print(f" - load_path: {args.load_path}")
    print(f" - bias_type: {args.bias_type}")

    kwargs = {}
    if args.bias_direction is not None:
        # Load the pre-computed bias direction for SentenceDebias.
        bias_direction = torch.load(args.bias_direction)
        kwargs["bias_direction"] = bias_direction

    if args.projection_matrix is not None:
        # Load the pre-computed projection matrix for INLP.
        projection_matrix = torch.load(args.projection_matrix)
        kwargs["projection_matrix"] = projection_matrix

    # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
    model = getattr(models, args.model)(
        args.load_path or args.model_name_or_path, **kwargs
    )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    runner = SEATRunner(
        experiment_id=experiment_id,
        tests=args.tests,
        data_dir=f"{args.persistent_dir}/data/seat",
        n_samples=args.n_samples,
        parametric=args.parametric,
        model=model,
        tokenizer=tokenizer,
    )
    results = runner()
    print(results)

    os.makedirs(f"{args.persistent_dir}/results/seat", exist_ok=True)
    with open(f"{args.persistent_dir}/results/seat/{experiment_id}.json", "w") as f:
        json.dump(results, f)
