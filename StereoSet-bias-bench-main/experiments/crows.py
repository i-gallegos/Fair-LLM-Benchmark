import argparse
import os
import json

import transformers

from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs CrowS-Pairs benchmark.")
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
    default="BertForMaskedLM",
    choices=[
        "BertForMaskedLM",
        "AlbertForMaskedLM",
        "RobertaForMaskedLM",
        "GPT2LMHeadModel",
    ],
    help="Model to evalute (e.g., BertForMaskedLM). Typically, these correspond to a HuggingFace "
    "class.",
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
    default=None,
    choices=["gender", "race", "religion"],
    help="Determines which CrowS-Pairs dataset split to evaluate against.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="crows",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
    )

    print("Running CrowS-Pairs benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_type: {args.bias_type}")

    # Load model and tokenizer.
    model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    runner = CrowSPairsRunner(
        model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized.csv",
        bias_type=args.bias_type,
        is_generative=_is_generative(args.model),  # Affects model scoring.
    )
    results = runner()

    print(f"Metric: {results}")

    os.makedirs(f"{args.persistent_dir}/results/crows", exist_ok=True)
    with open(f"{args.persistent_dir}/results/crows/{experiment_id}.json", "w") as f:
        json.dump(results, f)
