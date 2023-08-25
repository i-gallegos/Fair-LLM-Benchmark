import argparse
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os

import transformers
from transformers import GPT2TokenizerFast
from datasets import load_dataset

from bias_bench.model import models
from bias_bench.debias.self_debias.modeling import GPT2Wrapper
from bias_bench.debias.self_debias.self_debiasing import (
    DEBIASING_PREFIXES,
    DEBIASING_KEYWORDS,
)

# Prompts for self-debiasing.
DEBIASING_PREFIXES = {
    "race-color": "The following text discriminates against people because of their race/color: ",
    "gender": "The following text discriminates against people because of their gender: ",
    "socioeconomic": "The following text discriminates against people because of their socioeconomic status/occupation: ",
    "sexual-orientation": "The following text discriminates against people because of their sexual orientiation: ",
    "religion": "The following text discriminates against people because of their religion: ",
    "age": "The following text discriminates against people because of their age: ",
    "nationality": "The following text discriminates against people because of their nationality: ",
    "disability": "The following text discriminates against people because of their disability: ",
    "physical-appearance": "The following text discriminates against people because of their physical appearance: ",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where results are written.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
        help="The specific model to compute perplexity for (e.g., 'gpt2-medium')",
    )
    parser.add_argument(
        "--decay_constant",
        type=float,
        default=50,
        help="Value for the decay constant (lambda in the paper)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Minimum factor by which each probability is multiplied",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=-1,
        help="The maximum input length to be processed (-1 corresponds to the model's context window)",
    )
    parser.add_argument(
        "--max_length_pattern",
        type=int,
        default=32,
        help="The number of tokens to reserve for the self-diagnosis patterns",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=-1,
        help="If set, for the first --stride tokens no loss is computed",
    )
    parser.add_argument(
        "--use_keywords",
        action="store_true",
        help="If set to true, keywords are used instead of full sentences to construct self-debiasing inputs",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="If set to true, all computations are done on CPU",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, additional debugging output is printed to stdout",
    )
    parser.add_argument(
        "--model",
        action="store",
        type=str,
        default="BertForMaskedLM",
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
    parser.add_argument("--bias_type", action="store", type=str, default="gender")
    parser.add_argument(
        "--is_self_debias",
        action="store_true",
    )
    parser.add_argument(
        "--persistent_dir",
        action="store",
        type=str,
        help="Directory where all persistent data will be stored.",
    )

    args = parser.parse_args()
    print(f"Parameters: {args}")

    # Override loaded the model.
    kwargs = {}
    if args.bias_direction is not None:
        # Load the pre-computed bias direction for SentenceDebias.
        bias_direction = torch.load(args.bias_direction)
        kwargs["bias_direction"] = bias_direction

    if args.projection_matrix is not None:
        # Load the pre-computed projection matrix for INLP.
        projection_matrix = torch.load(args.projection_matrix)
        kwargs["projection_matrix"] = projection_matrix

    print("=" * 40)
    print(f"Loading: {args.model}")
    model = getattr(models, args.model)(args.model_name_or_path, **kwargs)
    print("=" * 40)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    device = "cuda" if not args.no_cuda else "cpu"

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    if args.is_self_debias:
        # Move the model to GPU, if available.
        model._model.to(device)

        max_length = (
            args.max_length if args.max_length > 0 else model._model.config.n_positions
        ) - args.max_length_pattern
    else:
        model.to(device)
        max_length = (
            args.max_length if args.max_length > 0 else model.config.n_positions
        ) - args.max_length_pattern

    if args.stride <= 0:
        args.stride = max_length

    lls = []
    ppl = None
    for i in tqdm(range(0, encodings.input_ids.size(1), args.stride)):
        begin_loc = max(i + args.stride - max_length, 0)
        end_loc = min(i + args.stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        debiasing_prefixes = [DEBIASING_PREFIXES[args.bias_type]]

        with torch.no_grad():
            if args.is_self_debias:
                loss = model.compute_loss_self_debiasing(
                    input_ids=input_ids,
                    trg_len=trg_len,
                    debiasing_prefixes=debiasing_prefixes,
                    decay_constant=args.decay_constant,
                    epsilon=args.epsilon,
                    debug=args.debug,
                )

            else:
                outputs = model(input_ids, labels=target_ids)
                lm_logits = outputs[1]

                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

            log_likelihood = loss * trg_len

        lls.append(log_likelihood)

        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        print(f"Perplexity after {i} tokens: {ppl}")

    print(f"Final perplexity: {ppl}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/results.txt", "w", encoding="utf8") as fh:
        fh.write(f"=== RESULT [{args.model}] ===\n")
        fh.write(f"Perplexity: {ppl}\n")
