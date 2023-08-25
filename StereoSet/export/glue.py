import argparse
import os
import glob
from pathlib import Path
import json

from tqdm import tqdm
import pandas as pd

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Export GLUE results.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    default=os.path.realpath(os.path.join(thisdir, "..")),
    type=str,
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--checkpoint_dir",
    action="store",
    type=str,
    required=True,
    help="Directory where GLUE evaluation results files are.",
)
parser.add_argument(
    "--model_type",
    action="store",
    type=str,
    choices=["bert", "albert", "roberta", "gpt2"],
    help="What model type to export results for.",
)


metric_mapping = {
    "cola": "eval_matthews_correlation",
    "mnli": "eval_accuracy",
    "mrpc": "eval_f1",
    "qnli": "eval_accuracy",
    "qqp": "eval_accuracy",
    "rte": "eval_accuracy",
    "sst2": "eval_accuracy",
    "stsb": "eval_pearson",
    "wnli": "eval_accuracy",
}


def _label_model_type(row):
    if "Bert" in row["model"]:
        return "bert"
    elif "Albert" in row["model"]:
        return "albert"
    elif "Roberta" in row["model"]:
        return "roberta"
    else:
        return "gpt2"


def _pretty_metric_value(row, baseline_metric):
    if row["avg_score"] == baseline_metric:
        return f"{row['avg_score']:.2f}"
    elif row["avg_score"] < baseline_metric:
        return (
            r"\dab{"
            + f"{baseline_metric - row['avg_score']:.2f}"
            + r"} "
            + f"{row['avg_score']:.2f}"
        )
    else:
        return (
            r"\uag{"
            + f"{row['avg_score'] - baseline_metric:.2f}"
            + r"} "
            + f"{row['avg_score']:.2f}"
        )


def _get_baseline_metric(df, model_type):
    model_type_to_baseline = {
        "bert": "BertForSequenceClassification",
        "albert": "AlbertForSequenceClassification",
        "roberta": "RobertaForSequenceClassification",
        "gpt2": "GPT2ForSequenceClassification",
    }
    baseline = model_type_to_baseline[model_type]
    return df[df["model"] == baseline]["avg_score"].values[0]


def _pretty_model_name(row):
    pretty_name_mapping = {
        "BertForSequenceClassification": "BERT",
        "SentenceDebiasBertForSequenceClassification": r"\, + \textsc{SentenceDebias}",
        "INLPBertForSequenceClassification": r"\, + \textsc{INLP}",
        "CDABertForSequenceClassification": r"\, + \textsc{CDA}",
        "DropoutBertForSequenceClassification": r"\, + \textsc{Dropout}",
        "AlbertForSequenceClassification": "ALBERT",
        "SentenceDebiasAlbertForSequenceClassification": r"\, + \textsc{SentenceDebias}",
        "INLPAlbertForSequenceClassification": r"\, + \textsc{INLP}",
        "CDAAlbertForSequenceClassification": r"\, + \textsc{CDA}",
        "DropoutAlbertForSequenceClassification": r"\, + \textsc{Dropout}",
        "RobertaForSequenceClassification": "RoBERTa",
        "SentenceDebiasRobertaForSequenceClassification": r"\, + \textsc{SentenceDebias}",
        "INLPRobertaForSequenceClassification": r"\, + \textsc{INLP}",
        "CDARobertaForSequenceClassification": r"\, + \textsc{CDA}",
        "DropoutRobertaForSequenceClassification": r"\, + \textsc{Dropout}",
        "GPT2ForSequenceClassification": "GPT-2",
        "SentenceDebiasGPT2ForSequenceClassification": r"\, + \textsc{SentenceDebias}",
        "INLPGPT2ForSequenceClassification": r"\, + \textsc{INLP}",
        "CDAGPT2ForSequenceClassification": r"\, + \textsc{CDA}",
        "DropoutGPT2ForSequenceClassification": r"\, + \textsc{Dropout}",
    }

    return pretty_name_mapping[row["model"]]


def _label_model_type(row):
    if "Bert" in row["model"]:
        return "bert"
    elif "Albert" in row["model"]:
        return "albert"
    elif "Roberta" in row["model"]:
        return "roberta"
    else:
        return "gpt2"


def _parse_experiment_id(experiment_id):
    model = None
    model_name_or_path = None
    bias_type = None

    items = experiment_id.split("_")[1:]
    for item in items:
        id_, val = item[:1], item[2:]
        if id_ == "m":
            model = val
        elif id_ == "c":
            model_name_or_path = val
        elif id_ == "t":
            bias_type = val
        else:
            raise ValueError(f"Unrecognized ID {id_}.")

    return model, model_name_or_path, bias_type


if __name__ == "__main__":
    args = parser.parse_args()

    print("Exporting GLUE results:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - checkpoint_dir: {args.checkpoint_dir}")

    result_files = glob.glob(
        f"{args.checkpoint_dir}/**/eval_results.json", recursive=True
    )

    records = []
    for result_file in tqdm(result_files, desc="Parsing GLUE results"):
        path = Path(result_file)

        # Extract experiment ID from path.
        parts = path.parts
        task_name, seed, experiment_id = parts[-2], parts[-3], parts[-4]
        model, model_name_or_path, bias_type = _parse_experiment_id(experiment_id)

        with open(result_file, "r") as f:
            results = json.load(f)

        eval_metric = metric_mapping[task_name]
        score = results[eval_metric] * 100  # Convert to percent.

        records.append(
            {
                "experiment_id": experiment_id,
                "seed": seed,
                "task_name": task_name,
                "model": model,
                "model_name_or_path": model_name_or_path,
                "bias_type": bias_type,
                "score": score,
            }
        )

    df = pd.DataFrame.from_records(records)
    base_df = df.copy()  # Save original df to join to later for metadata.

    df = df.groupby(by=["experiment_id", "task_name"], as_index=False).mean()
    df = df.pivot(index="experiment_id", columns="task_name", values="score")

    df = pd.merge(
        left=df,
        right=base_df[["experiment_id", "model"]],
        how="left",
        on="experiment_id",
    )
    df = df.drop_duplicates()

    # Get pretty model name.
    df["pretty_model_name"] = df.apply(lambda row: _pretty_model_name(row), axis=1)

    # Label model type (e.g., "bert").
    df["model_type"] = df.apply(lambda row: _label_model_type(row), axis=1)

    # Filter to subset of the results.
    df = df[df["model_type"] == args.model_type]

    df["avg_score"] = df.mean(axis=1, numeric_only=True)

    baseline_metric = _get_baseline_metric(df, args.model_type)

    # Get pretty metric values.
    df["pretty_metric_value"] = df.apply(
        lambda row: _pretty_metric_value(row, baseline_metric), axis=1
    )

    # To get proper ordering.
    df = df.sort_values(by="pretty_model_name")

    print(df)

    with pd.option_context("max_colwidth", 1000):
        print(
            df.to_latex(
                float_format="%.2f",
                columns=["pretty_model_name"] + ["pretty_metric_value"],
                index=False,
                escape=False,
            )
        )
