import argparse
import glob
import json
import os

import pandas as pd
import numpy as np

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Export SEAT results.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    default=os.path.realpath(os.path.join(thisdir, "..")),
    type=str,
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--alpha",
    action="store",
    type=float,
    default=0.01,
    help="Alpha value for reporting significant results.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    choices=["gender", "race", "religion"],
    default="gender",
    help="What bias type to export results for.",
)
parser.add_argument(
    "--model_type",
    action="store",
    type=str,
    choices=["bert", "albert", "roberta", "gpt2"],
    help="What model type to export results for.",
)


GENDER_TESTS = [
    "sent-weat6",
    "sent-weat6b",
    "sent-weat7",
    "sent-weat7b",
    "sent-weat8",
    "sent-weat8b",
]


RACE_TESTS = [
    "sent-angry_black_woman_stereotype",
    "sent-angry_black_woman_stereotype_b",
    "sent-weat3",
    "sent-weat3b",
    "sent-weat4",
    "sent-weat5",
    "sent-weat5b",
]


RELIGION_TESTS = [
    "sent-religion1",
    "sent-religion1b",
    "sent-religion2",
    "sent-religion2b",
]


def _label_model_type(row):
    if "Bert" in row["model"]:
        return "bert"
    elif "Albert" in row["model"]:
        return "albert"
    elif "Roberta" in row["model"]:
        return "roberta"
    else:
        return "gpt2"


def _pretty_model_name(row):
    pretty_name_mapping = {
        "BertModel": "BERT",
        "SentenceDebiasBertModel": r"\, + \textsc{SentenceDebias}",
        "INLPBertModel": r"\, + \textsc{INLP}",
        "CDABertModel": r"\, + \textsc{CDA}",
        "DropoutBertModel": r"\, + \textsc{Dropout}",
        "AlbertModel": "ALBERT",
        "SentenceDebiasAlbertModel": r"\, + \textsc{SentenceDebias}",
        "INLPAlbertModel": r"\, + \textsc{INLP}",
        "CDAAlbertModel": r"\, + \textsc{CDA}",
        "DropoutAlbertModel": r"\, + \textsc{Dropout}",
        "RobertaModel": "RoBERTa",
        "SentenceDebiasRobertaModel": r"\, + \textsc{SentenceDebias}",
        "INLPRobertaModel": r"\, + \textsc{INLP}",
        "CDARobertaModel": r"\, + \textsc{CDA}",
        "DropoutRobertaModel": r"\, + \textsc{Dropout}",
        "GPT2Model": "GPT-2",
        "SentenceDebiasGPT2Model": r"\, + \textsc{SentenceDebias}",
        "INLPGPT2Model": r"\, + \textsc{INLP}",
        "CDAGPT2Model": r"\, + \textsc{CDA}",
        "DropoutGPT2Model": r"\, + \textsc{Dropout}",
    }

    return pretty_name_mapping[row["model"]]


def _get_baseline_avg_absolute_effect_size(df, model_type):
    model_type_to_baseline = {
        "bert": "BertModel",
        "albert": "AlbertModel",
        "roberta": "RobertaModel",
        "gpt2": "GPT2Model",
    }
    baseline = model_type_to_baseline[model_type]
    return df[df["model"] == baseline]["avg_absolute_effect_size"].values[0]


def _pretty_avg_absolute_effect_size(row, baseline_avg_absolute_effect_size):
    if row["avg_absolute_effect_size"] - baseline_avg_absolute_effect_size == 0:
        return f"{row['avg_absolute_effect_size']:.3f}"
    elif row["avg_absolute_effect_size"] < baseline_avg_absolute_effect_size:
        return (
            r"\da{"
            + f"{abs(baseline_avg_absolute_effect_size - row['avg_absolute_effect_size']):.3f}"
            + r"} "
            + f"{row['avg_absolute_effect_size']:.3f}"
        )
    else:
        return (
            r"\ua{"
            + f"{abs(baseline_avg_absolute_effect_size- row['avg_absolute_effect_size']):.3f}"
            + r"} "
            + f"{row['avg_absolute_effect_size']:.3f}"
        )


def _pretty_test_score(row, df_significant, test):
    # Check if test result is significant.
    significant = df_significant[
        (df_significant["experiment_id"] == row["experiment_id"])
        & (df_significant["test"] == test)
    ]["significant"].values[0]

    if significant:
        return f"{row[test]:.3f}" + r" {$^*$}"
    else:
        return f"{row[test]:.3f}"


if __name__ == "__main__":
    args = parser.parse_args()

    print("Exporting SEAT results:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - alpha: {args.alpha}")
    print(f" - bias_type: {args.bias_type}")

    bias_type_to_tests = {
        "gender": GENDER_TESTS,
        "race": RACE_TESTS,
        "religion": RELIGION_TESTS,
    }

    # Filter to only a subset of the tests.
    tests = bias_type_to_tests[args.bias_type]

    # Load the results.
    results = []
    for file_path in glob.glob(f"{args.persistent_dir}/results/seat/*.json"):
        with open(file_path, "r") as f:
            results.extend(json.load(f))
    df = pd.DataFrame.from_records(results)

    # Get significant test results.
    df_significant = df.copy()
    df_significant["significant"] = df["p_value"] < args.alpha

    # Filter results.
    df = df[["experiment_id", "test", "effect_size"]]
    df = df[df["test"].isin(tests)]

    # Pivot table.
    df = df.pivot_table(df, index="experiment_id", columns="test")
    df.columns = df.columns.droplevel(0)
    df.columns.name = None
    df = df.reset_index()

    # Compute average absolute effect size in separate datafrmae.
    df_avg = df.copy()
    df_avg = df_avg.apply(lambda x: x.abs() if np.issubdtype(x.dtype, np.number) else x)
    df_avg["avg_absolute_effect_size"] = df_avg.iloc[:, 1:].mean(axis=1)
    df_avg = df_avg[["experiment_id", "avg_absolute_effect_size"]]

    # Join tables.
    df = pd.merge(df, df_avg, on="experiment_id", how="left")

    # Parse model type from experiment ID.
    df["model"] = df["experiment_id"].str.extract(
        r"seat_m-([A-Za-z0-9]+)_c-[A-Za-z0-9-]+_*[A-Za-z-]*"
    )

    # Parse model_name_or_path from experiment ID.
    df["model_name_or_path"] = df["experiment_id"].str.extract(
        r"seat_m-[A-Za-z0-9]+_c-([A-Za-z0-9-]+)_*[A-Za-z-]*"
    )

    # Parse bias type from experiment ID.
    df["bias_type"] = df["experiment_id"].str.extract(
        r"seat_m-[A-Za-z0-9]+_c-[A-Za-z0-9-]+_t-([A-Za-z-]*)"
    )

    # Label model type (e.g., "bert").
    df["model_type"] = df.apply(lambda row: _label_model_type(row), axis=1)

    # Filter to subset of results.
    df = df[
        (df["model_type"] == args.model_type)
        & ((df["bias_type"] == args.bias_type) | df["bias_type"].isnull())
    ]

    # Get pretty model name.
    df["pretty_model_name"] = df.apply(lambda row: _pretty_model_name(row), axis=1)

    baseline_avg_absolute_effect_size = _get_baseline_avg_absolute_effect_size(
        df, args.model_type
    )

    # Get pretty metric values.
    df["pretty_avg_absolute_effect_size"] = df.apply(
        lambda row: _pretty_avg_absolute_effect_size(
            row, baseline_avg_absolute_effect_size
        ),
        axis=1,
    )

    for test in tests:
        df[test] = df.apply(
            lambda row: _pretty_test_score(row, df_significant, test), axis=1
        )

    # To get proper ordering.
    df = df.sort_values(by="pretty_model_name")

    with pd.option_context("max_colwidth", 1000):
        print(
            df.to_latex(
                float_format="%.3f",
                columns=["pretty_model_name"]
                + tests
                + ["pretty_avg_absolute_effect_size"],
                index=False,
                escape=False,
            )
        )

    os.makedirs(f"{args.persistent_dir}/tables", exist_ok=True)
    with pd.option_context("max_colwidth", 1000):
        with open(
            f"{args.persistent_dir}/tables/seat_m-{args.model_type}_t-{args.bias_type}.tex",
            "w",
        ) as f:
            f.write(
                df.to_latex(
                    float_format="%.3f",
                    columns=["pretty_model_name"]
                    + tests
                    + ["pretty_avg_absolute_effect_size"],
                    index=False,
                    escape=False,
                )
            )
