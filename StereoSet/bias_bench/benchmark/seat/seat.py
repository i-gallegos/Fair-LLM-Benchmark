import json
import os
import random
import re

import numpy as np
import torch

from bias_bench.benchmark.seat import weat


class SEATRunner:
    """Runs SEAT tests for a given HuggingFace transformers model.

    Implementation taken from: https://github.com/W4ngatang/sent-bias.
    """

    # Extension for files containing SEAT tests.
    TEST_EXT = ".jsonl"

    def __init__(
        self,
        model,
        tokenizer,
        tests,
        data_dir,
        experiment_id,
        n_samples=100000,
        parametric=False,
        seed=0,
    ):
        """Initializes a SEAT test runner.

        Args:
            model: HuggingFace model (e.g., BertModel) to evaluate.
            tokenizer: HuggingFace tokenizer (e.g., BertTokenizer) used for pre-processing.
            tests (`str`): Comma separated list of SEAT tests to run. SEAT test files should
                be in `data_dir` and have corresponding names with extension ".jsonl".
            data_dir (`str`): Path to directory containing the SEAT tests.
            experiment_id (`str`): Experiment identifier. Used for logging.
            n_samples (`int`): Number of permutation test samples used when estimating p-values
                (exact test is used if there are fewer than this many permutations).
            parametric (`bool`): Use parametric test (normal assumption) to compute p-values.
            seed (`int`): Random seed.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._tests = tests
        self._data_dir = data_dir
        self._experiment_id = experiment_id
        self._n_samples = n_samples
        self._parametric = parametric
        self._seed = seed

    def __call__(self):
        """Runs specified SEAT tests.

        Returns:
            `list` of `dict`s containing the SEAT test results.
        """
        random.seed(self._seed)
        np.random.seed(self._seed)

        all_tests = sorted(
            [
                entry[: -len(self.TEST_EXT)]
                for entry in os.listdir(self._data_dir)
                if not entry.startswith(".") and entry.endswith(self.TEST_EXT)
            ],
            key=_test_sort_key,
        )

        # Use the specified tests, otherwise, run all SEAT tests.
        tests = self._tests or all_tests

        results = []
        for test in tests:
            print(f"Running test {test}")

            # Load the test data.
            encs = _load_json(os.path.join(self._data_dir, f"{test}{self.TEST_EXT}"))

            print("Computing sentence encodings")
            encs_targ1 = _encode(
                self._model, self._tokenizer, encs["targ1"]["examples"]
            )
            encs_targ2 = _encode(
                self._model, self._tokenizer, encs["targ2"]["examples"]
            )
            encs_attr1 = _encode(
                self._model, self._tokenizer, encs["attr1"]["examples"]
            )
            encs_attr2 = _encode(
                self._model, self._tokenizer, encs["attr2"]["examples"]
            )

            encs["targ1"]["encs"] = encs_targ1
            encs["targ2"]["encs"] = encs_targ2
            encs["attr1"]["encs"] = encs_attr1
            encs["attr2"]["encs"] = encs_attr2

            print("\tDone!")

            # Run the test on the encodings.
            esize, pval = weat.run_test(
                encs, n_samples=self._n_samples, parametric=self._parametric
            )

            results.append(
                {
                    "experiment_id": self._experiment_id,
                    "test": test,
                    "p_value": pval,
                    "effect_size": esize,
                }
            )

        return results


def _test_sort_key(test):
    """Return tuple to be used as a sort key for the specified test name.
    Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    """
    key = ()
    prev_end = 0
    for match in re.finditer(r"\d+", test):
        key = key + (test[prev_end : match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key


def _split_comma_and_check(arg_str, allowed_set, item_type):
    """Given a comma-separated string of items, split on commas and check if
    all items are in allowed_set -- item_type is just for the assert message.
    """
    items = arg_str.split(",")
    for item in items:
        if item not in allowed_set:
            raise ValueError(f"Unknown {item_type}: {item}!")
    return items


def _load_json(sent_file):
    """Load from json. We expect a certain format later, so do some post processing."""
    print(f"Loading {sent_file}...")
    all_data = json.load(open(sent_file, "r"))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
        v["examples"] = examples

    return all_data


def _encode(model, tokenizer, texts):
    encs = {}
    for text in texts:
        # Encode each example.
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)

        # Average over the last layer of hidden representations.
        enc = outputs["last_hidden_state"]
        enc = enc.mean(dim=1)

        # Following May et al., normalize the representation.
        encs[text] = enc.detach().view(-1).numpy()
        encs[text] /= np.linalg.norm(encs[text])

    return encs
