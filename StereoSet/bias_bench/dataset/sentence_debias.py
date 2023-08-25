import json
import os

import nltk
from tqdm import tqdm

# The original implementation uses SST, POM, WikiText-2, Reddit, Meld, and
# News-200.
DATASET_NAMES = ["wikipedia-2.5"]


def load_sentence_debias_data(persistent_dir, bias_type):
    data = []
    for dataset_name in DATASET_NAMES:
        if dataset_name == "sst":
            dataset = _SSTDataset(persistent_dir, bias_type)
        elif dataset_name == "pom":
            dataset = _POMDataset(persistent_dir, bias_type)
        else:
            dataset = _GenericDataset(persistent_dir, bias_type, dataset_name)
        data.extend(dataset.load_examples())
    return data


def _gender_augment_func(text, examples, attribute_words):
    words = text.split(" ")

    for i, (female_word, male_word) in enumerate(attribute_words):
        if female_word in words:
            female_example = text
            male_example = _replace_word_in_text(female_word, male_word, words)
            examples.append(
                {"female_example": female_example, "male_example": male_example}
            )

        if male_word in words:
            female_example = _replace_word_in_text(male_word, female_word, words)
            male_example = text
            examples.append(
                {"female_example": female_example, "male_example": male_example}
            )

    return examples


def _race_augment_func(text, examples, attribute_words):
    words = text.split(" ")

    for i, (r1_word, r2_word, r3_word) in enumerate(attribute_words):
        if r1_word in words:
            r1_example = text
            r2_example = _replace_word_in_text(r1_word, r2_word, words)
            r3_example = _replace_word_in_text(r1_word, r3_word, words)

            examples.append(
                {
                    "r1_example": r1_example,
                    "r2_example": r2_example,
                    "r3_example": r3_example,
                }
            )

        if r2_word in words:
            r1_example = _replace_word_in_text(r2_word, r1_word, words)
            r2_example = text
            r3_example = _replace_word_in_text(r2_word, r3_word, words)

            examples.append(
                {
                    "r1_example": r1_example,
                    "r2_example": r2_example,
                    "r3_example": r3_example,
                }
            )

        if r3_word in words:
            r1_example = _replace_word_in_text(r3_word, r1_word, words)
            r2_example = _replace_word_in_text(r3_word, r2_word, words)
            r3_example = text

            examples.append(
                {
                    "r1_example": r1_example,
                    "r2_example": r2_example,
                    "r3_example": r3_example,
                }
            )

    return examples


def _religion_augment_func(text, examples, attribute_words):
    words = text.split(" ")

    for i, (r1_word, r2_word, r3_word) in enumerate(attribute_words):
        if r1_word in words:
            r1_example = text
            r2_example = _replace_word_in_text(r1_word, r2_word, words)
            r3_example = _replace_word_in_text(r1_word, r3_word, words)

            examples.append(
                {
                    "r1_example": r1_example,
                    "r2_example": r2_example,
                    "r3_example": r3_example,
                }
            )

        if r2_word in words:
            r1_example = _replace_word_in_text(r2_word, r1_word, words)
            r2_example = text
            r3_example = _replace_word_in_text(r2_word, r3_word, words)

            examples.append(
                {
                    "r1_example": r1_example,
                    "r2_example": r2_example,
                    "r3_example": r3_example,
                }
            )

        if r3_word in words:
            r1_example = _replace_word_in_text(r3_word, r1_word, words)
            r2_example = _replace_word_in_text(r3_word, r2_word, words)
            r3_example = text

            examples.append(
                {
                    "r1_example": r1_example,
                    "r2_example": r2_example,
                    "r3_example": r3_example,
                }
            )

    return examples


class _SentenceDebiasDataset:

    _bias_type_to_func = {
        "gender": _gender_augment_func,
        "race": _race_augment_func,
        "religion": _religion_augment_func,
    }

    def __init__(self, persistent_dir, bias_type):
        self._persistent_dir = persistent_dir
        self._bias_type = bias_type
        self._augment_func = self._bias_type_to_func[self._bias_type]

        self._root_data_dir = f"{self._persistent_dir}/data/text"

        with open(f"{self._persistent_dir}/data/bias_attribute_words.json", "r") as f:
            self._attribute_words = json.load(f)[self._bias_type]

    def load_examples(self):
        raise NotImplementedError("load_examples method not implemented.")


class _SSTDataset(_SentenceDebiasDataset):
    def __init__(self, persistent_dir, bias_type):
        super().__init__(persistent_dir, bias_type)
        # Assumes text file containing SST exists.
        self._data_file = f"{self._root_data_dir}/sst.txt"

    def load_examples(self):
        examples = []
        for text in open(self._data_file, "r"):
            text = text.split("\t")[1:]
            text = " ".join(text)
            text = text.lower()
            text = text.strip()
            examples = self._augment_func(text, examples, self._attribute_words)

        return examples


class _POMDataset(_SentenceDebiasDataset):
    def __init__(self, persistent_dir, bias_type):
        super().__init__(persistent_dir, bias_type)
        # Assumes directory containing POM dataset exists.
        self._data_dir = f"{self._root_data_dir}/pom"

    def load_examples(self):
        examples = []
        for data_file in os.listdir(self._data_dir):
            with open(os.path.join(self._data_dir, data_file), "r") as f:
                data = f.read()

            lines = data.split(".")
            for line in lines:
                text = line.lower()
                text = text.strip()
                examples = self._augment_func(text, examples, self._attribute_words)

        return examples


class _GenericDataset(_SentenceDebiasDataset):
    def __init__(self, persistent_dir, bias_type, name):
        super().__init__(persistent_dir, bias_type)
        self._name = name
        self._data_file = f"{self._root_data_dir}/{name}.txt"

    def load_examples(self):
        examples = []

        with open(self._data_file, "r") as f:
            lines = f.readlines()

        data = []
        for line in tqdm(lines, desc=f"Sentence tokenizing {self._name}", leave=False):
            line = line.lower()
            data.extend(nltk.sent_tokenize(line))

        for sentence in tqdm(
            data, desc=f"Collecting counterfactual examples", leave=False
        ):
            sentence = sentence.lower()
            sentence = sentence.strip()
            examples = self._augment_func(sentence, examples, self._attribute_words)

        return examples


def _replace_word_in_text(word_to_replace, new_word, words):
    return " ".join([new_word if word == word_to_replace else word for word in words])
