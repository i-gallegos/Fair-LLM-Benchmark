import json
import random

import nltk
from tqdm import tqdm


def load_inlp_data(persistent_dir, bias_type, seed=0):
    """Loads sentences used to train INLP classifiers.

    Args:
        persistent_dir (`str`): Directory where all data is stored.
        bias_type (`str`): The bias type to generate the dataset for.
            Must be either gender, race, or religion.
    """
    random.seed(seed)

    if bias_type == "gender":
        data = _load_gender_data(persistent_dir)
    elif bias_type == "race":
        data = _load_race_data(persistent_dir)
    else:
        data = _load_religion_data(persistent_dir)
    return data


def _load_gender_data(persistent_dir):
    # Load the bias attribute words.
    with open(f"{persistent_dir}/data/bias_attribute_words.json", "r") as f:
        attribute_words = json.load(f)["gender"]

    male_biased_token_set = set([words[0] for words in attribute_words])
    female_biased_token_set = set([words[1] for words in attribute_words])

    male_sentences = []
    female_sentences = []

    male_sentences_clipped = []
    female_sentences_clipped = []
    neutral_sentences_clipped = []

    # We collect 10000 of each class of sentences.
    n_sentences = 10000
    count_male_sentences = 0
    count_female_sentences = 0
    count_neutral_sentences = 0

    with open(f"{persistent_dir}/data/text/wikipedia-2.5.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines, desc="Loading INLP data"):
        # Each line contains a paragraph of text.
        sentences = nltk.sent_tokenize(line)

        for sentence in sentences:
            male_flag = False
            female_flag = False

            idx = -1
            tokens = sentence.split(" ")

            # Convert tokens to lower case.
            tokens = [token.lower() for token in tokens]

            # Skip sentences that are too short.
            if len(tokens) < 5:
                continue

            for token in tokens:
                # Find male definitional token.
                if token in male_biased_token_set:
                    male_flag = True
                    idx = tokens.index(token)

                # Find female definitional token.
                if token in female_biased_token_set:
                    female_flag = True
                    idx = tokens.index(token)

                # Both female and male tokens appear.
                if male_flag and female_flag:
                    break

            # If the sentence doesn't contain male or female tokens we consider
            # it neutral.
            if (
                not male_flag
                and not female_flag
                and count_neutral_sentences < n_sentences
            ):
                # Start from the fourth token.
                index = random.randint(4, len(tokens))
                neutral_sentences_clipped.append(" ".join(tokens[:index]))
                count_neutral_sentences += 1
                continue

            # Both female and male tokens appear.
            if male_flag and female_flag:
                continue

            if male_flag and count_male_sentences < n_sentences:
                # Prevent duplicate sentences.
                if sentence not in male_sentences:
                    male_sentences.append(sentence)
                    index = random.randint(idx, len(tokens))
                    male_sentences_clipped.append(" ".join(tokens[: index + 1]))
                    count_male_sentences += 1

            if female_flag and count_female_sentences < n_sentences:
                if sentence not in female_sentences:
                    female_sentences.append(sentence)
                    index = random.randint(idx, len(tokens))
                    female_sentences_clipped.append(" ".join(tokens[: index + 1]))
                    count_female_sentences += 1

        if (
            count_male_sentences
            == count_female_sentences
            == count_neutral_sentences
            == n_sentences
        ):
            print("INLP dataset collected:")
            print(f" - Num. male sentences: {count_male_sentences}")
            print(f" - Num. female sentences: {count_female_sentences}")
            print(f" - Num. neutral sentences: {count_neutral_sentences}")
            break

    data = {
        "male": male_sentences_clipped,
        "female": female_sentences_clipped,
        "neutral": neutral_sentences_clipped,
    }

    return data


def _load_race_data(persistent_dir):
    # Load the bias attribute words.
    with open(f"{persistent_dir}/data/bias_attribute_words.json", "r") as f:
        attribute_words = json.load(f)["race"]

    # Flatten the list of race words.
    race_biased_token_set = set([word for words in attribute_words for word in words])

    race_sentences = []
    race_sentences_clipped = []
    neutral_sentences_clipped = []

    # We collect 10000 of each class of sentences.
    n_sentences = 10000
    count_race_sentences = 0
    count_neutral_sentences = 0

    with open(f"{persistent_dir}/data/text/wikipedia-2.5.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines, desc="Loading INLP data"):
        # Each line contains a paragraph of text.
        sentences = nltk.sent_tokenize(line)

        for sentence in sentences:
            race_flag = False

            idx = -1
            tokens = sentence.split(" ")

            # Convert tokens to lower case.
            tokens = [token.lower() for token in tokens]

            # Skip sentences that are too short.
            if len(tokens) < 5:
                continue

            for token in tokens:
                if token in race_biased_token_set:
                    race_flag = True
                    idx = tokens.index(token)

            # If the sentence doesn't contain a racial word we consider it neutral.
            if not race_flag and count_neutral_sentences < n_sentences:
                # Start from the fourth token.
                index = random.randint(4, len(tokens))
                neutral_sentences_clipped.append(" ".join(tokens[:index]))
                count_neutral_sentences += 1
                continue

            if race_flag and count_race_sentences < n_sentences:
                # Prevent duplicate sentences.
                if sentence not in race_sentences:
                    race_sentences.append(sentence)
                    index = random.randint(idx, len(tokens))
                    race_sentences_clipped.append(" ".join(tokens[: index + 1]))
                    count_race_sentences += 1

        if count_race_sentences == count_neutral_sentences == n_sentences:
            print("INLP dataset collected:")
            print(f" - Num. bias sentences: {count_race_sentences}")
            print(f" - Num. neutral sentences: {count_neutral_sentences}")
            break

    data = {"bias": race_sentences_clipped, "neutral": neutral_sentences_clipped}

    return data


def _load_religion_data(persistent_dir):
    # Load the bias attribute words.
    with open(f"{persistent_dir}/data/bias_attribute_words.json", "r") as f:
        attribute_words = json.load(f)["religion"]

    # Flatten the list of race words.
    religion_biased_token_set = set(
        [word for words in attribute_words for word in words]
    )

    religion_sentences = []
    religion_sentences_clipped = []
    neutral_sentences_clipped = []

    # We collect 10000 of each class of sentences.
    n_sentences = 10000
    count_religion_sentences = 0
    count_neutral_sentences = 0

    with open(f"{persistent_dir}/data/text/wikipedia-2.5.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines, desc="Loading INLP data"):
        # Each line contains a paragraph of text.
        sentences = nltk.sent_tokenize(line)

        for sentence in sentences:
            religion_flag = False

            idx = -1
            tokens = sentence.split(" ")

            # Convert tokens to lower case.
            tokens = [token.lower() for token in tokens]

            # Skip sentences that are too short.
            if len(tokens) < 5:
                continue

            for token in tokens:
                if token in religion_biased_token_set:
                    religion_flag = True
                    idx = tokens.index(token)

            # If the sentence doesn't contain a religious word we consider it neutral.
            if not religion_flag and count_neutral_sentences < n_sentences:
                index = random.randint(4, len(tokens))
                neutral_sentences_clipped.append(" ".join(tokens[:index]))
                count_neutral_sentences += 1
                continue

            if religion_flag and count_religion_sentences < n_sentences:
                # Prevent duplicate sentences.
                if sentence not in religion_sentences:
                    religion_sentences.append(sentence)
                    index = random.randint(idx, len(tokens))
                    religion_sentences_clipped.append(" ".join(tokens[: index + 1]))
                    count_religion_sentences += 1

        if count_religion_sentences == count_neutral_sentences == n_sentences:
            print("INLP dataset collected:")
            print(f" - Num. bias sentences: {count_religion_sentences}")
            print(f" - Num. neutral sentences: {count_neutral_sentences}")
            break

    data = {"bias": religion_sentences_clipped, "neutral": neutral_sentences_clipped}

    return data
