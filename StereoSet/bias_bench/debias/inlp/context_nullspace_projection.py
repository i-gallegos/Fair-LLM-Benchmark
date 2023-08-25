import numpy as np
import sklearn
from sklearn.svm import LinearSVC
import torch
from tqdm import tqdm

from bias_bench.debias.inlp import debias

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_gender_features(
    model,
    tokenizer,
    male_sentences,
    female_sentences,
    neutral_sentences,
):
    """Encodes gender sentences to create a set of representations to train classifiers
    for INLP on.

    Notes:
        * Implementation taken from  https://github.com/pliang279/LM_bias.
    """
    model.to(device)

    male_features = []
    female_features = []
    neutral_features = []

    # Encode the sentences.
    with torch.no_grad():
        for sentence in tqdm(male_sentences, desc="Encoding male sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            male_features.append(outputs)

        for sentence in tqdm(female_sentences, desc="Encoding female sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            female_features.append(outputs)

        for sentence in tqdm(neutral_sentences, desc="Encoding neutral sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            neutral_features.append(outputs)

    male_features = np.array(male_features)
    female_features = np.array(female_features)
    neutral_features = np.array(neutral_features)

    return male_features, female_features, neutral_features


def _extract_binary_features(model, tokenizer, bias_sentences, neutral_sentences):
    """Encodes race/religion sentences to create a set of representations to train classifiers
    for INLP on.

    Notes:
        * Sentences are split into two classes based upon if they contain *any* race/religion bias
          attribute words.
    """
    model.to(device)

    bias_features = []
    neutral_features = []

    # Encode the sentences.
    with torch.no_grad():
        for sentence in tqdm(bias_sentences, desc="Encoding bias sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            bias_features.append(outputs)

        for sentence in tqdm(neutral_sentences, desc="Encoding neutral sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            neutral_features.append(outputs)

    bias_features = np.array(bias_features)
    neutral_features = np.array(neutral_features)

    return bias_features, neutral_features


def _split_gender_dataset(male_feat, female_feat, neut_feat):
    np.random.seed(0)

    X = np.concatenate((male_feat, female_feat, neut_feat), axis=0)

    y_male = np.ones(male_feat.shape[0], dtype=int)
    y_female = np.zeros(female_feat.shape[0], dtype=int)
    y_neutral = -np.ones(neut_feat.shape[0], dtype=int)

    y = np.concatenate((y_male, y_female, y_neutral))

    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
        X_train_dev, y_train_dev, test_size=0.3, random_state=0
    )

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def _split_binary_dataset(bias_feat, neut_feat):
    np.random.seed(0)

    X = np.concatenate((bias_feat, neut_feat), axis=0)

    y_bias = np.ones(bias_feat.shape[0], dtype=int)
    y_neutral = np.zeros(neut_feat.shape[0], dtype=int)

    y = np.concatenate((y_bias, y_neutral))

    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
        X_train_dev, y_train_dev, test_size=0.3, random_state=0
    )

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def _apply_nullspace_projection(
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test, n_classifiers=80
):
    classifier_parameters = {
        "fit_intercept": False,
        "class_weight": None,
        "dual": False,
        "random_state": 0,
    }

    P, rowspace_projs, Ws = debias.get_debiasing_projection(
        classifier_class=LinearSVC,
        cls_params=classifier_parameters,
        num_classifiers=n_classifiers,
        input_dim=768,
        is_autoregressive=True,
        min_accuracy=0,
        X_train=X_train,
        Y_train=Y_train,
        X_dev=X_dev,
        Y_dev=Y_dev,
        Y_train_main=None,
        Y_dev_main=None,
        by_class=False,
        dropout_rate=0,
    )

    return P, rowspace_projs, Ws


def compute_projection_matrix(model, tokenizer, data, bias_type, n_classifiers=80):
    """Runs INLP.

    Notes:
        * We use the same classifier hyperparameters as Liang et al.

    Args:
        model: HuggingFace model (e.g., BertModel) to compute the projection
            matrix for.
        tokenizer: HuggingFace tokenizer (e.g., BertTokenizer). Used to pre-process
            examples for the INLP classifiers.
        data (`dict`): Dictionary of sentences used to train the INLP classifiers.
        bias_type (`str`): Type of bias to compute a projection matrix for.
        n_classifiers (`int`): How many classifiers to train when computing INLP
            projection matrix.
    """
    if bias_type == "gender":
        male_sentences = data["male"]
        female_sentences = data["female"]
        neutral_sentences = data["neutral"]

        male_features, female_features, neutral_features = _extract_gender_features(
            model, tokenizer, male_sentences, female_sentences, neutral_sentences
        )

        X_train, X_dev, X_test, Y_train, Y_dev, Y_test = _split_gender_dataset(
            male_features, female_features, neutral_features
        )

    else:
        bias_sentences = data["bias"]
        neutral_sentences = data["neutral"]

        bias_features, neutral_features = _extract_binary_features(
            model, tokenizer, bias_sentences, neutral_sentences
        )

        X_train, X_dev, X_test, Y_train, Y_dev, Y_test = _split_binary_dataset(
            bias_features, neutral_features
        )

    print("Dataset split sizes:")
    print(
        f"Train size: {X_train.shape[0]}; Dev size: {X_dev.shape[0]}; Test size: {X_test.shape[0]}"
    )

    P, rowspace_projs, Ws = _apply_nullspace_projection(
        X_train, X_dev, X_test, Y_train, Y_dev, Y_test, n_classifiers=n_classifiers
    )

    P = torch.tensor(P, dtype=torch.float32)

    return P
