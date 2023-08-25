import numpy as np
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_gender_subspace(data, model, tokenizer, batch_size=32):
    """Returns gender subspace components for SentenceDebias.

    Implementation based upon: https://github.com/pliang279/sent_debias.
    """
    # Use GPU, if available.
    model.to(device)

    all_embeddings_male = []
    all_embeddings_female = []

    n_batches = len(data) // batch_size
    for i in tqdm(range(n_batches), desc="Encoding gender examples"):
        offset = batch_size * i

        inputs_male = tokenizer(
            [example["male_example"] for example in data[offset : offset + batch_size]],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        inputs_female = tokenizer(
            [
                example["female_example"]
                for example in data[offset : offset + batch_size]
            ],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        male_input_ids = inputs_male["input_ids"].to(device)
        female_input_ids = inputs_female["input_ids"].to(device)

        male_attention_mask = inputs_male["attention_mask"].to(device)
        female_attention_mask = inputs_female["attention_mask"].to(device)

        with torch.no_grad():
            # Compute average representation from last layer.
            # embedding_male.shape == (batch_size, 128, 768).
            embedding_male = model(
                input_ids=male_input_ids, attention_mask=male_attention_mask
            )["last_hidden_state"]
            embedding_male *= male_attention_mask.unsqueeze(-1)
            embedding_male = embedding_male.sum(dim=1)
            embedding_male /= male_attention_mask.sum(dim=1, keepdims=True)

            embedding_female = model(
                input_ids=female_input_ids, attention_mask=female_attention_mask
            )["last_hidden_state"]
            embedding_female *= female_attention_mask.unsqueeze(-1)
            embedding_female = embedding_female.sum(dim=1)
            embedding_female /= female_attention_mask.sum(dim=1, keepdims=True)

        embedding_male /= torch.norm(embedding_male, dim=-1, keepdim=True)
        embedding_female /= torch.norm(embedding_female, dim=-1, keepdim=True)

        all_embeddings_male.append(embedding_male.cpu().numpy())
        all_embeddings_female.append(embedding_female.cpu().numpy())

    # all_embeddings_male.shape == (num_examples, dim).
    all_embeddings_male = np.concatenate(all_embeddings_male, axis=0)
    all_embeddings_female = np.concatenate(all_embeddings_female, axis=0)

    means = (all_embeddings_male + all_embeddings_female) / 2.0
    all_embeddings_male -= means
    all_embeddings_female -= means

    all_embeddings = np.concatenate(
        [all_embeddings_male, all_embeddings_female], axis=0
    )

    pca = PCA(n_components=1)
    pca.fit(all_embeddings)

    # We use only the first PCA component for debiasing.
    bias_direction = torch.tensor(pca.components_[0], dtype=torch.float32)

    return bias_direction


def compute_race_subspace(data, model, tokenizer, batch_size=32):
    """Returns race subspace components for SentenceDebias.

    Implementation based upon: https://github.com/pliang279/sent_debias.
    """
    # Use GPU, if available.
    model.to(device)

    all_embeddings_r1 = []
    all_embeddings_r2 = []
    all_embeddings_r3 = []

    n_batches = len(data) // batch_size
    for i in tqdm(range(n_batches), desc="Encoding race examples"):
        offset = batch_size * i

        inputs_r1 = tokenizer(
            [example["r1_example"] for example in data[offset : offset + batch_size]],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).to(device)

        inputs_r2 = tokenizer(
            [example["r2_example"] for example in data[offset : offset + batch_size]],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).to(device)

        inputs_r3 = tokenizer(
            [example["r3_example"] for example in data[offset : offset + batch_size]],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).to(device)

        r1_input_ids = inputs_r1["input_ids"].to(device)
        r1_attention_mask = inputs_r1["attention_mask"].to(device)

        r2_input_ids = inputs_r2["input_ids"].to(device)
        r2_attention_mask = inputs_r2["attention_mask"].to(device)

        r3_input_ids = inputs_r3["input_ids"].to(device)
        r3_attention_mask = inputs_r3["attention_mask"].to(device)

        with torch.no_grad():
            embedding_r1 = model(
                input_ids=r1_input_ids, attention_mask=r1_attention_mask
            )["last_hidden_state"]
            embedding_r1 *= r1_attention_mask.unsqueeze(-1)
            embedding_r1 = embedding_r1.sum(dim=1)
            embedding_r1 /= r1_attention_mask.sum(dim=1, keepdims=True)

            embedding_r2 = model(
                input_ids=r2_input_ids, attention_mask=r2_attention_mask
            )["last_hidden_state"]
            embedding_r2 *= r2_attention_mask.unsqueeze(-1)
            embedding_r2 = embedding_r2.sum(dim=1)
            embedding_r2 /= r2_attention_mask.sum(dim=1, keepdims=True)

            embedding_r3 = model(
                input_ids=r3_input_ids, attention_mask=r3_attention_mask
            )["last_hidden_state"]
            embedding_r3 *= r3_attention_mask.unsqueeze(-1)
            embedding_r3 = embedding_r3.sum(dim=1)
            embedding_r3 /= r3_attention_mask.sum(dim=1, keepdims=True)

        embedding_r1 /= torch.norm(embedding_r1, dim=-1, keepdim=True)
        embedding_r2 /= torch.norm(embedding_r2, dim=-1, keepdim=True)
        embedding_r3 /= torch.norm(embedding_r3, dim=-1, keepdim=True)

        all_embeddings_r1.append(embedding_r1.cpu().numpy())
        all_embeddings_r2.append(embedding_r2.cpu().numpy())
        all_embeddings_r3.append(embedding_r3.cpu().numpy())

    # all_embeddings_r1.shape == (num_examples, dim)
    all_embeddings_r1 = np.concatenate(all_embeddings_r1, axis=0)
    all_embeddings_r2 = np.concatenate(all_embeddings_r2, axis=0)
    all_embeddings_r3 = np.concatenate(all_embeddings_r3, axis=0)

    means = (all_embeddings_r1 + all_embeddings_r2 + all_embeddings_r3) / 3.0
    all_embeddings_r1 -= means
    all_embeddings_r2 -= means
    all_embeddings_r3 -= means

    all_embeddings = np.concatenate(
        [all_embeddings_r1, all_embeddings_r2, all_embeddings_r3], axis=0
    )

    pca = PCA(n_components=1)
    pca.fit(all_embeddings)

    # We use only the first PCA component for debiasing.
    bias_direction = torch.tensor(pca.components_[0], dtype=torch.float32)

    return bias_direction


def compute_religion_subspace(data, model, tokenizer, batch_size=32):
    """Returns religion subspace components for SentenceDebias.

    Implementation based upon: https://github.com/pliang279/sent_debias.
    """
    # Use GPU, if available.
    model.to(device)

    all_embeddings_r1 = []
    all_embeddings_r2 = []
    all_embeddings_r3 = []

    n_batches = len(data) // batch_size
    for i in tqdm(range(n_batches), desc="Encoding religion examples"):
        offset = batch_size * i

        inputs_r1 = tokenizer(
            [example["r1_example"] for example in data[offset : offset + batch_size]],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).to(device)

        inputs_r2 = tokenizer(
            [example["r2_example"] for example in data[offset : offset + batch_size]],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).to(device)

        inputs_r3 = tokenizer(
            [example["r3_example"] for example in data[offset : offset + batch_size]],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).to(device)

        r1_input_ids = inputs_r1["input_ids"].to(device)
        r1_attention_mask = inputs_r1["attention_mask"].to(device)

        r2_input_ids = inputs_r2["input_ids"].to(device)
        r2_attention_mask = inputs_r2["attention_mask"].to(device)

        r3_input_ids = inputs_r3["input_ids"].to(device)
        r3_attention_mask = inputs_r3["attention_mask"].to(device)

        with torch.no_grad():
            embedding_r1 = model(
                input_ids=r1_input_ids, attention_mask=r1_attention_mask
            )["last_hidden_state"]
            embedding_r1 *= r1_attention_mask.unsqueeze(-1)
            embedding_r1 = embedding_r1.sum(dim=1)
            embedding_r1 /= r1_attention_mask.sum(dim=1, keepdims=True)

            embedding_r2 = model(
                input_ids=r2_input_ids, attention_mask=r2_attention_mask
            )["last_hidden_state"]
            embedding_r2 *= r2_attention_mask.unsqueeze(-1)
            embedding_r2 = embedding_r2.sum(dim=1)
            embedding_r2 /= r2_attention_mask.sum(dim=1, keepdims=True)

            embedding_r3 = model(
                input_ids=r3_input_ids, attention_mask=r3_attention_mask
            )["last_hidden_state"]
            embedding_r3 *= r3_attention_mask.unsqueeze(-1)
            embedding_r3 = embedding_r3.sum(dim=1)
            embedding_r3 /= r3_attention_mask.sum(dim=1, keepdims=True)

        embedding_r1 /= torch.norm(embedding_r1, dim=-1, keepdim=True)
        embedding_r2 /= torch.norm(embedding_r2, dim=-1, keepdim=True)
        embedding_r3 /= torch.norm(embedding_r3, dim=-1, keepdim=True)

        all_embeddings_r1.append(embedding_r1.cpu().numpy())
        all_embeddings_r2.append(embedding_r2.cpu().numpy())
        all_embeddings_r3.append(embedding_r3.cpu().numpy())

    # all_embeddings_r1.shape == (num_examples, dim).
    all_embeddings_r1 = np.concatenate(all_embeddings_r1, axis=0)
    all_embeddings_r2 = np.concatenate(all_embeddings_r2, axis=0)
    all_embeddings_r3 = np.concatenate(all_embeddings_r3, axis=0)

    means = (all_embeddings_r1 + all_embeddings_r2 + all_embeddings_r3) / 3.0
    all_embeddings_r1 -= means
    all_embeddings_r2 -= means
    all_embeddings_r3 -= means

    all_embeddings = np.concatenate(
        [all_embeddings_r1, all_embeddings_r2, all_embeddings_r3], axis=0
    )

    pca = PCA(n_components=1)
    pca.fit(all_embeddings)

    # We use only the first PCA component for debiasing.
    bias_direction = torch.tensor(pca.components_[0], dtype=torch.float32)

    return bias_direction
