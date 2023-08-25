from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bias_bench.benchmark.stereoset import dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class StereoSetRunner:
    """Runs StereoSet intrasentence task.

    Notes:
        * We do not evaluate the intersentence task for simplicity. See the original
          implementation for intersentence details.
        * Implementation taken from: https://github.com/moinnadeem/StereoSet.
    """

    def __init__(
        self,
        intrasentence_model,
        tokenizer,
        model_name_or_path="bert-base-uncased",
        input_file="data/bias.json",
        batch_size=1,
        max_seq_length=128,
        is_generative=False,
        is_self_debias=False,
        bias_type=None,
    ):
        """Initializes StereoSet runner.

        Args:
            intrasentence_model: HuggingFace model (e.g., BertForMaskedLM) to evaluate on the
                StereoSet intrasentence task. This can potentially be a debiased model.
            tokenizer: HuggingFace tokenizer (e.g., BertTokenizer) used for pre-processing.
            model_name_or_path: HuggingFace model name (e.g., bert-base-uncased).
            input_file (`str`): Path to the file containing the dataset.
            batch_size (`int`): Batch size used for both the intrasentence and intersentence
                tasks.
            max_seq_length (`int`): Maximum sequence length used for pre-processing. If the
                `batch_size` is 1, there is no maximum.
            is_generative (`bool`): Whether to run the intrasentence task for a generative model or a
                discriminative model.
            is_self_debias (`bool`): Whether we are using a model with self-debiasing or not.
            bias_type (`str`): Bias type for self-debiasing. Determines which prompts are given
                to the model.
        """
        self._intrasentence_model = intrasentence_model
        self._tokenizer = tokenizer
        self._model_name_or_path = model_name_or_path
        self._input_file = input_file
        self._batch_size = batch_size
        self._max_seq_length = None if self._batch_size == 1 else max_seq_length
        self._is_generative = is_generative
        self._is_self_debias = is_self_debias
        # To align with self-debiasing prompt names.
        self._bias_type = "race-color" if bias_type == "race" else bias_type
        self._mask_token = self._tokenizer.mask_token
        self._mask_token_id = self._tokenizer.mask_token_id

    def __call__(self):
        bias = {}

        print("Evaluating intrasentence task.")
        intrasentence_bias = self.evaluate_intrasentence()
        bias["intrasentence"] = intrasentence_bias

        return bias

    def evaluate_intrasentence(self):
        # Use either the generative or discriminative version of likelihood scoring.
        if self._is_generative:
            sentence_probabilities = self._likelihood_score_generative()
        else:
            sentence_probabilities = self._likelihood_score()

        return sentence_probabilities

    def _likelihood_score(self):
        """Score intrasentence examples using likelihood scoring as proposed by Nadeem et al.

        Likelihood scoring computes the masked word probability of the stereotypical, anti-stereotypical,
        and unrelated associations for a given example. If a candidate consists of multiple subtokens,
        the score is computed by averaging the log probability of each subtoken.
        """
        # Use GPU, if available.
        if self._is_self_debias:
            self._intrasentence_model._model.to(device)
        else:
            model = self._intrasentence_model.to(device)

        pad_to_max_length = True if self._batch_size > 1 else False
        dataset = dataloader.IntrasentenceLoader(
            self._tokenizer,
            max_seq_length=self._max_seq_length,
            pad_to_max_length=pad_to_max_length,
            input_file=self._input_file,
            model_name_or_path=self._model_name_or_path,
        )

        loader = DataLoader(dataset, batch_size=self._batch_size)
        word_probabilities = defaultdict(list)

        # Calculate the logits for each prediction.
        for (
            sentence_id,
            next_token,
            input_ids,
            attention_mask,
            token_type_ids,
            target_tokens,
        ) in tqdm(loader, total=len(loader)):
            # Start by converting everything to a tensor.
            input_ids = torch.stack(input_ids).to(device).transpose(0, 1)
            attention_mask = torch.stack(attention_mask).to(device).transpose(0, 1)
            next_token = next_token.to(device)
            token_type_ids = torch.stack(token_type_ids).to(device).transpose(0, 1)

            mask_idxs = input_ids == self._mask_token_id

            if self._is_self_debias:
                # Get the logits for the masked token using self-debiasing.
                debiasing_prefixes = [DEBIASING_PREFIXES[self._bias_type]]
                with torch.no_grad():
                    hidden_states = (
                        self._intrasentence_model.get_token_logits_self_debiasing(
                            input_ids,
                            debiasing_prefixes=debiasing_prefixes,
                            decay_constant=50,
                            epsilon=0.01,
                        )
                    )
                output = hidden_states.softmax(dim=-1).unsqueeze(0)
            else:
                with torch.no_grad():
                    # Get the probabilities.
                    output = model(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )[0].softmax(dim=-1)

                output = output[mask_idxs]

            output = output.index_select(1, next_token).diag()
            for idx, item in enumerate(output):
                word_probabilities[sentence_id[idx]].append(item.item())

        # Reconcile the probabilities into sentences.
        sentence_probabilities = []
        for k, v in word_probabilities.items():
            pred = {}
            pred["id"] = k
            # score = np.sum([np.log2(i) for i in v]) + np.log2(len(v))
            score = np.mean(v)
            pred["score"] = score
            sentence_probabilities.append(pred)

        return sentence_probabilities

    def _likelihood_score_generative(self):
        """Score intrasentence examples using likelihood scoring as proposed by Nadeem et al. for
        generative models (e.g., GPT-2).
        """
        # Use GPU, if available.
        if self._is_self_debias:
            self._intrasentence_model._model.to(device)
        else:
            model = self._intrasentence_model.to(device)

        # Load the dataset.
        stereoset = dataloader.StereoSet(self._input_file)

        # Assume we are using GPT-2.
        unconditional_start_token = "<|endoftext|>"
        start_token = (
            torch.tensor(self._tokenizer.encode(unconditional_start_token))
            .to(device)
            .unsqueeze(0)
        )

        # Get the unconditional initial token prompts if not using self-debiasing.
        if not self._is_self_debias:
            with torch.no_grad():
                initial_token_probabilities = model(start_token)

            # initial_token_probabilities.shape == (1, 1, vocab_size).
            initial_token_probabilities = torch.softmax(
                initial_token_probabilities[0], dim=-1
            )

            # Ensure that our batch size is 1 and that our inital token isn't split into subwords.
            assert initial_token_probabilities.shape[0] == 1
            assert initial_token_probabilities.shape[1] == 1

        clusters = stereoset.get_intrasentence_examples()
        predictions = []
        for cluster in tqdm(clusters):
            joint_sentence_probability = []
            for sentence in cluster.sentences:
                probabilities = {}

                # Encode the sentence
                tokens = self._tokenizer.encode(sentence.sentence)
                tokens_tensor = torch.tensor(tokens).to(device).unsqueeze(0)

                if self._is_self_debias:
                    with torch.no_grad():
                        debiasing_prefixes = [DEBIASING_PREFIXES[self._bias_type]]
                        (
                            logits,
                            input_ids,
                        ) = self._intrasentence_model.compute_loss_self_debiasing(
                            tokens_tensor, debiasing_prefixes=debiasing_prefixes
                        )

                    # TODO Extract this to a global variable.
                    # Lengths of prompts:
                    # 13 for gender
                    # 15 for race
                    # 13 for religion
                    bias_type_to_position = {
                        "gender": 13,
                        "race-color": 15,
                        "religion": 13,
                    }

                    # Get the first token prob.
                    probs = torch.softmax(
                        logits[1, bias_type_to_position[self._bias_type] - 1], dim=-1
                    )
                    joint_sentence_probability = [probs[tokens[0]].item()]

                    # Don't include the prompt.
                    logits = logits[:, bias_type_to_position[self._bias_type] :, :]

                    output = torch.softmax(logits, dim=-1)

                else:
                    with torch.no_grad():
                        joint_sentence_probability = [
                            initial_token_probabilities[0, 0, tokens[0]].item()
                        ]

                        output = torch.softmax(model(tokens_tensor)[0], dim=-1)

                if self._is_self_debias:
                    for idx in range(1, len(tokens)):
                        joint_sentence_probability.append(
                            output[1, idx - 1, tokens[idx]].item()
                        )

                else:
                    for idx in range(1, len(tokens)):
                        joint_sentence_probability.append(
                            output[0, idx - 1, tokens[idx]].item()
                        )

                # Ensure that we have a probability on every token.
                assert len(tokens) == len(joint_sentence_probability)

                score = np.sum([np.log2(i) for i in joint_sentence_probability])
                score /= len(joint_sentence_probability)
                score = np.power(2, score)

                probabilities["id"] = sentence.ID
                probabilities["score"] = score

                predictions.append(probabilities)

        return predictions

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _get_mask_target_tokens(self, s1, s2):
        """Helper function for getting the indices of the target tokens to mask."""
        s1 = s1.tolist()
        if isinstance(s1, int):
            s1 = [s1]
        s2 = s2.tolist()

        idxs = []
        for idx in (i for i, e in enumerate(s2) if e == s1[0]):
            if s2[idx : idx + len(s1)] == s1:
                idxs.append([idx, idx + len(s1) - 1])

        return idxs
