from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    T5Tokenizer,
    GPT2Tokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForMaskedLM,
    RobertaForMaskedLM,
    BertForMaskedLM,
    AlbertForMaskedLM,
)

from bias_bench.debias.self_debias.generation import (
    SelfDebiasingLogitsProcessor,
    SelfDebiasingGPT2LMHeadModel,
    SelfDebiasingT5ForConditionalGeneration,
)


def get_top_k_tokens(logits: torch.Tensor, tokenizer: PreTrainedTokenizer, k: int = 5):
    values, indices = torch.topk(logits, k, dim=-1)
    if len(logits.shape) == 2:
        assert logits.shape[0] == 1
        values, indices = values[0], indices[0]
    return tokenizer.convert_ids_to_tokens(indices), values


class MaskedLMWrapper(ABC):
    """
    This class represents a wrapper for a masked language model that provides the ability to perform self-debiasing for sentences with
    a single masked token.
    """

    def __init__(self, model_name: str, use_cuda: bool = True):
        """
        :param use_cuda: whether to use CUDA
        """
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name).to(self._device)

    def get_token_logits(self, input_text: str) -> torch.Tensor:
        input_ids = self._tokenizer.encode(input_text, return_tensors="pt").to(
            self._device
        )
        assert (
            sum(1 for id_ in input_ids[0] if id_ == self._tokenizer.mask_token_id) == 1
        ), "Input text must contain exactly one mask token"
        scores = self._model(input_ids)["logits"]
        mask_positions = input_ids == self._tokenizer.mask_token_id
        return scores[mask_positions]

    def get_token_logits_batch(self, input_texts: List[str]) -> torch.Tensor:
        batch = self._tokenizer.batch_encode_plus(
            input_texts, return_tensors="pt", padding=True
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}

        mask_positions = batch["input_ids"] == self._tokenizer.mask_token_id
        assert torch.all(
            mask_positions.sum(axis=-1) == 1
        ), "Each input text must contain exactly one mask token"

        scores = self._model(**batch)["logits"]
        return scores[mask_positions]

    def get_token_logits_self_debiasing(
        self,
        input_ids: torch.Tensor,
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
    ) -> torch.Tensor:
        """
        Computes the token logits for the single masked position in the given input ids with self-debiasing.
        :param input_ids: the input ids
        :param debiasing_prefixes: the debiasing prefixes to be used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :return: the cross entropy loss
        """
        assert (
            input_ids.shape[0] == 1
        )  # TODO future versions should also work with batches

        logits_processor = SelfDebiasingLogitsProcessor(
            num_debiasing_prefixes=len(debiasing_prefixes),
            decay_constant=decay_constant,
            epsilon=epsilon,
            tokenizer=self._tokenizer,
        )

        input_prefixes = [self._tokenizer.cls_token] + [
            " ".join([self._tokenizer.cls_token, dp]) for dp in debiasing_prefixes
        ]

        input_prefixes = self._tokenizer.batch_encode_plus(
            input_prefixes, padding=True, return_tensors="pt", add_special_tokens=False
        )
        input_prefixes["attention_mask"] = torch.flip(
            input_prefixes["attention_mask"], dims=[1]
        )

        # remove leading [CLS] tokens
        input_ids = input_ids[:, 1:]

        shifts = input_prefixes["attention_mask"].shape[-1] - input_prefixes[
            "attention_mask"
        ].sum(dim=-1)
        for batch_idx in range(input_prefixes["input_ids"].shape[0]):
            input_prefixes["input_ids"][batch_idx] = input_prefixes["input_ids"][
                batch_idx
            ].roll(shifts[batch_idx].item())

        input_prefixes = {k: v.to(self._device) for k, v in input_prefixes.items()}

        input_ids_repeated = input_ids.repeat(len(debiasing_prefixes) + 1, 1)
        attention_mask = torch.ones_like(input_ids_repeated)

        attention_mask = torch.cat(
            [input_prefixes["attention_mask"], attention_mask], dim=-1
        )
        input_ids_repeated = torch.cat(
            [input_prefixes["input_ids"], input_ids_repeated], dim=-1
        )

        mask_positions = input_ids_repeated == self._tokenizer.mask_token_id

        position_ids = attention_mask.long().cumsum(-1)
        if isinstance(self._model, RobertaForMaskedLM):
            position_ids += self._model.base_model.embeddings.padding_idx
        elif isinstance(self._model, BertForMaskedLM):
            position_ids -= 1
        elif isinstance(self._model, AlbertForMaskedLM):
            position_ids -= 1
        else:
            raise ValueError(
                f"Position IDs shift is not implemented for {self._model.__class__}"
            )

        position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = self._model(
            input_ids=input_ids_repeated,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        lm_logits = outputs["logits"]

        for idx in range(lm_logits.shape[1]):
            if torch.any(mask_positions[:, idx]):
                lm_logits[:, idx, :] = logits_processor(
                    input_ids=None, scores=lm_logits[:, idx, :]
                )

        return lm_logits[mask_positions][0]

    def compute_loss(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        outputs = self._model(input_ids=input_ids, labels=labels)
        return outputs.loss

    def compute_loss_self_debiasing(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
    ) -> torch.Tensor:

        relevant_labels = labels[input_ids == self._tokenizer.mask_token_id]
        token_logits = self.get_token_logits_self_debiasing(
            input_ids,
            debiasing_prefixes=debiasing_prefixes,
            decay_constant=decay_constant,
            epsilon=epsilon,
        )
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            token_logits.view(-1, self._model.config.vocab_size),
            relevant_labels.view(-1),
        )
        return masked_lm_loss


class GenerativeLMWrapper(ABC):
    """
    This class represents a wrapper for a pretrained language model that provides some high-level functions, including zero-shot
    classification using cloze questions and the generation of texts with self-debiasing.
    """

    def __init__(self, use_cuda: bool = True):
        """
        :param use_cuda: whether to use CUDA
        """
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self._tokenizer = None  # type: Optional[PreTrainedTokenizer]
        self._model = None  # type: Optional[PreTrainedModel]

    def query_model(self, input_text: str) -> torch.FloatTensor:
        """For a given input text, returns the probability distribution over possible next tokens."""
        return self.query_model_batch([input_text])[0]

    @abstractmethod
    def query_model_batch(self, input_texts: List[str]) -> torch.FloatTensor:
        """For a batch of input texts, returns the probability distribution over possible next tokens."""
        pass

    @abstractmethod
    def generate(self, input_text: str, **kwargs) -> str:
        """Generates a continuation for a given input text."""
        pass

    @abstractmethod
    def generate_self_debiasing(
        self,
        input_texts: List[str],
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
        debug: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Generates continuations for the given input texts with self-debiasing.
        :param input_texts: the input texts to generate continuations for
        :param debiasing_prefixes: the debiasing prefixes to be used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :param kwargs: further arguments are passed on to the original generate function
        :return: the list of generated continuations
        """
        pass

    @abstractmethod
    def compute_loss(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        """Computes cross-entropy loss for the given input ids and corresponding labels."""
        pass

    @abstractmethod
    def compute_loss_self_debiasing(
        self,
        input_ids: torch.Tensor,
        trg_len: int,
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Computes cross-entropy loss for the given input ids with self-debiasing.
        :param input_ids: the input ids
        :param trg_len: only the last trg_len tokens are considered for computing the loss
        :param debiasing_prefixes: the debiasing prefixes to be used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :return: the cross entropy loss
        """
        pass

    def get_token_probability_distribution(
        self, input_texts: List[str], output_choices: List[str]
    ) -> List[List[Tuple[str, float]]]:
        """
        For a batch of input texts, returns the probability distribution over possible next tokens considering only the given list of
        output choices.
        :param input_texts: the input texts
        :param output_choices: the allowed output choices (must correspond to single tokens in the model's vocabulary)
        :return: a list of lists, where output[i][j] is a (output, probability) tuple for the ith input and jth output choice.
        """
        output_choice_ids = []
        kwargs = {"add_prefix_space": True} if isinstance(self, GPT2Wrapper) else {}
        for word in output_choices:
            tokens = self._tokenizer.tokenize(word, **kwargs)
            assert (
                len(tokens) == 1
            ), f"Word {word} consists of multiple tokens: {tokens}"
            assert (
                tokens[0] not in self._tokenizer.all_special_tokens
            ), f"Word {word} corresponds to a special token: {tokens[0]}"
            token_id = self._tokenizer.convert_tokens_to_ids(tokens)[0]
            output_choice_ids.append(token_id)

        logits = self.query_model_batch(input_texts)
        result = []

        for idx, _ in enumerate(input_texts):
            output_probabilities = logits[idx][output_choice_ids].softmax(dim=0)
            choices_with_probabilities = list(
                zip(output_choices, (prob.item() for prob in output_probabilities))
            )
            result.append(choices_with_probabilities)

        return result


class T5Wrapper(GenerativeLMWrapper):
    """A wrapper for the T5 model"""

    def __init__(self, model_name: str = "google/t5-v1_1-xl", use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained T5 model (default: "google/t5-v1_1-xl")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._tokenizer = T5Tokenizer.from_pretrained(model_name)
        self._model = SelfDebiasingT5ForConditionalGeneration.from_pretrained(
            model_name
        )
        if use_cuda:
            self._model.parallelize()

    def query_model_batch(self, input_texts: List[str]):
        assert all("<extra_id_0>" in input_text for input_text in input_texts)
        output_texts = ["<extra_id_0>"] * len(input_texts)
        inputs = self._tokenizer.batch_encode_plus(
            input_texts, padding=True, return_tensors="pt"
        )
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_ids = self._tokenizer.batch_encode_plus(
            output_texts, return_tensors="pt"
        )["input_ids"].to(self._device)
        return self._model(labels=output_ids, **inputs)["logits"][:, 1, :]

    def generate(self, input_text: str, **kwargs):
        assert "<extra_id_0>" in input_text
        input_ids = self._tokenizer.encode(input_text, return_tensors="pt").to(
            self._device
        )
        output_ids = self._model.generate(input_ids, **kwargs)[0]
        return self._tokenizer.decode(output_ids)

    def generate_self_debiasing(
        self,
        input_texts: List[str],
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
        debug: bool = False,
        **kwargs,
    ) -> List[str]:

        assert all("<extra_id_0>" in input_text for input_text in input_texts)

        self._model.init_logits_processor(
            num_debiasing_prefixes=len(debiasing_prefixes),
            decay_constant=decay_constant,
            epsilon=epsilon,
            debug=debug,
            tokenizer=self._tokenizer,
        )
        inputs = input_texts.copy()
        for debiasing_prefix in debiasing_prefixes:
            for input_text in input_texts:
                inputs += [debiasing_prefix + input_text]

        inputs = self._tokenizer.batch_encode_plus(
            inputs, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        output_ids = self._model.generate(**inputs, **kwargs)

        batch_size = output_ids.shape[0] // (1 + len(debiasing_prefixes))
        output_ids = output_ids[:batch_size, :]
        return self._tokenizer.batch_decode(output_ids)

    def compute_loss(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def compute_loss_self_debiasing(
        self,
        input_ids: torch.Tensor,
        trg_len: int,
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
        debug: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError()


class GPT2Wrapper(GenerativeLMWrapper):
    def __init__(self, model_name: str = "gpt2-xl", use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained GPT2 model (default: "gpt2-xl")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self._model = SelfDebiasingGPT2LMHeadModel.from_pretrained(
            model_name
        )  # type: SelfDebiasingGPT2LMHeadModel
        if use_cuda:
            # self._model.parallelize()
            pass
        self._model.to("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id

    def query_model_batch(self, input_texts: List[str]):
        inputs = self._tokenizer.batch_encode_plus(
            input_texts, padding=True, return_tensors="pt"
        )
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_indices = inputs["attention_mask"].sum(dim=1) - 1
        output = self._model(**inputs)["logits"]
        return torch.stack(
            [
                output[example_idx, last_word_idx, :]
                for example_idx, last_word_idx in enumerate(output_indices)
            ]
        )

    def generate(self, input_text: str, **kwargs):
        input_ids = self._tokenizer.encode(input_text, return_tensors="pt").to(
            self._device
        )
        output_ids = self._model.generate(input_ids, **kwargs)[0]
        return self._tokenizer.decode(output_ids)

    def generate_self_debiasing(
        self,
        input_texts: List[str],
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
        debug: bool = False,
        min_length: int = None,
        max_length: int = None,
        **kwargs,
    ) -> List[str]:

        self._model.init_logits_processor(
            num_debiasing_prefixes=len(debiasing_prefixes),
            decay_constant=decay_constant,
            epsilon=epsilon,
            debug=debug,
            tokenizer=self._tokenizer,
        )
        inputs = input_texts.copy()
        for debiasing_prefix in debiasing_prefixes:
            for input_text in input_texts:
                inputs += [debiasing_prefix + input_text]

        inputs = self._tokenizer.batch_encode_plus(
            inputs, padding=True, return_tensors="pt"
        )
        inputs["attention_mask"] = torch.flip(inputs["attention_mask"], dims=[1])
        shifts = inputs["attention_mask"].shape[-1] - inputs["attention_mask"].sum(
            dim=-1
        )
        for batch_idx in range(inputs["input_ids"].shape[0]):
            inputs["input_ids"][batch_idx] = inputs["input_ids"][batch_idx].roll(
                shifts[batch_idx].item()
            )

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length

        output_ids = self._model.generate(
            **inputs, min_length=min_length, max_length=max_length, **kwargs
        )

        batch_size = output_ids.shape[0] // (1 + len(debiasing_prefixes))
        output_ids = output_ids[:batch_size, inputs["input_ids"].shape[1] :]
        return self._tokenizer.batch_decode(output_ids)

    def compute_loss(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        outputs = self._model(input_ids, labels=labels)
        lm_logits = outputs[1]

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss

    def compute_loss_self_debiasing(
        self,
        input_ids: torch.Tensor,
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
        debug: bool = False,
    ) -> torch.Tensor:
        self._device = "cuda"

        self._model.init_logits_processor(
            num_debiasing_prefixes=len(debiasing_prefixes),
            decay_constant=decay_constant,
            epsilon=epsilon,
            debug=debug,
            tokenizer=self._tokenizer,
        )

        input_prefixes = [""] + debiasing_prefixes
        input_prefixes = self._tokenizer.batch_encode_plus(
            input_prefixes, padding=True, return_tensors="pt"
        )
        input_prefixes["attention_mask"] = torch.flip(
            input_prefixes["attention_mask"], dims=[1]
        )

        shifts = input_prefixes["attention_mask"].shape[-1] - input_prefixes[
            "attention_mask"
        ].sum(dim=-1)
        for batch_idx in range(input_prefixes["input_ids"].shape[0]):
            input_prefixes["input_ids"][batch_idx] = input_prefixes["input_ids"][
                batch_idx
            ].roll(shifts[batch_idx].item())

        input_prefixes = {k: v.to(self._device) for k, v in input_prefixes.items()}

        input_ids_repeated = input_ids.repeat(len(debiasing_prefixes) + 1, 1)
        attention_mask = torch.ones_like(input_ids_repeated)

        attention_mask = torch.cat(
            [input_prefixes["attention_mask"], attention_mask], dim=-1
        )
        input_ids_repeated = torch.cat(
            [input_prefixes["input_ids"], input_ids_repeated], dim=-1
        )

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = self._model(
            input_ids=input_ids_repeated,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        lm_logits = outputs["logits"]

        for idx in range(lm_logits.shape[1]):
            lm_logits[:, idx, :] = self._model.logits_processor(
                input_ids=None, scores=lm_logits[:, idx, :]
            )

        return lm_logits, input_ids_repeated
