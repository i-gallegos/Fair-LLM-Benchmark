import json
import string
from tqdm import tqdm


class IntrasentenceLoader(object):
    """Loads dataset containing StereoSet intrasentence examples."""

    def __init__(
        self,
        tokenizer,
        max_seq_length=None,
        pad_to_max_length=False,
        input_file="../../data/bias.json",
        model_name_or_path=None,
    ):
        stereoset = StereoSet(input_file)
        clusters = stereoset.get_intrasentence_examples()
        self._tokenizer = tokenizer
        self._sentences = []
        self._mask_token = self._tokenizer.mask_token
        self._max_seq_length = max_seq_length
        self._pad_to_max_length = pad_to_max_length
        self._model_name_or_path = model_name_or_path

        for cluster in clusters:
            for sentence in cluster.sentences:
                if (
                    self._model_name_or_path is not None
                    and "roberta" in self._model_name_or_path
                ):
                    insertion_tokens = self._tokenizer.encode(
                        f" {sentence.template_word}",
                        add_special_tokens=False,
                    )
                    target_tokens = self._tokenizer.encode(
                        f" {cluster.target}",
                        add_special_tokens=False,
                    )
                else:
                    insertion_tokens = self._tokenizer.encode(
                        sentence.template_word, add_special_tokens=False
                    )
                    target_tokens = self._tokenizer.encode(
                        cluster.target, add_special_tokens=False
                    )

                for idx in range(len(insertion_tokens)):
                    insertion = self._tokenizer.decode(insertion_tokens[:idx])
                    insertion_string = f"{insertion}{self._mask_token}"
                    new_sentence = cluster.context.replace("BLANK", insertion_string)
                    next_token = insertion_tokens[idx]
                    self._sentences.append(
                        (new_sentence, sentence.ID, next_token, target_tokens)
                    )

    def __len__(self):
        return len(self._sentences)

    def __getitem__(self, idx):
        sentence, sentence_id, next_token, target_tokens = self._sentences[idx]
        text = sentence
        text_pair = None
        tokens_dict = self._tokenizer.encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=True,
            max_length=self._max_seq_length,
            pad_to_max_length=self._pad_to_max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
        )
        input_ids = tokens_dict["input_ids"]
        attention_mask = tokens_dict["attention_mask"]
        token_type_ids = tokens_dict["token_type_ids"]
        return (
            sentence_id,
            next_token,
            input_ids,
            attention_mask,
            token_type_ids,
            target_tokens,
        )


class StereoSet(object):
    def __init__(self, location, json_obj=None):
        """Instantiates the StereoSet object.

        Args:
            location (`str`): Location of the StereoSet.json file.
        """

        if json_obj == None:
            with open(location, "r") as f:
                self.json = json.load(f)
        else:
            self.json = json_obj

        self.version = self.json["version"]
        self.intrasentence_examples = self.__create_intrasentence_examples__(
            self.json["data"]["intrasentence"]
        )

    def __create_intrasentence_examples__(self, examples):
        created_examples = []
        for example in examples:
            sentences = []
            for sentence in example["sentences"]:
                labels = []
                for label in sentence["labels"]:
                    labels.append(Label(**label))
                sentence_obj = Sentence(
                    sentence["id"], sentence["sentence"], labels, sentence["gold_label"]
                )
                word_idx = None
                for idx, word in enumerate(example["context"].split(" ")):
                    if "BLANK" in word:
                        word_idx = idx
                if word_idx is None:
                    raise Exception("No blank word found.")
                template_word = sentence["sentence"].split(" ")[word_idx]
                sentence_obj.template_word = template_word.translate(
                    str.maketrans("", "", string.punctuation)
                )
                sentences.append(sentence_obj)
            created_example = IntrasentenceExample(
                example["id"],
                example["bias_type"],
                example["target"],
                example["context"],
                sentences,
            )
            created_examples.append(created_example)
        return created_examples

    def get_intrasentence_examples(self):
        return self.intrasentence_examples


class Example(object):
    def __init__(self, ID, bias_type, target, context, sentences):
        """A generic example.

        Args:
            ID (`str`): Provides a unique ID for the example.
            bias_type (`str`): Provides a description of the type of bias that is
                represented. It must be one of [RACE, RELIGION, GENDER, PROFESSION].
            target (`str`): Provides the word that is being stereotyped.
            context (`str`): Provides the context sentence, if exists,  that
                sets up the stereotype.
            sentences (`list`): A list of sentences that relate to the target.
        """
        self.ID = ID
        self.bias_type = bias_type
        self.target = target
        self.context = context
        self.sentences = sentences

    def __str__(self):
        s = f"Domain: {self.bias_type} - Target: {self.target} \r\n"
        s += f"Context: {self.context} \r\n"
        for sentence in self.sentences:
            s += f"{sentence} \r\n"
        return s


class Sentence(object):
    def __init__(self, ID, sentence, labels, gold_label):
        """A generic sentence type that represents a sentence.

        Args:
            ID (`str`): Provides a unique ID for the sentence with respect to the example.
            sentence (`str`): The textual sentence.
            labels (`list` of `Label` objects): A list of human labels for the sentence.
            gold_label (`enum`): The gold label associated with this sentence,
                calculated by the argmax of the labels. This must be one of
                [stereotype, anti-stereotype, unrelated, related].
        """
        assert type(ID) == str
        assert gold_label in ["stereotype", "anti-stereotype", "unrelated"]
        assert isinstance(labels, list)
        assert isinstance(labels[0], Label)

        self.ID = ID
        self.sentence = sentence
        self.gold_label = gold_label
        self.labels = labels
        self.template_word = None

    def __str__(self):
        return f"{self.gold_label.capitalize()} Sentence: {self.sentence}"


class Label(object):
    def __init__(self, human_id, label):
        """Label, represents a label object for a particular sentence.

        Args:
            human_id (`str`): Provides a unique ID for the human that labeled the sentence.
            label (`enum`): Provides a label for the sentence. This must be one of
                [stereotype, anti-stereotype, unrelated, related].
        """
        assert label in ["stereotype", "anti-stereotype", "unrelated", "related"]
        self.human_id = human_id
        self.label = label


class IntrasentenceExample(Example):
    def __init__(self, ID, bias_type, target, context, sentences):
        """Implements the Example class for an intrasentence example.

        See Example's docstring for more information.
        """
        super(IntrasentenceExample, self).__init__(
            ID, bias_type, target, context, sentences
        )
