# HolisticBias

Source: [“I’m sorry to hear that”: Finding New Biases in Language Models with a Holistic Descriptor Dataset](https://aclanthology.org/2022.emnlp-main.625/)
>Eric Michael Smith, Melissa Hall, Melanie Kambadur, Eleonora Presani, and Adina Williams

Source dataset and documentation: https://github.com/facebookresearch/ResponsibleNLP

```
@inproceedings{smith-etal-2022-im,
    title = "{``}{I}{'}m sorry to hear that{''}: Finding New Biases in Language Models with a Holistic Descriptor Dataset",
    author = "Smith, Eric Michael  and
      Hall, Melissa  and
      Kambadur, Melanie  and
      Presani, Eleonora  and
      Williams, Adina",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.625",
    doi = "10.18653/v1/2022.emnlp-main.625",
    pages = "9180--9211"
}
```

License: Creative Commons Attribution-ShareAlike 4.0 International

## About

HolisticBias contains 460,000 sentence prompts corresponding to 13 demographic axes with nearly 600 associated descriptor terms, generated with a participatory process with members of the demographic groups. Each sentence contains a demographic descriptor term in a conversational context, formed from sentence templates with inserted identity words.

## Data

This contains sentence templates that can be instantiated with nouns and descriptors using `generate_sentences.py`.