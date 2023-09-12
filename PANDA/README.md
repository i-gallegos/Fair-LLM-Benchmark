# PANDA

Source: [Perturbation Augmentation for Fairer NLP](https://aclanthology.org/2022.emnlp-main.646/)
>Rebecca Qian, Candace Ross, Jude Fernandes, Eric Michael Smith, Douwe Kiela, and Adina Williams

Source dataset and documentation: https://github.com/facebookresearch/ResponsibleNLP

```
@inproceedings{qian-etal-2022-perturbation,
    title = "Perturbation Augmentation for Fairer {NLP}",
    author = "Qian, Rebecca  and
      Ross, Candace  and
      Fernandes, Jude  and
      Smith, Eric Michael  and
      Kiela, Douwe  and
      Williams, Adina",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.646",
    doi = "10.18653/v1/2022.emnlp-main.646",
    pages = "9496--9521"
}
```

License: N/A; Copyright (c) Facebook, Inc. and its affiliates.

## About

PANDA introduces a dataset of 98,583 text perturbations for gender, race/ethnicity, and age groups, with pairs of sentences with a demographic group changed but the semantic meaning preserved. PANDA includes annotations for the perturbed demographic words. Though originally proposed as a dataset for fine-tuning, the dataset can also be used to assess robustness to demographic perturbation, where a fair model produces two invariant outputs given an input sentence and its perturbation. 

## Data

This contains all annotated sentences and their perturbations.