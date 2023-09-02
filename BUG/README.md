#  BUG

Source: [Collecting a Large-Scale Gender Bias Dataset for Coreference Resolution and Machine Translation](https://aclanthology.org/2021.findings-emnlp.211)
>Shahar Levy, Koren Lazar, and Gabriel Stanovsky

Source dataset and documentation: https://github.com/SLAB-NLP/BUG

```
@inproceedings{levy-etal-2021-collecting-large,
    title = "Collecting a Large-Scale Gender Bias Dataset for Coreference Resolution and Machine Translation",
    author = "Levy, Shahar  and
      Lazar, Koren  and
      Stanovsky, Gabriel",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.211",
    doi = "10.18653/v1/2021.findings-emnlp.211",
    pages = "2470--2480"
}
```

License: MIT License

## About

BUG provides more syntactically diverse coreference templates, containing 108,419 sentences to measure stereotypical gender role assignments. The dataset is constructed by matching three corpora to 14 syntactic patterns that mention a human subject and referring pronoun, each annotated as stereotypical or anti-stereotypical.

## Data

This contains all BUG instances. `balanced_BUG.csv` is a subset of randomly sampled instances with equal male and female entities, and `gold_BUG.csv` is a subset of human-validated instances.