# HONEST

Source: [HONEST: Measuring Hurtful Sentence Completion in Language Models](https://aclanthology.org/2021.naacl-main.191/)
>Debora Nozza, Federico Bianchi, and Dirk Hovy

Source: [Measuring Harmful Sentence Completion in Language Models for LGBTQIA+ Individuals]()
>Debora Nozza, Federico Bianchi, Anne Lauscher, and Dirk Hovy

Source dataset and documentation: https://github.com/MilaNLProc/honest

```
@inproceedings{nozza-etal-2021-honest,
    title = "{HONEST}: Measuring Hurtful Sentence Completion in Language Models",
    author = "Nozza, Debora  and
      Bianchi, Federico  and
      Hovy, Dirk",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.191",
    doi = "10.18653/v1/2021.naacl-main.191",
    pages = "2398--2406"
}
```

```
@inproceedings{nozza-etal-2022-measuring,
    title = "Measuring Harmful Sentence Completion in Language Models for {LGBTQIA}+ Individuals",
    author = "Nozza, Debora  and
      Bianchi, Federico  and
      Lauscher, Anne  and
      Hovy, Dirk",
    booktitle = "Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.ltedi-1.4",
    doi = "10.18653/v1/2022.ltedi-1.4",
    pages = "26--34"
}
```

License: MIT License

## About

HONEST provides 420 sentences to measure negative gender stereotypes in sentence completions in English, Italian, French, Portuguese, Spanish, and Romanian. Each sentence follows a cloze-based form, with a gendered identity term in the prompt, that be can completed by prompting a free-text continuation or filling in the masked token.

## Data

This contains all instances from the original HONEST work (`data/binary`). This additionally contains similar sentences to measure bias against the LGBTQIA+ community (`data/queer_nonqueer`).

