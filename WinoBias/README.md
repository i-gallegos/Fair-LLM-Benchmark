# WinoBias

Source: [Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods](https://aclanthology.org/N18-2003/)
>Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang

Source dataset and documentation: https://github.com/uclanlp/corefBias

```
@inproceedings{zhao-etal-2018-gender,
    title = "Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods",
    author = "Zhao, Jieyu  and
      Wang, Tianlu  and
      Yatskar, Mark  and
      Ordonez, Vicente  and
      Chang, Kai-Wei",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-2003",
    doi = "10.18653/v1/N18-2003",
    pages = "15--20"
}
```

License: MIT License

## About

WinoBias is based on Winograd schemas, which present two sentences, differing only in one or two words, and ask the reader (human or machine) to disambiguate the referent of a pronoun or possessive adjective, with a different answer for each of the two sentences. WinoBias measures stereotypical gendered associations with 3,160 sentences over 40 occupations. Some sentences require linking gendered pronouns to its stereotypically-associated occupation, while others require linking pronouns to an anti-stereotypical occupation; an unbiased model should perform both of these tasks with equal accuracy. Each sentence mentions an interaction between two occupations. Some sentences contain no syntactic signals (Type 1), while others are resolvable from syntactic information (Type 2).

## Data

This contains all instances split by sterotype/anti-stereotype and Type 1/Type 2, with development and test sets.