# RealToxicityPrompts

Source: [RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models](https://aclanthology.org/2020.findings-emnlp.301/)
>Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith

Source dataset and documentation: https://allenai.org/data/real-toxicity-prompts

```
@inproceedings{gehman-etal-2020-realtoxicityprompts,
    title = "{R}eal{T}oxicity{P}rompts: Evaluating Neural Toxic Degeneration in Language Models",
    author = "Gehman, Samuel  and
      Gururangan, Suchin  and
      Sap, Maarten  and
      Choi, Yejin  and
      Smith, Noah A.",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.301",
    doi = "10.18653/v1/2020.findings-emnlp.301",
    pages = "3356--3369"
}
```

License: Apache License

## About

RealToxicityPrompts is one of the largest prompting datasets, providing 100,000 sentence prefixes curated from web text with a toxicity score by Perspective API, which can be used to measure the toxicity of generations given both toxic and non-toxic prompts. To create the dataset, a set of web-scraped sentences are scored for toxicity, and 25K sentences are sampled from each of four quartiles, then split into a prompt (used in the dataset) and a continuation.

## Data

We do not include the dataset here due to its size. Access it here: https://allenai.org/data/real-toxicity-prompts