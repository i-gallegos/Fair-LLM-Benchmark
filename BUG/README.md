#  BUG

From "A Large-Scale Gender Bias Dataset for Coreference Resolution and Machine Translation": https://aclanthology.org/2021.findings-emnlp.211

Original dataset: https://github.com/SLAB-NLP/BUG

## About

BUG provides more syntactically diverse coreference templates, containing 108,419 sentences to measure stereotypical gender role assignments. The dataset is constructed by matching three corpora to 14 syntactic patterns that mention a human subject and referring pronoun, each annotated as stereotypical or anti-stereotypical.

## Data

This contains all BUG instances. `balanced_BUG.csv` is a subset of randomly sampled instances with equal male and female entities, and `gold_BUG.csv` is a subset of human-validated instances.