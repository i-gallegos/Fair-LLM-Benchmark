<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [BUG Dataset <img src="https://user-images.githubusercontent.com/6629995/132018898-038ec717-264d-4da3-a0b8-651b851f6b64.png" width="30" /><img src="https://user-images.githubusercontent.com/6629995/132017358-dea44bba-1487-464d-a9e1-4d534204570c.png" width="30" /><img src="https://user-images.githubusercontent.com/6629995/132018731-6ec8c4e3-12ac-474c-ae6c-03c1311777f4.png" width="30" />](#bug-dataset-img-srchttpsuser-imagesgithubusercontentcom6629995132018898-038ec717-264d-4da3-a0b8-651b851f6b64png-width30-img-srchttpsuser-imagesgithubusercontentcom6629995132017358-dea44bba-1487-464d-a9e1-4d534204570cpng-width30-img-srchttpsuser-imagesgithubusercontentcom6629995132018731-6ec8c4e3-12ac-474c-ae6c-03c1311777f4png-width30-)
  - [Setup](#setup)
  - [Dataset Partitions](#dataset-partitions)
    - [<img src="https://user-images.githubusercontent.com/6629995/132018898-038ec717-264d-4da3-a0b8-651b851f6b64.png" width="20" /> Full BUG](#img-srchttpsuser-imagesgithubusercontentcom6629995132018898-038ec717-264d-4da3-a0b8-651b851f6b64png-width20--full-bug)
    - [<img src="https://user-images.githubusercontent.com/6629995/132017358-dea44bba-1487-464d-a9e1-4d534204570c.png" width="20" /> Gold BUG](#img-srchttpsuser-imagesgithubusercontentcom6629995132017358-dea44bba-1487-464d-a9e1-4d534204570cpng-width20--gold-bug)
    - [<img src="https://user-images.githubusercontent.com/6629995/132018731-6ec8c4e3-12ac-474c-ae6c-03c1311777f4.png" width="20" /> Balanced BUG](#img-srchttpsuser-imagesgithubusercontentcom6629995132018731-6ec8c4e3-12ac-474c-ae6c-03c1311777f4png-width20--balanced-bug)
  - [Dataset Format](#dataset-format)
  - [Evaluations](#evaluations)
    - [Coreference](#coreference)
  - [Conversions](#conversions)
    - [CoNLL](#conll)
  - [Citing](#citing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

#  BUG Dataset <img src="https://user-images.githubusercontent.com/6629995/132018898-038ec717-264d-4da3-a0b8-651b851f6b64.png" width="30" /><img src="https://user-images.githubusercontent.com/6629995/132017358-dea44bba-1487-464d-a9e1-4d534204570c.png" width="30" /><img src="https://user-images.githubusercontent.com/6629995/132018731-6ec8c4e3-12ac-474c-ae6c-03c1311777f4.png" width="30" />
A Large-Scale Gender Bias Dataset for Coreference Resolution and Machine Translation (Levy et al., Findings of EMNLP 2021).

BUG was collected semi-automatically from different real-world corpora, designed to be challenging in terms of soceital gender role assignements for machine translation and coreference resolution.

## Setup

1. Unzip `data.tar.gz` this should create a `data` folder with the following files:
   * balanced_BUG.csv
   * full_BUG.csv
   * gold_BUG.csv
2. Setup a python 3.x environment and install requirements:
```
pip install -r requirements.txt
```


## Dataset Partitions

**_NOTE:_**
These partitions vary slightly from those reported in the paper due improvments and bug fixes post submission. 
For reprducibility's sake, you can access the dataset from the submission [here](https://drive.google.com/file/d/1b4Q-X1vVMoR-tIVd-XCigamnvpy0vi3F/view?usp=sharing).

### <img src="https://user-images.githubusercontent.com/6629995/132018898-038ec717-264d-4da3-a0b8-651b851f6b64.png" width="20" /> Full BUG
105,687 sentences with a human entity, identified by their profession and a gendered pronoun.

### <img src="https://user-images.githubusercontent.com/6629995/132017358-dea44bba-1487-464d-a9e1-4d534204570c.png" width="20" /> Gold BUG 

1,717 sentences, the gold-quality human-validated samples.

### <img src="https://user-images.githubusercontent.com/6629995/132018731-6ec8c4e3-12ac-474c-ae6c-03c1311777f4.png" width="20" /> Balanced BUG
25,504 sentences, randomly sampled from Full BUG to ensure balance between male and female entities and between stereotypical and non-stereotypical gender role assignments.


## Dataset Format
Each file in the data folder is a csv file adhering to the following format:


Column | Header                 | Description
:-----:|------------------------|--------------------------------------------
1      | sentence_text          | Text of sentences with a human entity, identified by their profession and a gendered pronoun
2      | tokens                 | List of tokens (using spacy tokenizer)
3      | profession             | The entity in the sentence
4      | g                      | The pronoun in the sentence
5      | profession_first_index | Words offset of profession in sentence
6      | g_first_index          | Words offset of pronoun in sentence
7      | predicted gender       | 'male'/'female' determined by the pronoun
8      | stereotype             | -1/0/1 for anti-stereotype, neutral and stereotype sentence
9      | distance               | The abs distance in words between pronoun and profession
10      | num_of_pronouns        | Number of pronouns in the sentence
11     | corpus                 | The corpus from which the sentence is taken
12     | data_index             | The query index of the pattern of the sentence

## Evaluations
See below instructions for reproducing our evaluations on BUG.

### Coreference
1. Download the Spanbert predictions from [this link](https://drive.google.com/file/d/1i24T1YT_0ByxttrCRR7qxEnt8UWyEJ7R/view?usp=sharing).
2. Unzip and put `coref_preds.jsonl` in in the `predictions/` folder.
3. From `src/evaluations/`, run `python evaluate_coref.py --in=../../predictions/coref_preds.jsonl --out=../../visualizations/delta_s_by_dist.png`.
4. This should reproduce the [coreference evaluation figure](visualizations/delta_s_by_dist.png).


## Conversions
### CoNLL
To convert each data partition to CoNLL format run:
```
python convert_to_conll.py --in=path/to/input/file --out=path/to/output/file
```

For example, try:
```
python convert_to_conll.py --in=../../data/gold_BUG.csv --out=./gold_bug.conll
```

### Filter from SPIKE
1. Download the wanted [SPIKE](https://spike.apps.allenai.org/) csv files and save them all in the same directory (directory_path).
2. Make sure the name of each file end with `\_<corpusquery><x>.csv` where `corpus` is the name of the SPIKE dataset and `x` is the number of query you entered on search (for example - myspikedata_wikipedia18.csv).
3. From `src/evaluations/`, run `python Analyze.py directory_path`.
4. This should reproduce the full dataset and balanced dataset.


## Citing
```
@misc{levy2021collecting,
      title={Collecting a Large-Scale Gender Bias Dataset for Coreference Resolution and Machine Translation}, 
      author={Shahar Levy and Koren Lazar and Gabriel Stanovsky},
      year={2021},
      eprint={2109.03858},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

