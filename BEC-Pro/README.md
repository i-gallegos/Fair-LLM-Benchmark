# Gender bias in BERT
This repository holds the code for my master thesis entitled "The Association of Gender Bias with BERT - Measuring, Mitigating and Cross-lingual portability", written at the University of Groningen and the University of Malta and supervised by Prof. Malvina Nissim and Prof. Albert Gatt. The thesis work was published under the title "Unmasking Contextual Stereotypes: Measuring and Mitigating BERT's Gender Bias" as part of the 2nd Workshop on Gender Bias in Natural Language Processing at COLING 2020. ArXiv preprint: https://arxiv.org/abs/2010.14534
```
@inproceedings{bartl2020unmasking,
  title={Unmasking Contextual Stereotypes: Measuring and Mitigating BERT's Gender Bias},
  author={Bartl, Marion and Nissim, Malvina and Gatt, Albert},
  editor={Costa-jussà, Marta R. and Hardmeier, Christian and Webster, Kellie and Radford, Will},
  booktitle={Proceedings of the Second Workshop on Gender Bias in Natural Language Processing},
  year={2020}
}

```

## BEC-Pro

We created the **Bias Evaluation Corpus with Professions (BEC-Pro)**. This corpus is designed to measure gender bias for different groups of professions and contains English and German sentences built from templates. The corpus files are `BEC-Pro_EN.tsv` for English and `BEC-Pro_DE.tsv` for German.

## Code

In the code folder, run `main.py`, which requires the following arguments:
```
usage: main.py [-h] --lang LANG --eval EVAL [--tune TUNE] --out OUT [--model MODEL] [--batch BATCH]
               [--seed SEED]

  -h, --help     show this help message and exit
  --lang LANG    provide language, either EN or DE
  --eval EVAL    .tsv file with sentences for bias evaluation (BEC-Pro or transformed EEC)
  --tune TUNE    .tsv file with sentences for fine-tuning (GAP flipped)
  --out OUT      output directory and start of filename
  --model MODEL  which BERT model to use 
  --batch BATCH  fix batch-size for fine-tuning 
  --seed SEED
```

For English, run:
```
python3 main.py --lang EN --eval ../BEC-Pro/BEC-Pro_EN.tsv --tune ../data/gap_flipped.tsv --out ../data/results
```
For German, run:
```
python3 main.py --lang DE --eval ../BEC-Pro/BEC-Pro_DE.tsv --out ../data/results
```

<br/><br/>

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
