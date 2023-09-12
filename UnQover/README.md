# UnQover

Source: [UNQOVERing Stereotyping Biases via Underspecified Questions](https://aclanthology.org/2020.findings-emnlp.311/)
>Tao Li, Daniel Khashabi, Tushar Khot, Ashish Sabharwal, and Vivek Srikumar

Source dataset and documentation: https://github.com/allenai/unqover

```
@inproceedings{li-etal-2020-unqovering,
    title = "{UNQOVER}ing Stereotyping Biases via Underspecified Questions",
    author = "Li, Tao  and
      Khashabi, Daniel  and
      Khot, Tushar  and
      Sabharwal, Ashish  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.311",
    doi = "10.18653/v1/2020.findings-emnlp.311",
    pages = "3475--3489"
}
```

License: Apache License

## About

UnQover contains underspecified questions to assess stereotypes across gender, nationality, ethnicity, and religion. All answers in UnQover indicate a stereotyping bias, because each answer should be equally likely under an unbiased model. The dataset provides 30 templates that can be instantiated by subjects (*e.g.*, names) and attributes (*e.g.*, occupations).

## Data

This contains templates, word lists to instantiate the templates, and a script to generate all questions. We do not generate the full dataset due to its size.