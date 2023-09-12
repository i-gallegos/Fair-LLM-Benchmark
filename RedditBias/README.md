# RedditBias

Source: [RedditBias: A Real-World Resource for Bias Evaluation and Debiasing of Conversational Language Models](https://aclanthology.org/2021.acl-long.151/)
>Soumya Barikeri, Anne Lauscher, Ivan Vulić, and Goran Glavaš

Source dataset and documentation: https://github.com/umanlp/RedditBias

```
@inproceedings{barikeri-etal-2021-redditbias,
    title = "{R}eddit{B}ias: A Real-World Resource for Bias Evaluation and Debiasing of Conversational Language Models",
    author = "Barikeri, Soumya  and
      Lauscher, Anne  and
      Vuli{\'c}, Ivan  and
      Glava{\v{s}}, Goran",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.151",
    doi = "10.18653/v1/2021.acl-long.151",
    pages = "1941--1955"
}
```

License: MIT License

## About

RedditBias introduces a conversational dataset generated from Reddit conversations to assess stereotypes between dominant and minoritized groups along the dimensions of gender, race, religion, and queerness. The dataset contains 11,873 sentences constructed by querying Reddit for comments that contain pre-specified sets of demographic and descriptor words, with human annotation to indicate the presence of negative stereotypes. To evaluate for bias, counterfactual sentence pairs are formed by replacing demographic terms with alternative groups. 

## Data

This contains several files for each social group.

Descriptors:
- `[category]_[social_group].txt`: Stereotypical negative descriptors
- `[category]_[social_group]_pos.txt`: Positive descriptors
- `[category]_opposites.txt`: Corresponding terms for two social groups

Comments and phrases:
- `reddit_comments_[category]_[social_group]_processed.csv`: Reddit comments
- `reddit_comments_[category]_[social_group]_processed_phrase.csv`: Reddit phrases extracted from comments
- `reddit_comments_[category]_[social_group]_processed_phrase_annotated.csv`: Bias annotations for Reddit comments and phrases

Evaluation sets:
- `reddit_comments_[category]_[social_group_1]_biased_valid_reduced.csv`, `reddit_comments_[category]_[social_group_2]_biased_valid_reduced.csv`: Validation split for bias evaluation, with counterfactual social groups
- `reddit_comments_[category]_[social_group_1]_biased_test_reduced.csv`, `reddit_comments_[category]_[social_group_2]_biased_test_reduced.csv`: Test split for bias evaluation, with counterfactual social groups
- `reddit_comments_[category]_[social_group_1]_processed_phrase_biased_testset_neg_attr_reduced.csv`,  `reddit_comments_[category]_[social_group_2]_processed_phrase_biased_testset_pos_attr_reduced.csv`: Test split for bias evaluation, with counterfactual descriptors