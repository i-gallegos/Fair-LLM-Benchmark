# TrustGPT

Source: [TrustGPT: A Benchmark for Trustworthy and Responsible Large Language Models](https://arxiv.org/abs/2306.11507)
>Yue Huang, Qihui Zhang, Philip S. Y, and Lichao Sun

Source dataset and documentation: https://github.com/HowieHwong/TrustGPT

```
@article{huang2023trustgpt,
  title={TrustGPT: A Benchmark for Trustworthy and Responsible Large Language Models},
  author={Huang, Yue and Zhang, Qihui and Sun, Lichao and others},
  journal={arXiv preprint arXiv:2306.11507},
  year={2023}
}
```

License: MIT License

## About

TrustGPT provides prompts to evaluate toxicity and performance disparities between demographic groups. For toxicity evaluation, prompts asks the model to state something toxic, bad, or harmful, given some social norm; the completion can be scored for its level of toxicity. For bias evaluation, prompts ask for a toxic completion given both a demographic group and social norm; completions can be compared between demographic groups to measure disparities in toxicity levels.

## Data

Instead of providing a dataset, TrustGPT provides prompts that can be instantiated by the user using the provided code.