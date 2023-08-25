# Gender Representation-Bias for Information Retrieval (`GrepBiasIR`)

The `GrepBiasIR` provides a set of *bias-sensitive queries*, namely the gender-neutral queries for which biases in their retrieval results are considered socially problematic. The queries cover 7 gender dimensions on topics such as physical capabilities and child care. Each query is also accompanied by one relevant and one non-relevant document, where each document is expressed in neutral, male, and female wording.

```
@inproceedings{krieg2022grep,
  title={Grep-BiasIR: a dataset for investigating gender representation-bias in information retrieval results},
  author={Krieg, Klara and Parada-Cabaleiro, Emilia and Medicus, Gertraud and Lesota, Oleg and Schedl, Markus and Rekabsaz, Navid},
  booktitle={Proceeding of the 2023 ACM SIGIR Conference On Human Information Interaction And Retrieval (CHIIR)},
  year={2022}
}
```
Preprint: https://arxiv.org/pdf/2201.07754.pdf


## `GrepBiasIR` files

The dataset consists of the *queries.csv* file comprising all queries, and seven files with respective documents corresponding to the seven topics. The formatting of these files are explained below: 

**queries.csv:**
* `q_id` - unique ID of the query
* `category` - query category (one of seven)
* `query` - query text

**queries-documents_[CATEGORY].csv:**

[CATEGORY] - one of the seven query categories
* `q_id` - unique ID of the query
* `d_id` - unique ID of the document
* `relevant` - document to query relevance judgement (1 - relevant, 0 - not relevant)
* `query` - query text
* `doc_title` - title of the document
* `document` - text of the document
* `content_gender` - gender indication inferred from the text of the document (F - female, M - male, N - neutral)
* `exp_stereotype` - expected stereotype annotation

## *Show me a "Male Nurse"!* dataset
Using `GrepBiasIR`, Kopeinik et al. (citation below) conduct a user study to observe and measure the potential biases of the search engines' users, when formulating queries on gender-sensitive topics. The dataset consisting of these formulated queries is available here: https://github.com/CPJKU/user-interaction-gender-bias-IR

```
@inproceedings{Kopeinik2023Show,
  title={Show me a "Male Nurse"! How Gender Bias is Reflected in the Query Formulation of Search Engine Users},
  author={Kopeinik, Simone and Mara, Martina and Ratz, Linda and Krieg, Klara and Schedl, Markus and Rekabsaz, Navid},
  booktitle={Proceeding of the ACM Conference on Human Factors in Computing Systems (CHI),},
  year={2023}
}
```
