# Data Statements

## Curation Rationale

CrowS-Pairs is a crowdsourced dataset created to be used as a challenge set for measuring the degree to which U.S. stereotypical biases are present in large pretrained masked language models such as [BERT](https://www.aclweb.org/anthology/N19-1423/). The dataset consists of 1,508 examples that cover stereotypes dealing with nine type of social bias. Each example consists of a pair of sentences, where one sentence is always about a historically disadvantaged group in the United States and the other sentence is about a contrasting advantaged group. The sentence about a historically disadvantaged group can \textit{demonstrate} or \textit{violate} a stereotype. The paired sentence is a minimal edit of the first sentence: The only words that change between them are those that identify the group.

We collected this data through Amazon Mechanical Turk, where each example was written by a crowdworker and then validated by five other crowdworkers. We required all workers to be in the United States, to have completed at least 5,000 HITs, and to have greater than 98\% acceptance rate. We use the Fair Work tool \citep{fairwork} to ensure a minimum of \$15 hourly wage.

## Language Variety

We do not collect information on the varieties of English that workers use to create examples. However, as we require them to be in the United States, we assume that most of the examples are written in US-English (en-US). Manual analysis reveals that most, if not all, sentences in this dataset are written in Standard American English, and not African American Vernacular English (AAVE). 

## Speaker Demographic

We do not collect demographic information of the speakers, but we require them to be in the United States.

## Annotator Demographic

We do not collect demographic information of the annotators, but we require them to be in the United States.

## Speech Situation

N/A

## Text Characteristics

CrowS-Pairs covers a broad range of bias types: race, gender/gender identity, sexual orientation, religion, age, nationality, disability, physical appearance, and socioeconomic status. The top 3 most frequent types are race, gender/gender identity, and socioeconomic status.

## Recording Quality

N/A

## Other

This dataset contains statements that may be highly offensive and sensitive in nature. It should not be used to train any machine learning models. The main purpose of CrowS-Pairs is to serve as an evaluation set to measure the degree to which stereotypical biases are present in language models, a step towards building more fair NLP systems.

We aware of the risk of publishing CrowS-Pairs, especially given its limited scope and a single numeric bias measurement that we proposed. We strongly caution against the assumption that a lower score on our data means that a model is completely unbiased.  

## Provenance Appendix

N/A