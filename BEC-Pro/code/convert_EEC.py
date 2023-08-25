# File name: convert_EEC.py
# Description: create new columns to make EEC manageable for calculating associations with BERT language model
# Author: Marion Bartl
# Date: 15-7-2020

import argparse

import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', help='path to EEC corpus file', required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()

    eec = pd.read_csv(args.corpus)
    print("No. of sentences in EEC: {}".format(len(eec)))

    # get the non-emotional(neutral) sentences out of the picture, in new dataframe
    eec = eec.loc[eec.Emotion.notnull()]

    eec_male = eec[eec.Gender == 'male']
    eec_female = eec[eec.Gender == 'female']

    male_words = eec_male.Person.unique()[20:30]
    female_words = eec_female.Person.unique()[20:30]
    person_words = list(female_words) + list(male_words)

    print("No. of selected person words: {}".format(len(person_words)))
    print(list(zip(female_words, male_words)))

    eec = eec.loc[eec['Person'].isin(person_words)]
    eec.reset_index(drop=True, inplace=True)

    assert len(eec.Person.unique()) == len(person_words)

    mask = '[MASK]'
    person_templates = ['<person subject>', '<person object>']
    emotion_templates = ['<emotion word>', '<emotional situation word>']

    Sent_TM = []
    Sent_TAM = []
    for idx, row in eec.iterrows():
        emo_w = row['Emotion word']
        person = row['Person'].lower()
        if len(person.split()) == 1:
            person = person
        else:
            person = person.split()[1]

        sentence = row['Sentence'].lower()

        sent_tm = sentence.replace(person, mask)
        sent_tam = sent_tm.replace(emo_w, mask)

        Sent_TM.append(sent_tm)
        Sent_TAM.append(sent_tam)

    # add sentences with target and attribute masks to EEC dataframe
    eec = eec.assign(Sent_TM=Sent_TM)
    eec = eec.assign(Sent_TAM=Sent_TAM)

    print("No. of sentences in EEC after person word selection: {}".format(len(eec)))

    eec.to_csv("../data/EEC_TM_TAM.tsv", sep='\t')
