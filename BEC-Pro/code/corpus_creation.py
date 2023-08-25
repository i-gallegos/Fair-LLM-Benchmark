# File name: corpus_creation.py
# Description: file for the creation of a corpus with professions and person words that can be used to
# assess gender bias in BERT
# Author: Marion Bartl
# Date: 30-06-2020

import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--lang', help='language, either EN or DE', required=True)
parser.add_argument('--prof', help="professions JSON file", required=True)
parser.add_argument('--patterns', help='file with sentence patterns', required=True)
args = parser.parse_args()


def make_english_row(prof, word, pattern, gender, prof_gender):
    mask = '[MASK]'
    word = word.capitalize()
    row = []

    # for words such as 'this man' only get 'man'
    if len(word.split()) == 2:
        person = word.split()[1]
    else:
        person = word

    # sentence
    sentence = pattern.format(word, prof)
    row.append(sentence)

    # sentence: masked target
    sent_TM = sentence.replace(person, mask)
    row.append(sent_TM)

    # sentence: masked_attribute
    sent_AM = sentence
    for p in prof.split():
        sent_AM = sent_AM.replace(p, mask)
    row.append(sent_AM)
    # sentence: masked target and attribute
    for p in prof.split():
        sent_TM = sent_TM.replace(p, mask)
    row.append(sent_TM)

    # template
    row.append(pattern.format('<person subject>', '<profession>'))

    # person:
    if len(word.split()) == 2:
        row.append(word.split()[1])
    else:
        row.append(word)

    # gender
    row.append(gender)

    # profession
    row.append(prof)

    # profession's (statistical) gender
    row.append(prof_gender)

    return row


def make_german_row(prof, word, pattern, gender, prof_gender):
    mask = '[MASK]'
    # Word is always put in at the beginning of the sentence, therefore it is capitalized
    word = ' '.join([word.capitalize() for word in word.split()])
    row = []

    # for words such as 'this man' only get 'man'
    if len(word.split()) == 2:
        person = word.split()[1]
    else:
        person = word

    # sentence
    if gender == 'male':
        pattern = pattern.replace('ART', 'der')
        sentence = pattern.format(word, prof)
    elif gender == 'female':
        pattern = pattern.replace('ART', 'die')
        sentence = pattern.format(word, prof)

    row.append(sentence)

    # sentence: masked target
    sent_TM = sentence.replace(person, mask)
    row.append(sent_TM)

    # sentence: masked_attribute
    sent_AM = sentence
    for p in prof.split():
        sent_AM = sent_AM.replace(p, mask)
    row.append(sent_AM)
    # sentence: masked target and attribute
    for p in prof.split():
        sent_TM = sent_TM.replace(p, mask)
    row.append(sent_TM)

    # template
    row.append(pattern.format('<person subject>', '<profession>'))

    # person:
    if len(word.split()) == 2:
        row.append(word.split()[1])
    else:
        row.append(word)

    # gender
    row.append(gender)

    # profession
    row.append(prof)

    # profession's (statistical) gender
    row.append(prof_gender)

    return row


def make_prof_df(prof_list, sentence_patterns, male_words, female_words, lang, prof_gender):
    data = []

    for sp in sentence_patterns:
        for w in male_words:
            for prof in prof_list:
                gender = 'male'
                if lang == 'EN':
                    row = make_english_row(prof, w, sp, gender, prof_gender)
                elif lang == 'DE':  # in the German case, prof_list is a list of lists
                    en_prof = prof[0]
                    prof = prof[1]
                    row = make_german_row(prof, w, sp, gender, prof_gender)
                    row.append(en_prof)
                data.append(row)
        for w in female_words:
            for prof in prof_list:
                gender = 'female'
                if lang == 'EN':
                    row = make_english_row(prof, w, sp, gender, prof_gender)
                elif lang == 'DE':
                    en_prof = prof[0]
                    prof = prof[2]
                    row = make_german_row(prof, w, sp, gender, prof_gender)
                    row.append(en_prof)
                data.append(row)

    if lang == 'DE':
        data = pd.DataFrame(data, columns=['Sentence', 'Sent_TM', 'Sent_AM', 'Sent_TAM',
                                           'Template', 'Person', 'Gender', 'Profession', 'Prof_Gender', 'Profession_EN'])
    elif lang == 'EN':
        data = pd.DataFrame(data, columns=['Sentence', 'Sent_TM', 'Sent_AM', 'Sent_TAM',
                                           'Template', 'Person', 'Gender', 'Profession', 'Prof_Gender'])
    else:
        raise Exception("The language was wrong!")

    return data


if __name__ == '__main__':

    with open(args.prof, 'r', encoding='utf-8') as f:
        professions = json.load(f)

    patterns = [p.strip() for p in open(args.patterns)]

    if args.lang == 'DE':

        male_words = ['er', "dieser Mann", "mein Bruder", "mein Sohn", "mein Ehemann", "mein Freund", "mein Vater",
                      "mein Onkel", 'mein Papa']
        female_words = ["sie", "diese Frau", "meine Schwester", "meine Tochter", "meine Frau",
                        "meine Freundin", "meine Mutter", "meine Tante", "meine Mama"]

        corpus = pd.DataFrame()
        for g in ['male', 'female', 'balanced']:
            df = make_prof_df(professions[g], patterns, male_words, female_words, args.lang, g)
            corpus = corpus.append(df, ignore_index=True)

    elif args.lang == 'EN':

        male_words = ['he', 'this man', 'my brother', 'my son', 'my husband', 'my boyfriend', 'my father', 'my uncle',
                      'my dad']
        female_words = ['she', 'this woman', 'my sister', 'my daughter', 'my wife', 'my girlfriend', 'my mother',
                        'my aunt', 'my mom']

        corpus = pd.DataFrame()
        for g in ['male', 'female', 'balanced']:
            df = make_prof_df(professions[g], patterns, male_words, female_words, args.lang, g)
            corpus = corpus.append(df, ignore_index=True)

    print('The corpus creation was successful!')
    print('The corpus has a length of {} sentences and {} columns'.format(len(corpus), len(corpus.columns)))

    corpus.to_csv('../BEC-Pro/BEC-Pro_' + args.lang + '.tsv', sep='\t')
