# author: Marion Bartl
# date: 13-05-2020
# description: take professions as a tsv file
# and load them into json file, to retrieve them as dictionaries later

import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('file', help='filename with professions')
args = parser.parse_args()

if __name__ == '__main__':
    professions_en = dict()
    professions_de = dict()
    prof_df = pd.read_csv(args.file, sep='\t')

    ### The English case
    jobs_en = prof_df['simplified_professions']
    # check that there are 60 professions in total
    assert len(jobs_en) == 60, "Too many jobs"
    # split by order
    professions_en['female'] = list(jobs_en[:20])
    professions_en['male'] = list(jobs_en[20:40])
    professions_en['balanced'] = list(jobs_en[40:])

    ### The German case
    # get English profession and German masculine and feminine form of professions
    jobs_de = list(zip(jobs_en, prof_df['german_m'], prof_df['german_f']))
    # check that there are 60 professions in total here, too
    assert len(jobs_de) == 60, "Too many jobs"

    # split by order
    professions_de['female'] = jobs_de[:20]
    professions_de['male'] = jobs_de[20:40]
    professions_de['balanced'] = jobs_de[40:]

    # Write professions to JSON files
    with open('../BEC-Pro/english_professions.json', 'w', encoding='utf-8') as fn:
        json.dump(professions_en, fn)

    with open('../BEC-Pro/german_professions.json', 'w', encoding='utf-8') as fn:
        json.dump(professions_de, fn)
