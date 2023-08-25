import sys
sys.path.insert(0, './stereotypes')
from stereotypes.templates import Templates
from stereotypes.lists import Lists
import hashlib
import argparse
import csv


lists = Lists("word_lists", None)
templates = Templates(lists)

people_types = list(lists.people.keys())
adjective_types = list(lists.adjectives.keys())



parser = argparse.ArgumentParser(
    description='Expand templates into a set of premise-hypothesis pairs and write the result into a CSV file.')

parser.add_argument("--noun", dest="noun", action = "store_const",
                    const = True, default = False,
                    help = 'Use noun templates. With this argument, the premise and hypothesis types should be nouns.')

parser.add_argument("--adj", dest="adj", action = "store_const",
                    const = True, default = False,
                    help = 'Use adjective templates. With this argument, the premise and hypothesis types should be adjectives.')


noun_template_warning = 'If noun templates are used, this argument should be one of [' + ' '.join(people_types) + '].'

adj_template_warning = 'If adjective templates are used, this argument should be one of [' + ' '.join(adjective_types) + '].'

template_warning = noun_template_warning + " " + adj_template_warning

parser.add_argument('--p', dest='premise_type', action='store',
                    help='Premise word type. ' + template_warning,
                    type=str, required = True)

parser.add_argument('--h', dest='hyp_type', action='store',
                    help='Hypothesis word type. ' + template_warning,
                    type=str, required = True)

parser.add_argument("--output", dest="out_file", action= "store",
                    help= 'The name of the output CSV file',
                    required = True)

args = parser.parse_args()

if args.noun:
    word_lists = lists.people
    template_list = templates.noun_templates
elif args.adj:
    word_lists = lists.adjectives
    template_list = templates.adjective_templates
else:
    print("Either --noun or --adj should be chosen")

with open(args.out_file, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["id", "pair type", "premise_filler_word",
                     "hypothesis_filler_word", "template_type",
                     "premise", "hypothesis"])

premise_fillers = word_lists[args.premise_type]
hypothesis_fillers = word_lists[args.hyp_type]
pair_type = args.premise_type + " -> " + args.hyp_type
for p_filler in premise_fillers:
    for h_filler in hypothesis_fillers:
        for p_template, h_template in template_list:
            p_name = ' '.join(["X1_" + args.premise_type,
                               p_template.template_type,
                               "Y"])
            h_name = ' '.join(["X2_" + args.hyp_type,
                               h_template.template_type,
                               "Y"])
            
            tpe = p_name + " -> " + h_name
            p = p_template.apply(p_filler)
            h = h_template.apply(h_filler)

            ex_id = hashlib.sha224((pair_type + "\t" + p + "\t" + h).encode('utf-8')).hexdigest()
            row = [ex_id, pair_type, p_filler, h_filler, tpe, p, h]
            with open(args.out_file, 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)
    


