from .underspecified_templates import *
from .lists import Lists
import argparse
import csv
import json

def get_subject_pairs(opt, lists):
    rs = None

    if 'mixed_gender' in opt.subj:
        if opt.subj == 'mixed_gender':
            rs = lists.get_mixed_gender_pairs(f_file='female', m_file='male', add_flip=True)
        elif opt.subj.startswith('mixed_gender_'):
            suffix = opt.subj.split('mixed_gender_')[1]
            rs = lists.get_mixed_gender_pairs(f_file='female_'+suffix, m_file='male_'+suffix, add_flip=True)
    else:
        if opt.subj in lists.subjects:
            rs = lists.get_subject_pairs(opt.subj, add_flip=True)
        else:
            rs = lists.get_people_pairs(opt.subj, add_flip=True)
    return rs


def get_fillers(opt, lists):
    rs = ''
    if opt.filler.strip() == '' or opt.filler.strip().lower() == 'none':
        return rs

    types = opt.filler.split(',')
    for t in types:
        filler = lists.fillers[t]
        assert(len(filler) == 1)    # only support one filler each type
        rs += '{} '.format(filler[0])
    return rs


# make each line like:
#   John and Will wend to an Italian resturant, one ordered a pizza and the other ordered a drink.  [CHOICE] pizza | drink  [Q] Who ordered a pizza?    [Q] Who ordered a drink?
def compose_subj_act_bij_line(context, choices, spair, tid, apair, s_clusters, act_cluster, q_ls):
    line = context + '\t' + '|'.join(choices) + '\t' + '\t'.join(q_ls)
    return '{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}\t'.format(s_clusters[0], s_clusters[1], spair[0], spair[1], tid, act_cluster, apair[0], apair[1]) + line

def compose_subj_act_bij_json(context, choices, spair, tid, apair, s_clusters, act_cluster, q_ls):
    rs = {}
    key = '{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}'.format(s_clusters[0], s_clusters[1], spair[0], spair[1], tid, act_cluster, apair[0], apair[1])
    rs[key] = {}
    rs[key]['context'] = context
    rs[key]['q0'] = {'question': q_ls[0], 'pred': '', 'ans0': {'text': choices[0], 'start': '', 'end': ''}, 'ans1': {'text': choices[1], 'start': '', 'end': ''}} 
    rs[key]['q1'] = {'question': q_ls[1], 'pred': '', 'ans0': {'text': choices[0], 'start': '', 'end': ''}, 'ans1': {'text': choices[1], 'start': '', 'end': ''}}
    return rs


parser = argparse.ArgumentParser(
    description='Expand templates into a set of examples, each consists of a context and two questions and two answer choices.')
parser.add_argument("--template_type", help= 'The type of templates', required = True)
parser.add_argument("--subj", help= 'The type of subjects (located in word_lists/nouns/people/, or word_lists/nouns/subjects/, or some customized types', required = True)
parser.add_argument("--obj", help= 'The type of objects (located in word_lists/nouns/objects/', required = False, default='')
parser.add_argument("--verb", help= 'The type of objects (located in word_lists/verbs/', required = False, default='')
parser.add_argument("--subj_excl", help= 'The list of words to be excluded from subjects, separated by comma', required = False, default='')
parser.add_argument("--obj_excl", help= 'The list of words to be excluded from objects, separated by comma', required = False, default='')
parser.add_argument("--act", help= 'The type of activities (located in word_lists/activities/', required = False, default='')
#parser.add_argument("--hyper_act", help= 'The hyper activity', required = False, nargs='+', default='')
parser.add_argument("--wh_type", help= 'The type of questions: which/what/who', required = False)
parser.add_argument("--output", help= 'The name of the output txt file', required = True)
#
parser.add_argument("--filler", help= 'The type of fillers, separated by comma, located in word_lists/fillers', required = False, default='')
#
parser.add_argument("--slot", help= 'The type of slot template to use, located in word_lists/slots', required = False, default='')
parser.add_argument("--lm_mask", help= 'the mask token for lm tempaltes, defaulted to roberta <mask>', required = False, default='<mask>')

opt = parser.parse_args()
rs = {}

lists = Lists("word_lists", None)
spairs = get_subject_pairs(opt, lists)
filler = get_fillers(opt, lists)
templates = UnderspecifiedTemplates()

if opt.template_type == 'slot_act_map':
    act_repo = lists.activities[opt.act]
    slots = lists.slots[opt.slot]
    filler = None if opt.filler == '' else filler

    templates.spawn_slot_act_mapping(slots, spairs, act_repo, filler, opt.lm_mask)

    cnt = 0
    for t in templates.subj_templates:
        context, q1, q2 = t.apply()
        ex_json = compose_subj_act_bij_json(context, t.actual_spair, t.unique_spair, t.tid, (t.unique_act, t.unique_act), (t.spair[0]['cluster'], t.spair[1]['cluster']), t.act_cluster, (q1, q2))
        rs.update(ex_json)

        cnt += 1
        if cnt % 10000 == 0:
            print('generated {0} lines'.format(cnt))
    print('generated {0} lines'.format(cnt))

else:
    raise Exception('unrecognized template_type {0}'.format(opt.template_type))

print('writing to {0}'.format(opt.output))
json.dump(rs, open(opt.output, 'w'), indent=4)
