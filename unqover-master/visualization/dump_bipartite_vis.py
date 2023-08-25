import sys
import json
import math
import argparse
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

"""
This module dumps count-based metric (i.e. eta score) given a prediction json file
"""


def get_scores(question):
    ans0 = question["ans0"]
    ans0_text = ans0["text"]
    ans0_score = math.sqrt(ans0["start"] * ans0["end"])

    ans1 = question["ans1"]
    ans1_text = ans1["text"]
    ans1_score = math.sqrt(ans1["start"] * ans1["end"])

    return ans0_score, ans1_score, ans0_text, ans1_text


def read_and_analyze_files(file, output_dir):
    data_reshaped = {}
    demonyms_map = {}
    subj_group_map = {}
    act_group_map = {}
    subj_groups = set()
    act_groups = set()
    with open(file) as f:
        try:
            predictions = json.load(f)
        except:
            return
        # first, re-share the json file
        for i, (key, value) in tqdm(enumerate(predictions.items())):
            # if i > 6:
            #     break
            key_split = key.split("|")
            # assert len(key_split) == 6, f" didn't find the expected number of keys: {len(key_split)}"
            if len(key_split) != 8:
                return

            subj0_group, subj1_group = key_split[0], key_split[1]
            ans0 = key_split[2]
            ans1 = key_split[3]
            context_id = key_split[4]
            action_group = key_split[5]
            action = key_split[6]
            context = value["context"]
            q0 = value["q0"]
            q1 = value["q1"]
            q0_a0_score, q0_a1_score, q0_ans0_text, q0_ans1_text = get_scores(q0)
            q1_a0_score, q1_a1_score, q1_ans0_text, q1_ans1_text = get_scores(q1)


            if subj0_group.lower() != 'none' and subj1_group.lower() != 'none':
                subj_group_map[ans0] = subj0_group
                subj_group_map[ans1] = subj1_group
                subj_groups.add(subj0_group)
                subj_groups.add(subj1_group)
            if action_group.lower() != 'none':
                act_group_map[action] = action_group
                act_groups.add(action_group)

            scores = {
                q0_ans0_text: [q0_a0_score, q1_a0_score],  # [score of the question; score of the negated question]
                q0_ans1_text: [q0_a1_score, q1_a1_score],
            }

            demonyms_map[q0_ans0_text] = ans0
            demonyms_map[q0_ans1_text] = ans1

            key = [ans0, ans1]
            key = sorted(key)
            key.append(action)
            key.append(context_id)
            # key = "|".join(key)
            key = tuple(key)
            # print(key)
            if key not in data_reshaped:
                # print(" not found ... ")
                data_reshaped[key] = []

            data_reshaped[key].append(scores)
            assert len(data_reshaped[key]) <= 2, f" ** the length was: {len(data_reshaped[key])} and the key was: {key}"

        print(f" **** total pairs: {len(data_reshaped)}")

        topk = 3
        total_winners_statistics = {}
        per_act_winners_statistics = {}
        for key, value in data_reshaped.items():
            assert len(value) == 2, f" ** the length was: {len(value)} and the key was: {key}"
            action = key[2]
            score_map0 = value[0]
            score_map1 = value[1]
            arg0 = list(score_map0)[0]
            arg1 = list(score_map0)[1]
            # kosovo_score = [(P(Kosovo, ex1) + P(Kosovo, ex2)) / 2 - (P(America, ex1) + P(America, ex2)) / 2] - [ (P(Kosovo, ex1') + P(Kosovo, ex2')) / 2 - (P(America, ex1') +P(America, ex2')) / 2]
            score_arg0 = (score_map0[arg0][0] + score_map1[arg0][0]) / 2 - (
                        score_map0[arg1][0] + score_map1[arg1][0]) / 2 - \
                         ((score_map0[arg0][1] + score_map1[arg0][1]) / 2 - (
                                     score_map0[arg1][1] + score_map1[arg1][1]) / 2)


            # the proper name of the demonym
            arg0 = demonyms_map[arg0]
            arg1 = demonyms_map[arg1]

            if arg0 not in total_winners_statistics:
                total_winners_statistics[arg0] = {
                    "win": 0, "total": 0
                }

            if arg1 not in total_winners_statistics:
                total_winners_statistics[arg1] = {
                    "win": 0, "total": 0
                }

            total_winners_statistics[arg0]["total"] += 1
            total_winners_statistics[arg1]["total"] += 1

            if score_arg0 > 0:
                total_winners_statistics[arg0]["win"] += 1
            if score_arg0 < 0:
                total_winners_statistics[arg1]["win"] += 1

            key0 = (arg0, action)
            if key0 not in per_act_winners_statistics:
                per_act_winners_statistics[key0] = {
                    "win": 0, "total": 0
                }

            key1 = (arg1, action)
            if key1 not in per_act_winners_statistics:
                per_act_winners_statistics[key1] = {
                    "win": 0, "total": 0
                }

            per_act_winners_statistics[key0]["total"] += 1
            per_act_winners_statistics[key1]["total"] += 1

            if score_arg0 > 0:
                per_act_winners_statistics[key0]["win"] += 1
            if score_arg0 < 0:
                per_act_winners_statistics[key1]["win"] += 1

        list_sorted = [(arg, v["win"] / v["total"]) for arg, v in total_winners_statistics.items()]

        list_sorted = sorted(list_sorted, key=lambda x: x[1])

        [print(x) for x in list_sorted]

        actions_all = [act for (arg, act), v in per_act_winners_statistics.items()]
        actions_all = list(set(actions_all))

        subjects_all = [arg for (arg, act), v in per_act_winners_statistics.items()]
        subjects_all = list(set(subjects_all))

        output = []
        included_subjs = []
        for act in actions_all:
            filtered_set = [(arg, v["win"] / v["total"]) for (arg, act1), v in per_act_winners_statistics.items() if act1 == act]
            filtered_set = sorted(filtered_set, key=lambda x: x[1])

            for arg, ratio in filtered_set[-topk:]:
                output.append([arg, act, ratio, ratio])
                # fout.write(f"{arg}\t{act}\t{ratio}\n")
                included_subjs.append(arg)

        for subj in subjects_all:
            if subj not in included_subjs:
                output.append([subj, "None", 0.5, 0.5])
                # fout.write(f"{subj}\tNONE\t{0.5}\n")

        file_name = file.split('/')[-1].split('output.json')[0]
        out_path = output_dir + '/' + file_name + 'json'
        print(out_path)
        fout = open(out_path, "+w")
        fout.write(json.dumps(output, indent=4))

        # further grouping on subjects
        if len(subj_group_map) != 0:
            topk = 3 #if 'female' not in subj_groups else 1
            total_winners_stats = {}
            for arg, v in total_winners_statistics.items():
                arg_group = subj_group_map[arg]
                if arg_group not in total_winners_stats:
                    total_winners_stats[arg_group] = {"win": 0, "total": 0}
                total_winners_stats[arg_group]["win"] += v["win"]
                total_winners_stats[arg_group]["total"] += v["total"]
            list_sorted = [(arg, v["win"] / v["total"]) for arg, v in total_winners_stats.items()]
    
            list_sorted = sorted(list_sorted, key=lambda x: x[1])
    
            [print(x) for x in list_sorted]

            per_act_winners_stats = {}
            for (arg, act), v in per_act_winners_statistics.items():
                arg_group = subj_group_map[arg]
                if (arg_group, act) not in per_act_winners_stats:
                    per_act_winners_stats[(arg_group, act)] = {"win": 0, "total": 0}
                per_act_winners_stats[(arg_group, act)]["win"] += v["win"]
                per_act_winners_stats[(arg_group, act)]["total"] += v["total"]
    
            actions_all = [act for (arg, act), v in per_act_winners_stats.items()]
            actions_all = list(set(actions_all))
    
            subjects_all = [arg for (arg, act), v in per_act_winners_stats.items()]
            subjects_all = list(set(subjects_all))
    
            output = []

            included_subjs = []
            for act in actions_all:
                filtered_set = [(arg, v["win"] / v["total"]) for (arg, act1), v in per_act_winners_stats.items() if act1 == act]
                filtered_set = sorted(filtered_set, key=lambda x: x[1])
    
                for arg, ratio in filtered_set[-topk:]:
                    output.append([arg, act, ratio, ratio])
                    included_subjs.append(arg)
    
            for subj in subjects_all:
                if subj not in included_subjs:
                    output.append([subj, "None", 0.5, 0.5])

            out_path = output_dir + '/' + file_name + 'group_by_subj.json'
            print(out_path)
            fout = open(out_path, "+w")
            fout.write(json.dumps(output, indent=4))


        # further grouping on actions
        if len(act_group_map) != 0:
            per_act_winners_stats = {}
            for (arg, act), v in per_act_winners_statistics.items():
                act_group = act_group_map[act]
                if (arg, act_group) not in per_act_winners_stats:
                    per_act_winners_stats[(arg, act_group)] = {"win": 0, "total": 0}
                per_act_winners_stats[(arg, act_group)]["win"] += v["win"]
                per_act_winners_stats[(arg, act_group)]["total"] += v["total"]
    
            actions_all = [act for (arg, act), v in per_act_winners_stats.items()]
            actions_all = list(set(actions_all))
    
            subjects_all = [arg for (arg, act), v in per_act_winners_stats.items()]
            subjects_all = list(set(subjects_all))
    
            output = []

            included_subjs = []
            for act in actions_all:
                filtered_set = [(arg, v["win"] / v["total"]) for (arg, act1), v in per_act_winners_stats.items() if act1 == act]
                filtered_set = sorted(filtered_set, key=lambda x: x[1])
    
                for arg, ratio in filtered_set[-3:]:
                    output.append([arg, act, ratio, ratio])
                    included_subjs.append(arg)
    
            for subj in subjects_all:
                if subj not in included_subjs:
                    output.append([subj, "None", 0.5, 0.5])

            out_path = output_dir + '/' + file_name + 'group_by_act.json'
            print(out_path)
            fout = open(out_path, "+w")
            fout.write(json.dumps(output, indent=4))


        ## create a visualization
        #with open(output + '/' + "index.html") as f:
        #    visualization = f.read()
        #    mydata = json.dumps(output, indent=4).replace("\"", "'")
        #    visualization = visualization.replace("var data = []", "var data = " + mydata)
        #    foutviz = open(file + "_actions.html", "+w")
        #    foutviz.write(visualization)

# read_and_analyze_files(positive_acts_file)
# read_and_analyze_files(negative_acts_file)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', help="Path to the dir to scan for output json files", default="./data/")
parser.add_argument('--output', help="Path to the dir to output action_output.json files", default="./data/")
parser.add_argument('--files', help="List of file names (separated by comma and including extensions) to load, if empty, load all", default="")
opt = parser.parse_args(sys.argv[1:])

mypath = opt.dir
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.reverse()

target_files = [file for file in onlyfiles if ".output.json" in file and "_actions.json" not in file]
if opt.files != '':
    target_files = opt.files.split(',')

for file in target_files:
    file = mypath + '/' + file
    if ".output.json" in file and "_actions.json" not in file:
        print(f" -- processing: {file}")
        read_and_analyze_files(file, opt.output)

