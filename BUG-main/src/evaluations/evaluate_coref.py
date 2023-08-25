""" Usage:
    <file-name> [--in=INPUT_FILE] [--out=OUTPUT_FILE] [--debug]

Options:
  --help                           Show this message and exit
  -i INPUT_FILE --in=INPUT_FILE    Input file
                                   [default: ../../predictions/coref_preds.jsonl]
  -o INPUT_FILE --out=OUTPUT_FILE  Input file
                                   [default: ../../visualizations/delta_s_by_dist.png]
  --debug                          Whether to debug
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict
from math import floor
import matplotlib.pyplot as plt
from operator import itemgetter

# Local imports


#----

BIN_SIZE = 5

def find_cluster_ind(clusters, word_ind):
    """
    find the cluster ind for the given word, or -1 if not found
    """
    found_in_clusters = []
    for cluster_ind, cluster in enumerate(clusters):
        for ent_ind, ent in enumerate(cluster):
            ent_start, ent_end = ent
            if (word_ind >= ent_start) and (word_ind <= ent_end):
                # found a cluster
                found_in_clusters.append(cluster_ind)

    # no cluster found
    return found_in_clusters


def is_correct_pred(line):
    """
    return True iff this line represents a correct prediction.
    """
    pred = line["pred"]
    row = line["row"]
    clusters = pred["clusters"]
    ent_id = find_cluster_ind(clusters, row["profession_first_index"])
    pron_id = find_cluster_ind(clusters, row["g_first_index"])
    
    # prediction is correct if it assigns pronoun and entity to the same cluster
    is_correct = len(set(ent_id).intersection(pron_id))
    
    return is_correct


def get_acc(vals):
    """
    return the accuracy given binary scores
    """
    acc = (sum(vals) / len(vals)) * 100
    return acc

def get_delta_s_by_dist(dist_metric):
    """
    Compute delta s by distance
    """
    dists = sorted(dist_metric.keys())
    delta_s_by_dist = []
    for dist in dists:
        cur_metric = dist_metric[dist]
        ste = cur_metric[1]
        ant = cur_metric[-1]
        delta_s_by_dist.append((dist,
                                get_acc(ste) - get_acc(ant),
                                len(ste) + len(ant)))
    return delta_s_by_dist

def get_simple_metric_by_dist(dist_metric, metric_name):
    """
    Aggregate over a given metric by dist.
    """
    dists = sorted(dist_metric.keys())
    met_by_dist = []
    for dist in dists:
        met_by_dist.append(get_acc(dist_metric[dist][metric_name]))
    return met_by_dist

# Simple instantiations
get_acc_by_dist = lambda dist_metric: get_simple_metric_by_dist(dist_metric, "acc")
get_ste_by_dist = lambda dist_metric: get_simple_metric_by_dist(dist_metric, 1)
get_ant_by_dist = lambda dist_metric: get_simple_metric_by_dist(dist_metric, -1)

def average_buckets(b1, b2):
    """
    average two buckets
    """
    b1_ind, b1_val, b1_cnt = b1
    b2_ind, b2_val, b2_cnt = b2

    cnt = b1_cnt + b2_cnt
    val = ((b1_val * b1_cnt) + (b2_val * b2_cnt)) / cnt

    new_bucket = (b1_ind, val, cnt)
    return new_bucket
    
if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = Path(args["--in"])
    out_fn = Path(args["--out"])

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    logging.info(f"Input file: {inp_fn}, Output file: {out_fn}.")

    # Start computation
    metrics = {"acc": [],
               "ste": [],
               "ant": [],
               "masc": [],
               "femn": [],
               "num_of_pronouns": defaultdict(list),
               "distance": defaultdict(lambda: defaultdict(list))}

    for line in tqdm(open(inp_fn, encoding = "utf8")):
        line = json.loads(line.strip())
        is_correct = is_correct_pred(line)
        row = line["row"]
        metrics["acc"].append(is_correct)
        gender = row["predicted gender"].lower()
        stereotype = row["stereotype"]
        
        if gender == "male":
            metrics["masc"].append(is_correct)
        elif gender == "female":
            metrics["femn"].append(is_correct)

        if stereotype == 1:
            metrics["ste"].append(is_correct)
        elif stereotype == -1:
            metrics["ant"].append(is_correct)

        num_of_prons = row["num_of_pronouns"]
        dist = floor(row["distance"] / BIN_SIZE)
        metrics["num_of_pronouns"][num_of_prons].append(is_correct)
        metrics["distance"][dist][stereotype].append(is_correct)
        metrics["distance"][dist]["acc"].append(is_correct)
            
        
    acc = get_acc(metrics["acc"])
    delta_g = get_acc(metrics["masc"]) - get_acc(metrics["femn"])
    delta_s = get_acc(metrics["ste"]) - get_acc(metrics["ant"])
    delta_s_by_dist = get_delta_s_by_dist(metrics["distance"])
    acc_by_dist = get_acc_by_dist(metrics["distance"])
    ste_by_dist = get_ste_by_dist(metrics["distance"])
    ant_by_dist = get_ant_by_dist(metrics["distance"])

    # average last bucket
    # delta_s_by_dist[-2] = average_buckets(delta_s_by_dist[-2], delta_s_by_dist[-1])
    # delta_s_by_dist = delta_s_by_dist[:-1]

    logging.info(f"acc = {acc:.1f}; delta_g = {delta_g:.1f}; delta_s = {delta_s:.1f}")
    logging.info(f"delta s by dist = {delta_s_by_dist}")
    logging.info(f"acc by dist = {acc_by_dist}")

    # plot
    plt.rcParams.update({'font.size': 15})
    ranges = [f"{(x*5) + 1}-{(x +1) * 5}" for x in  map(itemgetter(0), delta_s_by_dist)]
    ranges[-1] = ranges[-1].split("-")[0] + ">"
    values_ds = list(map(itemgetter(1), delta_s_by_dist))
    y_pos = np.arange(len(ranges))
    width = 1 # the width for the bars

    plt.plot(y_pos, ste_by_dist, label = "stereotypical",
             color = "orange", linestyle = "dashed")
    plt.scatter(y_pos, ste_by_dist, color = "orange")

    plt.plot(y_pos, ant_by_dist, label = "anti-stereotypical",
             color = "blue", linestyle = "dotted")
    plt.scatter(y_pos, ant_by_dist, color = "blue")

    plt.xticks(y_pos, ranges)
    plt.ylabel("coreference acc")
    plt.xlabel("distance [words] between pronoun and antecedent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fn)

    
    # End
    logging.info("DONE")
