""" Usage:
    <file-name> [--in=INPUT_FILE] [--out=OUTPUT_FILE] [--debug]

Options:
  --help                           Show this message and exit
  -i INPUT_FILE --in=INPUT_FILE    Input file
                                   [default: ../data/full_bug.csv]
  -o INPUT_FILE --out=OUTPUT_FILE  Input file
                                   [default: outfile.tmp]
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
import pandas as pd
import numpy as np


# Local imports


#----

def dist_between_prof_ant(df):
    """
    Return average dist and std between profession and
    antecedent in the given data frame.
    """
    dist = np.abs(df.profession_first_index - df.g_first_index)
    ave = np.average(dist)
    std = np.std(dist)
    dist_dict = {"average": ave,
                 "std": std}
    return dist_dict

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

    # Load data
    df = pd.read_csv(inp_fn)

    # Compute distance between pronoun and antecedent between
    # dataset partitions
    ste = df[df.stereotype == 1]
    ant = df[df.stereotype == -1]

    ste_dist = dist_between_prof_ant(ste)
    ant_dist = dist_between_prof_ant(ant)
    
    logging.info("stereotype dist: {}".format(pformat(ste_dist)))
    logging.info("anti-stereotype dist: {}".format(pformat(ant_dist)))

    
    # End
    logging.info("DONE")
