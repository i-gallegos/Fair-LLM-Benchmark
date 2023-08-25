""" Usage:
    <file-name> [--in=INPUT_FILE] [--out=OUTPUT_FILE] [--debug]

Options:
  --help                           Show this message and exit
  -i INPUT_FILE --in=INPUT_FILE    Input file
                                   [default: infile.tmp]
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
import json
import pandas as pd

# Local imports


#----


HEADER = "#begin document ({doc_name}); part 000"
FOOTER = "\n#end document"
BOILER = ["-"] * 5 + ["Speaker#1"] + ["*"] * 4
ENTITY = "(1)"
BLIST = [13055, 13996, ] # indices which the converter doesn't like for some reason
GENRE = "nw" # conll parsing requires some genre, "nw" follows the convention
             # in winobias, but is probably arbitrary otherwise.


def validate_row(row):
    """
    run sanity checks on row, return true iff they pass
    """
    prof_ind = row.profession_first_index
    pron_ind = row.g_first_index
    
    words = row.sentence_text.lstrip().rstrip().split(" ")
    num_of_words = len(words)
    prof = row["profession"].lower()
    pron = row["g"].lower()


    # make sure inner references in the line make sense
    if prof_ind >= len(words):
        return False
    if pron_ind >= len(words):
        return False
    if words[prof_ind].lower() != prof:
        logging.debug(f"prof doesn't match")
        return False
    if words[pron_ind].lower() != pron:
        logging.debug(f"pron longer than a single word")
        return False

    # don't deal with weird empty tokens
    if any([(str.isspace(word) or (not word)) for word in words]):
        return False

    # all tests passed
    return True
    

def convert_row_to_conll(row, doc_name):
    """
    get a conll multi-line string representing a csv row
    """
    # find prof_index
    prof_ind = row.profession_first_index

    # find pronoun
    pron_ind = row.g_first_index

    # construct conll rows
    conll = []
    words = row.sentence_text.lstrip().rstrip().split(" ")
    prof = row["profession"].lower()
    pron = row["g"].lower()
    for word_ind, word in enumerate(words):
        word_lower = word.lower()
        coref_flag = "-"
        
        if word_ind == prof_ind:
            coref_flag = ENTITY

        elif word_ind == pron_ind:
            coref_flag = ENTITY
                   
        metadata = list(map(str, [doc_name, 0, word_ind, word]))
        conll_row = metadata + BOILER + [coref_flag]
        conll.append("\t".join(conll_row))


    conll_data_str = "\n".join(conll)
    header = HEADER.format(doc_name = doc_name)
    full_conll = "\n".join([header,conll_data_str,FOOTER])
    return full_conll
    

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
    df = pd.read_csv(inp_fn)
    top_doc_name = out_fn.stem

    err_cnt = 0
    
    with open(out_fn, "w", encoding = "utf8") as fout:
        for row_index, row in tqdm(df.iterrows()):
            try:
                valid = validate_row(row)
            except:
                err_cnt += 1
                continue
            if not valid:
                # Something is wrong with this row
                # recover and continue
                err_cnt += 1
                continue

            if row_index in BLIST:
                err_cnt += 1
                continue
            
            doc_name = f"{GENRE}/{top_doc_name}/{row_index}"
            try:
                conll = convert_row_to_conll(row, doc_name)
            except:
                err_cnt += 1
                continue
            fout.write(f"{conll}\n")

    total_rows = len(df)
    perc = round((err_cnt / total_rows)*100)
    rows_written = total_rows - err_cnt
    logging.debug(f"""Wrote a total of {rows_written} to {out_fn}.
    Filtered out {err_cnt} ({perc}%) rows out of {total_rows} total rows.""")
    
    # End
    logging.info("DONE")
