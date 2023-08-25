import pickle
import pandas as pd
import spacy
import sys
from tqdm import tqdm

# ARGUMENTS
# sys.argv[1] dataset path (pickled pandas df)
# sys.argv[2] save location

# define some helper functions
# sentence segmentation for news
nlp = spacy.load("en_core_web_sm")
def spacy_seg(row):
    doc = nlp(row["text"])
    return ([sent.text for sent in doc.sents])

tqdm.pandas()

# load dataset
print("loading dataset...")
df = pickle.load(open(sys.argv[1], 'rb'))

# do sentence segmentation
print("Segmenting sentences...")
df['sentence'] = df.progress_apply(lambda row: spacy_seg(row), axis=1)
df = df.explode("sentence")

# deal with whitespace and empty strings
df['sentence'] = df['sentence'].str.strip()
df = df[df.sentence != '']
df = df[df.sentence != None]
df = df.dropna()

# save
pickle.dump(df, open(sys.argv[2], 'wb'))
