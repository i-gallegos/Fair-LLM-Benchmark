import sys
import argparse

# This script fully instantiates the 120 templates in ../data/templates.tsv
# to generate the 720 sentences in ../data/all_sentences.tsv
# By default this script prints to stdout, and can be run with no arguments:

# python instantiate.py

def load_templates(path):
  fp = open(path, 'r')
  S = []
  headers = fp.next().strip().split('\t')
  for line in fp:
    line = line.strip().split('\t')
    occupation, other_participant, answer, sentence = line[0], line[1], int(line[2]), line[3]
    S.append((occupation, other_participant, answer, sentence))
  return S

def generate(occupation, other_participant, answer, sentence, someone=False, context=None):
  toks = sentence.split(" ")
  occ_index = toks.index("$OCCUPATION")
  part_index = toks.index("$PARTICIPANT")
  toks[occ_index] = occupation
  if not someone: # we are using the instantiated participant, e.g. "client", "patient", "customer",...
    toks[part_index] = other_participant
  else: # we are using the bleached NP "someone" for the other participant
    # first, remove the token that precedes $PARTICIPANT, i.e. "the"
    toks = toks[:part_index-1]+toks[part_index:]
    # recompute participant index (it should be part_index - 1)
    part_index = toks.index("$PARTICIPANT")
    if part_index == 0:
      toks[part_index] = "Someone"
    else:
      toks[part_index] = "someone"
  NOM = "$NOM_PRONOUN"
  POSS = "$POSS_PRONOUN"
  ACC = "$ACC_PRONOUN"
  special_toks = set({NOM, POSS, ACC})
  female_map = {NOM: "she", POSS: "her", ACC: "her"}
  male_map = {NOM: "he", POSS: "his", ACC: "him"}
  neutral_map = {NOM: "they", POSS: "their", ACC: "them"}
  female_toks = [x if not x in special_toks else female_map[x] for x in toks]
  male_toks = [x if not x in special_toks else male_map[x] for x in toks]
  neutral_toks = [x if not x in special_toks else neutral_map[x] for x in toks]
  male_sent, female_sent, neutral_sent = " ".join(male_toks), " ".join(female_toks), " ".join(neutral_toks)
  neutral_sent = neutral_sent.replace("they was", "they were")
  neutral_sent = neutral_sent.replace("They was", "They were")
  return male_sent, female_sent, neutral_sent


if __name__ == "__main__":
  sentence_path = "../data/templates.tsv"
  S = load_templates(sentence_path)
  print "sentid\tsentence"
  for s in S:
    occupation, other_participant, answer, sentence = s

    male_sent, female_sent, neutral_sent = generate(occupation, other_participant, answer, sentence)
    male_sentid, female_sentid, neutral_sentid = [occupation+'.'+other_participant+'.'+str(answer)+'.'+gender+".txt" for gender in ["male","female","neutral"]]

    male_sent_some1, female_sent_some1, neutral_sent_some1 = generate(occupation, other_participant, answer, sentence, someone=True)
    male_sentid_some1, female_sentid_some1, neutral_sentid_some1 = [occupation+'.'+"someone"+'.'+str(answer)+'.'+gender+".txt" for gender in ["male","female","neutral"]]

    # other participant is specific
    print male_sentid+"\t"+male_sent
    print female_sentid+"\t"+female_sent
    print neutral_sentid+"\t"+neutral_sent

    # other participant is "someone"
    print male_sentid_some1+"\t"+male_sent_some1
    print female_sentid_some1+"\t"+female_sent_some1
    print neutral_sentid_some1+"\t"+neutral_sent_some1
