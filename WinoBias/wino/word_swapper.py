import copy
import csv
import re 
import sys

### assumes generalized_swaps.txt and extra_gendered_words.txt are in directoy
### reads file conll formatted file in from stdin and writes swapped file to stdout

### example: cat input.conll | python word_swapper.py > output.conll
 
lines = open(sys.argv[1]).readlines()

#baseline_swap_list = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["Mary", "John"]]

#def check_swap(word_list):
#  for w in word_list:
#    w = w.lower()
#    for other in baseline_swap_list:
#      if w == other[0] or w == other[1]: return True
#  return False

swaps = open("generalized_swaps.txt").readlines()
swap_dict = {}
for line in swaps:
  tabs = line.split("\t")
  swap_dict[tabs[-2].strip()] = tabs[-1].strip()

swaps = open("extra_gendered_words.txt").readlines()
for line in swaps:
  tabs = line.split("\t")
  swap_dict[tabs[-2].strip()] = tabs[-1].strip()
  swap_dict[tabs[-1].strip()] = tabs[-2].strip()


docs = []
cur_doc = []
cur_sent = []
clusters = {}
active_spans = {}
in_cluster = False
for line in lines:
  #print line
  if line.strip() == "#end document": continue
  if len(line.strip()) == 0:
    cur_doc.append(cur_sent)
    cur_sent = []
  elif line.startswith("#begin"): 
    if len(cur_doc) > 0 :
      cur_doc.append(cur_sent)
      docs.append((doc_id, part_id, clusters, cur_doc))
      #print docs[-1]
    clusters = {}  
    cur_doc = []
    cur_sent = []
  else:
    tabs = re.split("\s+",line)
    doc_id = tabs[0]
    part_id = tabs[1]
    word = tabs[3]
    cur_sent.append([word, tabs])
    if tabs[-2] != "-":
      c = tabs[-2].split("|") #potentially multiple cluster references
      for _c in c:
        cluster_id = int(_c.replace(")","").replace("(",""))
        if "(" in _c:
          if cluster_id in active_spans:
            #print active_spans 
            active_spans[cluster_id].append([]) 
            #print "opening embedded span"
            #print line
            #exit()
          else : active_spans[cluster_id] = [[]]#.add(cluster_id)
      for (k,v) in active_spans.items():
        for _v in v:
          _v.append((len(cur_doc), int(tabs[2])))
      for _c in c:
        cluster_id = int(_c.replace(")","").replace("(",""))
        if ")" in _c: #close the last cluster
          if cluster_id not in clusters: clusters[cluster_id] = []
          #if cluster_id not in active_spans: print "embedded cluster close"
          #else :
          #if len(active_spans[cluster_id]) > 1: print "closing embedded span" 
          clusters[cluster_id].append(active_spans[cluster_id][-1])
          del active_spans[cluster_id][-1]
          if len(active_spans[cluster_id]) == 0: 
            del active_spans[cluster_id]
    else:
       for (k,v) in active_spans.items():
        for _v in v:
          _v.append((len(cur_doc), int(tabs[2])))

def to_str(sentence):
  rv = ""
  for _s in sentence:
    if len(rv) > 0: rv += " "
    rv += _s[0]
  return rv

def pad_zero(n, s):
  s = str(s)
  while len(s) < n:
    s = "0" + s
  return s
#print swap_dict
#read in flip doc
docs_saved = copy.deepcopy(docs)
total_swaps =0
total_casing=0
total_sentences = 0
total_changed_sentences = 0
her_hits = 0
for doci in range(0,len(docs)):
  doc_id = docs[doci][0]
  part_id = docs[doci][1]
  clusters = docs[doci][2]
  cur_doc = docs[doci][3]
  saved_doc = docs_saved[doci][3]
  changed_sentences = set()
  swapped_ids = set()

  for i in range(0, len(cur_doc)):
    curr_sent = cur_doc[i]
    for w, tabs in curr_sent: #w is in tabs[3]
      q = w.lower()
      swap = None
      if q in swap_dict:
        swap = swap_dict[q]
      #a bit of obvious special casing
      #her/PRP him      
      #her/PRP$        his
      if q == "her":
        her_hits += 1
        if tabs[4] == "PRP": swap = "him"
        else: swap = "his"
      #if swap is None:
        #if "woman" in q: swap = q.replace("woman", "!!!man")
        #if "man" in q: swap = q.replace("man", "!!!woman*")
        #if "women" in q: swap = q.replace("women", "!!!men")
        #if "men" in q: swap = q.replace("men", "!!!women")
      if swap is not None and w[0].isupper(): 
        swap = swap[0].upper()+swap[1:]
        total_casing+=1
      if swap is not None: 
        tabs[3] = swap
        changed_sentences.add(i)
        total_swaps += 1
       
  total_changed_sentences += len(changed_sentences) 
  print "#begin document ("+cur_doc[0][0][1][0]+"); part "+ pad_zero(3,cur_doc[0][0][1][1])
  for i in range(0,len(cur_doc)):
     total_sentences += 1
     #if i not in changed_sentences: continue
     #kept_sentences += 1     
     sent = cur_doc[i] 
     for _w in sent:
       print " ".join(_w[1])
     if i < (len(cur_doc)-1): print ""
  print "#end document"

#print "baseline swaps: {}; swaps: {} ; spans: {}; {:.1f}% of spans, baseline covers {:.1f}% ".format(baseline_swaps, total_swaps, total_spans, 100*float(total_swaps) / total_spans, 100*float(baseline_swaps)/total_swaps)
#print "changed clusters {} total cluster {} {:.1f}%".format(cluster_changed, total_cluster, 100*float(cluster_changed)/total_cluster)
#print "changed sentences {} total sentences {} {:.1f}%".format(total_changed_sentences, total_sentences, 100*float(total_changed_sentences)/total_sentences)
#print "total swaps {}, her_hits = {}".format(total_swaps, her_hits)
