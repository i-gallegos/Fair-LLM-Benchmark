import argparse
import codecs
import stanza

from depparse import *
from genderlang import genderneutral
import language_tool_python
import re

#Step 1: Clean OpenSubtitles Data
#OK => write down how it was cleaned

#Step 2: process
def process_sentences(lang, advanced=True):
    stanza_init(lang)
    lang, sentences = rewriteheshe(testsentences, lang)
    they_sentences = rewritehisher(sentences)
    print(they_sentences)
    if advanced:
        they_sentences = genderneutral(they_sentences)
    correctgram(they_sentences)

def rewriteheshe(sentences,language):
    nsents=[]
    for sent in sentences:
        nsent = [word if word.lower() != 'she' and word.lower() != 'he' else 'they' if word =="she" or word =="he" else "They" for word in sent.split()]
        nsents.append(" ".join(nsent))
    return language,nsents

tool = language_tool_python.LanguageTool('en-US')
def correctgram(sents):
    correct_s=[]
    for s in sents:
        s = s.replace('they is ', 'they are ')
        s = s.replace('They is ', ' They are ')
        s = s.replace('They was ', 'They were ')
        s = s.replace('they was ', 'they were ')
        s = s.replace('They wasn ', 'They weren ')
        s = s.replace('they wasn ', 'they weren ')
        s = s.replace("they 's ", "they are ")
        s = s.replace("they ' s ", "they are ")
        s = s.replace("They ' s ", "They are ")
        s = s.replace("They 's ", "They are ")
        s = s.replace("They does ", "They do ")
        s = s.replace("they does ", "they do ")
        matches=tool.check(s)
        new_matches = [match for match in matches if match.category == 'GRAMMAR'] #correct only grammar issues
        s=language_tool_python.utils.correct(s,new_matches)  #
        s = s.replace("'t 't", " 't")
        s = s.replace("'t ' t", " 't")
        s = s.replace("' t ' t", " 't")
        s = s.replace("'t ’ t", " 't")
        s = s.replace("they doesn", "they don")
        s = s.replace("They doesn", "They don")
        s = s.replace('they isn ', 'they aren ')
        s = s.replace('They isn ', ' They aren ')
        s = s.replace("they hasn", "they haven")
        s = s.replace("They hasn", "They haven")
        correct_s.append(s)
    return correct_s

testsentences=["If sales have dropped off , I do hope she decides to diversify and offer different kinds of videos - like you said she ’ s been around a while , and clearly has her own MO , but I hope she doesn ’ t decide to simply stop if sales are falling ."]

process_sentences(lang="en")



