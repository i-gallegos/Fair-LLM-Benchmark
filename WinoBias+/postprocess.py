import codecs
import argparse
import stanza

def postprocess(inFile,outFile):
    nlp = stanza.Pipeline(lang='en', processors='truecase')
    doc = nlp('this is a test sentence for stanza. this is another sentence.')
    tokenizedsents=[]

    for sent in doc.sentences:
        tok_s=[]
        for word in sent.words:
            print("word " + str(word))
            tok_s.append(word.text)
            print(tok_s)
            tok_sent = " ".join(tok_s)
            print(tok_s)
        tokenizedsents.append(tok_s)
    print(tokenizedsents)

def read_file(input):
    with codecs.open(input,'r') as inF:
        input_sents=inF.readlines()
    return input_sents

def write_output(outputfile,output):
     with codecs.open(outputfile, 'w') as outF:
         outF.write('\n'.join(output))

if __name__ == '__main__':
     # USAGE: python postprocess-lc-tk.py -i inputF -l en -o outputF
     parser = argparse.ArgumentParser(description='tokenize sentences using Stanza')
     parser.add_argument("-i", "--input_file", required=True)
     parser.add_argument("-o", "--output_file", required=True)
     args = parser.parse_args()
     postprocess(args.input_file, args.output_file)