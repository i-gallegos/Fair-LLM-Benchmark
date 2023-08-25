# Retrieve he, she, him, his, her, hers, himself, themself sentences
import codecs
import argparse

def maketestset(infile, outfile, total_counts):
     sentences = read_file(infile)
     testset= find_sents(sentences, total_counts)
     print(len(testset))
     write_output(outfile,testset)

def read_file(input):
    with codecs.open(input,'r') as inF:
        input_sents=inF.readlines()
    return input_sents

def write_output(outputfile,output):
     with codecs.open(outputfile, 'w') as outF:
         outF.write(''.join(output))

def find_sents(sentences, total_counts):
     hecounter=0
     shecounter=0
     hercounter=0
     herscounter=0
     hiscounter=0
     himcounter=0
     himselfcounter=0
     herselfcounter=0

     totalcounter=0
     maxcount = total_counts // 8 #8 is the number of all pronouns

     testset = []

     for sent in sentences:
          addSent=False

          if hecounter < maxcount:
              if ' he ' in sent.lower():
                  addSent = True
                  hecount = countOccurences(sent, 'he')
                  hecounter += hecount
                  totalcounter += hecount

          if shecounter < maxcount:
              if ' she ' in sent.lower():
                   addSent = True
                   shecount = countOccurences(sent, 'she')
                   shecounter += shecount
                   totalcounter += shecount

          if hercounter < maxcount:
              if ' her ' in sent.lower():
                  addSent = True
                  hercount = countOccurences(sent, 'her')
                  hercounter += hercount
                  totalcounter += hercount

          if hiscounter < maxcount:
              if ' his ' in sent.lower():
                  addSent = True
                  hiscount = countOccurences(sent, 'his')
                  hiscounter += hiscount
                  totalcounter += hiscount

          if herscounter < maxcount:
              if ' hers ' in sent.lower():
                  addSent = True
                  herscount = countOccurences(sent, 'hers')
                  herscounter += herscount
                  totalcounter += herscount

          if himcounter < maxcount:
              if ' him ' in sent.lower():
                  addSent = True
                  himcount = countOccurences(sent, 'him')
                  himcounter += himcount
                  totalcounter += himcount

          if herselfcounter < maxcount:
              if ' herself ' in sent.lower():
                  addSent = True
                  herselfcount = countOccurences(sent, 'herself')
                  herselfcounter += herselfcount
                  totalcounter += herselfcount

          if himselfcounter < maxcount:
              if ' himself ' in sent.lower():
                  addSent = True
                  himselfcount = countOccurences(sent, 'himself')
                  himselfcounter += himselfcount
                  totalcounter += himselfcount

          if addSent == True:
              testset.append(sent)

     print("hecounter: " + str(hecounter) + "\n" + "shecounter: " + str(shecounter) + "\n" \
          "hercounter: " + str(hercounter) + "\n" + "himcounter: " + str(himcounter) + "\n" \
          "herscounter: " + str(herscounter) + "\n" + "hiscounter: " + str(hiscounter) + "\n" \
          "herselfcounter: " + str(herselfcounter) + "\n" + "himselfcounter: " + str(himselfcounter) + "\n" \
          "totalcounter: " + str(totalcounter))
     return testset

def countOccurences(str, word):
     # split the string by spaces in a
     a = str.lower().split(" ")
     # print(str)
     # search for pattern in a
     count = 0
     for i in range(0, len(a)):

          # if match found increase count
          if (word == a[i]):
               count = count + 1
     return count

if __name__ == '__main__':
     # USAGE: python createtestset.py -i inputF -o outputF
     parser = argparse.ArgumentParser(description='parse sentences using stanzaNLP')
     parser.add_argument("-i", "--input_file", required=True)
     parser.add_argument("-o", "--output_file", required=True)
     parser.add_argument("-t", "--total_count", required=True, default=2000)
     args = parser.parse_args()
     maketestset(args.input_file, args.output_file, int(args.total_count))
