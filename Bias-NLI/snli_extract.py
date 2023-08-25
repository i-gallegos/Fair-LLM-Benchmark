import ujson
import sys
import argparse
import re
import spacy

spacy_nlp = spacy.load('en_core_web_sm')

# tokenize and tag pos
def tokenize_spacy(text):
	tokenized = spacy_nlp(text)
	# use universal pos tags
	toks = [tok.text for tok in tokenized if not tok.is_space]
	pos = [tok.pos_ for tok in tokenized if not tok.is_space]
	lemma = [tok.lemma_.replace(' ','') for tok in tokenized if not tok.is_space]
	lemma = [l if l != '' else t for l, t in zip(lemma, toks)]
	return toks, pos, lemma			


def filter_by_pos(keys, toks, pos, lemma):
	filtered_toks = []
	filtered_pos = []
	filtered_lemma = []
	for t, p, l in zip(toks, pos, lemma):
		if p not in keys:
			filtered_toks.append(t)
			filtered_pos.append(p)
			filtered_lemma.append(l)
	return filtered_toks, filtered_pos, filtered_lemma

def write_to(ls, out_file):
	print('writing to {0}'.format(out_file))
	with open(out_file, 'w+') as f:
		for l in ls:
			f.write((l + '\n'))

def extract(opt, csv_file):
	all_sent1 = []
	all_sent2 = []
	all_label = []
	all_sent1_pos = []
	all_sent2_pos = []
	all_sent1_lemma = []
	all_sent2_lemma = []
	max_sent_l = 0

	skip_cnt = 0

	with open(csv_file, 'r') as f:
		line_idx = 0
		for l in f:
			line_idx += 1
			if line_idx == 1 or l.strip() == '':
				continue

			cells = l.rstrip().split('\t')
			label = cells[0]
			sent1 = cells[5]
			sent2 = cells[6]

			if label == '-':
				print('skipping label {0}'.format(label))
				skip_cnt += 1
				continue
			else:
				print(label)

			assert(label in ['entailment', 'neutral', 'contradiction'])

			sent1_toks, sent1_pos, sent1_lemma = tokenize_spacy(sent1)
			sent2_toks, sent2_pos, sent2_lemma = tokenize_spacy(sent2)

			if opt.filter != '':
				keys = opt.filter.split(',')
				sent1_toks, sent1_pos, sent1_lemma = filter_by_pos(keys, sent1_toks, sent1_pos, sent1_lemma)
				sent2_toks, sent2_pos, sent1_lemma = filter_by_pos(keys, sent2_toks, sent2_pos, sent2_lemma)

			assert(len(sent1_toks) == len(sent1_pos))
			assert(len(sent2_toks) == len(sent2_pos))
			assert(len(sent1_toks) == len(sent1_lemma))
			max_sent_l = max(max_sent_l, len(sent1_toks), len(sent2_toks))

			all_sent1.append(' '.join(sent1_toks))
			all_sent2.append(' '.join(sent2_toks))
			all_sent1_pos.append(' '.join(sent1_pos))
			all_sent2_pos.append(' '.join(sent2_pos))
			all_sent1_lemma.append(' '.join(sent1_lemma))
			all_sent2_lemma.append(' '.join(sent2_lemma))
			all_label.append(label)

	print('skipped {0} examples'.format(skip_cnt))

	return (all_sent1, all_sent2, all_sent1_pos, all_sent2_pos, all_sent1_lemma, all_sent2_lemma, all_label)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', help="Path to SNLI txt file", default="data/nli_bias/snli_1.0_dev.txt")
parser.add_argument('--output', help="Prefix to the path of output", default="data/nli_bias/dev")
parser.add_argument('--filter', help="List of pos tags to filter out", default="")


def main(args):
	opt = parser.parse_args(args)
	all_sent1, all_sent2, all_sent1_pos, all_sent2_pos, all_sent1_lemma, all_sent2_lemma, all_label = extract(opt, opt.data)
	print('{0} examples processed.'.format(len(all_sent1)))

	write_to(all_sent1, opt.output + '.sent1.txt')
	write_to(all_sent2, opt.output + '.sent2.txt')
	write_to(all_sent1_pos, opt.output + '.sent1_pos.txt')
	write_to(all_sent2_pos, opt.output + '.sent2_pos.txt')
	write_to(all_sent1_lemma, opt.output + '.sent1_lemma.txt')
	write_to(all_sent2_lemma, opt.output + '.sent2_lemma.txt')
	write_to(all_label, opt.output + '.label.txt')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))


