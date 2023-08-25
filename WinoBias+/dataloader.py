import json
import random
import re
from glob import glob

from tqdm import tqdm


class DataLoader(object):

    def __init__(self, file_in, gender_filter=True, language='en'):
        """General data loader class for loading and filtering sentences.

        Parameters
        ----------
        file_in : str
            full path to input file to read
        gender_filter : bool, optional
            filter data based on gender, by default True
        language : str, optional
            language code for stanza, by default 'en'
        """
        self.file_in = file_in
        self.filter = gender_filter

    @staticmethod
    def find_gender(text, join=False):
        """Regex to find gendered text.

        Parameters
        ----------
        text : str
            piece of text (sentence)
        join : bool, optional
            pre-concatenate regex parts, by default False

        Returns
        -------
        list
            list with hit groups if not join else list with concatenated hits
        """
        hits = re.findall(r'(?:[^a-z0-9])' +
                          r'(she|he|his|him|her|hers|himself|herself){1}' +
                          r'(?=[^a-z0-9])', text.lower())
        return [''.join(hit) for hit in hits] if join else hits

    def __iter__(self):
        """Iter through sentences yielding sent if it contains gender."""
        with open(self.file_in, 'r') as fi:
            for sent in fi.readlines():
                if self.filter and self.find_gender(sent):
                    yield sent
                elif not self.filter:
                    yield sent

    def to_file(self, file_out, output=None):
        """Dump full dataset as text file in file_out location.

        Parameters
        ----------
        file_out : str
            directory path of dump file
        output : iter
            iterable with output per line
        """
        with open(file_out, 'w') as fo:
            for sent in tqdm(output if output else self):
                fo.write(sent)

    def _get_samples(self):
        """Get indexed data and post indices per gender word."""
        data, samples = {}, {}
        for ix, sent in enumerate(self):
            data[ix] = sent
            for hit in self.find_gender(sent, join=True):
                form = hit.lower()
                if form == 'hisself':  # nope
                    continue
                if not samples.get(form):
                    samples[form] = []
                samples[form].append(ix)
        return data, samples

    def to_train_test_splits(self, file_out, n_samples=11, randomize=True):
        """Dump train and test splits in dir_out location.

        Parameters
        ----------
        file_out : str
            full txt file path where to dump the splits (.txt will be renamed)
        n_samples : int, optional
            n amount sampled per gendered word, by default 10
        randomize : bool, optional
            randomize sampling procedure, otherwise head, by default True
        """
        (data, samples), sample_idx = self._get_samples(), []
        out = re.sub(r'\.[^\/].*', '.', file_out)
        with open(out + 'test', 'w') as fo:
            selection = []
            for sample in samples:
                if randomize:
                    selection += random.sample(set(samples[sample]), n_samples)
                else:
                    selection += sorted(samples[sample])[:n_samples]

            for instance in set(sorted(selection)):
                sample_idx.append(instance)
                fo.write(data[instance])

        with open(out + 'train', 'w') as fo:
            for ix, sent in tqdm(data.items()):
                if ix not in sample_idx:
                    fo.write(sent)

        with open(out + 'idx', 'w') as fo:
            fo.write('\n'.join([str(i) for i in sample_idx]))


class OpenSubsLoader(DataLoader):

    def __init__(self, file_name='/data/opensubs/opensubs.txt'):
        """Loader for the OpenSubs corpus.

        Parameters
        ----------
        file_name : str, optional
            full path to sub file, by default '/data/opensubs/opensubs.txt'
        """
        self.file_name = file_name


class RedditLoader(DataLoader):

    def __init__(self, snap_dir='/data/reddit', period='2019-12'):
        """Loads and filters Reddit data based on snap zips.

        Parameters
        ----------
        snap_dir : str, optional
            directory containing snaps, by default '/data/reddit'
        year : int, optional
            year of snaps to extract data from, by default 2019
        """
        self.snap_dir = snap_dir
        self.year = period

    def _extract_sentences(self, doc):
        """Detects simple sentence boundaries and returns them as splits."""
        matches = re.findall(r'([\?\!\.] |\n)', doc)
        splits = re.split(r'([\?\!\.] |\n)', doc)
        if len(splits) + 1 == len(matches):
            matches.append('')
        for sent, suffix in zip(splits, matches):
            sent = (sent + suffix).replace('\n', ' ')
            if len(sent) > 4:
                yield sent + '\n'

    def _filter_sents(self, doc):
        """Split doc in sentences, find_gender, preprocess sents."""
        if self.find_gender(doc):
            for sent in self._extract_sentences(doc):
                if self.find_gender(sent):
                    yield sent

    def __iter__(self):
        """Iter through snaps yielding id and body per post."""
        for file_name in glob(self.snap_dir + '/*' + str(self.year) + '*'):
            try:
                if 'zst' not in file_name and 'bz2' not in file_name:
                    open_func = open
                elif 'bz2' in file_name:  # NOTE: old format compatibility
                    import bz2
                    open_func = bz2.open
                else:
                    continue
                for line in tqdm(open_func(file_name, 'r')):
                    reddit_object = json.loads(line)
                    for sent in self._filter_sents(reddit_object['body']):
                        yield sent
            except Exception as e:
                print(f"Error in: {e}")


if __name__ == '__main__':
    # dl = DataLoader('./evaluation/orig-subs.txt')
    # dl.to_train_test_splits('./evaluation/edit-subs.txt', randomize=False)
    dl = RedditLoader()
    dl.to_train_test_splits('./evaluation/reddit-gender.txt')
