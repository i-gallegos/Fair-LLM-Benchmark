import argparse
import re

import language_tool_python
import stanza

# reproducibility bit ----------------
from random import seed
from numpy.random import seed as np_seed
import os
seed(42)
np_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
# -----------------------------------


class NeutralRewriter(object):

    def __init__(self, language='en', parse=False, advanced=False):
        """Applies rule-based neutral rewrite operations.

        Parameters
        ----------
        language : str, optional
            language identifier for stanza, by default 'en'
        parse : bool, optional
            if doc requires parsing using stanza, by default False
        advanced : bool, optional
            invoke advanced gender neutral word replacements, by default False
        """
        self.stanza = self._stanza_init(language, parse)
        self.tool = language_tool_python.LanguageTool('en-US')
        self.parse = parse
        self.advanced = advanced

    def _stanza_init(self, language, parse):
        """Initalizes stanza either for tokenization or not."""
        return stanza.Pipeline(
            lang=language, processors='tokenize,mwt,pos,lemma,depparse',
            tokenize_pretokenized='true' if not parse else 'false')

    @staticmethod
    def dict_replace(sent, d):
        """Replaces strings in sent based on key value pairs in d.

        Parameters
        ----------
        sent : str
            a sentence from a document
        d : dict
            key is query, value is replacement string

        Returns
        -------
        str
            a sentence with word keys replaced
        """
        for hit in re.findall(r'|'.join(d), sent):
            sent = sent.replace(hit, d[hit])
        return sent

    def _genderneutral(self, sent):
        """Rewrite specific words to their gender neutral variants."""
        gender_lang = {
            #################################
            # 1. CHANGE INTO GENDER NEUTRAL #
            #################################
            # chairman/woman
            'chairman': 'chairperson',
            'chairmen': 'chairpeople',
            'chairwoman': 'chairperson',
            'chairwomen': 'chairpeople',
            # anchorman/woman
            'anchorman': 'anchor',
            'anchormen': 'anchors',
            'anchorwoman': 'anchor',
            'anchorwomen': 'anchors',
            # congresswoman/congressman
            'congressman': 'member of congress',
            'congressmen': 'members of congress',
            'congresswoman': 'member of congress',
            'congresswomen': 'members of congress',
            # policeman/woman
            'policeman': 'police officer',
            'policemen': 'police officers',
            'policewoman': 'police officer',
            'policewomen': 'police officers',
            # spokesman/woman
            'spokesman': 'spokesperson',
            'spokesmen': 'spokespersons',
            'spokeswoman': 'spokesperson',
            'spokeswomen': 'spokespersons',
            # steward/stewardess
            'steward': 'flight attendant',
            'stewards': 'flight attendants',
            'stewardess': 'flight attendant',
            'stewardesses': 'flight attendants',
            # headmaster/mistress
            'headmaster': 'principal',
            'headmasters': 'principals',
            'headmistress': 'principal',
            'headmistresses': 'principals',
            # business man/woman
            'businessman': 'business person',
            'businessmen': 'business people',
            'businesswoman': 'business person',
            'businesswomen': 'business persons',
            # postman/postwoman
            'postman': 'mail carrier',
            'postmen': 'mail carriers',
            'postwoman': 'mail carrier',
            'postwomen': 'mail carriers',
            # mailman/mailwoman
            'mailman': 'mail carrier',
            'mailmen': 'mail carriers',
            'mailwoman': 'mail carrier',
            'mailwomen': 'mail carriers',
            # salesman/saleswoman
            'salesman': 'salesperson',
            'salesmen': 'salespersons',
            'saleswoman': 'salesperson',
            'saleswomen': 'salespersons',
            # fireman/firewoman
            'fireman': 'firefighter',
            'firemen': 'firefighters',
            'firewoman': 'firefighter',
            'firewomen': 'firefighter',
            # barman/barwoman
            'barman': 'bartender',
            'barmen': 'bartenders',
            'barwoman': 'bartender',
            'barwomen': 'bartenders',
            # cleaning lady
            'cleaning man': 'cleaner',
            'cleaning lady': 'cleaners',
            'cleaning men': 'cleaner',
            'cleaning ladies': 'cleaners',
            # foreman/woman
            'foreman': 'supervisor',
            'foremen': 'supervisors',
            'forewoman': 'supervisor',
            'forewomen': 'supervisors',
            #######################################
            # 2. AVOID UNNECESSARY FEMININE FORMS #
            #######################################
            # actor/actress
            'actress': 'actor',
            'actresses': 'actors',
            # hero/heroine
            'heroine': 'hero',
            'heroines': 'heros',
            # comedian/comedienne
            'comedienne': 'comedian',
            'comediennes': 'comedians',
            # executrix/executor
            'executrix': 'executor',
            'executrices': 'executors',
            'executrixes': 'executors',
            # poetess/poet
            'poetess': 'poet',
            'poetesses': 'poets',
            # usherette/usher
            'usherette': 'usher',
            'usherettes': 'ushers',
            # authoress/author
            'authoress': 'author',
            'authoresses': 'authors',
            # boss lady
            'boss lady': 'boss',
            'boss ladies': 'bosses',
            # boss lady
            'waitress': 'waiter',
            'waitresses': 'waiters',
            #################################
            # 3. AVOIDANCE OF GENERIC 'MAN' #
            #################################
            # average man
            'average man': 'average person',
            'average men': 'average people',
            # best man for the job
            'best man for the job': ' best person for the job',
            'best men for the job': ' best people for the job',
            # layman
            'layman': 'layperson',
            'laymen': 'laypeople',
            # man and wife
            # left space (otherwise e.g. woman and wife => wohusband and wife,
            ' man and wife': ' husband and wife',
            # mankind
            # left space (otherwise e.g. humankind => huhumankind,
            ' mankind': ' humankind',
            # man-made
            # left space (otherwise e.g. human-made => huhuman-made,
            ' man-made': ' human-made',
            # manpower
            # 'manpower': 'staff',  Depends on context
            # workmanlike
            'workmanlike': 'skillful',
            # workmanlike
            'freshman': 'first-year student'
        }
        return self.dict_replace(sent, gender_lang)

    def _correctgram(self, sent):
        """Manual correct word grams and automatic correct grammar."""
        d1 = {
            "’": "'",
            'they is ': 'they are ',
            'They is ': ' They are ',
            'They was ': 'They were ',
            'they was ': 'they were ',
            'They wasn ': 'They weren ',
            'they wasn ': 'they weren ',
            "they 's ": "they are ",
            "they ' s ": "they are ",
            "They ' s ": "They are ",
            "They 's ": "They are ",
            "They does ": "They do ",
            "they does ": "they do "
        }
        d2 = {
            "'t 't": " 't",
            "'t ' t": " 't",
            "' t ' t": " 't",
            "they doesn": "they don",
            "They doesn": "They don",
            'they isn ': 'they aren ',
            'They isn ': ' They aren ',
            "they hasn": "they haven",
            "They hasn": "They haven"
        }
        sent = self.dict_replace(sent, d1)
        matches = self.tool.check(sent)
        new_matches = [match for match in matches if match.category ==
                       'GRAMMAR']  # correct only grammar issues
        sent = language_tool_python.utils.correct(sent, new_matches)
        return self.dict_replace(sent, d2)

    @staticmethod
    def match_case(word, query):
        """Case query based on casing of word."""
        return query.capitalize() if word.text[0].isupper() else query

    def process_sentence(self, sent, parse=False):
        """Process sentence (runs full rule-based neutralizer).

        Parameters
        ----------
        sent : str
            single sentence to be converted into gender neutral variants
        parse : bool, optional
            if sentence still requires stanza parsing, by default False

        Returns
        -------
        str
            gender-neutral sentence
        """
        if parse:
            sent = self.stanza(sent).sentences[0]
        sent_map = [word.text for word in sent.words]
        for i, word in enumerate(sent.words):
            _word = word.text.lower()
            if _word == 'he' or _word == 'she':
                sent_map[i] = self.match_case(word, 'they')

            elif _word == 'his':
                sent_map[i] = self.match_case(word, 'their')
                if word.deprel != "nmod:poss":
                    sent_map[i] += 's'  # theirs

            elif _word == 'her' or _word == 'her.':
                if word.xpos == "PRP$" and word.text == 'her':
                    sent_map[i] = self.match_case(word, 'their')
                elif word.xpos == "PRP" and "Poss=Yes" in word.feats:
                    sent_map[i] = self.match_case(word, 'their')
                elif word.text == 'her.':
                    sent_map[i] = self.match_case(word, 'them')
                else:
                    sent_map[i] = self.match_case(word, 'them')

            elif word.text == 'himself' or word.text == 'herself':
                sent_map[i] = self.match_case(word, 'themselves')

            elif word.text == 'hers':
                sent_map[i] = self.match_case(word, 'theirs')

            elif word.text == 'him':
                sent_map[i] = self.match_case(word, 'them')

        new_sent = " ".join(sent_map)
        if self.advanced:
            new_sent = self._genderneutral(new_sent)
        return self._correctgram(new_sent)

    def process_document(self, document):
        """Splits document in sentences and rewrites those to gender neutral.

        Parameters
        ----------
        document : str
            a document with multiple sentences, to be parsed

        Yields
        -------
        str
            a sentence that has been made gender neutral
        """
        for sent in self.stanza(document).sentences:
            yield self.process_sentence(sent)

    def process_file(self, file_in):
        """Process file_in line by line to rewrite gender-neutral.

        Parameters
        ----------
        file_in : str
            full path to file to be rewritten

        Yields
        -------
        str
            a sentence that has been made gender neutral
        """
        with open(file_in, 'r') as fi:
            for line in fi.readlines():
                if self.parse:
                    for sent in self.process_document(line):
                        yield sent
                else:
                    yield self.process_sentence(line, parse=True)

    def save(self, output, output_file):
        """Save output to output_file.

        Parameters
        ----------
        output : iterable
            an interable containing sentences
        output_file : str
            full file path to file to be saved
        """
        with open(output_file, 'w') as fo:
            for sent in output:
                fo.write(sent + '\n')


if __name__ == '__main__':
    # USAGE: python rewrite_neutral.py -i inputF -l en -o outputF
    parser = argparse.ArgumentParser(
        description='parse sentences using stanzaNLP')
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-d", "--documents", required=False, default=False,
                        action='store_true',
                        help="Inputs are documents rather than sentences.")
    parser.add_argument("-l", "--language", required=False,  # EN only v
                        default='en',
                        help="Specify language code, e.g. en, es, fr...")
    parser.add_argument("-a", "--advanced", required=False, default=False,
                        action='store_true',
                        help="Invokes the more advanced rewriting")
    parser.add_argument("-o", "--output_file", required=False)
    args = parser.parse_args()
    nr = NeutralRewriter(args.language, args.documents, args.advanced)
    nr.save(nr.process_file(args.input_file), args.output_file)
