import os
import sys

class Lists:
    """A container class for lists that loads data from a directory
    referred to as data_dir and exposes the following members:

    1. people: A dictionary containing lists describing people. Each
    element in the collection corresponds to a file in the directory
    data_dir/nouns/people. The file name is the key and the value is
    the list of entries.

    2. objects: A dictionary containing lists that can serve as
    objects in a sentence, loaded from the directory
    data_dir/nouns/objects. The file name is the key and the value
    is the list of entries.

    3. all_objects: A flattened version of objects. A list.

    4. adjectives: A dictionary containing lists containing
    adjectives that can describe singular people. These are loaded
    from the directory data_dir/adjectives. The file name is the key
    and the value is the list of entries.

    5. verbs: A dictionary of common verbs, grouped by their
    type. These are loaded from data_dir/verbs.

    """

    
    known_words = None
    data_dir = None

    def __init__(self, data_dir, known_words = None):
        """Arguments:

        data_dir -- the word list directory

        known_words -- a set of known words to keep. If this
        argument is None, all words are kept.
        
        """
        self.known_words = known_words
        self.people = self.load_dir(os.path.join(data_dir, 'nouns', 'people'))
        self.objects = self.load_dir(os.path.join(data_dir, 'nouns', 'objects'))
        self.all_objects = []
        for o in self.objects.keys():
            self.all_objects.extend(self.objects[o])
        self.all_objects = list(set(self.all_objects))

        self.adjectives = self.load_dir(os.path.join(data_dir, 'adjectives'))
        self.verbs = self.load_dir(os.path.join(data_dir, 'verbs'))

    def load_dir(self, dir_name):
        out = {}
        print("Loading " + dir_name)
        for name in os.listdir(dir_name):
            file_name = os.path.join(dir_name, name)
            out[name] = self.load_list(file_name)
            print("\t" + str(len(out[name])) + " " + name + " words ")
        return out
                                
    
    def load_list(self, file_name):
        with open(file_name) as f:
            lines = list(set(f.readlines()))
            lines.sort()
            lines = map(lambda x: x.strip(), lines)
            out = []
            for item in lines:
                if self.known_words == None:
                    out.append(item)
                elif item in self.known_words:
                    out.append(item)
                # else:
                #     print >> sys.stderr, ("Ignoring unknown word: " + item)
            return out
