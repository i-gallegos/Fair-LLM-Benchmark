import os
import sys

def articlize(x, capitalize):
    """A utility function to assign the correct article to a word

    Arguments:
    x -- a word
    capitalize -- a flag to indicate whether the article should be capitalized
    """
    is_vowel = x[0].lower() in set(['a', 'e', 'i', 'o', 'u'])
    if is_vowel:
        v = "An"
    else:
        v = "A"
 
    if capitalize:
        return v + " " + x
    else:
        return v.lower() + " " + x

def make_type(name):
    """
    A utility to get a template type from its name
    """
    return '_'.join(name.split("_")[0:-1])
        
class SubjectTemplate:
    """
    A template of the form "____ verb object", with proper articles.
    """
    def __init__(self, name, verb, obj):
        self.name = name
        self.template_type = make_type(name)
        self.ending = verb.strip() + " " + obj.strip() + " ."

    def apply(self, x):
        return articlize(x.strip(), True) + " " + self.ending.strip()

class ObjectTemplate:
    """
    A template of the form "start ____"
    """
    def __init__(self, name, beginning):
        self.name = name
        self.template_type = make_type(name)
        self.beginning = beginning 

    def apply(self, x):
        return self.beginning.strip() + " " + articlize(x.strip(), False)
    
    
class Templates:
    """A container for templates that consists two lists of pairs. Each
    element in both lists consists of two templates --- p and h ---
    which are meant to be a template for generating a premise and
    hypothesis respectively.

    The two lists exposed are

    1. adjective_templates: templates where the slot filler is an
    adjective. These templates are of the form "____ person verb
    object".

    2. noun_templates: templates where the slot fillers are
    nouns. These templates are of the form "____ verb object".

    All templates are created using lists of verbs, nouns and
    adjectives.

    """

    def __init__(self, lists):
        self.adjective_templates = []
        self.noun_templates = []
        commerce_verbs = lists.verbs['commerce_verbs']

        for v in commerce_verbs:
            for o in lists.all_objects:
                self.add_templates(v, o)

        interaction_verbs = lists.verbs['interaction_verbs']
        other_people = []
        other_people.extend(lists.people['person_hyponyms'])
        if 'rulers' in lists.people:
            other_people.extend(lists.people['rulers'])
        other_people = list(set(other_people))
        other_people.sort()
        for v in interaction_verbs:
            for o in other_people:
                self.add_templates(v, o)

        for v in lists.verbs['driving_verbs']:
            for o in lists.objects['vehicles']:
                self.add_templates(v, o)

        for v in lists.verbs['eating_verbs']:
            for o in lists.objects['food']:
                self.add_templates(v, o)

    def add_templates(self, v, o):
        ao = articlize(o, False)
        n_template = SubjectTemplate(v + "_" + o,
                                     v, ao)
        a_template = SubjectTemplate('person_' + v + "_" + o,
                                     'person ' + v, ao)

        self.noun_templates.append([n_template, n_template])
        self.adjective_templates.append([a_template, a_template])
