import pandas as pd
import numpy as np
import random
import re


def return_list_from_string(inputx):
    """
    This function takes a string that's formatted as
    WORD1: [word, word]; WORD2: [word, word]
    and separates it into two iterable lists

    Args:
        input: string

    Returns:
        two lists
    """
    x = inputx.split(";")
    for wrd in x:
        if "WORD1" in wrd or "NAME1" in wrd:
            wrd2 = (
                wrd.replace("WORD1:", "")
                .replace("{{WORD1}}:", "")
                .replace("NAME1:", "")
                .replace("{{NAME1}}:", "")
            )
            wrd3 = wrd2.strip()
            wrds = wrd3.replace("[", "").replace("]", "")
            wrds1 = wrds.split(",")
            wrds1 = [w.strip() for w in wrds1]
        if "WORD2" in wrd or "NAME2" in wrd:
            wrd2 = (
                wrd.replace("WORD2:", "")
                .replace("{{WORD2}}:", "")
                .replace("NAME2:", "")
                .replace("{{NAME2}}:", "")
            )
            wrd3 = wrd2.strip()
            wrds = wrd3.replace("[", "").replace("]", "")
            wrds2 = wrds.split(",")
            wrds2 = [w.strip() for w in wrds2]
        else:
            wrds2 = ""
    return wrds1, wrds2


def do_slotting(
    fr_row,
    frame_cols,
    this_word,
    gs_word,
    this_word_2,
    gs_word2,
    lex_div,
    rand_wrd1,
    rand_wrd2,
):
    """
    Does a bunch of string replacement to make the templates look like normal English sentences based on whatever
    the input words are. Works only with pre-defined slots of {{NAME1}}, {{NAME2}}, etc.

    Args:
        fr_row: dataframe row
        frame_cols: names of the columns in the dataframe
        this_word: the target word
        gs_word: for intersectional templates, the gender or SES of the target word, otherwise None
        this_word_2: the non-target word
        gs_word2: for intersectional templates, the gender or SES of the non-target word, otherwise None
        lex_div: the string of words that need to be supplanted into {{WORD1}} or {{WORD2}}, if provided
        rand_wrd1: the word that replaces {{WORD1}} when lex_div is defined
        rand_wrd2: the word that replaces {{WORD2}} when lex_div is defined

    Returns:
        the dataframe row with all slots filled
    """
    fr_row2 = fr_row.copy()
    for a in range(len(frame_cols)):
        # this_txt = fr_row[frame_cols[a]][0]
        this_txt = fr_row.loc[0, frame_cols[a]]
        # don't try to string replace on things that aren't strings
        if isinstance(this_txt, str):
            this_txt = this_txt.replace("{{NAME1}}", this_word)
            this_txt = this_txt.replace("{{NAME2}}", this_word_2)
            if gs_word is not None:
                this_txt = this_txt.replace("{{GEN1}}", gs_word)
                this_txt = this_txt.replace("{{GEN2}}", gs_word2)
                this_txt = this_txt.replace("{{OCC1}}", gs_word)
                this_txt = this_txt.replace("{{OCC2}}", gs_word2)
            if len(lex_div) > 1:
                this_txt = this_txt.replace("{{WORD1}}", rand_wrd1)
                this_txt = this_txt.replace("{{WORD2}}", rand_wrd2)
            if (
                "lesbian man" in this_txt
            ):  # just the easiest way to fix an issue in sexual_orientation
                this_txt = this_txt.replace("lesbian man", "lesbian woman")
            if "Latino woman" in this_txt:  # just the easiest way to fix this issue too
                this_txt = this_txt.replace("Latino woman", "Latina woman")
            # need to switch a to an when there's a vowel
            # (note: this doesn't work if 'a' is the first word in a sentence)
            if (
                re.search("^[aeiou]", this_word.lower()) is not None
                or re.search("^[aeiou]", this_word_2.lower())
                or re.search("^[aeiou]", str(gs_word).lower())
                or re.search("^[aeiou]", str(gs_word2).lower())
            ):
                this_txt = re.sub(r"(\s[aA])(\s[aAeEiIoOuU])", r"\1n\2", this_txt)
                # fix European
                this_txt = this_txt.replace(" an European", " a European")
            fr_row2.loc[0, frame_cols[a]] = this_txt
    return fr_row2


def make_dict(
    nn,
    q_id,
    polarity,
    context_cond,
    cat,
    subcat,
    answer_info,
    bias_targets,
    version,
    notes,
    context,
    question,
    ans_list,
    ans_place,
):
    """
    Formats information into a standardized dict that can be saved to jsonl

    Args:
        nn: item number created for output
        q_id: item number from the template file
        polarity: neg/nonneg indicating what type of question is used
        context_cond: ambig/disambig indicating the type of context used
        cat: category of bias
        subcat: subcategory of bias, if defined
        answer_info: information about the answer items, provided as a list of three lists
        bias_targets: list of bias targets for that template
        version: version info from the template file
        notes: additional info provided in the template file
        context: string, full context provided for model input
        question: string, question provided for model input
        ans_list: list of answer options
        ans_place: label corresponding to which answer option is correct

    Returns:
        dictionary
    """
    this_dict = {
        "example_id": nn,
        "question_index": q_id,
        "question_polarity": polarity,
        "context_condition": context_cond,
        "category": cat,
        "answer_info": answer_info,
        "additional_metadata": {
            "subcategory": subcat,
            "stereotyped_groups": bias_targets,
            "version": version,
            "source": notes,
        },
        "context": context.strip(),
        "question": question.strip(),
        "ans0": ans_list[0],
        "ans1": ans_list[1],
        "ans2": ans_list[2],
        "label": ans_place,
    }
    return this_dict


def create_templating_dicts(
    cat,
    this_frame_row,
    subcat,
    unknown_options,
    frame_cols,
    bias_targets,
    name1,
    name2,
    name1_info,
    name2_info,
    nn,
):
    """
    takes in all the information needed to create a single set of four items for a given instance,
    calls make_dict() for formatting, and returns the four dicts

    Args:
        cat: bias category
        this_frame_row: the frame row with slots already filled
        subcat: the subcategory of bias, if provided
        unknown_options: the string meaning 'unknown'
        frame_cols: names of the dataframe columns
        bias_targets: list of bias targets associated with this example
        name1: list or string of first name variable
        name2: list or string of second name variable
        name1_info: list or string, category info associated with first name variable
        name2_info: list or string, category info associated with first name variable
        nn: item number created for output

    Returns:
        list of four dicts created by crossing context condition x polarity for this example set
    """
    # get all the info for this frame, should all be formatted now
    q_id = this_frame_row.Q_id[0]
    text_ambig = this_frame_row.Ambiguous_Context[0]
    text_disambig = this_frame_row.Disambiguating_Context[0]
    q_neg = this_frame_row.Question_negative_stereotype[0]
    q_non_neg = this_frame_row.Question_non_negative[0]
    ans_neg = this_frame_row.Answer_negative[0]
    ans_non_neg = this_frame_row.Answer_non_negative[0]
    ans_ambig = random.choice(unknown_options)
    if "version" in frame_cols:
        version = this_frame_row.version[0]
    else:
        version = "None"

    notes = this_frame_row.Notes[0]

    # for answer option order randomization
    ans_list = [ans_neg, ans_non_neg, ans_ambig]
    random.shuffle(ans_list)
    # need to still get the position of each answer after randomization to know correct answer label
    ans_neg_place = ans_list.index(ans_neg)
    ans_non_neg_place = ans_list.index(ans_non_neg)
    ans_unknown_place = ans_list.index(ans_ambig)

    answer_info = {"ans0": ["", ""], "ans1": ["", ""], "ans2": ["", ""]}

    # need to pull the answer info differently depending on whether the strings are part of a list or not
    if isinstance(name1, list):
        # this is annoying, but when one of the names is 'man', it returns true for 'woman'
        # so just add a space before it, so that the conditionals below won't trigger at 'woman'
        if name1[1] == "man":
            name1[1] = " man"
        if name2[1] == "man":
            name2[1] = " man"
        if name1[0] in ans_neg and name1[1] in ans_neg:
            ans_neg_key = list(answer_info.keys())[ans_neg_place]
            answer_info[ans_neg_key] = [name1[0] + " " + name1[1].lstrip(), name1_info]
        if name1[0] in ans_non_neg and name1[1] in ans_non_neg:
            ans_non_neg_key = list(answer_info.keys())[ans_non_neg_place]
            answer_info[ans_non_neg_key] = [
                name1[0] + " " + name1[1].lstrip(),
                name1_info,
            ]
        if name2[0] in ans_neg and name2[1] in ans_neg:
            ans_neg_key = list(answer_info.keys())[ans_neg_place]
            answer_info[ans_neg_key] = [name2[0] + " " + name2[1].lstrip(), name2_info]
        if name2[0] in ans_non_neg and name2[1] in ans_non_neg:
            ans_non_neg_key = list(answer_info.keys())[ans_non_neg_place]
            answer_info[ans_non_neg_key] = [
                name2[0] + " " + name2[1].lstrip(),
                name2_info,
            ]
        ans_unknown_key = list(answer_info.keys())[ans_unknown_place]
        answer_info[ans_unknown_key] = [ans_ambig, "unknown"]
    else:
        # same issue noted above
        if name1 == "man":
            name1 = " man"
        if name2 == "man":
            name2 = " man"
        if name1 in ans_neg:
            ans_neg_key = list(answer_info.keys())[ans_neg_place]
            answer_info[ans_neg_key] = [name1.lstrip(), name1_info]
        if name1 in ans_non_neg:
            ans_non_neg_key = list(answer_info.keys())[ans_non_neg_place]
            answer_info[ans_non_neg_key] = [name1.lstrip(), name1_info]
        if name2 in ans_neg:
            ans_neg_key = list(answer_info.keys())[ans_neg_place]
            answer_info[ans_neg_key] = [name2.lstrip(), name2_info]
        if name2 in ans_non_neg:
            ans_non_neg_key = list(answer_info.keys())[ans_non_neg_place]
            answer_info[ans_non_neg_key] = [name2.lstrip(), name2_info]
        ans_unknown_key = list(answer_info.keys())[ans_unknown_place]
        answer_info[ans_unknown_key] = [ans_ambig, "unknown"]

    dict1 = make_dict(
        nn,
        q_id,
        "neg",
        "ambig",
        cat,
        subcat,
        answer_info,
        bias_targets,
        version,
        notes,
        text_ambig,
        q_neg,
        ans_list,
        ans_unknown_place,
    )
    nn += 1
    dict2 = make_dict(
        nn,
        q_id,
        "neg",
        "disambig",
        cat,
        subcat,
        answer_info,
        bias_targets,
        version,
        notes,
        "%s %s" % (text_ambig.strip(), text_disambig),
        q_neg,
        ans_list,
        ans_neg_place,
    )
    nn += 1
    dict3 = make_dict(
        nn,
        q_id,
        "nonneg",
        "ambig",
        cat,
        subcat,
        answer_info,
        bias_targets,
        version,
        notes,
        text_ambig,
        q_non_neg,
        ans_list,
        ans_unknown_place,
    )
    nn += 1
    dict4 = make_dict(
        nn,
        q_id,
        "nonneg",
        "disambig",
        cat,
        subcat,
        answer_info,
        bias_targets,
        version,
        notes,
        "%s %s" % (text_ambig.strip(), text_disambig),
        q_non_neg,
        ans_list,
        ans_non_neg_place,
    )

    return [dict1, dict2, dict3, dict4]
