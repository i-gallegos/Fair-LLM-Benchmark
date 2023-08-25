import pandas as pd
import numpy as np
import io
import json
from BBQ_Full.utils import *
import random
import re
import ast

# all possible values of cat
cats = ["Race_x_gender", "Race_x_SES"]

# read in vocabulary files
vocab = pd.read_csv("BBQ_full/templates/vocabulary.csv")
vocab = vocab[vocab.Pilot_include != "No"]
names_vocab = pd.read_csv("BBQ_full/templates/vocabulary_proper_names.csv")

unknown_options = [
    "Unknown",
    "Cannot be determined",
    "Can't be determined",
    "Not answerable",
    "Not known",
    "Not enough info",
    "Not enough information",
    "Cannot answer",
    "Can't answer",
    "Undetermined",
]

for cat in cats:
    frames = pd.read_csv(
        "BBQ_full/templates/new_templates - %s.csv" % cat, na_filter=False
    )  # get the right template file

    dat_file = io.open(
        "BBQ_full/data/%s.jsonl" % cat, "w"
    )  # open the file that all the data will be saved in

    the_frames = frames[
        frames.Ambiguous_Context != ""
    ].reset_index()  # get rid of ones that are empty
    frame_cols = the_frames.columns

    # initialize item id counter
    nn = 0

    for i in range(len(the_frames)):

        # by default, take the words from the vocab that match this category
        # this needs to be done at each iteration because some rows use custom vocab and some don't
        words = vocab[vocab.Category == "Race_ethnicity"]

        # make sure that one of the words used is the target of the bias
        possible_word_list = words.Name.unique().tolist()
        bias_targets = the_frames.Known_stereotyped_race[i]

        #  Stereotyped groups info not relevant for subsetting Names in some categories
        if len(bias_targets) > 0:
            bias_targets = ast.literal_eval(the_frames.Known_stereotyped_race[i])
            targeted_word_list = [x for x in possible_word_list if x in bias_targets]
        else:
            targeted_word_list = possible_word_list
            bias_targets = []

        if len(targeted_word_list) > 4:
            word_list = random.sample(targeted_word_list, 5)
        else:
            word_list = targeted_word_list

        if the_frames.Proper_nouns_only[i] == "TRUE":
            has_proper_name = True
        else:
            has_proper_name = False

        # if the word needs to be further pulled apart with subcategory, do that here
        if "Subcategory" in frame_cols:
            this_subcat = the_frames.Subcategory[i]
        else:
            this_subcat = "None"

        # get info from the Names column
        critical_words = the_frames.Names[i]
        if len(critical_words) > 1:
            critical_words = ast.literal_eval(critical_words)

        # get corresponding info about the words
        critical_words_info = the_frames.Var2_info[i]
        if len(critical_words_info) > 1:
            critical_words_info = ast.literal_eval(critical_words_info)

        # get the bias target of the second variable (gender or SES)
        target_var2 = the_frames.Known_stereotyped_var2[i]

        # Race_x_Gender always has Names, Race_x_SES only sometimes
        if (cat == "Race_x_gender" and not has_proper_name) or the_frames.Names[
            i
        ] != "":
            t_loc = critical_words_info.index(target_var2)
            t_word_gs = critical_words[t_loc]
            non_t_word_gs = critical_words[
                1 - t_loc
            ]  # there's only ever 2 possibilities, take the other one
        # else need to grab relevant info from the vocab
        else:
            t_word_gs = (
                vocab[(vocab.Category == "SES") & (vocab.Info == target_var2)]
                .Name.unique()
                .tolist()
            )
            non_t_word_gs = (
                vocab[(vocab.Category == "SES") & (vocab.Info != target_var2)]
                .Name.unique()
                .tolist()
            )

        # check if this one needs to use proper names
        if has_proper_name:
            # grab possible first names
            first_names_full = names_vocab[names_vocab.First_last == "first"]
            # get first names that match bias target ethnicity
            first_names = first_names_full[
                first_names_full.ethnicity.isin(bias_targets)
            ]
            if cat == "Race_x_gender":
                # further pare down based on gender when needed
                first_names = first_names[first_names.gender == target_var2]
            if cat == "Race_x_SES" and the_frames.Stated_gender_info[i] != "":
                # sometimes the template needs a certain gender because of the way it's written
                first_names = first_names[
                    first_names.gender == the_frames.Stated_gender_info[i]
                ]
            word_list = first_names.Name.tolist()
            word_list = random.sample(word_list, 5)  # for downsampling

        # iterate over each word in word_list
        for j in range(len(word_list)):
            # initialize variables for later
            Name1_info = None
            Name2_info = None
            rand_wrd1 = ""
            rand_wrd2 = ""
            Name2_info_dict = {}

            t_word = word_list[j]

            # create new word list
            new_word_list = [n for n in possible_word_list if n not in word_list]
            new_word_list = random.sample(new_word_list, 5)  # downsample

            # getting proper names
            if has_proper_name:
                # get target name info
                the_word_gender = first_names.loc[
                    first_names["Name"] == t_word, "gender"
                ].iloc[0]
                the_word_eth = first_names.loc[
                    first_names["Name"] == t_word, "ethnicity"
                ].iloc[0]
                Name1_info = the_word_gender + "-" + the_word_eth
                # add appropriate last name
                last_names = names_vocab[
                    (names_vocab.First_last == "last")
                    & (names_vocab.ethnicity == the_word_eth)
                ]
                last_names_list = last_names.Name.unique().tolist()
                this_last_name = random.choice(last_names_list)
                t_word = t_word + " " + this_last_name

                # create lists of non-target names
                if cat == "Race_x_SES":
                    # need to create lists for both same and different races
                    # in this case, always keep the gender the same, since that isn't supposed to vary in these
                    other_first_names_diff_race = first_names_full[
                        (~first_names_full.ethnicity.isin(bias_targets))
                        & (first_names_full.gender == the_word_gender)
                    ].reset_index()
                    other_first_names_diff_race = other_first_names_diff_race.sample(
                        n=5, replace=False
                    ).reset_index()  # for downsampling
                    other_first_names_same_race = first_names_full[
                        (first_names_full.ethnicity == the_word_eth)
                        & (first_names_full.gender == the_word_gender)
                        & (first_names_full.Name != word_list[j])
                    ].reset_index()
                    other_first_names_same_race = other_first_names_same_race.sample(
                        n=5, replace=False
                    ).reset_index()  # for downsampling

                    all_last_names = names_vocab[
                        (names_vocab.First_last == "last")
                    ].reset_index()

                    # initialize empty list and dict to store names and the important info about those names
                    new_word_list = []
                    Name2_info_dict = {}
                    for nam in range(len(other_first_names_same_race)):
                        # initialize new list at each iteration to contain two names, one that's the same race
                        # as the target and one that's different
                        same_diff = []

                        frst_1 = other_first_names_same_race.Name[nam]
                        eth_1 = other_first_names_same_race.ethnicity[nam]
                        gen_1 = other_first_names_same_race.gender[nam]
                        Name2_info_1 = gen_1 + "-" + eth_1
                        lst_list_1 = (
                            all_last_names[all_last_names.ethnicity == eth_1]
                            .Name.unique()
                            .tolist()
                        )
                        lst_1 = random.choice(lst_list_1)
                        same_diff.append(frst_1 + " " + lst_1)
                        Name2_info_dict[frst_1 + " " + lst_1] = Name2_info_1

                        frst_2 = other_first_names_diff_race.Name[nam]
                        eth_2 = other_first_names_diff_race.ethnicity[nam]
                        gen_2 = other_first_names_diff_race.gender[nam]
                        Name2_info_2 = gen_2 + "-" + eth_2
                        lst_list_2 = (
                            all_last_names[all_last_names.ethnicity == eth_2]
                            .Name.unique()
                            .tolist()
                        )
                        lst_2 = random.choice(lst_list_2)
                        same_diff.append(frst_2 + " " + lst_2)
                        Name2_info_dict[frst_2 + " " + lst_2] = Name2_info_2

                        new_word_list.append(same_diff)

                if cat == "Race_x_gender":
                    # need to create separate lists of names that match the three possible conditions
                    # a) same race, diff gender
                    other_first_names_a = first_names_full[
                        (first_names_full.ethnicity == the_word_eth)
                        & (first_names_full.gender != the_word_gender)
                    ].reset_index()
                    other_first_names_a = other_first_names_a.sample(
                        n=5, replace=False
                    ).reset_index()  # for downsampling

                    # b) same gender, diff race
                    other_first_names_b = first_names_full[
                        (first_names_full.ethnicity != the_word_eth)
                        & (first_names_full.gender == the_word_gender)
                    ].reset_index()
                    other_first_names_b = other_first_names_b.sample(
                        n=5, replace=False
                    ).reset_index()  # for downsampling

                    # c) diff race, diff gender
                    other_first_names_c = first_names_full[
                        (first_names_full.ethnicity != the_word_eth)
                        & (first_names_full.gender != the_word_gender)
                    ].reset_index()
                    other_first_names_c = other_first_names_c.sample(
                        n=5, replace=False
                    ).reset_index()  # for downsampling

                    all_last_names = names_vocab[
                        (names_vocab.First_last == "last")
                    ].reset_index()

                    # initialize empty list and dict to hold new names and info about them
                    new_word_list = []
                    Name2_info_dict = {}
                    for nam in range(len(other_first_names_a)):
                        # need a list that stores three names, one for each of the three types of variation
                        abc = []

                        frst_a = other_first_names_a.Name[nam]
                        eth_a = other_first_names_a.ethnicity[nam]
                        gen_a = other_first_names_a.gender[nam]
                        Name2_info_a = gen_a + "-" + eth_a
                        lst_list_a = (
                            all_last_names[all_last_names.ethnicity == eth_a]
                            .Name.unique()
                            .tolist()
                        )
                        lst_a = random.choice(lst_list_a)
                        abc.append(frst_a + " " + lst_a)
                        Name2_info_dict[frst_a + " " + lst_a] = Name2_info_a

                        frst_b = other_first_names_b.Name[nam]
                        eth_b = other_first_names_b.ethnicity[nam]
                        gen_b = other_first_names_b.gender[nam]
                        Name2_info_b = gen_b + "-" + eth_b
                        lst_list_b = (
                            all_last_names[all_last_names.ethnicity == eth_b]
                            .Name.unique()
                            .tolist()
                        )
                        lst_b = random.choice(lst_list_b)
                        abc.append(frst_b + " " + lst_b)
                        Name2_info_dict[frst_b + " " + lst_b] = Name2_info_b

                        frst_c = other_first_names_c.Name[nam]
                        eth_c = other_first_names_c.ethnicity[nam]
                        gen_c = other_first_names_c.gender[nam]
                        Name2_info_c = gen_c + "-" + eth_c
                        lst_list_c = (
                            all_last_names[all_last_names.ethnicity == eth_c]
                            .Name.unique()
                            .tolist()
                        )
                        lst_c = random.choice(lst_list_c)
                        abc.append(frst_c + " " + lst_c)
                        Name2_info_dict[frst_c + " " + lst_c] = Name2_info_c

                        new_word_list.append(abc)

            name1_info_static = Name1_info
            # iterate over each word in the second word list
            # results in full pairings of every possible vocab pair
            for k in range(len(new_word_list)):
                word2_race = new_word_list[k]

                if the_frames.Names[i] == "" and cat == "Race_x_SES":
                    # need to recreate these to grab a new item in the next set
                    t_word_gs = (
                        vocab[(vocab.Category == "SES") & (vocab.Info == target_var2)]
                        .Name.unique()
                        .tolist()
                    )
                    non_t_word_gs = (
                        vocab[(vocab.Category == "SES") & (vocab.Info != target_var2)]
                        .Name.unique()
                        .tolist()
                    )
                # get word or word subset
                if isinstance(t_word_gs, list) and len(t_word_gs) > 0:
                    t_word_gs = random.choice(t_word_gs)
                    non_t_word_gs = random.choice(non_t_word_gs)

                # get the second word for when race is held constant
                word2_a_race = t_word
                word2_a_var2 = non_t_word_gs

                # get the second word for when gender/ses is held constant
                word2_b_race = word2_race
                word2_b_var2 = t_word_gs

                # get the second word for when both gender/ses and race change
                word2_c_race = word2_race
                word2_c_var2 = non_t_word_gs

                this_frame_row = the_frames.iloc[[i]].reset_index()
                lex_div = this_frame_row.Lexical_diversity[0]

                # Only need to create these values when there's text in lexical diversity
                if len(lex_div) > 1:
                    wrdlist1, wrdlist2 = return_list_from_string(lex_div)
                    rand_wrd1 = random.choice(wrdlist1)
                    if len(wrdlist2) > 1:  # sometimes there's not a WORD2
                        rand_wrd2 = random.choice(wrdlist2)

                # need to record info about the names that were used for easier analysis later
                if cat == "Race_x_gender":
                    if Name1_info is not None and has_proper_name:
                        Name2_info_a = Name2_info_dict[word2_race[0]]
                        Name2_info_b = Name2_info_dict[word2_race[1]]
                        Name2_info_c = Name2_info_dict[word2_race[2]]
                    # covers all Race_x_gender and Race_x_SES cases where Names is specified
                    elif this_frame_row.Var2_info[0] != "":
                        if target_var2 == "M":
                            nontarget_var2 = "F"
                        elif target_var2 == "F":
                            nontarget_var2 = "M"
                        Name1_info = target_var2 + "-" + t_word  # add gender/SES info
                        Name2_info_a = nontarget_var2 + "-" + t_word  # add gender/SES info
                        Name2_info_b = target_var2 + "-" + word2_race  # add gender/SES info
                        Name2_info_c = nontarget_var2 + "-" + word2_race  # add gender/SES info
                if cat == "Race_x_SES":
                    if this_frame_row.Var2_info[0] != "":
                        if target_var2 == "lowSES":
                            nontarget_var2 = "highSES"
                        elif target_var2 == "highSES":
                            nontarget_var2 = "lowSES"

                        if Name1_info is not None and has_proper_name:
                            Name1_info = (
                                target_var2 + "-" + name1_info_static
                            )  # add gender/SES info
                            Name2_info_a = (
                                nontarget_var2 + "-" + Name2_info_dict[word2_race[0]]
                            )  # add gender/SES info
                            Name2_info_b = (
                                target_var2 + "-" + Name2_info_dict[word2_race[1]]
                            )  # add gender/SES info
                            Name2_info_c = (
                                nontarget_var2 + "-" + Name2_info_dict[word2_race[1]]
                            )  # add gender/SES info
                        else:
                            Name1_info = target_var2 + "-" + t_word  # add gender/SES info
                            Name2_info_a = nontarget_var2 + "-" + t_word  # add gender/SES info
                            Name2_info_b = (
                                target_var2 + "-" + word2_race
                            )  # add gender/SES info
                            Name2_info_c = (
                                nontarget_var2 + "-" + word2_race
                            )  # add gender/SES info
                    else:
                        # there's only two options for low/high SES, so store the nontarget name as the other one
                        if target_var2 == "lowSES":
                            nontarget_var2 = "highSES"
                        elif target_var2 == "highSES":
                            nontarget_var2 = "lowSES"
                        if Name1_info is not None and has_proper_name:
                            Name1_info = (
                                target_var2 + "-" + name1_info_static
                            )  # add gender/SES info
                            Name2_info_a = (
                                nontarget_var2 + "-" + Name2_info_dict[word2_race[0]]
                            )  # add gender/SES info
                            Name2_info_b = (
                                target_var2 + "-" + Name2_info_dict[word2_race[1]]
                            )  # add gender/SES info
                            Name2_info_c = (
                                nontarget_var2 + "-" + Name2_info_dict[word2_race[1]]
                            )  # add gender/SES info
                        else:
                            Name1_info = (
                                target_var2 + "-" + t_word
                            )  # add gender/SES info
                            Name2_info_a = (
                                nontarget_var2 + "-" + t_word
                            )  # add gender/SES info
                            Name2_info_b = (
                                target_var2 + "-" + word2_race
                            )  # add gender/SES info
                            Name2_info_c = (
                                nontarget_var2 + "-" + word2_race
                            )  # add gender/SES info

                # replace frame text info with value of {{NAME}} and {{WORD}}.
                # then set everything into the formatting needed to save the data
                # then repeat with names in the reverse order
                # this has to be done slightly differently depending on which category is being used and whether
                # the template uses a proper name. No matter what, there's just a handful of fiddly differences
                # in the way the names are stored, so each one is handled a little differently using the same functions,
                # but this gets very repetitive.
                if not has_proper_name:
                    new_frame_row_a = do_slotting(
                        this_frame_row,
                        frame_cols,
                        t_word,
                        t_word_gs,
                        word2_a_race,
                        word2_a_var2,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_b = do_slotting(
                        this_frame_row,
                        frame_cols,
                        t_word,
                        t_word_gs,
                        word2_b_race,
                        word2_b_var2,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_c = do_slotting(
                        this_frame_row,
                        frame_cols,
                        t_word,
                        t_word_gs,
                        word2_c_race,
                        word2_c_var2,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )

                    # create four sets of data, each as a dictionary
                    dat_formatted_a = create_templating_dicts(
                        cat,
                        new_frame_row_a,
                        "intersectional_a",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [t_word, t_word_gs],
                        [word2_a_race, word2_a_var2],
                        Name1_info,
                        Name2_info_a,
                        nn,
                    )
                    nn += 4
                    dat_formatted_b = create_templating_dicts(
                        cat,
                        new_frame_row_b,
                        "intersectional_b",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [t_word, t_word_gs],
                        [word2_b_race, word2_b_var2],
                        Name1_info,
                        Name2_info_b,
                        nn,
                    )
                    nn += 4
                    dat_formatted_c = create_templating_dicts(
                        cat,
                        new_frame_row_c,
                        "intersectional_c",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [t_word, t_word_gs],
                        [word2_c_race, word2_c_var2],
                        Name1_info,
                        Name2_info_c,
                        nn,
                    )
                    nn += 4

                    # now reverse everything for counterbalancing
                    new_frame_row_a_rev = do_slotting(
                        this_frame_row,
                        frame_cols,
                        word2_a_race,
                        word2_a_var2,
                        t_word,
                        t_word_gs,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_b_rev = do_slotting(
                        this_frame_row,
                        frame_cols,
                        word2_b_race,
                        word2_b_var2,
                        t_word,
                        t_word_gs,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_c_rev = do_slotting(
                        this_frame_row,
                        frame_cols,
                        word2_c_race,
                        word2_c_var2,
                        t_word,
                        t_word_gs,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )

                    # create four sets of data, each as a dictionary
                    dat_formatted_a_rev = create_templating_dicts(
                        cat,
                        new_frame_row_a_rev,
                        "intersectional_a",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [word2_a_race, word2_a_var2],
                        [t_word, t_word_gs],
                        Name2_info_a,
                        Name1_info,
                        nn,
                    )
                    nn += 4
                    dat_formatted_b_rev = create_templating_dicts(
                        cat,
                        new_frame_row_b_rev,
                        "intersectional_b",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [word2_b_race, word2_b_var2],
                        [t_word, t_word_gs],
                        Name2_info_b,
                        Name1_info,
                        nn,
                    )
                    nn += 4
                    dat_formatted_c_rev = create_templating_dicts(
                        cat,
                        new_frame_row_c_rev,
                        "intersectional_c",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [word2_c_race, word2_c_var2],
                        [t_word, t_word_gs],
                        Name2_info_c,
                        Name1_info,
                        nn,
                    )
                    nn += 4

                if has_proper_name and cat == "Race_x_gender":
                    new_frame_row_a = do_slotting(
                        this_frame_row,
                        frame_cols,
                        t_word,
                        None,
                        word2_race[0],
                        None,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_b = do_slotting(
                        this_frame_row,
                        frame_cols,
                        t_word,
                        None,
                        word2_race[1],
                        None,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_c = do_slotting(
                        this_frame_row,
                        frame_cols,
                        t_word,
                        None,
                        word2_race[2],
                        None,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    # create four sets of data, each as a dictionary
                    dat_formatted_a = create_templating_dicts(
                        cat,
                        new_frame_row_a,
                        "intersectional_a",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        t_word,
                        word2_race[0],
                        Name1_info,
                        Name2_info_a,
                        nn,
                    )
                    nn += 4
                    dat_formatted_b = create_templating_dicts(
                        cat,
                        new_frame_row_b,
                        "intersectional_b",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        t_word,
                        word2_race[1],
                        Name1_info,
                        Name2_info_b,
                        nn,
                    )
                    nn += 4
                    dat_formatted_c = create_templating_dicts(
                        cat,
                        new_frame_row_c,
                        "intersectional_c",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        t_word,
                        word2_race[2],
                        Name1_info,
                        Name2_info_c,
                        nn,
                    )
                    nn += 4

                    # now reverse them all for counterbalancing
                    new_frame_row_a_rev = do_slotting(
                        this_frame_row,
                        frame_cols,
                        word2_race[0],
                        None,
                        t_word,
                        None,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_b_rev = do_slotting(
                        this_frame_row,
                        frame_cols,
                        word2_race[1],
                        None,
                        t_word,
                        None,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_c_rev = do_slotting(
                        this_frame_row,
                        frame_cols,
                        word2_race[2],
                        None,
                        t_word,
                        None,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    # create four sets of data, each as a dictionary
                    dat_formatted_a_rev = create_templating_dicts(
                        cat,
                        new_frame_row_a_rev,
                        "intersectional_a",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        word2_race[0],
                        t_word,
                        Name2_info_a,
                        Name1_info,
                        nn,
                    )
                    nn += 4
                    dat_formatted_b_rev = create_templating_dicts(
                        cat,
                        new_frame_row_b_rev,
                        "intersectional_b",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        word2_race[1],
                        t_word,
                        Name2_info_b,
                        Name1_info,
                        nn,
                    )
                    nn += 4
                    dat_formatted_c_rev = create_templating_dicts(
                        cat,
                        new_frame_row_c_rev,
                        "intersectional_c",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        word2_race[2],
                        t_word,
                        Name2_info_c,
                        Name1_info,
                        nn,
                    )
                    nn += 4
                if has_proper_name and cat == "Race_x_SES":
                    new_frame_row_a = do_slotting(
                        this_frame_row,
                        frame_cols,
                        t_word,
                        t_word_gs,
                        word2_race[0],
                        word2_a_var2,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_b = do_slotting(
                        this_frame_row,
                        frame_cols,
                        t_word,
                        t_word_gs,
                        word2_race[1],
                        word2_b_var2,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_c = do_slotting(
                        this_frame_row,
                        frame_cols,
                        t_word,
                        t_word_gs,
                        word2_race[1],
                        word2_c_var2,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    # create four sets of data, each as a dictionary
                    dat_formatted_a = create_templating_dicts(
                        cat,
                        new_frame_row_a,
                        "intersectional_a",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [t_word, t_word_gs],
                        [word2_race[0], word2_a_var2],
                        Name1_info,
                        Name2_info_a,
                        nn,
                    )
                    nn += 4
                    dat_formatted_b = create_templating_dicts(
                        cat,
                        new_frame_row_b,
                        "intersectional_b",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [t_word, t_word_gs],
                        [word2_race[1], word2_b_var2],
                        Name1_info,
                        Name2_info_b,
                        nn,
                    )
                    nn += 4
                    dat_formatted_c = create_templating_dicts(
                        cat,
                        new_frame_row_c,
                        "intersectional_c",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [t_word, t_word_gs],
                        [word2_race[1], word2_c_var2],
                        Name1_info,
                        Name2_info_c,
                        nn,
                    )
                    nn += 4

                    # now reverse everything for counterbalancing
                    new_frame_row_a_rev = do_slotting(
                        this_frame_row,
                        frame_cols,
                        word2_race[0],
                        word2_a_var2,
                        t_word,
                        t_word_gs,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_b_rev = do_slotting(
                        this_frame_row,
                        frame_cols,
                        word2_race[1],
                        word2_b_var2,
                        t_word,
                        t_word_gs,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    new_frame_row_c_rev = do_slotting(
                        this_frame_row,
                        frame_cols,
                        word2_race[1],
                        word2_c_var2,
                        t_word,
                        t_word_gs,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    # create four sets of data, each as a dictionary
                    dat_formatted_a_rev = create_templating_dicts(
                        cat,
                        new_frame_row_a_rev,
                        "intersectional_a",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [word2_race[0], word2_a_var2],
                        [t_word, t_word_gs],
                        Name2_info_a,
                        Name1_info,
                        nn,
                    )
                    nn += 4
                    dat_formatted_b_rev = create_templating_dicts(
                        cat,
                        new_frame_row_b_rev,
                        "intersectional_b",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [word2_race[1], word2_b_var2],
                        [t_word, t_word_gs],
                        Name2_info_b,
                        Name1_info,
                        nn,
                    )
                    nn += 4
                    dat_formatted_c_rev = create_templating_dicts(
                        cat,
                        new_frame_row_c_rev,
                        "intersectional_c",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [word2_race[1], word2_c_var2],
                        [t_word, t_word_gs],
                        Name2_info_c,
                        Name1_info,
                        nn,
                    )
                    nn += 4

                for dat_formatted in [
                    dat_formatted_a,
                    dat_formatted_b,
                    dat_formatted_c,
                    dat_formatted_a_rev,
                    dat_formatted_b_rev,
                    dat_formatted_c_rev,
                ]:
                    for item in dat_formatted:
                        dat_file.write(json.dumps(item, default=str))
                        dat_file.write("\n")
                    dat_file.flush()

        print("generated %s sentences total for %s" % (str(nn), cat))

    dat_file.close()
